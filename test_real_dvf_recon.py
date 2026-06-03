"""
De-risk 1: does the DVF VAE reconstruct a REAL respiratory DVF (not synthetic)?
Compute an independent DIR (SimpleITK demons) between two real 4D-Lung phase
volumes, convert to the VAE's normalized-grid units, round-trip through the VAE,
and measure reconstruction corr + folding. If corr is high and the input/recon
are both diffeomorphic, the VAE faithfully represents real motion structure.
"""
import os, sys, json, glob
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk

LDM = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LDM)
from args.classes import EncoderArgs
from backbones.models import VariationalAutoencoder
from inference.utils.load_models import revert_model_weight_names

D, H, W = 50, 256, 256


def load_vae():
    base = os.path.join(LDM, "pretrained_models", "dvf_vae")
    margs = json.load(open(os.path.join(base, "args.json")))["model_args"]
    vae = VariationalAutoencoder(encoder_args=EncoderArgs(**margs))
    vae.load_state_dict(revert_model_weight_names(torch.load(
        os.path.join(base, "vae.pt"), map_location="cpu", weights_only=True)), strict=False)
    vae.eval(); vae.sampling_layer.identity_sampling = True
    return vae, float(margs["dvf_scale_factor"])


def find_pair():
    """Find a study with both a 0.0% and a 50.0% volume."""
    studies = glob.glob(os.path.join(LDM, "data", "idc_downloads", "patient_100", "*"))
    for st in studies:
        v0 = glob.glob(os.path.join(st, "*_0.0%*", "volume.pt"))
        v5 = glob.glob(os.path.join(st, "*_50.0%*", "volume.pt"))
        if v0 and v5:
            return v0[0], v5[0]
    raise RuntimeError("no 0%/50% pair found")


def to50(vol):
    vol = vol.squeeze().float()
    if vol.max() > 10:
        vol = torch.clamp(vol, -1024, 3071); vol = 2 * (vol + 1024) / 4095 - 1
    if tuple(vol.shape) != (D, H, W):
        vol = F.interpolate(vol[None, None], size=(D, H, W), mode="trilinear", align_corners=False)[0, 0]
    return vol


def sitk_demons(fixed_np, moving_np):
    """Demons DIR (spacing=1 -> displacement in voxels). fixed/moving: numpy [D,H,W]."""
    f = sitk.GetImageFromArray(fixed_np.astype(np.float32))
    m = sitk.GetImageFromArray(moving_np.astype(np.float32))
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(256); matcher.SetNumberOfMatchPoints(16)
    matcher.ThresholdAtMeanIntensityOn()
    m = matcher.Execute(m, f)
    demons = sitk.FastSymmetricForcesDemonsRegistrationFilter()
    demons.SetNumberOfIterations(60)
    demons.SetStandardDeviations(2.0)
    field = demons.Execute(f, m)            # vector image, components (x,y,z), voxel units
    arr = sitk.GetArrayFromImage(field)     # [D,H,W,3], comps (dx,dy,dz)
    return arr


def folding_and_mag(dvf_norm):
    u = torch.stack([dvf_norm[0, 0] * (D - 1) / 2, dvf_norm[0, 1] * (H - 1) / 2,
                     dvf_norm[0, 2] * (W - 1) / 2], 0)

    def d(t, ax):
        return 0.5 * (torch.roll(t, -1, ax) - torch.roll(t, 1, ax))
    det = ((1 + d(u[0], 0)) * ((1 + d(u[1], 1)) * (1 + d(u[2], 2)) - d(u[1], 2) * d(u[2], 1))
           - d(u[0], 1) * (d(u[1], 0) * (1 + d(u[2], 2)) - d(u[1], 2) * d(u[2], 0))
           + d(u[0], 2) * (d(u[1], 0) * d(u[2], 1) - (1 + d(u[1], 1)) * d(u[2], 0)))
    inner = det[2:-2, 2:-2, 2:-2]
    return float((inner <= 0).float().mean()), u.abs().max().item()


def corr(a, b):
    a = a.flatten() - a.flatten().mean(); b = b.flatten() - b.flatten().mean()
    return float((a @ b / (a.norm() * b.norm() + 1e-12)))


def main():
    vae, sf = load_vae()
    p0, p5 = find_pair()
    print("registering (moving=0%, fixed=50%):\n  ", os.path.relpath(p5, LDM))
    fixed = to50(torch.load(p5, map_location="cpu")).numpy()
    moving = to50(torch.load(p0, map_location="cpu")).numpy()
    arr = sitk_demons(fixed, moving)        # [D,H,W,3], (dx,dy,dz) voxels

    # to torch [1,3,D,H,W] channels (dz,dy,dx) in voxels
    dz = torch.from_numpy(arr[..., 2]); dy = torch.from_numpy(arr[..., 1]); dx = torch.from_numpy(arr[..., 0])
    dvf_vox = torch.stack([dz, dy, dx], 0)[None].float()
    # voxel -> normalized grid (per axis)
    dvf_norm = torch.stack([dvf_vox[0, 0] * 2 / (D - 1), dvf_vox[0, 1] * 2 / (H - 1),
                            dvf_vox[0, 2] * 2 / (W - 1)], 0)[None]
    fin, magin = folding_and_mag(dvf_norm)
    print("\nREAL DVF (SimpleITK demons): norm|max|={:.3f}  vox|max|={:.2f}  folding={:.5f}  "
          "mean|u|_vox={:.2f}".format(dvf_norm.abs().max().item(), magin, fin,
          (dvf_vox[0, 0].pow(2) + dvf_vox[0, 1].pow(2) + dvf_vox[0, 2].pow(2)).sqrt().mean().item()))

    with torch.no_grad():
        z, _ = vae.encode(dvf_norm * sf)
        rec = vae.decode(z) / sf
    frec, magrec = folding_and_mag(rec)
    print("\nVAE reconstruction of the REAL DVF:")
    print("  recon_corr={:.4f}  z_std={:.3f}".format(corr(dvf_norm, rec), z.std().item()))
    print("  recon folding={:.5f}  recon vox|max|={:.2f}  out/in mag={:.3f}".format(
        frec, magrec, rec.abs().max().item() / (dvf_norm.abs().max().item() + 1e-9)))
    print("\n(synthetic-smooth recon was ~0.97; this is the REAL-motion check)")


if __name__ == "__main__":
    main()
