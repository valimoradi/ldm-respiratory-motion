"""
De-risk 2 (part b): domain gap. Round-trip the DynaGAN P7 DVF (saved by
run_save_p7_dvf.py) through the 4D-Lung-trained DVF VAE. If recon corr is high
(near the in-domain 0.99) and folding stays ~0, P7 is usable with this VAE;
if poor, the domain gap argues for running on a 4D-Lung patient instead.
"""
import os, sys, json
import torch
import torch.nn.functional as F

LDM = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LDM)
from args.classes import EncoderArgs
from backbones.models import VariationalAutoencoder
from inference.utils.load_models import revert_model_weight_names

D, H, W = 50, 256, 256
P7 = os.path.join(os.path.dirname(os.path.dirname(LDM)), "results", "sro", "p7_nominal_dvf.pt")


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


base = os.path.join(LDM, "pretrained_models", "dvf_vae")
margs = json.load(open(os.path.join(base, "args.json")))["model_args"]
vae = VariationalAutoencoder(encoder_args=EncoderArgs(**margs))
vae.load_state_dict(revert_model_weight_names(torch.load(os.path.join(base, "vae.pt"),
                    map_location="cpu", weights_only=True)), strict=False)
vae.eval(); vae.sampling_layer.identity_sampling = True
sf = float(margs["dvf_scale_factor"])

dvf_vox = torch.load(P7, map_location="cpu").float()    # [1,3,Dp,256,256] voxel units (dz,dy,dx)
Dp = dvf_vox.shape[2]
# voxel -> normalized grid using the ORIGINAL grid size (normalized is grid-relative)
dvf_norm = torch.stack([dvf_vox[0, 0] * 2 / (Dp - 1), dvf_vox[0, 1] * 2 / (H - 1),
                        dvf_vox[0, 2] * 2 / (W - 1)], 0)[None]
# resample normalized field to the VAE's 50-slice depth (normalized values are grid-relative)
dvf_norm = F.interpolate(dvf_norm, size=(D, H, W), mode="trilinear", align_corners=True)

fin, magin = folding_and_mag(dvf_norm)
print("P7 DynaGAN DVF -> normalized grid (resampled to {} slices):".format(D))
print("  norm|max|={:.3f}  vox-equiv|max|={:.2f}  folding={:.5f}".format(
    dvf_norm.abs().max().item(), magin, fin))
print("  (note: P7 norm|max| is small ~ low end of the VAE's training range)")

with torch.no_grad():
    z, _ = vae.encode(dvf_norm * sf)
    rec = vae.decode(z) / sf
frec, magrec = folding_and_mag(rec)
print("\nVAE reconstruction of the P7 DVF (4D-Lung-trained VAE):")
print("  recon_corr={:.4f}   z_std={:.3f}".format(corr(dvf_norm, rec), z.std().item()))
print("  recon folding={:.5f}  out/in mag={:.3f}".format(frec, rec.abs().max().item() / (dvf_norm.abs().max().item() + 1e-9)))
print("\nCOMPARE: in-domain real 4D-Lung DVF recon was corr 0.989.")
