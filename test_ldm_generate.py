"""
DECISIVE TEST: does the LDM actually generate PLAUSIBLE, DIVERSE respiratory
motion? Run the full pipeline (image AE encode -> DDIM UNet -> DVF VAE decode)
on a REAL 4D-Lung CT and measure:
  - magnitude (normalized grid units + converted to voxels)
  - smoothness
  - folding fraction det(I + grad u) <= 0  (diffeomorphism / plausibility)
  - DIVERSITY: different noise -> different but still-plausible DVFs?
No data-pipeline machinery; models loaded directly from pretrained_models/.
"""
import os, sys, json, glob
from math import prod
import torch
import torch.nn.functional as F

LDM = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LDM)
from args.classes import EncoderArgs, DiffusionArgs
from backbones.models import Autoencoder, VariationalAutoencoder, UNet
from inference.utils.load_models import revert_model_weight_names
from diffusion.utils.noise_schedules import get_noise_schedule

device = "cpu"
PM = os.path.join(LDM, "pretrained_models")


def load_ae(folder, fname, variational):
    margs = json.load(open(os.path.join(PM, folder, "args.json")))["model_args"]
    ea = EncoderArgs(**margs)
    m = VariationalAutoencoder(ea) if variational else Autoencoder(ea)
    sd = revert_model_weight_names(torch.load(os.path.join(PM, folder, fname),
                                              map_location=device, weights_only=True))
    miss, unexp = m.load_state_dict(sd, strict=False)
    print("  {:12s} load: missing={} unexpected={}".format(folder, len(miss), len(unexp)))
    m.eval()
    return m, ea


def folding_fraction(dvf_norm, D, H, W):
    """frac of voxels with det(I + grad u_vox) <= 0. dvf_norm: [1,3,D,H,W] (z,y,x), normalized grid units."""
    # convert normalized -> voxel displacement (align_corners convention: d_vox = d_norm*(size-1)/2)
    u = torch.stack([dvf_norm[0, 0] * (D - 1) / 2.0,
                     dvf_norm[0, 1] * (H - 1) / 2.0,
                     dvf_norm[0, 2] * (W - 1) / 2.0], 0)  # [3,D,H,W]

    def d(t, ax):
        return 0.5 * (torch.roll(t, -1, ax) - torch.roll(t, 1, ax))
    j11 = 1 + d(u[0], 0); j12 = d(u[0], 1); j13 = d(u[0], 2)
    j21 = d(u[1], 0); j22 = 1 + d(u[1], 1); j23 = d(u[1], 2)
    j31 = d(u[2], 0); j32 = d(u[2], 1); j33 = 1 + d(u[2], 2)
    det = (j11 * (j22 * j33 - j23 * j32) - j12 * (j21 * j33 - j23 * j31)
           + j13 * (j21 * j32 - j22 * j31))
    inner = det[2:-2, 2:-2, 2:-2]
    return float((inner <= 0).float().mean()), u.abs().max().item()


def corr(a, b):
    a = a.flatten() - a.flatten().mean(); b = b.flatten() - b.flatten().mean()
    return float((a @ b / (a.norm() * b.norm() + 1e-12)))


def main():
    print("Loading models...")
    img_ae, _ = load_ae("image_autoenc", "model.pt", variational=False)
    dvf_vae, dvf_ea = load_ae("dvf_vae", "vae.pt", variational=True)
    dvf_vae.sampling_layer.identity_sampling = True
    dargs = json.load(open(os.path.join(PM, "ldm", "args.json")))["model_args"]
    da = DiffusionArgs(**dargs)
    unet = UNet(da)
    sd = revert_model_weight_names(torch.load(os.path.join(PM, "ldm", "ldm.pt"),
                                              map_location=device, weights_only=True))
    miss, unexp = unet.load_state_dict(sd, strict=False)
    print("  unet         load: missing={} unexpected={}".format(len(miss), len(unexp)))
    unet.eval()
    sf = da.dvf_scale_factor

    # ---- real CT: a 0.0% (reference/EOI) volume from 4D-Lung ----
    cands = glob.glob(os.path.join(LDM, "data", "idc_downloads", "patient_100",
                                   "**", "*0.0%*", "volume.pt"), recursive=True)
    assert cands, "no 0.0% volume found"
    vol = torch.load(cands[0], map_location=device)
    print("\nCT volume: {}  shape={}  range=[{:.1f},{:.1f}]".format(
        os.path.relpath(cands[0], LDM), tuple(vol.shape), vol.min().item(), vol.max().item()))
    vol = vol.squeeze().float()
    if vol.max() > 10:   # looks like HU -> normalize to [-1,1]
        vol = torch.clamp(vol, -1024, 3071); vol = 2 * (vol + 1024) / (3071 + 1024) - 1
    if tuple(vol.shape) != (50, 256, 256):
        vol = F.interpolate(vol[None, None], size=(50, 256, 256), mode="trilinear",
                            align_corners=False)[0, 0]
    ct = vol[None, None]   # [1,1,50,256,256]
    with torch.no_grad():
        img_lat = img_ae.encode(ct)
    while img_lat.dim() > 5:
        img_lat = img_lat.squeeze(0)
    print("image latent: {}".format(tuple(img_lat.shape)))

    # ---- DDIM sampler ----
    T = 1000; delta_t = 20; eps = 1e-8
    beta = get_noise_schedule(T, da.noise_schedule)
    alpha = [1 - beta[i] for i in range(T)]
    ps_alpha = [prod(alpha[:i + 1]) for i in range(T)]
    D, W = da.image_depth, da.image_width

    def sample(phase_idx, seed):
        torch.manual_seed(seed)
        noise = torch.randn(1, da.out_channels, D, W, W)
        x = torch.cat([img_lat, noise], dim=1)            # [1,12,50,32,32]
        phase = torch.tensor([phase_idx])
        for t in range(T, 0, -delta_t):
            with torch.no_grad():
                pred = unet(x, torch.tensor([t]) / T, phase)
            tn = t - delta_t
            psa = ps_alpha[t - 1]; psan = ps_alpha[tn] if tn >= 1 else 1.0
            xsf = (psan / (psa + eps)) ** 0.5
            nf = (1 - psan) ** 0.5 - ((1 - psa) * psan / (psa + eps)) ** 0.5
            x = torch.cat([x[:, :3], x[:, 3:] * xsf + nf * pred], dim=1)
        with torch.no_grad():
            dvf = dvf_vae.decode(x[:, 3:]) / sf            # normalized-grid DVF [1,3,50,256,256]
        return dvf

    print("\nGenerating DVFs (phase=4, EOE-ish), DDIM {} steps...".format(T // delta_t))
    dvfs = []
    for s in range(3):
        dvf = sample(phase_idx=4, seed=100 + s)
        fn, maxvox = folding_fraction(dvf, 50, 256, 256)
        nmax = dvf.abs().max().item()
        # roughness (in-plane smoothness proxy): mean |laplacian| / mean|u|
        rough = (dvf - 0.25 * (torch.roll(dvf, 1, 3) + torch.roll(dvf, -1, 3)
                               + torch.roll(dvf, 1, 4) + torch.roll(dvf, -1, 4)) ).abs().mean().item()
        print("  sample {}: norm|max|={:.3f}  vox|max|={:.2f}  folding_frac={:.5f}  "
              "mean|u|_vox={:.2f}  roughness={:.4f}".format(
                  s, nmax, maxvox, fn,
                  ( (dvf[0,0]*(50-1)/2).pow(2)+(dvf[0,1]*255/2).pow(2)+(dvf[0,2]*255/2).pow(2)
                   ).sqrt().mean().item(), rough))
        dvfs.append(dvf)

    print("\nDIVERSITY (pairwise corr between samples, same phase/CT, diff noise):")
    for i in range(3):
        for j in range(i + 1, 3):
            print("  corr(sample{}, sample{}) = {:.3f}".format(i, j, corr(dvfs[i], dvfs[j])))

    print("\nPHASE dependence (corr of DVF at phase p vs phase 0, same noise):")
    base = sample(phase_idx=0, seed=7)
    for p in [0, 4, 8]:
        d = sample(phase_idx=p, seed=7)
        print("  phase {}: vox|max|={:.2f}  corr_to_phase0={:.3f}".format(
            p, folding_fraction(d, 50, 256, 256)[1], corr(d, base)))


if __name__ == "__main__":
    main()
