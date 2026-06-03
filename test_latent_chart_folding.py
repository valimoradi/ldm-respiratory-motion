"""
THE decisive test for our use: is the DVF VAE latent a CLEAN, plausibility-
preserving chart? Perturb the latent in a ball around a good (low-folding)
encoded DVF and measure FOLDING (det(I+grad u)<=0), not just corr/magnitude.

If folding stays ~0 across a usable ball -> the latent IS a clean chart, and a
latent-space adversary stays diffeomorphic by construction (the property weight
space and convex proxies lacked).
If folding rises with the ball -> the chart leaks; same fundamental problem.
"""
import os, sys, json
import torch
import torch.nn.functional as F

LDM = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LDM)
from args.classes import EncoderArgs
from backbones.models import VariationalAutoencoder
from inference.utils.load_models import revert_model_weight_names

torch.manual_seed(0)
base = os.path.join(LDM, "pretrained_models", "dvf_vae")
margs = json.load(open(os.path.join(base, "args.json")))["model_args"]
vae = VariationalAutoencoder(encoder_args=EncoderArgs(**margs))
vae.load_state_dict(revert_model_weight_names(torch.load(os.path.join(base, "vae.pt"),
                    map_location="cpu", weights_only=True)), strict=False)
vae.eval(); vae.sampling_layer.identity_sampling = True
sf = float(margs["dvf_scale_factor"])
D, H, W = 50, 256, 256


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


def smooth_norm_dvf(maxmag, coarse=8):
    c = torch.randn(1, 3, D // 4, coarse, coarse)
    v = F.interpolate(c, size=(D, H, W), mode="trilinear", align_corners=True)
    return v / (v.abs().max() + 1e-9) * maxmag


# encode a realistic, low-folding smooth DVF -> z0
dvf0 = smooth_norm_dvf(0.10)
with torch.no_grad():
    z0, _ = vae.encode(dvf0 * sf)
    base_dec = vae.decode(z0) / sf
zstd = z0.std().item()
f0, m0 = folding_and_mag(base_dec)
print("base decoded DVF: folding={:.5f}  vox|max|={:.2f}  z_std={:.3f}".format(f0, m0, zstd))

print("\nLATENT-BALL FOLDING (perturb z0 by rel*z_std, 3 seeds each):")
print("  rel    folding(mean/max over seeds)   vox|max|   corr_to_base")
for rel in [0.25, 0.5, 1.0, 1.5, 2.0]:
    fs, ms, cs = [], [], []
    for s in range(3):
        torch.manual_seed(1000 + s)
        with torch.no_grad():
            dec = vae.decode(z0 + rel * zstd * torch.randn_like(z0)) / sf
        f, m = folding_and_mag(dec)
        fs.append(f); ms.append(m); cs.append(corr(dec, base_dec))
    print("  {:.2f}   {:.5f} / {:.5f}            {:.1f}      {:.3f}".format(
        rel, sum(fs) / 3, max(fs), sum(ms) / 3, sum(cs) / 3))

print("\n(plausibility threshold used elsewhere in project: folding < 0.005)")
