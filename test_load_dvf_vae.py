"""
Re-test the DVF VAE with CORRECT units.

The repo's DVFs are airlab/grid-normalized ([-1,1] coords: warp_image does
grid_sample(img, dvf+grid)), then multiplied by dvf_scale_factor (=10). So the
VAE input lives at ~ normalized_disp * 10, i.e. a realistic motion of ~0.05-0.15
normalized -> VAE input ~0.5-1.5. My first test fed voxel-scale (|max|~100) * 10
-> ~1000x too large, which saturated it. Redo at the right scale.
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
enc_args = EncoderArgs(**margs)
vae = VariationalAutoencoder(encoder_args=enc_args)
sd = revert_model_weight_names(torch.load(os.path.join(base, "vae.pt"),
                                          map_location="cpu", weights_only=True))
vae.load_state_dict(sd, strict=False)
vae.eval()
vae.sampling_layer.identity_sampling = True
sf = float(margs["dvf_scale_factor"])


def corr(a, b):
    a = a.flatten().float() - a.flatten().float().mean()
    b = b.flatten().float() - b.flatten().float().mean()
    return float((a @ b / (a.norm() * b.norm() + 1e-12)))


def smooth_norm_dvf(D, H, W, maxmag, coarse=8):
    """Low-frequency (plausible) field in NORMALIZED grid units, no edge spikes."""
    c = torch.randn(1, 3, max(2, D // 4), coarse, coarse)
    v = F.interpolate(c, size=(D, H, W), mode="trilinear", align_corners=True)
    v = v / (v.abs().max() + 1e-9) * maxmag
    return v


D, H, W = 50, 256, 256
print("RECON in correct (normalized) units, sf={}:".format(sf))
print("  norm|max|  VAEin|max|  recon_corr  out/in_mag  z_std")
for mm in [0.02, 0.05, 0.10, 0.20, 0.40]:
    dvf = smooth_norm_dvf(D, H, W, mm)
    with torch.no_grad():
        z, _ = vae.encode(dvf * sf)
        rec = vae.decode(z) / sf
    print("   {:5.2f}     {:6.2f}      {:.4f}      {:.3f}       {:.2f}".format(
        mm, (dvf * sf).abs().max().item(), corr(dvf, rec),
        rec.abs().max().item() / (dvf.abs().max().item() + 1e-9), z.std().item()))

print("\nself-consistency (decode prior latent -> encode -> decode):")
for zs in [0.5, 1.0, 2.0, 4.0]:
    with torch.no_grad():
        z0 = zs * torch.randn(1, 9, D, 32, 32)
        d1 = vae.decode(z0)
        z1, _ = vae.encode(d1)
        d2 = vae.decode(z1)
    print("  z_std={:.1f}  decoded|max|(VAE)={:.2f}  roundtrip_corr={:.4f}".format(
        zs, d1.abs().max().item(), corr(d1, d2)))

print("\nlatent-ball decode (encode a 0.1-norm field -> perturb z):")
dvf = smooth_norm_dvf(D, H, W, 0.10)
with torch.no_grad():
    z0, _ = vae.encode(dvf * sf)
zstd = z0.std().item()
base_dec = vae.decode(z0) / sf
for rel in [0.0, 0.25, 0.5, 1.0]:
    with torch.no_grad():
        dec = vae.decode(z0 + rel * zstd * torch.randn_like(z0)) / sf
    print("  |dz|={:.2f}*std  decoded norm|max|={:.3f}  corr_to_base={:.3f}".format(
        rel, dec.abs().max().item(), corr(dec, base_dec)))
