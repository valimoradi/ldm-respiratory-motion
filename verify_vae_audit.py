"""
Rigorous audit of the pretrained models — BEFORE building AGRO.

Goals (per user: "make sure the model really does what we expect and it has 70+M
parameters"):
  1. GROUND TRUTH: count parameters directly from each .pt checkpoint (sum of
     numel over every tensor), independent of any model construction. This is the
     true size of the saved model and cannot be fooled by strict=False.
  2. CONSTRUCT each model from its args.json and load with strict=True (the way
     the repo's own load_models.py does it). Report missing / unexpected keys.
     If strict=True succeeds with 0/0, the architecture matches the checkpoint
     exactly and the constructed param count == checkpoint param count.
  3. FUNCTION: DVF VAE encode->decode round-trip (correct normalized units),
     latent shape, reconstruction corr, folding.
  4. RECONCILE: report DVF VAE / image AE / diffusion sizes and the TOTAL, so we
     know what "70M+" refers to.
"""
import os, sys, json
import torch
import torch.nn.functional as F

LDM = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LDM)
from args.classes import EncoderArgs, DiffusionArgs
from backbones.models import VariationalAutoencoder, Autoencoder, UNet
from inference.utils.load_models import revert_model_weight_names

PM = os.path.join(LDM, "pretrained_models")


def ckpt_param_count(path):
    """Ground-truth: sum numel over every tensor in the saved checkpoint."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = revert_model_weight_names(sd)
    total = 0
    n_tensors = 0
    for k, v in sd.items():
        if torch.is_tensor(v):
            total += v.numel()
            n_tensors += 1
    return total, n_tensors, sd


def model_param_count(m):
    return sum(p.numel() for p in m.parameters())


def fmt(n):
    return "{:,} ({:.2f}M)".format(n, n / 1e6)


def strict_load_report(model, sd, name):
    """Load strictly; report what doesn't line up."""
    msd = model.state_dict()
    model_keys = set(msd.keys())
    ckpt_keys = set(sd.keys())
    missing = model_keys - ckpt_keys      # in model, absent from ckpt
    unexpected = ckpt_keys - model_keys   # in ckpt, absent from model
    # shape mismatches among shared keys
    shape_mismatch = []
    for k in (model_keys & ckpt_keys):
        if tuple(msd[k].shape) != tuple(sd[k].shape):
            shape_mismatch.append((k, tuple(msd[k].shape), tuple(sd[k].shape)))
    print(f"  [{name}] strict-load check:")
    print(f"    model keys={len(model_keys)}  ckpt keys={len(ckpt_keys)}")
    print(f"    missing (in model, not ckpt)   = {len(missing)}")
    print(f"    unexpected (in ckpt, not model)= {len(unexpected)}")
    print(f"    shape mismatches               = {len(shape_mismatch)}")
    if missing:
        print("      e.g. missing:", list(sorted(missing))[:5])
    if unexpected:
        print("      e.g. unexpected:", list(sorted(unexpected))[:5])
    if shape_mismatch:
        print("      e.g. shape:", shape_mismatch[:5])
    ok = (len(missing) == 0 and len(unexpected) == 0 and len(shape_mismatch) == 0)
    # try the real strict load
    try:
        model.load_state_dict(sd, strict=True)
        print("    load_state_dict(strict=True): OK")
        strict_ok = True
    except Exception as e:
        print("    load_state_dict(strict=True): FAILED ->", str(e)[:200])
        strict_ok = False
    return ok and strict_ok


def corr(a, b):
    a = a.flatten().float() - a.flatten().float().mean()
    b = b.flatten().float() - b.flatten().float().mean()
    return float((a @ b / (a.norm() * b.norm() + 1e-12)))


print("=" * 72)
print("STEP 1 — GROUND-TRUTH checkpoint parameter counts (raw tensor numel)")
print("=" * 72)
sizes = {}
sds = {}
for name, sub, fn in [("DVF VAE", "dvf_vae", "vae.pt"),
                      ("Image AE", "image_autoenc", "model.pt"),
                      ("Diffusion UNet", "ldm", "ldm.pt")]:
    p = os.path.join(PM, sub, fn)
    n, nt, sd = ckpt_param_count(p)
    sizes[name] = n
    sds[name] = sd
    print(f"  {name:16s}: {fmt(n):28s}  across {nt} tensors   [{fn}]")
print(f"  {'-'*60}")
print(f"  {'TOTAL (all 3)':16s}: {fmt(sum(sizes.values()))}")

print()
print("=" * 72)
print("STEP 2 — construct from args.json + STRICT load (repo's own method)")
print("=" * 72)

# DVF VAE
margs = json.load(open(os.path.join(PM, "dvf_vae", "args.json")))["model_args"]
vae = VariationalAutoencoder(encoder_args=EncoderArgs(**margs))
print(f"  DVF VAE constructed params: {fmt(model_param_count(vae))}")
vae_ok = strict_load_report(vae, sds["DVF VAE"], "DVF VAE")

# Image AE
iargs = json.load(open(os.path.join(PM, "image_autoenc", "args.json")))["model_args"]
iargs.setdefault("variational", False)
img_ae = Autoencoder(encoder_args=EncoderArgs(**iargs))
print(f"  Image AE constructed params: {fmt(model_param_count(img_ae))}")
img_ok = strict_load_report(img_ae, sds["Image AE"], "Image AE")

# Diffusion UNet
dargs = json.load(open(os.path.join(PM, "ldm", "args.json")))["model_args"]
unet = UNet(model_args=DiffusionArgs(**dargs))
print(f"  Diffusion UNet constructed params: {fmt(model_param_count(unet))}")
unet_ok = strict_load_report(unet, sds["Diffusion UNet"], "Diffusion UNet")

print()
print("=" * 72)
print("STEP 3 — DVF VAE functional check (encode->decode, correct units)")
print("=" * 72)
vae.eval()
vae.sampling_layer.identity_sampling = True
sf = float(margs["dvf_scale_factor"])
D, H, W = 50, 256, 256


def smooth_norm_dvf(D, H, W, maxmag, coarse=8):
    c = torch.randn(1, 3, max(2, D // 4), coarse, coarse)
    v = F.interpolate(c, size=(D, H, W), mode="trilinear", align_corners=True)
    return v / (v.abs().max() + 1e-9) * maxmag


def folding_frac(dvf_norm):
    u = torch.stack([dvf_norm[0, 0] * (D - 1) / 2, dvf_norm[0, 1] * (H - 1) / 2,
                     dvf_norm[0, 2] * (W - 1) / 2], 0)

    def d(t, ax):
        return 0.5 * (torch.roll(t, -1, ax) - torch.roll(t, 1, ax))
    det = ((1 + d(u[0], 0)) * ((1 + d(u[1], 1)) * (1 + d(u[2], 2)) - d(u[1], 2) * d(u[2], 1))
           - d(u[0], 1) * (d(u[1], 0) * (1 + d(u[2], 2)) - d(u[1], 2) * d(u[2], 0))
           + d(u[0], 2) * (d(u[1], 0) * d(u[2], 1) - (1 + d(u[1], 1)) * d(u[2], 0)))
    inner = det[2:-2, 2:-2, 2:-2]
    return float((inner <= 0).float().mean())


torch.manual_seed(0)
dvf = smooth_norm_dvf(D, H, W, 0.10)
with torch.no_grad():
    z, _ = vae.encode(dvf * sf)
    rec = vae.decode(z) / sf
print(f"  input norm|max|={dvf.abs().max():.3f}  latent z shape={tuple(z.shape)}  z_std={z.std():.3f}")
print(f"  recon corr={corr(dvf, rec):.4f}  out/in mag={rec.abs().max()/(dvf.abs().max()+1e-9):.3f}")
print(f"  folding(input)={folding_frac(dvf):.5f}  folding(recon)={folding_frac(rec):.5f}")

print()
print("=" * 72)
print("VERDICT")
print("=" * 72)
print(f"  DVF VAE  : {fmt(sizes['DVF VAE'])}   strict-load OK={vae_ok}")
print(f"  Image AE : {fmt(sizes['Image AE'])}   strict-load OK={img_ok}")
print(f"  Diffusion: {fmt(sizes['Diffusion UNet'])}   strict-load OK={unet_ok}")
print(f"  TOTAL    : {fmt(sum(sizes.values()))}")
print("  (AGRO uses the DVF VAE decoder as the frozen chart; the diffusion UNet")
print("   is only needed if we later add a GAS-DRO-style diffusion baseline.)")
