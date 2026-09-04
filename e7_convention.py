"""Decide the LDM DVF channel-order/sign convention by warping, not by argument.

warp_image() does grid_sample(image, dvf.permute(1,2,3,0) + grid), so DVF channel c
maps onto grid last-dim c. For 5-D grid_sample the last dim is (x,y,z) = (W,H,D).
Test: warp real phase-0 onto real phase-10 with the model DVF under each candidate
ordering; the correct one minimises MSE to the real phase-10 volume.
"""
import os, sys, glob, json, itertools
import torch, torch.nn.functional as F
LDM = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, LDM)
torch.compile = lambda m, *a, **k: m
from args.classes import SamplingArgs
from inference.utils.load_models import load_autoencoder
from diffusion.sampling import DiffusionSampler

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
P = LDM.replace(os.sep, '/') + '/pretrained_models'
img_ae, _, _ = load_autoencoder(P + '/image_autoenc/model.pt', dev)
dvf_ae, _, _ = load_autoencoder(P + '/dvf_vae/vae.pt', dev)
ma = json.load(open(P + '/ldm/args.json'))['model_args']
smp = DiffusionSampler(SamplingArgs(epsilon=0, delta_t=5, n_samples=1, T=1000,
                                    sampling_algo='ddim'), P + '/ldm/ldm.pt', dev)

st = sorted(glob.glob(os.path.join(LDM, 'data/idc_downloads/patient_100/*')))[0]
def vol_at(pct):
    g = glob.glob(os.path.join(st, '*_Gated,_%.1f%%_*' % pct, 'volume.pt'))
    return torch.load(g[0], map_location='cpu').float()
v0, v10 = vol_at(0.0), vol_at(10.0)
print('phase0', tuple(v0.shape), ' phase10', tuple(v10.shape))

torch.manual_seed(0)
with torch.no_grad():
    lat = img_ae.encode(v0[None].to(dev))
    lat = lat.unsqueeze(0) if lat.dim() == 4 else lat
    z = torch.randn(1, ma['out_channels'], ma['image_depth'], ma['image_width'],
                    ma['image_width'], device=dev)
    out = smp.generate_sample(torch.cat((lat, z), 1),
                              torch.tensor([0], device=dev, dtype=torch.long))
    dvf = (dvf_ae.decode(out[:, ma['in_channels']-ma['out_channels']:])
           / ma['dvf_scale_factor'])[0].cpu()          # (3,D,H,W), grid-normalised
D, H, W = v0.shape
body = v0 > -0.85
print('\nper-channel RMS inside body (grid-normalised):')
for c in range(3):
    print('   ch%d  %.5f   -> voxels if this axis were D/H/W: %.3f / %.3f / %.3f'
          % (c, dvf[c][body].pow(2).mean().sqrt(),
             dvf[c][body].pow(2).mean().sqrt()*(D-1)/2,
             dvf[c][body].pow(2).mean().sqrt()*(H-1)/2,
             dvf[c][body].pow(2).mean().sqrt()*(W-1)/2))

base = F.affine_grid(torch.eye(3, 4)[None], (1, 1, D, H, W), align_corners=False)
def warp(v, d):
    g = base + d.permute(1, 2, 3, 0)[None]
    return F.grid_sample(v[None, None], g, padding_mode='border',
                         mode='bilinear', align_corners=False)[0, 0]

mse0 = float((v0 - v10).pow(2).mean())
print('\nno warp                       MSE %.6f' % mse0)
rows = []
for perm in itertools.permutations(range(3)):
    for sgn in [1, -1]:
        d = torch.stack([dvf[perm[0]], dvf[perm[1]], dvf[perm[2]]], 0) * sgn
        m = float((warp(v0, d) - v10).pow(2).mean())
        rows.append((m, perm, sgn))
rows.sort()
print('  ordering (as fed to grid_sample x,y,z)   sign      MSE     vs no-warp')
for m, perm, sgn in rows:
    print('   ch%d,ch%d,ch%d                              %+d    %.6f   %+.1f%%'
          % (perm[0], perm[1], perm[2], sgn, m, 100*(m-mse0)/mse0))
