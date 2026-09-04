import os, sys, glob, time, json
import torch
LDM = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, LDM)
torch.compile = lambda m, *a, **k: m          # Windows: no Triton
from args.classes import SamplingArgs
from inference.utils.load_models import load_autoencoder
from diffusion.sampling import DiffusionSampler

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
P = LDM.replace(os.sep, '/') + '/pretrained_models'   # load_models rsplit('/') needs fwd slashes
img_ae, img_a, _ = load_autoencoder(P + '/image_autoenc/model.pt', dev)
dvf_ae, dvf_a, _ = load_autoencoder(P + '/dvf_vae/vae.pt', dev)
print('img_ae latent_ch', img_a.latent_channels, ' dvf_ae latent_ch', dvf_a.latent_channels,
      ' dvf_sf', dvf_a.dvf_scale_factor)

v = sorted(glob.glob(os.path.join(LDM,'data','idc_downloads','patient_100','*','*_Gated,_0.0%_*','volume.pt')))
print('found volume:', v[0] if v else None)
vol = torch.load(v[0], map_location='cpu').float()
print('stored volume shape', tuple(vol.shape), 'min %.3f max %.3f' % (vol.min(), vol.max()))

x = vol if vol.dim() == 4 else vol[None]        # (1,D,H,W)
with torch.no_grad():
    lat = img_ae.encode(x.to(dev))
print('encode((1,D,H,W)) ->', tuple(lat.shape))
lat_b = lat.unsqueeze(0) if lat.dim() == 4 else lat
print('after unsqueeze ->', tuple(lat_b.shape))

ma = json.load(open(P + '/ldm/args.json'))['model_args']
sa = SamplingArgs(epsilon=0, delta_t=20, n_samples=1, T=1000, sampling_algo='ddim')
smp = DiffusionSampler(sa, P + '/ldm/ldm.pt', dev)
z = torch.randn(1, ma['out_channels'], ma['image_depth'], ma['image_width'], ma['image_width'], device=dev)
inp = torch.cat((lat_b, z), dim=1)
print('sampler input', tuple(inp.shape), '(expect in_channels=%d)' % ma['in_channels'])
ph = torch.tensor([0], device=dev, dtype=torch.long)
t0 = time.time()
with torch.no_grad():
    out = smp.generate_sample(inp, ph)
print('ddim delta_t=20 (50 steps): %.1fs -> %s' % (time.time()-t0, tuple(out.shape)))
with torch.no_grad():
    dvf = dvf_ae.decode(out[:, ma['in_channels']-ma['out_channels']:]) / ma['dvf_scale_factor']
print('decoded DVF', tuple(dvf.shape), 'absmax %.4f' % dvf.abs().max())
