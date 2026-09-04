"""Is cos(draw1,draw2)=0.25 the MODEL, or my under-resolved DDIM?
Sweep delta_t and the sampling algorithm; measure self-consistency between two
draws at identical conditioning, and agreement with the DIR field."""
import glob
import os
import sys

import torch

LDM = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LDM)
torch.compile = lambda m, *a, **k: m
from e7_gate1_prediction import LDMGen, demons, SCALE   # noqa: E402

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st = sorted(glob.glob(os.path.join(LDM, 'data/idc_downloads/patient_100/*')))[0]
vols = {}
for p in glob.glob(os.path.join(st, '*_Gated,_*')):
    v = os.path.join(p, 'volume.pt')
    if os.path.exists(v):
        vols[float(os.path.basename(p).split('Gated,_')[1].split('%')[0])] = v
ks = sorted(vols)[:10]
real = [torch.load(vols[k], map_location='cpu').float() for k in ks]
v0 = real[0]
lung = (v0 < -0.60) & (v0 > -0.96)
true = demons(real[1].numpy(), real[0].numpy())
a = true.reshape(3, -1)[:, lung.reshape(-1)]
print('true 0->10 RMS(lung) %.3f vox' % float(a.pow(2).sum(0).mean().sqrt()))


def cosv(x, y):
    return float((x.flatten() @ y.flatten()) / (x.norm() * y.norm() + 1e-12))


print('\n  %-8s %6s   %10s %10s %10s %10s' % (
    'algo', 'dt', 'RMS d1', 'RMS d2', 'cos(d1,d2)', 'cos(d1,true)'))
print('  ' + '-' * 62)
for algo, dt in [('ddim', 1), ('ddim', 5), ('ddim', 20), ('ddpm', 1)]:
    gen = LDMGen(dev, dt)
    gen.smp.sampling_algo = algo
    torch.manual_seed(0)
    d1 = gen.step(v0, 0) * SCALE
    d2 = gen.step(v0, 0) * SCALE
    b1 = d1.reshape(3, -1)[:, lung.reshape(-1)]
    b2 = d2.reshape(3, -1)[:, lung.reshape(-1)]
    print('  %-8s %6d   %10.4f %10.4f %10.3f %10.3f' % (
        algo, dt, float(b1.pow(2).sum(0).mean().sqrt()),
        float(b2.pow(2).sum(0).mean().sqrt()), cosv(b1, b2), cosv(b1, a)))
    del gen
    torch.cuda.empty_cache()
