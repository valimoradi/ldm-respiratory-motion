"""Why is STEP cos ~ 0? Diagnose the predicted field against the DIR field
directly (not the weak warp proxy). Tests every channel permutation and sign
against ground truth, and reports per-channel magnitudes."""
import glob
import itertools
import json
import os
import sys

import numpy as np
import torch

LDM = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LDM)
torch.compile = lambda m, *a, **k: m
from e7_gate1_prediction import (LDMGen, demons, D, H, W, SCALE, warp)  # noqa: E402

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
gen = LDMGen(dev, 5)

st = sorted(glob.glob(os.path.join(LDM, 'data/idc_downloads/patient_100/*')))[0]
vols = {}
for p in glob.glob(os.path.join(st, '*_Gated,_*')):
    v = os.path.join(p, 'volume.pt')
    if os.path.exists(v):
        vols[float(os.path.basename(p).split('Gated,_')[1].split('%')[0])] = v
ks = sorted(vols)[:10]
real = [torch.load(vols[k], map_location='cpu').float() for k in ks]
v0 = real[0]
body = v0 > -0.85
lung = (v0 < -0.60) & (v0 > -0.96)
print('body voxels %d   lung voxels %d' % (int(body.sum()), int(lung.sum())))

pred = gen.step(v0, 0) * SCALE                       # voxels, model order
true = demons(real[1].numpy(), real[0].numpy())      # voxels, (x,y,z)
true_big = demons(real[5].numpy(), real[0].numpy())  # 0 -> 50%, largest motion

for nm, m in [('body', body), ('lung', lung)]:
    print('\n--- mask=%s ---' % nm)
    for c, lbl in enumerate(['ch0 (x,W)', 'ch1 (y,H)', 'ch2 (z,D)']):
        pr = float(pred[c][m].pow(2).mean().sqrt())
        tr = float(true[c][m].pow(2).mean().sqrt())
        cc = float((pred[c][m] @ true[c][m])
                   / (pred[c][m].norm() * true[c][m].norm() + 1e-12))
        print('  %-10s pred RMS %7.4f   true RMS %7.4f   cos %+.3f'
              % (lbl, pr, tr, cc))

print('\n--- all permutations/signs of PRED vs DIR truth (lung mask, phase 0->10) ---')
a = true.reshape(3, -1)[:, lung.reshape(-1)]
rows = []
for perm in itertools.permutations(range(3)):
    for sgn in (1, -1):
        b = torch.stack([pred[perm[0]], pred[perm[1]], pred[perm[2]]], 0) * sgn
        b = b.reshape(3, -1)[:, lung.reshape(-1)]
        cos = float((a.flatten() @ b.flatten()) / (a.norm() * b.norm() + 1e-12))
        rows.append((cos, perm, sgn))
rows.sort(reverse=True)
for cos, perm, sgn in rows:
    print('   pred[%d,%d,%d] * %+d   cos %+.3f' % (perm[0], perm[1], perm[2], sgn, cos))

print('\n--- is the DIR step field itself sane? (0->10 vs 0->50) ---')
for nm, t in [('0->10', true), ('0->50', true_big)]:
    print('  %s  RMS(lung) %.3f vox   per-ch %.3f %.3f %.3f'
          % (nm, float(t.reshape(3, -1)[:, lung.reshape(-1)].pow(2).sum(0).mean().sqrt()),
             float(t[0][lung].pow(2).mean().sqrt()),
             float(t[1][lung].pow(2).mean().sqrt()),
             float(t[2][lung].pow(2).mean().sqrt())))

print('\n--- does the model give the SAME field for different phase indices? ---')
torch.manual_seed(0)
fields = [gen.step(v0, i) * SCALE for i in range(4)]
for i in range(4):
    r = float(fields[i].reshape(3, -1)[:, lung.reshape(-1)].pow(2).sum(0).mean().sqrt())
    c01 = float((fields[i].flatten() @ fields[0].flatten())
                / (fields[i].norm() * fields[0].norm() + 1e-12))
    print('  phase %d  RMS(lung) %.4f vox   cos-with-phase0 %+.3f' % (i, r, c01))

print('\n--- two draws at the SAME phase (sampler variability) ---')
d1 = gen.step(v0, 0) * SCALE
d2 = gen.step(v0, 0) * SCALE
print('  cos(draw1,draw2) %+.3f   RMS %.4f / %.4f'
      % (float((d1.flatten() @ d2.flatten()) / (d1.norm() * d2.norm() + 1e-12)),
         float(d1.reshape(3, -1)[:, lung.reshape(-1)].pow(2).sum(0).mean().sqrt()),
         float(d2.reshape(3, -1)[:, lung.reshape(-1)].pow(2).sum(0).mean().sqrt())))
