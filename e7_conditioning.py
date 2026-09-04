"""Does the LDM actually USE the CT conditioning, or has it learned the marginal
distribution of plausible DVFs?

Same seed (same z), same phase, different patient CT. If the model conditions on
anatomy, the fields must differ. If cos ~ 1, the CT is being ignored and the model
is sampling a patient-independent marginal -- which would explain a clean training
loss alongside cos ~ 0 prediction.

Control: same CT, same phase, DIFFERENT z (already measured at 0.251).
"""
import glob
import os
import sys

import torch

LDM = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LDM)
torch.compile = lambda m, *a, **k: m
from e7_gate1_prediction import LDMGen, demons, SCALE   # noqa: E402

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gen = LDMGen(dev, 5)


def phase0_vol(pt):
    st = sorted(glob.glob(os.path.join(LDM, 'data/idc_downloads/patient_%s/*' % pt)))[0]
    g = glob.glob(os.path.join(st, '*_Gated,_0.0%*', 'volume.pt'))
    return torch.load(g[0], map_location='cpu').float()


def cosv(x, y):
    return float((x.flatten() @ y.flatten()) / (x.norm() * y.norm() + 1e-12))


pts = ['100', '101', '105', '110']
vols = {p: phase0_vol(p) for p in pts}
print('CT pairwise cos (how different are the anatomies?):')
for i in range(len(pts)):
    for j in range(i + 1, len(pts)):
        print('   CT %s vs %s   cos %+.3f' % (pts[i], pts[j],
                                              cosv(vols[pts[i]], vols[pts[j]])))

# SAME z for every patient: any difference in output is caused by the CT alone.
fields = {}
for p in pts:
    torch.manual_seed(1234)
    fields[p] = gen.step(vols[p], 0) * SCALE

print('\nSAME z, SAME phase, DIFFERENT patient CT -> cos between generated DVFs')
print('  (cos ~ 1 => CT conditioning ignored; model samples a patient-independent marginal)')
for i in range(len(pts)):
    for j in range(i + 1, len(pts)):
        print('   DVF %s vs %s   cos %+.3f   RMS %.4f / %.4f' % (
            pts[i], pts[j], cosv(fields[pts[i]], fields[pts[j]]),
            float(fields[pts[i]].pow(2).sum(0).mean().sqrt()),
            float(fields[pts[j]].pow(2).sum(0).mean().sqrt())))

print('\nCONTROL: same CT (100), same phase, DIFFERENT z')
outs = []
for s in (1234, 999):
    torch.manual_seed(s)
    outs.append(gen.step(vols['100'], 0) * SCALE)
print('   cos %+.3f' % cosv(outs[0], outs[1]))

print('\nREFERENCE: how different are the TRUE fields across patients? (0->10 DIR)')
tr = {}
for p in pts[:3]:
    st = sorted(glob.glob(os.path.join(LDM, 'data/idc_downloads/patient_%s/*' % p)))[0]
    g10 = glob.glob(os.path.join(st, '*_Gated,_10.0%*', 'volume.pt'))
    if not g10:
        continue
    v10 = torch.load(g10[0], map_location='cpu').float()
    tr[p] = demons(v10.numpy(), vols[p].numpy())
kk = list(tr)
for i in range(len(kk)):
    for j in range(i + 1, len(kk)):
        print('   TRUE %s vs %s   cos %+.3f' % (kk[i], kk[j], cosv(tr[kk[i]], tr[kk[j]])))
