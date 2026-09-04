"""
============================================================================
  E8 -- COVERAGE, not accuracy.
============================================================================
  The argument being tested: if the model emits PLAUSIBLE breathing fields,
  then one of them could be THIS patient's real breathing. For robust
  optimisation that is all we need -- the uncertainty set must CONTAIN the
  truth (SRO Assumption A0), it does not have to predict it. Gate 1 measured
  accuracy (the conditional mean) and therefore does NOT settle this.

  Three measures, weakest requirement last:

    1. BEST-OF-N   min over N draws of  ||true - s_i|| / ||true||
                   Does any single sample land near the truth?

    2. SPAN        min over c of ||true - sum_i c_i s_i|| / ||true||
                   Does the truth lie in the LINEAR SPAN of the samples?
                   This is the most generous coverage notion available: an
                   uncertainty set built as combinations of model outputs.

    3. SPAN+       same, but with the samples' own mean allowed as a free
                   offset (affine span).

  Baseline to beat: ratio 1.0 is "no better than assuming zero motion".
  A set that covers the truth should drive SPAN well below 1.0.

  CONTROL: the same span test using N random Gaussian smooth fields. If
  random fields span the truth just as well, the model contributes nothing
  beyond dimension counting.
============================================================================
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

LDM = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LDM)
torch.compile = lambda m, *a, **k: m
from e7_gate1_prediction import LDMGen, demons, SCALE, D, H, W   # noqa: E402


def span_residual(true_v, samples, ridge=1e-8):
    """min_c ||true - S c|| / ||true||, least squares over the sample span."""
    S = torch.stack([s for s in samples], 1)          # (M, N)
    t = true_v
    A = S.T @ S
    A = A + ridge * torch.eye(A.shape[0]) * float(A.diagonal().mean())
    c = torch.linalg.solve(A, S.T @ t)
    return float((t - S @ c).norm() / (t.norm() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--patients', nargs='+', default=['100', '101'])
    ap.add_argument('--phases', nargs='+', type=int, default=[0, 4])
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--delta_t', type=int, default=5)
    a = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen = LDMGen(dev, a.delta_t)
    torch.manual_seed(0)

    print('=' * 96)
    print('  E8 -- does the model SET contain the patient motion?  N=%d draws '
          'per phase' % a.n)
    print('  ratio 1.0 = no better than assuming zero motion')
    print('=' * 96)
    print('  %-8s %6s %9s %10s %10s %10s %10s %10s' % (
        'patient', 'phase', 'true_vox', 'best-of-N', 'median-N', 'SPAN',
        'SPAN+', 'rand SPAN'))
    print('  ' + '-' * 92)

    for pt in a.patients:
        sts = sorted(glob.glob(os.path.join(LDM, 'data/idc_downloads',
                                            'patient_%s' % pt, '*')))
        vols = {}
        for st in sts:
            vols = {}
            for p in glob.glob(os.path.join(st, '*_Gated,_*')):
                v = os.path.join(p, 'volume.pt')
                if os.path.exists(v):
                    try:
                        vols[float(os.path.basename(p).split('Gated,_')[1]
                                   .split('%')[0])] = v
                    except Exception:
                        pass
            if 0.0 in vols and len(vols) >= 10:
                break
        if not vols:
            continue
        ks = sorted(vols)[:10]
        real = [torch.load(vols[k], map_location='cpu').float() for k in ks]

        for ph in a.phases:
            v_in = real[ph]
            lung = (v_in < -0.60) & (v_in > -0.96)
            lf = lung.reshape(-1)
            true = demons(real[ph + 1].numpy(), real[ph].numpy())
            t_v = true.reshape(3, -1)[:, lf].reshape(-1)
            tn = float(t_v.norm())
            true_rms = float(true.reshape(3, -1)[:, lf].pow(2).sum(0).mean().sqrt())

            samples, ratios = [], []
            for _ in range(a.n):
                s = (gen.step(v_in, ph) * SCALE).reshape(3, -1)[:, lf].reshape(-1)
                samples.append(s)
                ratios.append(float((t_v - s).norm() / (tn + 1e-12)))

            sp = span_residual(t_v, samples)
            mu = torch.stack(samples, 1).mean(1)
            sp_plus = span_residual(t_v, samples + [mu])

            # CONTROL: N smooth random fields of matched RMS
            rnd = []
            for _ in range(a.n):
                g = torch.randn(1, 3, D // 4, H // 8, W // 8)
                g = torch.nn.functional.interpolate(
                    g, size=(D, H, W), mode='trilinear', align_corners=False)[0]
                g = g / g.reshape(3, -1)[:, lf].pow(2).sum(0).mean().sqrt() * true_rms
                rnd.append(g.reshape(3, -1)[:, lf].reshape(-1))
            sp_rnd = span_residual(t_v, rnd)

            print('  %-8s %6d %9.3f %10.3f %10.3f %10.3f %10.3f %10.3f' % (
                pt, ph, true_rms, min(ratios), float(np.median(ratios)),
                sp, sp_plus, sp_rnd))


if __name__ == '__main__':
    main()
