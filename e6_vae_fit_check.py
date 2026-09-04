"""
============================================================================
  E6 — TIER 1 FIT CHECK: can the DVF VAE represent REAL respiratory motion?
============================================================================

  WHY THIS TEST, AND WHY ON TRAINING DATA.
  A model that cannot reproduce the data it was trained on is broken, and no
  held-out evaluation can rescue it. So the FIRST question for any candidate
  generator is a fit check, and training data is exactly the right place to
  run it. (This is also the correct reading of DynaGAN's failure: its
  resid/real = 0.73 was measured on 4D-Lung, the dataset its own README
  names. That is not a generalization failure, it is a fit failure.)

  The three-tier plan this belongs to:
    TIER 1 (this script) : the LDM's TRAINING patients  -> does it work at all?
    TIER 2               : its 2 held-out test patients -> within-dataset generalization
    TIER 3               : a different 4DCT cohort      -> distribution shift
  The DEGRADATION BETWEEN TIERS is a principled calibration of a DRO
  ambiguity radius -- the first non-arbitrary way this project has had to set
  one.

  WHAT IS MEASURED HERE.
  The DVF VAE is the necessary condition for the whole latent-diffusion model:
  if the latent chart cannot REPRESENT a real DVF, the diffusion prior on top
  of it cannot generate one. So we bypass sampling entirely and test the chart:

      real DVF (SimpleITK demons, 0% -> k%)  --encode--> z --decode--> recon

  and report, in the SAME units as the DynaGAN measurement so the two are
  directly comparable:

      resid/real = RMS|dvf - recon| / RMS|dvf|         (DynaGAN: 0.73 median)
      cos        = cosine(dvf, recon)                  (DynaGAN: 0.65-0.79)
      folding    = fraction of voxels with det(J) <= 0, input and recon

  NOTE ON WHAT THIS IS NOT. A good VAE reconstruction does NOT mean the LDM
  predicts motion -- reconstruction is given the true DVF as input. It is a
  CEILING on what the generative model can express, not a measure of its
  predictive accuracy. Reported as such.

  USAGE
    cd /e/Cancer/curse_of_optimization/motion/ldm-respiratory-motion
    python -u e6_vae_fit_check.py                 # all patients, all phases
    python -u e6_vae_fit_check.py --patients 100 101 --max_phases 3
============================================================================
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk

LDM = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LDM)

from args.classes import EncoderArgs                              # noqa: E402
from backbones.models import VariationalAutoencoder               # noqa: E402
from inference.utils.load_models import revert_model_weight_names  # noqa: E402
from data_processing.utils.dp_utils import split_patient_folders   # noqa: E402

D, H, W = 50, 256, 256
DATA = os.path.join(LDM, 'data', 'idc_downloads')


def load_vae(device):
    base = os.path.join(LDM, 'pretrained_models', 'dvf_vae')
    margs = json.load(open(os.path.join(base, 'args.json')))['model_args']
    vae = VariationalAutoencoder(encoder_args=EncoderArgs(**margs))
    vae.load_state_dict(revert_model_weight_names(torch.load(
        os.path.join(base, 'vae.pt'), map_location='cpu', weights_only=True)),
        strict=False)
    vae.eval()
    vae.sampling_layer.identity_sampling = True
    return vae.to(device), float(margs['dvf_scale_factor'])


def to50(vol):
    vol = vol.squeeze().float()
    if vol.max() > 10:
        vol = torch.clamp(vol, -1024, 3071)
        vol = 2 * (vol + 1024) / 4095 - 1
    if tuple(vol.shape) != (D, H, W):
        vol = F.interpolate(vol[None, None], size=(D, H, W),
                            mode='trilinear', align_corners=False)[0, 0]
    return vol


def sitk_demons(fixed_np, moving_np):
    f = sitk.GetImageFromArray(fixed_np.astype(np.float32))
    m = sitk.GetImageFromArray(moving_np.astype(np.float32))
    mt = sitk.HistogramMatchingImageFilter()
    mt.SetNumberOfHistogramLevels(256)
    mt.SetNumberOfMatchPoints(16)
    mt.ThresholdAtMeanIntensityOn()
    m = mt.Execute(m, f)
    dm = sitk.FastSymmetricForcesDemonsRegistrationFilter()
    dm.SetNumberOfIterations(60)
    dm.SetStandardDeviations(2.0)
    return sitk.GetArrayFromImage(dm.Execute(f, m))    # [D,H,W,3] (dx,dy,dz)


def folding(dvf_norm):
    u = torch.stack([dvf_norm[0, 0] * (D - 1) / 2, dvf_norm[0, 1] * (H - 1) / 2,
                     dvf_norm[0, 2] * (W - 1) / 2], 0)

    def d(t, ax):
        return 0.5 * (torch.roll(t, -1, ax) - torch.roll(t, 1, ax))
    det = ((1 + d(u[0], 0)) * ((1 + d(u[1], 1)) * (1 + d(u[2], 2))
                               - d(u[1], 2) * d(u[2], 1))
           - d(u[0], 1) * (d(u[1], 0) * (1 + d(u[2], 2)) - d(u[1], 2) * d(u[2], 0))
           + d(u[0], 2) * (d(u[1], 0) * d(u[2], 1) - (1 + d(u[1], 1)) * d(u[2], 0)))
    inner = det[2:-2, 2:-2, 2:-2]
    return float((inner <= 0).float().mean())


def studies_for(pt):
    return sorted(glob.glob(os.path.join(DATA, 'patient_{}'.format(pt), '*')))


def phase_vols(study):
    out = {}
    for p in sorted(glob.glob(os.path.join(study, '*_Gated,_*'))):
        v = os.path.join(p, 'volume.pt')
        if not os.path.exists(v):
            continue
        try:
            tag = os.path.basename(p).split('Gated,_')[1].split('%')[0]
            out[float(tag)] = v
        except Exception:
            pass
    return out


def run_patient(pt, vae, sf, device, max_phases=None):
    sts = studies_for(pt)
    if not sts:
        return None
    for st in sts:
        pv = phase_vols(st)
        if 0.0 in pv and len(pv) >= 2:
            break
    else:
        return None

    moving = to50(torch.load(pv[0.0], map_location='cpu')).numpy()
    ks = sorted(k for k in pv if k > 1e-6)
    if max_phases:
        ks = ks[:max_phases]

    rows = []
    for k in ks:
        fixed = to50(torch.load(pv[k], map_location='cpu')).numpy()
        arr = sitk_demons(fixed, moving)
        dz = torch.from_numpy(arr[..., 2]); dy = torch.from_numpy(arr[..., 1])
        dx = torch.from_numpy(arr[..., 0])
        dvf_vox = torch.stack([dz, dy, dx], 0)[None].float()
        dvf_norm = torch.stack([dvf_vox[0, 0] * 2 / (D - 1),
                                dvf_vox[0, 1] * 2 / (H - 1),
                                dvf_vox[0, 2] * 2 / (W - 1)], 0)[None]

        with torch.no_grad():
            z, _ = vae.encode(dvf_norm.to(device) * sf)
            rec = (vae.decode(z) / sf).cpu()

        # back to voxels for a magnitude-meaningful residual
        rec_vox = torch.stack([rec[0, 0] * (D - 1) / 2, rec[0, 1] * (H - 1) / 2,
                               rec[0, 2] * (W - 1) / 2], 0)[None]
        mag_real = dvf_vox.pow(2).sum(1).sqrt()
        mag_res = (dvf_vox - rec_vox).pow(2).sum(1).sqrt()
        # lung-ish: parenchyma on the normalized CT (approx -900..-400 HU)
        nf = torch.from_numpy(fixed)
        lung = (nf < -0.60) & (nf > -0.96)
        if lung.sum() < 5000:
            lung = torch.ones_like(nf, dtype=torch.bool)

        a = dvf_vox.reshape(3, -1)[:, lung.reshape(-1)]
        b = rec_vox.reshape(3, -1)[:, lung.reshape(-1)]
        cos = float((a.flatten() @ b.flatten())
                    / (a.norm() * b.norm() + 1e-12))
        real = float(mag_real[0][lung].pow(2).mean().sqrt())
        resid = float(mag_res[0][lung].pow(2).mean().sqrt())

        rows.append({
            'phase': k, 'real_rms_vox': real, 'resid_rms_vox': resid,
            'ratio': resid / max(real, 1e-9), 'cos': cos,
            'fold_in': folding(dvf_norm), 'fold_rec': folding(rec),
            'z_std': float(z.std()),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--patients', nargs='+', default=None)
    ap.add_argument('--max_phases', type=int, default=None)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    vae, sf = load_vae(device)

    try:
        tr, va, te = split_patient_folders(DATA)
        norm = lambda L: set(os.path.basename(p).replace('patient_', '')
                             for p in L)
        TR, VA, TE = norm(tr), norm(va), norm(te)
    except Exception as e:
        print("  [warn] could not reproduce split: {}".format(e))
        TR, VA, TE = set(), set(), set()

    pats = args.patients or sorted(
        os.path.basename(p).replace('patient_', '')
        for p in glob.glob(os.path.join(DATA, 'patient_*')))

    print("=" * 92)
    print("  E6 — TIER 1 FIT CHECK: DVF VAE reconstruction of REAL 4D-Lung "
          "motion  (device={})".format(device))
    print("  LDM split reproduced: train={} val={} test={}".format(
        sorted(TR), sorted(VA), sorted(TE)))
    print("  comparison point -- DynaGAN on the same cohort: resid/real "
          "median 0.73, cos 0.65-0.79")
    print("=" * 92)
    print("  {:<8s} {:>5s} {:>10s} {:>10s} {:>8s} {:>7s} {:>9s} {:>9s} {:>6s}"
          .format("patient", "split", "real_vox", "resid_vox", "ratio", "cos",
                  "fold_in", "fold_rec", "n"))
    print("  " + "-" * 86)

    allrows, out = [], {}
    t0 = time.time()
    for pt in pats:
        try:
            rows = run_patient(pt, vae, sf, device, args.max_phases)
        except Exception as e:
            print("  {:<8s} FAILED {}".format(pt, str(e)[:60]))
            continue
        if not rows:
            continue
        out[pt] = rows
        allrows.extend([dict(r, patient=pt) for r in rows])
        sp = ('train' if pt in TR else 'val' if pt in VA
              else 'TEST' if pt in TE else '?')
        m = lambda k: float(np.mean([r[k] for r in rows]))
        print("  {:<8s} {:>5s} {:>10.3f} {:>10.3f} {:>8.3f} {:>7.3f} "
              "{:>9.5f} {:>9.5f} {:>6d}".format(
                  pt, sp, m('real_rms_vox'), m('resid_rms_vox'), m('ratio'),
                  m('cos'), m('fold_in'), m('fold_rec'), len(rows)))

    if not allrows:
        print("\n  nothing ran")
        return

    def agg(rows, name):
        if not rows:
            return
        r = np.array([x['ratio'] for x in rows])
        c = np.array([x['cos'] for x in rows])
        f = np.array([x['fold_rec'] for x in rows])
        print("  {:<12s} n={:<4d} ratio mean {:.3f} median {:.3f}  |  "
              "cos mean {:.3f}  |  fold_rec mean {:.5f}".format(
                  name, len(rows), r.mean(), np.median(r), c.mean(), f.mean()))

    print("\n" + "=" * 92)
    print("  SUMMARY  ({:.0f}s)".format(time.time() - t0))
    print("=" * 92)
    agg(allrows, 'ALL')
    agg([r for r in allrows if r['patient'] in TR], 'TRAIN (tier1)')
    agg([r for r in allrows if r['patient'] in VA], 'VAL')
    agg([r for r in allrows if r['patient'] in TE], 'TEST (tier2)')
    print("\n  DynaGAN reference on this cohort: ratio median 0.73, "
          "cos 0.65-0.79")
    print("  NOTE: this is RECONSTRUCTION (the true DVF is given as input). "
          "It is a CEILING\n        on what the LDM can express, NOT its "
          "predictive accuracy.")

    op = os.path.join(LDM, 'e6_vae_fit_check.json')
    json.dump({'split': {'train': sorted(TR), 'val': sorted(VA),
                         'test': sorted(TE)},
               'per_patient': out}, open(op, 'w'), indent=2)
    print("\n  Saved -> {}".format(op))


if __name__ == '__main__':
    main()
