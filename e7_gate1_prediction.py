"""
============================================================================
  GATE 1 -- Does the LDM PREDICT real respiratory motion?
============================================================================
  E6 measured RECONSTRUCTION (true DVF handed in) and used the WRONG channel
  order. This measures PREDICTION, in the model's own convention, the way the
  authors' own inference loop uses it (visualization/dvf_movie.py):

      diffusion_vol <- real phase-0 CT           (the ONLY real CT used)
      for i in 0..8:
          dvf_i         = LDM(diffusion_vol, phase=i, z~N(0,I))   # consecutive
          cum_dvf      += dvf_i                                   # additive
          diffusion_vol = warp(initial_vol, cum_dvf)              # feed back

  Convention (settled empirically in e7_convention.py): DVF channel c feeds
  grid_sample last-dim c, i.e. (x,y,z) = (W,H,D); sign as emitted. ch2 carries
  the SI motion; every ordering with ch2 not last warps WORSE than no warp.

  Ground truth: SimpleITK demons, same filter and parameters as E6 and as the
  DynaGAN measurement, so resid/real is directly comparable to DynaGAN's
  0.703 cohort median.

  Reported per phase step:
    STEP  predicted consecutive dvf_i   vs demons(phase i -> i+1)
    CUM   predicted cumulative cum_dvf  vs demons(phase 0 -> i+1)
    VAE   recon control: encode/decode the TRUE consecutive DVF in the CORRECT
          channel order. This supersedes E6 and is a CEILING, not a prediction.
  plus folding of cum_dvf, which tests whether the authors' additive
  accumulation stays diffeomorphic (proper composition would be phi_k o phi_k-1).

  PASS: CUM resid/real clearly below DynaGAN's 0.703 median.

  USAGE
    python -u e7_gate1_prediction.py                      # all patients
    python -u e7_gate1_prediction.py --patients 100 101
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
torch.compile = lambda m, *a, **k: m          # Windows: no Triton

from args.classes import SamplingArgs                              # noqa: E402
from inference.utils.load_models import load_autoencoder           # noqa: E402
from diffusion.sampling import DiffusionSampler                    # noqa: E402
from data_processing.utils.dp_utils import split_patient_folders   # noqa: E402

D, H, W = 50, 256, 256
DATA = os.path.join(LDM, 'data', 'idc_downloads')
# grid-normalised -> voxels, per channel, in model (x,y,z) = (W,H,D) order
SCALE = torch.tensor([(W - 1) / 2., (H - 1) / 2., (D - 1) / 2.]).view(3, 1, 1, 1)


def demons(fixed_np, moving_np):
    """Same filter and parameters as E6 and the DynaGAN measurement.
    Returns (3,D,H,W) in model channel order (x,y,z), VOXEL units."""
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
    a = sitk.GetArrayFromImage(dm.Execute(f, m))          # [D,H,W,3] = (dx,dy,dz)
    return torch.from_numpy(np.stack([a[..., 0], a[..., 1], a[..., 2]], 0)).float()


def folding(dvf_vox):
    """Fraction of voxels with det(J) <= 0. dvf_vox is (3,D,H,W) in (x,y,z)."""
    def d(t, ax):
        return 0.5 * (torch.roll(t, -1, ax) - torch.roll(t, 1, ax))
    ux, uy, uz = dvf_vox[0], dvf_vox[1], dvf_vox[2]
    det = ((1 + d(uz, 0)) * ((1 + d(uy, 1)) * (1 + d(ux, 2)) - d(uy, 2) * d(ux, 1))
           - d(uz, 1) * (d(uy, 0) * (1 + d(ux, 2)) - d(uy, 2) * d(ux, 0))
           + d(uz, 2) * (d(uy, 0) * d(ux, 1) - (1 + d(uy, 1)) * d(ux, 0)))
    return float((det[2:-2, 2:-2, 2:-2] <= 0).float().mean())


def metrics(pred_vox, true_vox, mask):
    a = true_vox.reshape(3, -1)[:, mask.reshape(-1)]
    b = pred_vox.reshape(3, -1)[:, mask.reshape(-1)]
    real = float(a.pow(2).sum(0).mean().sqrt())
    resid = float((a - b).pow(2).sum(0).mean().sqrt())
    cos = float((a.flatten() @ b.flatten()) / (a.norm() * b.norm() + 1e-12))
    return real, resid, resid / max(real, 1e-9), cos


class LDMGen:
    def __init__(self, device, delta_t):
        P = LDM.replace(os.sep, '/') + '/pretrained_models'
        self.img_ae, _, _ = load_autoencoder(P + '/image_autoenc/model.pt', device)
        self.dvf_ae, _, _ = load_autoencoder(P + '/dvf_vae/vae.pt', device)
        self.ma = json.load(open(P + '/ldm/args.json'))['model_args']
        self.smp = DiffusionSampler(
            SamplingArgs(epsilon=0, delta_t=delta_t, n_samples=1, T=1000,
                         sampling_algo='ddim'), P + '/ldm/ldm.pt', device)
        self.dev = device

    @torch.no_grad()
    def step(self, vol, phase_idx):
        lat = self.img_ae.encode(vol[None].to(self.dev))
        lat = lat.unsqueeze(0) if lat.dim() == 4 else lat
        z = torch.randn(1, self.ma['out_channels'], self.ma['image_depth'],
                        self.ma['image_width'], self.ma['image_width'],
                        device=self.dev)
        out = self.smp.generate_sample(
            torch.cat((lat, z), 1),
            torch.tensor([phase_idx], device=self.dev, dtype=torch.long))
        keep = self.ma['in_channels'] - self.ma['out_channels']
        dvf = (self.dvf_ae.decode(out[:, keep:]) / self.ma['dvf_scale_factor'])[0]
        return dvf.cpu()                    # (3,D,H,W) grid-normalised, (x,y,z)

    @torch.no_grad()
    def vae_roundtrip(self, dvf_norm):
        sf = self.ma['dvf_scale_factor']
        z, _ = self.dvf_ae.encode(dvf_norm[None].to(self.dev) * sf)
        return (self.dvf_ae.decode(z) / sf)[0].cpu()


BASE = F.affine_grid(torch.eye(3, 4)[None], (1, 1, D, H, W), align_corners=False)


def warp(vol, dvf_norm):
    g = BASE + dvf_norm.permute(1, 2, 3, 0)[None]
    return F.grid_sample(vol[None, None], g, padding_mode='border',
                         mode='bilinear', align_corners=False)[0, 0]


def find_study(pt):
    for st in sorted(glob.glob(os.path.join(DATA, 'patient_%s' % pt, '*'))):
        vols = {}
        for p in glob.glob(os.path.join(st, '*_Gated,_*')):
            v = os.path.join(p, 'volume.pt')
            if not os.path.exists(v):
                continue
            try:
                tag = os.path.basename(p).split('Gated,_')[1].split('%')[0]
                vols[float(tag)] = v
            except Exception:
                pass
        if 0.0 in vols and len(vols) >= 10:
            return vols
    return None


def run_patient(pt, gen, n_draws=1):
    vols = find_study(pt)
    if vols is None:
        return None
    ks = sorted(vols)[:10]                                   # 0,10,...,90
    real = [torch.load(vols[k], map_location='cpu').float() for k in ks]
    v0 = real[0]
    body = v0 > -0.85

    cum = torch.zeros(3, D, H, W)
    cur = v0.clone()
    rows = []
    for i in range(9):                                       # phase index 0..8
        # the model is stochastic; the conditional MEAN is the optimal point
        # predictor under MSE, so give it its best shot rather than one draw
        pred_step = gen.step(cur, i)
        for _ in range(n_draws - 1):
            pred_step = pred_step + gen.step(cur, i)
        pred_step = pred_step / n_draws
        cum = cum + pred_step                                # authors' additive accum
        cur = warp(v0, cum)

        true_step = demons(real[i + 1].numpy(), real[i].numpy())
        true_cum = demons(real[i + 1].numpy(), real[0].numpy())

        r = {'phase': i}
        r['step_real'], r['step_resid'], r['step_ratio'], r['step_cos'] = \
            metrics(pred_step * SCALE, true_step, body)
        r['cum_real'], r['cum_resid'], r['cum_ratio'], r['cum_cos'] = \
            metrics(cum * SCALE, true_cum, body)
        # VAE reconstruction control in the CORRECT channel order (supersedes E6)
        rec = gen.vae_roundtrip(true_step / SCALE) * SCALE
        _, _, r['vae_ratio'], r['vae_cos'] = metrics(rec, true_step, body)
        r['fold_cum'] = folding(cum * SCALE)
        r['fold_true_cum'] = folding(true_cum)
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--patients', nargs='+', default=None)
    ap.add_argument('--delta_t', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_draws', type=int, default=4)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    gen = LDMGen(dev, a.delta_t)

    try:
        tr, va, te = split_patient_folders(DATA)
        nm = lambda L: set(os.path.basename(p).replace('patient_', '') for p in L)
        TR, VA, TE = nm(tr), nm(va), nm(te)
    except Exception:
        TR, VA, TE = set(), set(), set()

    pats = a.patients or sorted(os.path.basename(p).replace('patient_', '')
                                for p in glob.glob(os.path.join(DATA, 'patient_*')))

    print('=' * 106)
    print('  GATE 1 -- LDM PREDICTION of real 4D-Lung motion   device=%s  '
          'ddim delta_t=%d  seed=%d  n_draws=%d (conditional mean)' % (dev, a.delta_t, a.seed, a.n_draws))
    print('  DynaGAN reference on this cohort: resid/real median 0.703   '
          '(PASS = clearly below)')
    print('=' * 106)
    print('  %-8s %-5s %9s %8s %8s %9s %8s %9s %9s %9s' % (
        'patient', 'split', 'STEPratio', 'STEPcos', 'CUMreal', 'CUMratio',
        'CUMcos', 'VAEratio', 'fold_cum', 'fold_true'))
    print('  ' + '-' * 102)

    out, allr = {}, []
    t0 = time.time()
    outp = os.path.join(LDM, 'e7_gate1.json')
    for pt in pats:
        try:
            rows = run_patient(pt, gen, a.n_draws)
        except Exception as e:
            print('  %-8s FAILED %s' % (pt, str(e)[:70]))
            continue
        if not rows:
            continue
        out[pt] = rows
        allr += [dict(r, patient=pt) for r in rows]
        m = lambda k: float(np.mean([r[k] for r in rows]))
        sp = ('train' if pt in TR else 'val' if pt in VA
              else 'TEST' if pt in TE else '?')
        print('  %-8s %-5s %9.3f %8.3f %8.3f %9.3f %8.3f %9.3f %9.5f %9.5f' % (
            pt, sp, m('step_ratio'), m('step_cos'), m('cum_real'),
            m('cum_ratio'), m('cum_cos'), m('vae_ratio'), m('fold_cum'),
            m('fold_true_cum')))
        json.dump({'per_patient': out}, open(outp, 'w'), indent=2)

    if not allr:
        print('\n  nothing ran')
        return

    def agg(rows, name):
        if not rows:
            return
        f = lambda k: np.array([x[k] for x in rows])
        print('  %-13s n=%-4d STEP ratio %.3f cos %.3f | CUM ratio mean %.3f '
              'median %.3f cos %.3f | VAE ratio %.3f | fold_cum %.5f'
              % (name, len(rows), f('step_ratio').mean(), f('step_cos').mean(),
                 f('cum_ratio').mean(), np.median(f('cum_ratio')),
                 f('cum_cos').mean(), f('vae_ratio').mean(),
                 f('fold_cum').mean()))

    print('\n' + '=' * 106)
    print('  SUMMARY (%.0fs)' % (time.time() - t0))
    print('=' * 106)
    agg(allr, 'ALL')
    agg([r for r in allr if r['patient'] in TR], 'TRAIN')
    agg([r for r in allr if r['patient'] in VA], 'VAL')
    agg([r for r in allr if r['patient'] in TE], 'TEST')
    cm = float(np.median([r['cum_ratio'] for r in allr]))
    print('\n  DynaGAN median 0.703   vs   LDM CUM median %.3f   ->   %s'
          % (cm, 'PASS' if cm < 0.703 else 'FAIL'))
    json.dump({'split': {'train': sorted(TR), 'val': sorted(VA),
                         'test': sorted(TE)}, 'per_patient': out},
              open(outp, 'w'), indent=2)
    print('  Saved -> %s' % outp)


if __name__ == '__main__':
    main()
