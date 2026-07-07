"""Pick single-mode phase-velocity reference curves from the mode-separated network images.

Input : vsg_modesep_stacks_sign{+-1}.npz in paths.ref_dir (produced by vsg_modesep.py) -- the
        network-stacked phase-velocity images for ZZ, G_LR0 (fundamental) and G_LR1 (1st overtone).
Method: per-frequency amplitude maximum (argmax) inside a mode-specific search window; no
        Viterbi, no smoothing, no forward model. QC = dominance trim + roughness gate + longest
        contiguous run. Windows and gates come from the YAML `pick_windows` section and are
        genuinely network-tuned: read them off the stacked figure before picking.
Output: (in paths.ref_dir) picked_reference_curves.csv, ref_fundamental_phase.txt,
        ref_overtone_phase.txt (consumed by dispersion_curves_V6_modesep.py /
        dispersion_batch_modesep.py for the 2piN phase resolution), and two diagnostic PNGs.
See VSG_REFERENCE_METHODOLOGY.md for the full description.

Usage:  python pick_reference_ridges.py --config ../../param_files/modesep_params.yaml [--sign 1]
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import modesep_config

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--config", required=True, help="network YAML (param_files/modesep_params.yaml)")
ap.add_argument("--sign", type=int, choices=(1, -1), default=None,
                help="which sign's stacked npz to pick from (default: vsg_modesep.sign)")
args = ap.parse_args()

cfg = modesep_config.load_config(args.config)
modesep_config.apply_overrides(cfg, args, {"sign": ("vsg_modesep", "sign")})
OUT = cfg["paths"]["ref_dir"]
SIGN = int(cfg["vsg_modesep"].get("sign", 1))
pw = cfg["pick_windows"]
WINDOWS = {b: dict(pw[b]) for b in ("fundamental", "overtone")}
ROUGH_WIN = int(pw.get("rough_win", 9))    # samples in the rolling roughness window
MIN_RUN = int(pw.get("min_run", 10))       # min samples per kept contiguous run

z = np.load(os.path.join(OUT, f'vsg_modesep_stacks_sign{SIGN:+d}.npz'))
f, vel = z['f'], z['vel']


def pick(img, fmin, fmax, vmin, vmax, dom_min, rough_max, max_gap=2, **_):
    """Per-frequency argmax within the window. Returns (freq, c, dominance) for kept columns.

    QC filters (no smoothing is ever applied to the retained picks):
      1. dominance = in-window peak / full-column maximum (0..1): =1 when the targeted mode is
         the strongest arrival at that frequency. Picks with dominance < dom_min are dropped.
      2. roughness: rolling mean of |c_{k+1}-c_k| over ROUGH_WIN consecutive picks. On a real
         ridge consecutive argmax picks move by ~1-2 velocity-grid steps (dvel=0.01 km/s);
         where the ridge dissolves into noise the argmax jumps by many steps, so the local
         roughness rises an order of magnitude. Picks where it exceeds rough_max are dropped.
         This is the automated replacement for a manual high-frequency clip.
      3. contiguous runs of surviving frequencies shorter than MIN_RUN samples are dropped
         (gaps <= max_gap bridged) -- removes isolated spurious columns while keeping every
         substantial clean segment.
    """
    fi = np.where((f >= fmin) & (f <= fmax))[0]
    vi = np.where((vel >= vmin) & (vel <= vmax))[0]
    col_max = img[:, fi].max(axis=0)                           # full-column max per frequency
    win = img[np.ix_(vi, fi)]
    kmax = np.argmax(win, axis=0)                              # argmax within window
    peak = win[kmax, np.arange(len(fi))]
    dom = peak / np.where(col_max > 0, col_max, np.nan)
    cpick = vel[vi][kmax]
    good = dom >= dom_min
    # rolling roughness of the raw argmax curve
    dc = np.abs(np.diff(cpick))
    dc = np.concatenate([dc[:1], dc])                          # align to samples
    kernel = np.ones(ROUGH_WIN) / ROUGH_WIN
    rough = np.convolve(dc, kernel, mode='same')
    good &= rough <= rough_max
    if good.any():                                            # drop short isolated runs
        idx = np.where(good)[0]
        runs = np.split(idx, np.where(np.diff(idx) > max_gap)[0] + 1)
        good = np.zeros_like(good)
        for r in runs:
            if len(r) >= MIN_RUN:
                good[r] = True
    return f[fi][good], cpick[good], dom[good]


picks_rows = []
curves = {}
for branch, w in WINDOWS.items():
    fp, cp, sp = pick(z[w['img']], w['fmin'], w['fmax'], w['vmin'], w['vmax'],
                      w['dom_min'], w['rough_max'])
    curves[branch] = (fp, cp, w['img'])
    for fi, ci, si in zip(fp, cp, sp):
        picks_rows.append(('rayleigh', branch, w['img'], fi, 1 / fi, ci, si))

picks = pd.DataFrame(picks_rows, columns=['wave', 'branch', 'from_image',
                                          'frequency', 'period', 'c_phase', 'dominance'])
picks.to_csv(os.path.join(OUT, 'picked_reference_curves.csv'), index=False, float_format='%.4f')

# 2-column (period[s]  c[km/s]) reference files for noisepy load_reference_curve / V6
for branch, fname in (('fundamental', 'ref_fundamental_phase.txt'),
                      ('overtone', 'ref_overtone_phase.txt')):
    fp, cp, _ = curves[branch]
    order = np.argsort(1 / fp)
    np.savetxt(os.path.join(OUT, fname),
               np.column_stack([(1 / fp)[order], cp[order]]), fmt='%.4f',
               header='period[s]  phase_velocity[km/s]  (picked from mode-separated VSG stack)')

# ---------- figures: frequency version and period version, both log-x ----------
# layered-model overlay is optional (paths.model_reference_csv)
_mod_csv = cfg["paths"].get("model_reference_csv")
if _mod_csv and os.path.exists(_mod_csv):
    mod = pd.read_csv(_mod_csv)
    mod = mod[(mod.wave == 'rayleigh') & (mod.component == 'pooled')]
else:
    mod = None


def make_fig(xaxis):
    """xaxis = 'frequency' or 'period'."""
    xf = (lambda q: q) if xaxis == 'frequency' else (lambda q: 1.0 / q)
    xgrid = xf(f)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.2))
    for ax, branch in zip(axs[:2], ('fundamental', 'overtone')):
        fp, cp, imk = curves[branch]
        ax.pcolormesh(xgrid, vel, z[imk], cmap='jet', shading='auto')
        w = WINDOWS[branch]
        x0, x1 = sorted([xf(w['fmin']), xf(w['fmax'])])
        ax.add_patch(plt.Rectangle((x0, w['vmin']), x1 - x0, w['vmax'] - w['vmin'],
                                   fill=False, ec='w', ls=':', lw=1))
        ax.plot(xf(fp), cp, 'w.', ms=4, label=f'picked {branch}')
        ax.set(title=f'{branch} from {imk}  (argmax in dotted window)', xscale='log',
               xlabel=('Frequency [Hz]' if xaxis == 'frequency' else 'Period [s]'),
               ylim=(0.5, 5))
        ax.set_xlim(sorted([xf(0.16), xf(2.0)]))
        ax.legend(loc='upper right', fontsize=9)
    axs[0].set_ylabel('Phase velocity [km/s]')
    ax = axs[2]
    for branch, col in (('fundamental', 'r'), ('overtone', 'b')):
        fp, cp, _ = curves[branch]
        ax.plot(xf(fp), cp, col + '.', ms=5, label=f'picked {branch}')
        if mod is not None:
            m = mod[mod.branch == branch].sort_values('frequency')
            ax.plot(xf(np.asarray(m.frequency)), m.c_ref, col + '--', lw=1.3, alpha=0.7,
                    label=f'model {branch} (CSV)')
    ax.set(title='picked (data) vs model (layered forward model)', xscale='log',
           xlabel=('Frequency [Hz]' if xaxis == 'frequency' else 'Period [s]'), ylim=(1, 4.5))
    ax.set_xlim(sorted([xf(0.16), xf(2.0)]))
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    tag = 'freq' if xaxis == 'frequency' else 'period'
    fig.savefig(os.path.join(OUT, f'picked_reference_curves_{tag}.png'), dpi=130)
    plt.close(fig)


make_fig('frequency')
make_fig('period')

for branch in ('fundamental', 'overtone'):
    fp, cp, _ = curves[branch]
    if len(fp) == 0:
        print(f'{branch:>11}: 0 picks (window/QC rejected all -- check pick_windows[{branch!r}])')
        continue
    print(f'{branch:>11}: {len(fp)} picks | {1/fp.max():.2f}-{1/fp.min():.2f} s '
          f'({fp.min():.2f}-{fp.max():.2f} Hz) | c {cp.min():.2f}-{cp.max():.2f} km/s')
