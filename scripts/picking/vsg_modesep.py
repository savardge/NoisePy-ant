"""Network-scale mode separation in the phase-velocity (VSG phase-shift) domain.

For each virtual source: load the symmetric CCF traces of ZZ/RR/RZ/ZR (aligned on the receivers
common to all four components), synthesize G_LR0/G_LR1 per receiver (Nayak & Thurber 2020 eqs 3/4,
linear stack), phase-only beamform (Park phase-shift) on the SAME (frequency, velocity) grid used
by the per-source VSG analysis (scripts/postprocess_stacks/phaseshift_dispersion.py), per-frequency
normalize, and stack |E| linearly across sources. Control: ZZ processed identically. If a
layered-model reference CSV is configured, overlays its branches and prints the mean
column-normalized amplitude along each branch in each image (suppression contrast).

Outputs (to paths.ref_dir unless --out): vsg_modesep_stacks_sign{+-1}.npz, vsg_modesep_sign{+-1}.png

Usage:  python vsg_modesep.py --config ../../param_files/modesep_params.yaml [--sign -1]
                              [--max-sources 20] [--out DIR]
Sanity check: rerun with the opposite --sign; the fundamental/overtone branches must swap.
"""
import argparse
import glob
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
                help="sign of the +/-pi/2 phase corrections (+1 = paper sign)")
ap.add_argument("--max-sources", type=int, default=None, help="limit sources for quick tests")
ap.add_argument("--out", default=None, help="output directory (default: paths.ref_dir)")
args = ap.parse_args()

cfg = modesep_config.load_config(args.config)
modesep_config.apply_overrides(cfg, args, {"sign": ("vsg_modesep", "sign"),
                                           "max_sources": ("vsg_modesep", "max_sources")})
vc = cfg["vsg_modesep"]
D = cfg["paths"]["vsg_dir"]
OUT = args.out or cfg["paths"]["ref_dir"]
os.makedirs(OUT, exist_ok=True)
SIGN = int(vc.get("sign", 1))
NSRC = vc.get("max_sources") or 10**9
MIN_RX = int(vc.get("min_common_receivers", 20))
NET = cfg["network"].get("name", "")

srcs = sorted(os.path.basename(p).replace('.npz', '')
              for p in glob.glob(os.path.join(D, 'ZZ', 'sources', '*.npz')))[:NSRC]

z0 = np.load(os.path.join(D, 'ZZ', 'sources', srcs[0] + '.npz'), allow_pickle=True)
fgrid, vel, dt = z0['f'], z0['vel'], float(z0['dt'])
# NFFT of the per-source analysis: the stored f grid is a contiguous band of
# rfftfreq(nfft, dt) bins, so df = 1/(nfft*dt). YAML vsg_modesep.nfft overrides.
NFFT = int(vc.get("nfft") or round(1.0 / ((fgrid[1] - fgrid[0]) * dt)))
print(f"NFFT = {NFFT} ({'from config' if vc.get('nfft') else 'derived from f grid'})")
fr = np.fft.rfftfreq(NFFT, d=dt)
fidx = np.array([int(np.argmin(np.abs(fr - fv))) for fv in fgrid])
assert np.max(np.abs(fr[fidx] - fgrid)) < 1e-6
w = 2 * np.pi * fgrid


def beamform(traces, x):
    """Phase-only Park phase-shift: E(c, f) = sum_j e^{i phi_j(f)} e^{+i w x_j / c}, |E| returned."""
    S = np.fft.rfft(traces, n=NFFT, axis=1)[:, fidx]     # (nrx, nf) complex spectra
    mag = np.abs(S)
    U = np.where(mag > 0, S / mag, 0)                    # unit phasors (phase only)
    E = np.empty((len(vel), len(fgrid)), dtype=complex)
    for ic, c in enumerate(vel):
        ph = np.exp(1j * np.outer(x, w / c))
        E[ic] = (U * ph).sum(axis=0)
    return np.abs(E)


acc = {k: np.zeros((len(vel), len(fgrid))) for k in ('G_LR0', 'G_LR1', 'ZZ')}
nused = 0
for s in srcs:
    try:
        d = {}
        for comp in ('ZZ', 'RR', 'RZ', 'ZR'):
            z = np.load(os.path.join(D, comp, 'sources', s + '.npz'), allow_pickle=True)
            d[comp] = dict(sym=np.asarray(z['sym'], float), x=np.asarray(z['x'], float),
                           rx=[str(r) for r in z['rx_codes']])
        common = set(d['ZZ']['rx'])
        for c in ('RR', 'RZ', 'ZR'):
            common &= set(d[c]['rx'])
        common = sorted(common)
        if len(common) < MIN_RX:
            continue
        idx = {c: [d[c]['rx'].index(r) for r in common] for c in d}
        tr = {c: d[c]['sym'][idx[c]] for c in d}
        x = d['ZZ']['x'][idx['ZZ']]
        n = tr['ZZ'].shape[1]
        rz_c = np.fft.rfft(tr['RZ'], axis=1) * np.exp(1j * np.pi / 2 * SIGN)
        zr_c = np.fft.rfft(tr['ZR'], axis=1) * np.exp(-1j * np.pi / 2 * SIGN)
        g0 = tr['ZZ'] + tr['RR'] + np.fft.irfft(rz_c + zr_c, n=n, axis=1)   # eq (3)
        g1 = tr['ZZ'] + tr['RR'] + np.fft.irfft(-rz_c - zr_c, n=n, axis=1)  # eq (4)
        for key, T in (('G_LR0', g0), ('G_LR1', g1), ('ZZ', tr['ZZ'])):
            E = beamform(T, x)
            mx = E.max(axis=0, keepdims=True)            # per-frequency normalize
            acc[key] += E / np.where(mx > 0, mx, 1)
        nused += 1
        if nused % 25 == 0:
            print(f'{nused} sources done', flush=True)
    except Exception as e:
        print(f'{s}: skip ({e})', flush=True)

for k in acc:
    acc[k] /= max(nused, 1)
np.savez(os.path.join(OUT, f'vsg_modesep_stacks_sign{SIGN:+d}.npz'),
         f=fgrid, vel=vel, n_sources=nused, **acc)

# ---- figure + branch contrast vs the model reference curves (overlay optional) ----
_ref_csv = cfg["paths"].get("model_reference_csv")
if _ref_csv and os.path.exists(_ref_csv):
    ref = pd.read_csv(_ref_csv)
    ref = ref[(ref.wave == 'rayleigh') & (ref.component == 'pooled')]
    br = {b: ref[ref.branch == b].sort_values('frequency') for b in ('fundamental', 'overtone')}
else:
    br = None


def branch_amp(img, b):
    vals = []
    for _, r in br[b].iterrows():
        i_f = int(np.argmin(np.abs(fgrid - r.frequency)))
        i_v = int(np.argmin(np.abs(vel - r.c_ref)))
        col = img[:, i_f]
        vals.append(col[i_v] / col.max() if col.max() > 0 else np.nan)
    return float(np.nanmean(vals))


flim = tuple(vc.get("plot_flim", (0.15, 2.0)))
fig, axs = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
for ax, k in zip(axs, ('ZZ', 'G_LR0', 'G_LR1')):
    ax.pcolormesh(fgrid, vel, acc[k], cmap='jet', shading='auto')
    if br is not None:
        for b, colr in (('fundamental', 'w'), ('overtone', 'k')):
            ax.plot(br[b].frequency, br[b].c_ref, colr, lw=2,
                    label=f'{b} c_ref' if k == 'ZZ' else None)
    ax.set(title=f'{k}  (stack of {nused} sources, sign {SIGN:+d})',
           xlabel='Frequency [Hz]', xlim=flim, ylim=(0.5, 5.0))
axs[0].set_ylabel('Phase velocity [km/s]')
if br is not None:
    axs[0].legend(loc='upper right', fontsize=8)
if NET:
    fig.suptitle(f'{NET}: network-stacked mode-separated phase-velocity images', y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(OUT, f'vsg_modesep_sign{SIGN:+d}.png'), dpi=120)

print(f'\nsources used: {nused}   (sign {SIGN:+d})')
if br is not None:
    print(f'{"image":>7} | {"amp@fund":>9} | {"amp@overtone":>12}   (col-normalized, mean along branch)')
    for k in ('ZZ', 'G_LR0', 'G_LR1'):
        print(f'{k:>7} | {branch_amp(acc[k], "fundamental"):9.3f} | {branch_amp(acc[k], "overtone"):12.3f}')
