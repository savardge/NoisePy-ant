#!/usr/bin/env python3
"""Publication-quality SMOOTH dispersion pick-density image (no striping).

Why the standard histograms stripe: after QC's group_scale_dedupe there is one group pick
per CWT scale, and above ~2 s the log-spaced scales are further apart than the 0.1 s
nominal grid, so nominal-period columns that no scale reaches are near-empty. Those
stripes are real on the NOMINAL axis -- but the nominal period is only a label. Each
pick's true period is its CWT scale (`T_scale`).

This script therefore bins picks at (T_scale, U) on a FINE log-period grid and convolves
with a Gaussian kernel matched to the measurement resolution:
    sigma_T(T) = the local MEASUREMENT SPACING: one CWT rung (5.95%%) above the ~1.7 s
                 crossover, but the 0.1 s nominal step below it (the picker measured on
                 the nominal grid there, and the populated rungs inherit its spacing --
                 a fixed one-rung kernel leaves residual banding at short T)
    sigma_U    = --sigma-u km/s (default 0.02 = two nodes of the 0.01 pick grid)
i.e. the image is smoothed AT the data's intrinsic resolution, not beyond it -- the
honest equivalent of drawing each measurement as its resolution cell.

Usage:
  python plot_dispersion_density_smooth.py --qc-csv .../picks_unified_QCd.csv \
         --out density.png --title "Riehen ..." [--sigma-u 0.02] [--per-period-norm]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

RUNG = 2.0 ** (1.0 / 12.0)          # CWT ladder ratio (dj = 1/12)
CLIP_PCT = 99.0
PANELS = [("rayleigh", "fundamental", "Rayleigh fundamental"),
          ("rayleigh", "overtone", "Rayleigh overtone"),
          ("love", "fundamental", "Love fundamental")]
MEDIAN_MIN_FRAC, MEDIAN_MIN_ABS = 0.25, 30


def smooth_var_t(h, sig_t_bins, sig_v):
    """Velocity smoothing (fixed sigma) + period smoothing with per-column sigma."""
    hv = gaussian_filter1d(h, sigma=sig_v, axis=1)
    nT = h.shape[0]
    out = np.zeros_like(hv)
    x = np.arange(nT)
    for i in range(nT):
        if not hv[i].any():
            continue
        g = np.exp(-0.5 * ((x - i) / sig_t_bins[i]) ** 2)
        out += g[:, None] * (hv[i][None, :] / g.sum())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--qc-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--tmin", type=float, default=0.2)
    ap.add_argument("--tmax", type=float, default=6.0)
    ap.add_argument("--vmin", type=float, default=0.2)
    ap.add_argument("--vmax", type=float, default=3.0)
    ap.add_argument("--nt", type=int, default=260, help="fine period bins (log-spaced)")
    ap.add_argument("--sigma-u", type=float, default=0.02,
                    help="velocity kernel sigma [km/s]")
    ap.add_argument("--per-period-norm", action="store_true",
                    help="normalise each period column by its max (FTAN-image look) "
                         "instead of showing absolute density")
    a = ap.parse_args()

    tedges = np.geomspace(a.tmin, a.tmax, a.nt + 1)
    vedges = np.arange(a.vmin - 0.005, a.vmax + 0.015, 0.01)
    # kernel sigmas in BINS: one rung in log-period, sigma-u in velocity
    dlogt = np.log(tedges[1] / tedges[0])
    tmid_all = np.sqrt(tedges[:-1] * tedges[1:])
    # per-column period kernel [bins]: measurement spacing = max(one rung, nominal 0.1 s)
    sig_t_bins = np.maximum(np.log(RUNG), 0.6 * 0.1 / tmid_all) / dlogt
    sig_v = a.sigma_u / 0.01

    hists = {}
    for ch in pd.read_csv(a.qc_csv, usecols=["T_scale", "group_velocity", "wave_type",
                                             "mode", "group_ok"], chunksize=2_000_000):
        ch = ch[ch["group_ok"] == 1].dropna(subset=["T_scale", "group_velocity"])
        for key, s in ch.groupby(["wave_type", "mode"], observed=True):
            h, _, _ = np.histogram2d(s["T_scale"], s["group_velocity"],
                                     bins=[tedges, vedges])
            hists[key] = hists.get(key, 0) + h

    fig, axes = plt.subplots(1, len(PANELS), figsize=(6.3 * len(PANELS), 5.4),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (wt, md, title) in zip(axes, PANELS):
        h = np.asarray(hists.get((wt, md), np.zeros((a.nt, len(vedges) - 1))), float)
        n = int(h.sum())
        hs = smooth_var_t(h, sig_t_bins, sig_v)
        occ = hs[hs > 0]
        vmax = np.percentile(occ, CLIP_PCT) if occ.size else 1.0
        img = hs.copy()
        if a.per_period_norm:
            mx = img.max(axis=1, keepdims=True)
            img = np.divide(img, mx, out=np.zeros_like(img), where=mx > 0)
            vmax = 1.0
        mesh = ax.pcolormesh(tedges, vedges, img.T, cmap="magma", vmin=0, vmax=vmax,
                             rasterized=True)
        # median curve from the UNsmoothed histogram, gated on column population,
        # contiguous run around the peak (see plot_dispersion_hist2d for rationale)
        c = 0.5 * (vedges[:-1] + vedges[1:])
        tmid = np.sqrt(tedges[:-1] * tedges[1:])
        tots = hs.sum(axis=1)
        ok = tots >= max(MEDIAN_MIN_ABS, MEDIAN_MIN_FRAC * tots.max())
        pk = int(np.argmax(tots))
        lo = pk
        while lo - 1 >= 0 and ok[lo - 1]:
            lo -= 1
        hi = pk
        while hi + 1 < len(tots) and ok[hi + 1]:
            hi += 1
        med = np.full(len(tmid), np.nan)
        for i in range(lo, hi + 1):
            med[i] = c[np.searchsorted(np.cumsum(hs[i]), tots[i] / 2.0)]
        ax.plot(tmid, med, color="deepskyblue", lw=1.6)
        ax.set_xscale("log")
        ax.set_xticks([0.3, 0.5, 1, 2, 3, 5])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlim(a.tmin, a.tmax)
        ax.set_ylim(a.vmin, a.vmax)
        ax.set_xlabel("Period [s]")
        ax.set_title("%s  (%s picks)" % (title, format(n, ",")), fontsize=10)
        cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("column-normalised density" if a.per_period_norm
                     else "pick density (p%g clip)" % CLIP_PCT, fontsize=8)
    axes[0].set_ylabel("Group velocity U [km/s]")
    if a.title:
        fig.suptitle("%s -- picks at their CWT-scale periods, kernel = max(1 rung, 0.1 s) x %.2f km/s"
                     % (a.title, a.sigma_u), fontsize=12)
    fig.tight_layout()
    fig.savefig(a.out, dpi=180, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
