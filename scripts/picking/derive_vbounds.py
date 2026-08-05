#!/usr/bin/env python3
"""Derive a period-dependent UPPER velocity bound from the pick density itself.

Above the dispersion branch the histogram thins out into a sparse high-velocity haze --
body-wave arrivals, spurious cross-correlation peaks from local noise sources. Those cells
have a LOW PAIR COUNT, which is what makes them separable without hand-drawing a line.

Method, per (wave, mode) and per period column of the rung histogram:
  1. find the column's peak count (the branch),
  2. walk UP from the peak until the count falls below `--frac` of it (and below
     `--min-count` in absolute terms), and take that velocity as the raw bound,
  3. running-median smooth over `--smooth` rungs, because a per-column edge is noisy,
  4. multiply by `--margin` so the bound sits just clear of the branch rather than on it.

Written in the pick_vbounds.py schema, so the existing --bounds-file machinery applies it:
    {"bounds": {"rayleigh|fundamental": {"upper": [[T, v], ...]}, ...}}
Only `upper` is written: the slow side is left alone deliberately -- unconsolidated
sediments genuinely produce very slow short-period velocities, and those picks are real.

ONE FILE PER MEASURE. The schema has no measure axis, and phase velocity exceeds group, so
group and phase need separate files fed to the `export` and `export_phase` blocks.

CAVEAT: derived from the EXPORTED tables (already aggregated per pair and rung), then
applied during a later export, before aggregation. The distributions are close but not
identical; the QC figure shows what the bound actually cuts, so check it before trusting it.

Usage:
  python derive_vbounds.py --picks-dir <.../inputs_tspws> --net riehen --measure group \
      [--frac 0.05] [--smooth 5] [--margin 1.05] [--out FILE]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WAVES = {"fund": ("rayleigh", "fundamental"), "overtone": ("rayleigh", "overtone"),
         "love": ("love", "fundamental")}


def running_median(y, w):
    if w < 2 or len(y) < 3:
        return y
    out = np.copy(y)
    h = w // 2
    for i in range(len(y)):
        s = slice(max(0, i - h), min(len(y), i + h + 1))
        seg = y[s][np.isfinite(y[s])]
        if seg.size:
            out[i] = np.median(seg)
    return out


def edge_mad(v, k):
    """median + k*1.4826*MAD. The robust twin of mean+k*sigma: MAD is not inflated by the
    outlier tail, and 1.4826*MAD == sigma for a Gaussian, so k=2 reproduces '2 sigma'.
    Multi-peaked distributions survive this as long as the second geological mode sits
    within k robust-sigma of the median -- unlike a density walk-up, which stops at the
    first trough and can amputate a real second branch."""
    med = np.median(v)
    return med + k * 1.4826 * np.median(np.abs(v - med))


def edge_quantile(v, q):
    """Distribution-free: keep the lowest q of the column. Shape-agnostic, so multimodality
    is irrelevant -- but it removes the same FRACTION at every period, even where there is
    no haze to remove."""
    return np.percentile(v, q)


def enforce_nondecreasing(y):
    """Suffix minimum: y[i] <- min(y[i..end]). Makes the bound non-decreasing with period
    WITHOUT ever raising it, so a spurious short-period spike is pulled down to the level
    the longer periods support instead of propagating forward.

    Physical basis: in a normally dispersive medium the branch rises with period, so its
    upper envelope must not fall. A bound that climbs steeply toward the SHORT-period end
    -- where the medium is shallowest and slowest -- is an artifact of the column having no
    clear peak-and-gap there, not a real feature."""
    y = np.asarray(y, float)
    return np.minimum.accumulate(y[::-1])[::-1]


def derive_track(T, V, k, smooth, margin):
    """bound(T) = median(T) + k*1.4826*MADsmooth(T).

    The per-period MAD is smoothed ACROSS periods before use. Deriving each period
    independently lets a single broad column dominate: on Riehen fund the raw median+3MAD
    reaches 5.2 km/s at T<0.5 s, which then forced a crude non-decreasing constraint that
    flattened the bound to 1.96 -- BELOW the 2.73 branch median at T=0.2 s, cutting over
    half of those picks by accident.

    Smoothing the spread and adding it to the median means the bound TRACKS the branch:
    it inherits the branch's own period dependence, cannot invert, and needs no
    monotonicity hack. The median is robust to the outlier haze and to multimodality; the
    MAD sets how much geological spread is allowed around it."""
    rungs = np.unique(T)
    med = np.array([np.median(V[T == t]) for t in rungs])
    mad = np.array([np.median(np.abs(V[T == t] - m)) for t, m in zip(rungs, med)])
    n = np.array([np.sum(T == t) for t in rungs])
    ok = n >= 20
    if ok.sum() < 3:
        return None, None
    med_s = running_median(med[ok], smooth)
    mad_s = running_median(mad[ok], smooth)
    return rungs[ok], (med_s + k * 1.4826 * mad_s) * margin


def derive_one(T, V, frac, min_count, smooth, margin, vstep=0.02,
               method="density", k=2.0, q=97.5, monotonic=True):
    """(periods, upper) from the density edge above the branch."""
    if method == "track":
        return derive_track(T, V, k, smooth, margin)
    rungs = np.unique(T)
    vedges = np.arange(0.195, max(6.0, np.nanmax(V) + 0.1), vstep)
    vcent = 0.5 * (vedges[1:] + vedges[:-1])
    raw = np.full(len(rungs), np.nan)
    for i, t in enumerate(rungs):
        v = V[T == t]
        if v.size < 20:
            continue
        if method == "mad":
            raw[i] = edge_mad(v, k)
            continue
        if method == "quantile":
            raw[i] = edge_quantile(v, q)
            continue
        h, _ = np.histogram(v, bins=vedges)
        if not h.any():
            continue
        kk = int(np.argmax(h))                     # the branch for this period
        thr = max(frac * h[kk], min_count)
        j = kk
        while j + 1 < len(h) and h[j + 1] >= thr:  # walk up while still dense
            j += 1
        raw[i] = vcent[j]
    ok = np.isfinite(raw)
    if ok.sum() < 3:
        return None, None
    sm = running_median(raw[ok], smooth) * margin
    if monotonic:
        sm = enforce_nondecreasing(sm)
    return rungs[ok], sm


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--picks-dir", required=True)
    ap.add_argument("--net", required=True)
    ap.add_argument("--measure", default="group", choices=("group", "phase"))
    ap.add_argument("--frac", type=float, default=0.05,
                    help="column-peak fraction defining the density edge")
    ap.add_argument("--min-count", type=float, default=3.0)
    ap.add_argument("--smooth", type=int, default=5, help="running-median window, in rungs")
    ap.add_argument("--margin", type=float, default=1.05,
                    help="multiply the edge so the bound clears the branch")
    ap.add_argument("--method", default="track", choices=("track", "mad", "quantile", "density"),
                    help="track: median(T) + k*1.4826*MADsmooth(T) -- follows the branch, "
                         "no monotonic hack needed (DEFAULT). "
                         "mad: same but per-period, unsmoothed spread. "
                         "quantile: per-period upper percentile. density: walk up from the "
                         "column peak to where the count drops below --frac of it -- "
                         "adaptive, but it stops at the FIRST trough, which in a "
                         "multi-domain area can be the valley between two real branches.")
    ap.add_argument("--k", type=float, default=3.0, help="MAD multiplier for --method mad")
    ap.add_argument("--q", type=float, default=97.5, help="percentile for --method quantile")
    ap.add_argument("--no-monotonic", action="store_true",
                    help="disable the non-decreasing constraint (see enforce_nondecreasing)")
    ap.add_argument("--k-sweep", default="",
                    help="comma-separated k values to overlay (MAD method) so the "
                         "multiplier can be judged against the 2D pick distribution. "
                         "Writes no JSON.")
    ap.add_argument("--compare", action="store_true",
                    help="overlay all three methods on the figure and write no JSON")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    suffix = "_phase" if a.measure == "phase" else ""
    bounds, report = {}, []
    _cmp, _pend = {}, []
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 5), squeeze=False)
    for j, (key, (wt, md)) in enumerate(WAVES.items()):
        fn = os.path.join(a.picks_dir, "picks_%s_uni%s.csv" % (key, suffix))
        ax = axs[0][j]
        if not os.path.exists(fn):
            ax.axis("off"); continue
        d = pd.read_csv(fn)
        if not len(d):
            ax.axis("off"); continue
        T = d["inst_period"].to_numpy(float)
        V = d["group_velocity"].to_numpy(float)   # holds phase in the _phase tables
        if a.k_sweep:
            ks = [float(x) for x in a.k_sweep.split(",") if x.strip()]
            cols = plt.cm.viridis(np.linspace(0.15, 0.95, len(ks)))
            for kk, col in zip(ks, cols):
                Pk, Uk = derive_one(T, V, a.frac, a.min_count, a.smooth, a.margin,
                                    method=a.method, k=kk,
                                    monotonic=not a.no_monotonic)
                if Pk is None:
                    continue
                cutk = int((V > np.interp(T, Pk, Uk, left=Uk[0], right=Uk[-1])).sum())
                _pend.append((ax, Pk, Uk, col, "k=%g  cuts %.1f%%" % (kk, 100*cutk/len(d))))
                _cmp.setdefault(key, []).append(("k=%g" % kk, cutk, len(d)))
        if a.compare:
            for meth, col, lab in (("mad", "deepskyblue", "median+%.1f MAD" % a.k),
                                   ("quantile", "lime", "p%.1f" % a.q),
                                   ("density", "red", "density edge")):
                Pc, Uc = derive_one(T, V, a.frac, a.min_count, a.smooth, a.margin,
                                    method=meth, k=a.k, q=a.q,
                                    monotonic=not a.no_monotonic)
                if Pc is not None:
                    cutc = int((V > np.interp(T, Pc, Uc, left=Uc[0], right=Uc[-1])).sum())
                    _cmp.setdefault(key, []).append((meth, cutc, len(d)))
                    _pend.append((ax, Pc, Uc, col, "%s (%.1f%%)" % (lab, 100 * cutc / len(d))))
        P, U = derive_one(T, V, a.frac, a.min_count, a.smooth, a.margin,
                          method=a.method, k=a.k, q=a.q,
                          monotonic=not a.no_monotonic)
        rungs = np.unique(T)
        redges = np.concatenate(([rungs[0] ** 2 / np.sqrt(rungs[1] * rungs[0])],
                                 np.sqrt(rungs[1:] * rungs[:-1]),
                                 [rungs[-1] ** 2 / np.sqrt(rungs[-1] * rungs[-2])]))
        vedges = np.arange(0.195, np.nanmax(V) + 0.1, 0.02)
        H, _, _ = np.histogram2d(T, V, bins=[redges, vedges])
        pos = H[H > 0]
        ax.pcolormesh(redges, vedges, np.where(H > 0, H, np.nan).T, cmap="magma",
                      vmin=0, vmax=np.percentile(pos, 99) if pos.size else 1,
                      shading="flat")
        for _ax, _P, _U, _c, _l in [x for x in _pend if x[0] is ax]:
            _ax.plot(_P, _U, "-", color=_c, lw=1.8, label=_l)
        if a.compare or a.k_sweep:
            ax.legend(fontsize=7, loc="upper left", framealpha=0.85)
            ax.set_title("%s %s (%s)" % (a.net, key, a.measure), fontsize=10)
            ax.set_xlabel("period [s]"); ax.set_ylabel("%s velocity [km/s]" % a.measure)
            continue
        if P is not None:
            ax.plot(P, U, "-", color="deepskyblue", lw=2, label="derived upper bound")
            cut = int((V > np.interp(T, P, U, left=U[0], right=U[-1])).sum())
            report.append((key, len(d), cut))
            bounds["%s|%s" % (wt, md)] = {"upper": [[float(t), float(u)] for t, u in zip(P, U)]}
            ax.legend(fontsize=8, loc="upper left")
            ax.set_title("%s %s (%s)\ncuts %s of %s (%.1f%%)"
                         % (a.net, key, a.measure, format(cut, ","), format(len(d), ","),
                            100 * cut / len(d)), fontsize=10)
        ax.set_xlabel("period [s]"); ax.set_ylabel("%s velocity [km/s]" % a.measure)

    if a.k_sweep:
        fig.suptitle("%s %s: upper bound = median + k x 1.4826 MAD per period, "
                     "non-decreasing, x%.2f margin -- choose k against the pick density"
                     % (a.net, a.measure, a.margin), y=1.02)
    else:
        fig.suptitle("%s %s: density-derived upper velocity bound "
                     "(edge where the column count falls below %g x its peak, x%.2f margin)"
                     % (a.net, a.measure, a.frac, a.margin), y=1.02)
    fig.tight_layout()
    out = a.out or os.path.join(a.picks_dir, "vbounds_%s_%s.json" % (a.net, a.measure))
    fig.savefig(out.replace(".json", ".png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    if a.compare or a.k_sweep:
        print("comparison only -- no JSON written")
        for k_, rows in _cmp.items():
            print("  %-9s %s" % (k_, "  ".join("%s %.1f%%" % (m, 100 * c / n)
                                               for m, c, n in rows)))
        return
    json.dump({"bounds": bounds}, open(out, "w"), indent=2)
    print("wrote %s and %s" % (os.path.basename(out),
                               os.path.basename(out.replace(".json", ".png"))))
    for k, n, c in report:
        print("  %-9s %8s picks -> %7s above the bound (%.1f%%)"
              % (k, format(n, ","), format(c, ","), 100 * c / n))


if __name__ == "__main__":
    main()
