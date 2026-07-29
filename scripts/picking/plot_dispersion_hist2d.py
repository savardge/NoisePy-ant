#!/usr/bin/env python3
"""2D dispersion histograms (period vs group velocity) before and/or after QC.

Works on any unified-picking tree:
    --pre-dir   directory holding the per-pair <pair>_unified.csv files (pre-QC)
    --qc-csv    picks_unified_QCd.csv written by qc_unified_picks.py (post-QC)

Give either or both; one figure row per input. Counts are accumulated in CHUNKS and only
the 2D histograms are kept, so a 31-million-row tree (Haute-Sorne) costs the same memory
as a small one. The per-period median is derived from the histogram itself rather than
from the rows, which keeps the whole thing streaming.

Panels are discovered from the data, so a network carrying a love-overtone stream gets an
extra column automatically.

Examples:
    python plot_dispersion_hist2d.py \
        --pre-dir Projects/riehen/dispersion_unified \
        --qc-csv  Projects/riehen/dispersion_unified/picks_unified_QCd.csv \
        --out     Projects/riehen/dispersion_unified/hist2d_preQC_vs_postQC.png \
        --title   "Riehen - unified picking"
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Both axes are DISCRETE grids: nominal_period on an exact 0.1 s step, group_velocity on
# an exact 0.01 km/s step. Bin edges must fall BETWEEN nodes, never on them -- an edge
# generated as 1.1000000000000003 by np.arange puts the 1.10 node in the bin below,
# starving one row and doubling the next. That produced fake horizontal lines at 1.09/1.13
# and 2.23/2.25 km/s (deficit 0.43x next to excess 1.58x) that read as data features.
# So: period edges offset by half a 0.1 s step, velocity edges by half a 0.01 km/s step.
V_EDGES = np.arange(0.095, 3.025, 0.02)

# PERIOD AXIS. Two grids exist in these files and they are NOT interchangeable:
#   nominal_period : uniform 0.1 s grid, complete, what the picker reports
#   T_scale        : the log-spaced CWT scale the measurement actually came from
# Above ~2 s the CWT scales are more than 0.1 s apart (Aargau: 2.032, 2.153, 2.281, 2.416,
# 2.560, 2.712, 2.874, 3.044 ...), so no scale rounds onto nominal 2.1, 2.5, 2.8, 3.1, 3.3.
# qc_unified_picks.py's group_scale_dedupe keeps ONE pick per CWT scale, so after QC those
# nominal periods are strongly depleted (Aargau Rayleigh fund: 5,638 at T=2.1 against
# ~20,000 at 2.0 and 2.2). Those columns in the post-QC panels are REAL, not a binning bug.
#
# The rule is exact: a nominal column is FULL if some scale lies within 0.05 s (half the
# nominal step) of it, near-empty otherwise -- each depleted node loses its nearest scale to
# the adjacent node (2.153 is 0.047 from 2.2 but 0.053 from 2.1, so 2.2 takes it).
#
# Binning on T_scale does NOT fix it: the streams populate DIFFERENT scale sets (Aargau
# Rayleigh fund 51, Rayleigh overtone 41), so union-derived bins go empty in whichever panel
# lacks that scale, and below ~1.5 s the scales are irregular (0.3805 -> 0.4032 is ratio
# 1.06, -> 0.4795 is 1.19). Use --period-axis merged for a stripe-free view that moves no
# data: variable-width columns, each depleted node folded into the neighbour that took its
# scale. The plain nominal grid stays the default because it is uniform and complete.
T_NOMINAL_EDGES = np.arange(0.15, 6.30, 0.10)


def scale_edges(scales, min_share=0.05):
    """Bin edges at geometric midpoints between the WELL-POPULATED CWT scales.

    The ladder is not uniformly used. Below ~1.7 s it is finer than the 0.1 s nominal grid,
    so each nominal pick snaps to its nearest rung and the populated set inherits the 0.1 s
    spacing; the rungs in between are reached only by a few long paths (Riehen: 17 of 46
    rungs carry 2-41 pairs out of 19,017). Giving those their own column produces alternating
    16k / 5-pick columns and a see-sawing median. Rungs carrying < min_share of the median
    rung's picks are dropped from the EDGE set, so their picks land in the neighbouring
    column instead -- nothing is discarded, only merged.
    """
    if isinstance(scales, pd.Series):
        keep = scales[scales >= min_share * scales.median()]
        scales = keep.index.values if len(keep) > 2 else scales.index.values
    s = np.sort(np.unique(scales))
    mid = np.sqrt(s[:-1] * s[1:])
    r = s[1] / s[0]
    return np.concatenate([[s[0] / np.sqrt(r)], mid, [s[-1] * np.sqrt(r)]])


def merged_nominal_edges(scales, nodes, step=0.10):
    """Variable-width columns on the nominal axis -- one column per CWT scale.

    Post-QC each scale contributes one pick, landing in the nominal node nearest to it, so
    nodes that no scale claims are left near-empty (see the T_scale note above). Here every
    nominal node is assigned to the node that claimed ITS nearest scale, which folds each
    depleted node into exactly the neighbour that took its scale; runs of nodes sharing an
    assignment become one wide column. Edges land midway between nodes (…x.x5), never on a
    node, so the float-on-node aliasing fixed for V_EDGES cannot come back.
    """
    # discover_scales returns a Series (scale -> pick count); the SCALES are its index. Taking
    # np.asarray() of it grabs the counts instead and silently destroys the period axis.
    if isinstance(scales, pd.Series):
        scales = scales.index.values
    s = np.sort(np.unique(np.asarray(scales, dtype=float)))
    nodes = np.sort(np.unique(np.round(np.asarray(nodes, dtype=float), 3)))
    # each node -> the nominal node that claims the scale nearest to it
    owner = np.round(s[np.abs(nodes[:, None] - s[None, :]).argmin(axis=1)] / step) * step
    edges = [nodes[0] - step / 2.0]
    for i in range(1, len(nodes)):
        if abs(owner[i] - owner[i - 1]) > 1e-9:      # ownership changes -> cut here
            edges.append(0.5 * (nodes[i - 1] + nodes[i]))
    edges.append(nodes[-1] + step / 2.0)
    return np.array(edges)

# Linear colour scale (no log), vmax clipped at a high percentile of the occupied cells so
# a single hot cell cannot flatten everything else. Colourbar is extended to say so.
CLIP_PCT = 99.0

# Median curve: a column must hold this fraction of the busiest column before its median is
# drawn. Tuned so the curve stops before the long-period count collapse that drags it down.
MEDIAN_MIN_FRAC = 0.25

VMIN_PROD = 0.5          # production velocity floor, drawn for reference
CHUNK = 500_000

COLS = ["nominal_period", "T_scale", "group_velocity", "wave_type", "mode"]
PCOL = "T_scale"      # set from --period-axis in main()
T_EDGES = None        # set once the scale grid is known
AXIS_LABEL = "nominal"
FOLD_LOVE_OT = False

LABELS = {
    ("rayleigh", "fundamental"): "Rayleigh fundamental",
    ("rayleigh", "overtone"):    "Rayleigh overtone",
    ("love",     "fundamental"): "Love fundamental",
    ("love",     "overtone"):    "Love overtone",
}
ORDER = [("rayleigh", "fundamental"), ("rayleigh", "overtone"),
         ("love", "fundamental"), ("love", "overtone")]


def _accumulate(hists, df):
    """Add one dataframe's rows into the per-(wave, mode) histogram dict."""
    df = df.dropna(subset=[PCOL, "group_velocity"])
    if FOLD_LOVE_OT:
        df = df.copy()
        df.loc[(df["wave_type"] == "love") & (df["mode"] == "overtone"),
               "mode"] = "fundamental"
    for key, s in df.groupby(["wave_type", "mode"], observed=True):
        h, _, _ = np.histogram2d(s[PCOL], s["group_velocity"],
                                 bins=[T_EDGES, V_EDGES])
        if key in hists:
            hists[key] += h
        else:
            hists[key] = h


def from_pre_dir(d):
    """Stream every <pair>_unified.csv under d into histograms."""
    fs = sorted(glob.glob(os.path.join(d, "**", "*_unified.csv"), recursive=True))
    print("  reading %s pair files ..." % format(len(fs), ","))
    hists, buf, nrow = {}, [], 0
    for i, f in enumerate(fs):
        try:
            t = pd.read_csv(f, usecols=COLS)
        except Exception:
            continue
        if len(t):
            buf.append(t)
            nrow += len(t)
        # flush periodically so the buffer never grows to the size of the tree
        if len(buf) >= 2000:
            _accumulate(hists, pd.concat(buf, ignore_index=True))
            buf = []
        if (i + 1) % 25000 == 0:
            print("    %s/%s" % (format(i + 1, ","), format(len(fs), ",")))
    if buf:
        _accumulate(hists, pd.concat(buf, ignore_index=True))
    print("  %s rows" % format(nrow, ","))
    return hists


def from_qc_csv(path):
    """Stream the QC'd merge file, keeping only surviving GROUP picks."""
    print("  reading %s ..." % os.path.basename(path))
    hists, nrow = {}, 0
    use = COLS + ["group_ok"]
    for chunk in pd.read_csv(path, usecols=use, chunksize=CHUNK):
        chunk = chunk[chunk["group_ok"] == 1]
        nrow += len(chunk)
        _accumulate(hists, chunk)
    print("  %s surviving group picks" % format(nrow, ","))
    return hists


def discover_scales(src):
    """Collect the CWT scale grid EXHAUSTIVELY.

    Sampling is not safe here: a 400-file sample of Aargau found 48 of the 52 scales, and
    picks at the 4 missed scales then fell outside every bin and gapped the figure. One
    full pass over a single column is cheap, so always take it.
    """
    cnt = {}
    def add(v):
        for k, n in v.dropna().round(4).value_counts().items():
            cnt[k] = cnt.get(k, 0) + int(n)
    if os.path.isdir(src):
        for f in sorted(glob.glob(os.path.join(src, "**", "*_unified.csv"), recursive=True)):
            try:
                add(pd.read_csv(f, usecols=["T_scale"])["T_scale"])
            except Exception:
                continue
    else:
        for ch in pd.read_csv(src, usecols=["T_scale"], chunksize=CHUNK):
            add(ch["T_scale"])
    return pd.Series(cnt).sort_index()


def median_from_hist(h, min_frac=MEDIAN_MIN_FRAC, min_abs=30.0):
    """Per-period median velocity, read off the histogram columns.

    The curve is drawn ONLY over the contiguous run of well-populated columns containing the
    busiest one. Two failure modes this avoids:
      * a column with a handful of picks gives a median that lands anywhere -> see-saw;
      * where the count collapses at long period the surviving picks skew SLOW, so the median
        bends back down and fakes a decreasing branch (Riehen Love fund: 1.765 km/s at T=3.0 s
        decaying to 1.465 at 4.7 s as the count falls to 7% of peak; Rayleigh fund turns over
        below 3% of peak). A group-velocity branch does not reverse -- that is the tail of the
        distribution, not the physics.
    A column qualifies with >= min_abs picks AND >= min_frac of the busiest column; the run is
    then grown outwards from the peak and stops at the first column that fails.
    """
    centers = 0.5 * (V_EDGES[:-1] + V_EDGES[1:])
    T = 0.5 * (T_EDGES[:-1] + T_EDGES[1:])
    med = np.full(len(T), np.nan)
    tots = h.sum(axis=1)
    if not tots.size or tots.max() <= 0:
        return T, med
    ok = (tots >= max(min_abs, min_frac * tots.max()))
    pk = int(np.argmax(tots))
    lo = pk
    while lo - 1 >= 0 and ok[lo - 1]:
        lo -= 1
    hi = pk
    while hi + 1 < len(tots) and ok[hi + 1]:
        hi += 1
    for i in range(lo, hi + 1):
        col = h[i]
        med[i] = centers[np.searchsorted(np.cumsum(col), tots[i] / 2.0)]
    return T, med


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pre-dir", default=None)
    ap.add_argument("--qc-csv", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--fold-love-overtone", action="store_true",
                    help="relabel love/overtone as love/fundamental while reading, matching "
                         "qc_unified_picks.py --fold-love-overtone, so the pre-QC row is "
                         "drawn on the same convention as a folded QC merge.")
    ap.add_argument("--vmax-plot", type=float, default=3.0,
                    help="upper edge of the velocity axis [km/s]. Default 3.0 shows the "
                         "branch; raise to 5.0 to see the whole range QC's vbounds admits, "
                         "which is what you need in order to choose a velocity ceiling.")
    ap.add_argument("--raw-counts", action="store_true",
                    help="with --period-axis merged, plot raw counts instead of picks per "
                         "0.1 s. Raw = one CWT scale per column, so the POST-QC row is "
                         "flattest; normalised (default) is flattest for the PRE-QC row, "
                         "which is still uniformly sampled on the nominal grid. Neither is "
                         "flat for both -- the two rows have genuinely different sampling.")
    ap.add_argument("--use-cache", action="store_true",
                    help="reuse the accumulated histograms from <out>.npz if present "
                         "(re-reading a 232k-pair tree takes ~30 min; restyling does not)")
    ap.add_argument("--period-axis", choices=["nominal", "merged", "scale"],
                    default="nominal",
                    help="nominal (default): uniform, complete 0.1 s grid; post-QC shows "
                         "REAL depleted columns above ~2 s (see the T_scale note at the "
                         "top). merged: variable-width columns, one per CWT scale, folding "
                         "each depleted node into the neighbour that took its scale -- no "
                         "stripes, no data moved, but columns above ~2 s are wider. "
                         "scale: bin on T_scale; gappy, since streams differ in which "
                         "scales they populate (Aargau fund 51, overtone 41).")
    a = ap.parse_args()
    if not a.pre_dir and not a.qc_csv:
        ap.error("give --pre-dir and/or --qc-csv")

    global PCOL, T_EDGES, AXIS_LABEL, FOLD_LOVE_OT, V_EDGES
    V_EDGES = np.arange(0.095, a.vmax_plot + 0.025, 0.02)
    FOLD_LOVE_OT = a.fold_love_overtone
    if a.period_axis == "nominal":
        PCOL, T_EDGES, AXIS_LABEL = "nominal_period", T_NOMINAL_EDGES, "nominal"
    elif a.period_axis == "merged":
        PCOL, AXIS_LABEL = "nominal_period", "nominal, merged per CWT scale"
        src = a.qc_csv or a.pre_dir
        T_EDGES = merged_nominal_edges(discover_scales(src),
                                       0.5 * (T_NOMINAL_EDGES[:-1] + T_NOMINAL_EDGES[1:]))
        w = np.diff(T_EDGES)
        print("period axis: nominal, merged to %d columns "
              "(width %.1f-%.1f s; %d nodes folded into a neighbour)"
              % (len(w), w.min(), w.max(), len(T_NOMINAL_EDGES) - 1 - len(w)))
    else:
        PCOL, AXIS_LABEL = "T_scale", "CWT scale"
        T_EDGES = scale_edges(discover_scales(a.qc_csv or a.pre_dir))
        print("period axis: T_scale, %d CWT scales (%.3f - %.3f s)"
              % (len(T_EDGES) - 1, T_EDGES[0], T_EDGES[-1]))

    cache = a.out + ".hist.npz"
    rows = []
    if a.use_cache and os.path.exists(cache):
        print("reusing histograms from %s" % os.path.basename(cache))
        z = np.load(cache, allow_pickle=True)
        for label in z["labels"]:
            hs = {tuple(k.split("|")): z["h_%s_%s" % (label, k)]
                  for k in z["keys_%s" % label]}
            rows.append((str(label), hs))
        T_EDGES = z["t_edges"]
    else:
        if a.pre_dir:
            print("[pre-QC]")
            rows.append(("before QC", from_pre_dir(a.pre_dir)))
        if a.qc_csv:
            print("[post-QC]")
            rows.append(("after QC", from_qc_csv(a.qc_csv)))
        save = {"labels": np.array([lab for lab, _ in rows]), "t_edges": T_EDGES}
        for lab, hs in rows:
            save["keys_%s" % lab] = np.array(["|".join(k) for k in hs])
            for k, h in hs.items():
                save["h_%s_%s" % (lab, "|".join(k))] = h
        np.savez_compressed(cache, **save)
        print("cached histograms -> %s" % os.path.basename(cache))

    # Columns = the (wave, mode) streams actually present, in a stable order
    present = [k for k in ORDER if any(k in h for _, h in rows)]
    if not present:
        raise SystemExit("no picks found")

    # Variable-width columns pool several nominal nodes, so a raw count makes the wide ones
    # look hot for no physical reason. Normalise to picks per 0.1 s of period; on the
    # uniform axes every width is 0.1 s and this is a no-op.
    wnorm = (np.diff(T_EDGES) / 0.10)[:, None]
    if a.raw_counts:
        wnorm = np.ones_like(wnorm)
    if not np.allclose(wnorm, 1.0):
        for _, hs in rows:
            for k in hs:
                hs[k] = hs[k] / wnorm

    # One colour limit PER ROW. QC removes ~40-60% of the picks, so a shared scale renders the
    # post-QC row uniformly dim and hides its structure; the rows are separate populations and
    # each is clipped to its own p99. Cross-row brightness is therefore NOT comparable -- read
    # the counts in the panel titles for that.
    vmaxes = []
    for _, hs in rows:
        occ = np.concatenate([h[h > 0].ravel() for h in hs.values()])
        vmaxes.append(np.percentile(occ, CLIP_PCT) if occ.size else 1.0)

    nr, nc = len(rows), len(present)
    fig, axes = plt.subplots(nr, nc, figsize=(5.5 * nc, 4.6 * nr),
                             sharex=True, sharey=True, squeeze=False)

    row_mesh = [None] * nr
    for i, (label, hs) in enumerate(rows):
        for j, key in enumerate(present):
            ax = axes[i][j]
            h = hs.get(key)
            if h is None:
                ax.set_facecolor("0.95")
                ax.set_title("%s -- %s  (none)" % (label, LABELS[key]), fontsize=10)
                continue
            # pcolormesh, NOT imshow: on the T_scale axis the period edges are GEOMETRIC,
            # and imshow assumes uniform cells -- it spreads log-spaced columns evenly
            # across a linear extent and mangles the branch. pcolormesh takes real edges.
            masked = np.ma.masked_where(h.T == 0, h.T)   # empty cells stay white
            row_mesh[i] = ax.pcolormesh(T_EDGES, V_EDGES, masked, cmap="magma",
                                        vmin=0, vmax=vmaxes[i])
            ax.axhline(VMIN_PROD, color="cyan", lw=1.1, ls="--")
            T, med = median_from_hist(h)
            ax.plot(T, med, color="deepskyblue", lw=1.5)
            ax.set_title("%s -- %s  (%s)" % (label, LABELS[key],
                                             format(int(round((h * wnorm).sum())), ",")),
                         fontsize=10)
            if i == nr - 1:
                ax.set_xlabel("Period T [s]  (%s)" % AXIS_LABEL)
            if j == 0:
                ax.set_ylabel("U [km/s]")
            ax.set_xlim(T_EDGES[0], T_EDGES[-1])
            if AXIS_LABEL == "CWT scale":
                ax.set_xscale("log")
                ax.set_xticks([0.3, 0.5, 1, 2, 3, 5])
                ax.get_xaxis().set_major_formatter(
                    matplotlib.ticker.ScalarFormatter())
            ax.set_ylim(V_EDGES[0], V_EDGES[-1])

    unit = ("picks per 0.1 s" if not np.allclose(wnorm, 1.0) else "number of picks")
    for i, (label, _) in enumerate(rows):
        if row_mesh[i] is None:
            continue
        cb = fig.colorbar(row_mesh[i], ax=list(axes[i]), fraction=0.015, pad=0.012,
                          extend="max")
        cb.set_label("%s -- %s (p%g)" % (label, unit, CLIP_PCT), fontsize=9)

    if a.title:
        fig.suptitle("%s  (cyan line = %.1f km/s production floor)"
                     % (a.title, VMIN_PROD), fontsize=13)

    fig.savefig(a.out, dpi=140, bbox_inches="tight")
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
