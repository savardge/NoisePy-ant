#!/usr/bin/env python3
"""Drop picks that fall in sparsely-populated cells of the period-velocity histogram.

The idea: a pick sitting in a 2D-histogram cell that almost nothing else occupies is
either a genuine outlier branch or a mis-pick, and either way it is not supported by the
rest of the measurement population at that period. Cutting those cells should sharpen the
dispersion image the inversion is asked to fit.

CELLS ARE THE ONES YOU SEE IN picks_hist2d_*.png -- one column per CWT scale rung
(geometric midpoints between rungs) x 0.02 km/s in velocity, imported from
plot_exported_picks_hist2d so the filter cannot drift away from the figure it is read off.
The velocity edges are deliberately offset half a node (0.195, not 0.20): `group_velocity`
steps by exactly 0.01 km/s, and edges landing ON a node push picks into the bin below.

THRESHOLD IS A PERCENTILE, NOT A COUNT. A flat count is not comparable across tables: at
25 picks/cell it removed 65% of Riehen's Rayleigh overtone but 1% of Haute-Sorne's
fundamental, because the tables differ by an order of magnitude in size and in how widely
their picks scatter. Taking the p-th percentile of each table's OWN occupied-cell counts
removes a comparable fraction everywhere.

Usage:
  python filter_picks_hist2d.py --picks-dir <inputs_tspws> --out-dir <inputs_tspws_histfilt>
                                [--pct 75] [--net riehen]
"""
import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_exported_picks_hist2d import V_EDGES, rung_edges

WAVES = ("fund", "overtone", "love", "love_ot")
MEASURES = (("group", ""), ("phase", "_phase"))
TITLES = {"fund": "Rayleigh fundamental", "overtone": "Rayleigh overtone",
          "love": "Love fundamental", "love_ot": "Love overtone"}


def cell_index(d):
    """(period edges, per-pick cell counts, histogram) on the figure's own bins."""
    te = rung_edges(d["inst_period"].values)
    h, _, _ = np.histogram2d(d["inst_period"], d["group_velocity"], bins=[te, V_EDGES])
    ip = np.clip(np.digitize(d["inst_period"], te) - 1, 0, h.shape[0] - 1)
    iv = np.clip(np.digitize(d["group_velocity"], V_EDGES) - 1, 0, h.shape[1] - 1)
    return te, h[ip, iv], h


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--picks-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pct", type=float, default=75.0,
                    help="percentile of the occupied-cell count distribution used as the "
                         "per-table threshold (default 75 = drop the sparsest quartile)")
    ap.add_argument("--net", default="")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    # the station file is not filtered but the tomo config reads it from this directory
    for extra in ("stations_all.csv", "stations.csv", "stations_keepflag.csv"):
        s = os.path.join(a.picks_dir, extra)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(a.out_dir, extra))

    summary = []
    for measure, suf in MEASURES:
        panels = []
        for wave in WAVES:
            f = os.path.join(a.picks_dir, f"picks_{wave}_uni{suf}.csv")
            if not os.path.exists(f):
                continue
            d = pd.read_csv(f)
            out = os.path.join(a.out_dir, f"picks_{wave}_uni{suf}.csv")
            if len(d) < 100:                       # empty table (e.g. love_ot): pass through
                d.to_csv(out, index=False)
                continue
            te, cnt, h = cell_index(d)
            occ = h[h > 0]
            thr = float(np.percentile(occ, a.pct))
            keep = cnt >= thr
            d[keep].to_csv(out, index=False)

            # sidecar: carry the original provenance forward and record what we did
            ms, md = os.path.join(a.picks_dir, f"picks_{wave}_uni{suf}.csv.meta.json"), {}
            if os.path.exists(ms):
                md = json.load(open(ms))
            md["hist2d_cell_filter"] = dict(
                percentile=a.pct, threshold_picks_per_cell=thr,
                rows_before=int(len(d)), rows_after=int(keep.sum()),
                pct_dropped=round(100 * (1 - keep.mean()), 2),
                cells_occupied=int((h > 0).sum()), cells_kept=int((h >= thr).sum()),
                bins="one column per CWT rung x 0.02 km/s (plot_exported_picks_hist2d)",
                tool="filter_picks_hist2d.py")
            json.dump(md, open(out + ".meta.json", "w"), indent=1)

            summary.append(dict(measure=measure, wave=wave, thr=thr,
                                before=len(d), after=int(keep.sum()),
                                dropped=100 * (1 - keep.mean())))
            print("  %-6s %-9s thr=%5.1f picks/cell  %7d -> %7d  (-%.1f%%)"
                  % (measure, wave, thr, len(d), keep.sum(), 100 * (1 - keep.mean())),
                  flush=True)
            panels.append((wave, d, keep, te, thr))

        if panels:
            plot_panels(panels, a, measure, suf)

    if summary:
        pd.DataFrame(summary).to_csv(os.path.join(a.out_dir, "hist2d_filter_summary.csv"),
                                     index=False)


def plot_panels(panels, a, measure, suf):
    """Before / after / removed, one column per wave, on the filter's own bins."""
    n = len(panels)
    fig, axs = plt.subplots(3, n, figsize=(6.0 * n, 12.5), squeeze=False)
    for j, (wave, d, keep, te, thr) in enumerate(panels):
        for i, (sel, lab) in enumerate(((slice(None), "BEFORE"), (keep, "AFTER filter"),
                                        (~keep, "REMOVED"))):
            ax = axs[i][j]
            s = d[sel] if not isinstance(sel, slice) else d
            hh, _, _ = np.histogram2d(s["inst_period"], s["group_velocity"],
                                      bins=[te, V_EDGES])
            hh = np.where(hh > 0, hh, np.nan)
            vmax = np.nanpercentile(hh, 99) if np.isfinite(hh).any() else 1
            pc = ax.pcolormesh(te, V_EDGES, hh.T, cmap="inferno", vmin=0, vmax=vmax)
            ax.set_xscale("log")
            ax.set_xticks([0.3, 0.5, 1, 2, 3, 5])
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.set_xlabel("period [s]  (CWT scale rungs)")
            ax.set_ylabel("%s velocity [km/s]" % measure)
            ax.set_title("%s %s\n%s  --  %d picks" % (TITLES.get(wave, wave), measure,
                                                      lab, len(s)), fontsize=10)
            plt.colorbar(pc, ax=ax, shrink=0.85, label="picks per cell (p99 clip)")
    fig.suptitle("%s  %s: histogram-cell filter, threshold = p%g of each table's "
                 "occupied-cell counts" % (a.net or "", measure, a.pct),
                 fontsize=14, fontweight="bold", y=0.997)
    fig.tight_layout()
    o = os.path.join(a.out_dir, "picks_hist2d_filter_%s.png" % measure)
    fig.savefig(o, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", os.path.basename(o), flush=True)


if __name__ == "__main__":
    main()
