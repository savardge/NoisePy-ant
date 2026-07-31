#!/usr/bin/env python3
"""2D histograms of the EXPORTED tomography pick tables (picks_<wave>_uni*.csv).

These are the files the tomography actually inverts, one row per (station pair, period)
after median aggregation -- NOT the per-pick QC output. Period is the CWT scale ladder
(`inst_period` holds T_scale on the default --period-axis scale), so the period bins are
built from the rungs present in the file: no uniform-grid striping, no smoothing.

Panels: one per wave table found. Also plots ray count vs period, which is what actually
limits each period map.

    python plot_exported_picks_hist2d.py --dir <.../tomo/1_velocity_maps/inputs> \
        --title "Riehen" [--suffix ""]
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WAVES = [("fund", "Rayleigh fundamental"), ("overtone", "Rayleigh overtone"),
         ("love", "Love fundamental"), ("love_ot", "Love overtone")]
CLIP_PCT = 99.0
V_EDGES = np.arange(0.195, 5.025, 0.02)


def rung_edges(periods):
    """Bin edges at geometric midpoints between the period rungs present in the file."""
    s = np.sort(np.unique(np.round(periods, 4)))
    if len(s) < 2:
        return np.array([s[0] * 0.97, s[0] * 1.03])
    mid = np.sqrt(s[:-1] * s[1:])
    r = s[1] / s[0]
    return np.concatenate([[s[0] / np.sqrt(r)], mid, [s[-1] * np.sqrt(r)]])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--suffix", default="", help="pick table suffix (e.g. _phase)")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--xscale", default="linear", choices=("linear", "log"),
                    help="period axis scale. linear (default) shows the true period "
                         "spacing; log spreads the short periods where the CWT rungs are "
                         "dense. The BINS are the rung ladder either way.")
    a = ap.parse_args()

    found = []
    for key, label in WAVES:
        fn = os.path.join(a.dir, "picks_%s_uni%s.csv" % (key, a.suffix))
        if not os.path.exists(fn):
            continue
        d = pd.read_csv(fn)
        if not len(d):
            print("  %-9s empty -- skipped" % key)
            continue
        meta = {}
        if os.path.exists(fn + ".meta.json"):
            meta = json.load(open(fn + ".meta.json"))
        found.append((key, label, d, meta))
        print("  %-9s %s rows | %s pairs | T %.2f-%.2f | U %.2f-%.2f"
              % (key, format(len(d), ","), format(d.station_pair.nunique(), ","),
                 d.inst_period.min(), d.inst_period.max(),
                 d.group_velocity.min(), d.group_velocity.max()))
    if not found:
        raise SystemExit("no pick tables in %s" % a.dir)

    n = len(found)
    fig, axes = plt.subplots(2, n, figsize=(6.0 * n, 9.4), squeeze=False,
                             gridspec_kw={"height_ratios": [2.2, 1]})
    for j, (key, label, d, meta) in enumerate(found):
        te = rung_edges(d["inst_period"].values)
        h, _, _ = np.histogram2d(d["inst_period"], d["group_velocity"],
                                 bins=[te, V_EDGES])
        occ = h[h > 0]
        vmax = np.percentile(occ, CLIP_PCT) if occ.size else 1.0
        ax = axes[0][j]
        mesh = ax.pcolormesh(te, V_EDGES, np.ma.masked_where(h.T == 0, h.T),
                             cmap="magma", vmin=0, vmax=vmax, rasterized=True)
        # median curve over the rungs
        c = 0.5 * (V_EDGES[:-1] + V_EDGES[1:])
        tmid = np.sqrt(te[:-1] * te[1:])
        tots = h.sum(axis=1)
        med = np.array([c[np.searchsorted(np.cumsum(h[i]), tots[i] / 2.0)]
                        if tots[i] >= 30 else np.nan for i in range(len(tmid))])
        ax.plot(tmid, med, color="deepskyblue", lw=1.6)
        vb = meta.get("vbounds_km_s")
        if vb:
            for y in vb:
                ax.axhline(y, color="cyan", lw=0.9, ls="--")
        if a.xscale == "log":
            ax.set_xscale("log")
            ax.set_xticks([0.3, 0.5, 1, 2, 3, 5])
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlim(te[0], te[-1])
        ax.set_ylim(V_EDGES[0], min(V_EDGES[-1], (vb[1] + 0.4) if vb else 4.0))
        ax.set_title("%s\n%s rows | %s pairs%s"
                     % (label, format(len(d), ","), format(d.station_pair.nunique(), ","),
                        ("  | bounds %.1f-%.1f" % tuple(vb)) if vb else ""), fontsize=10)
        if j == 0:
            ax.set_ylabel("Group velocity U [km/s]")
        cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("pairs per cell (p%g clip)" % CLIP_PCT, fontsize=8)
        # ray count per period -- what limits each tomographic map
        ax2 = axes[1][j]
        cnt = d.groupby(d["inst_period"].round(4)).size()
        ax2.semilogy(cnt.index, cnt.values, "o-", ms=3, color="crimson")
        if a.xscale == "log":
            ax2.set_xscale("log")
            ax2.set_xticks([0.3, 0.5, 1, 2, 3, 5])
            ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax2.set_xlim(te[0], te[-1])
        ax2.grid(alpha=0.3, which="both")
        ax2.set_xlabel("Period [s]  (CWT scale rungs)")
        if j == 0:
            ax2.set_ylabel("rays per period map")
        ax2.set_title("%d period maps" % len(cnt), fontsize=9)
    fig.suptitle("%s -- exported tomography pick tables" % (a.title or a.dir), fontsize=13)
    fig.tight_layout()
    out = a.out or os.path.join(a.dir, "picks_hist2d%s.png" % a.suffix)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
