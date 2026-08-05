"""2D histograms of the EXPORTED tomography pick tables (picks_<wave>_uni*.csv).

These are the files the tomography actually inverts, one row per (station pair, period)
after median aggregation -- NOT the per-pick QC output. Period is the CWT scale ladder
(`inst_period` holds T_scale on the default --period-axis scale), so the period bins are
built from the rungs present in the file: no uniform-grid striping, no smoothing.

Panels: one per wave table found. Also plots ray count vs period, which is what actually
limits each period map.

    python plot_exported_picks_hist2d.py --dir <.../tomo/1_velocity_maps/inputs> \
        --title "Riehen" [--suffix ""]

RECOVERED 2026-08-05. This file was found truncated to 0 bytes (mtime 2026-08-04 11:00,
cause unknown; the repo has no git). `WAVES`, `CLIP_PCT`, `V_EDGES` and `rung_edges` are
restored EXACTLY from the surviving `__pycache__` bytecode (cpython-311, 2026-08-03) --
those are the names other modules import (`filter_picks_hist2d.py`,
`pick_vs_cell_hist2d.py`). `main()` is a RECONSTRUCTION built from the bytecode's constant
and attribute tables plus the figures it had already produced (`picks_hist2d_group.png`,
`picks_hist2d_phase.png`): same CLI, same two-row layout, same overlays. Cosmetic details
may differ from the original.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WAVES = (("fund", "Rayleigh fundamental"), ("overtone", "Rayleigh overtone"),
         ("love", "Love fundamental"), ("love_ot", "Love overtone"))
CLIP_PCT = 99.0
# Velocity edges offset half a node: `group_velocity` steps by exactly 0.01 km/s, and edges
# landing ON a node push picks into the bin below (the documented discrete-axis trap).
V_EDGES = np.arange(0.195, 5.025, 0.02)


def rung_edges(periods):
    """Bin edges at geometric midpoints between the period rungs present in the file."""
    s = np.sort(np.unique(np.round(periods, 4)))
    if len(s) < 2:
        return np.array([s[0] * 0.97, s[0] * 1.03])
    mid = np.sqrt(s[:-1] * s[1:])
    return np.concatenate([[s[0] ** 2 / mid[0]], mid, [s[-1] ** 2 / mid[-1]]])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--suffix", default="", help="pick table suffix (e.g. _phase)")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--measure", default=None,
                    help="label only; inferred from --suffix when omitted. The phase tables "
                         "reuse swtomotv's group_velocity COLUMN (see the sidecar "
                         ".meta.json), so the label cannot come from the column name.")
    ap.add_argument("--xscale", default="linear",
                    help="period axis scale. linear (default) shows the true period spacing; "
                         "log spreads the short periods where the CWT rungs crowd together.")
    a = ap.parse_args()
    measure = a.measure or ("phase" if a.suffix.lower() == "_phase" else "group")

    panels = []
    for wave, lab in WAVES:
        f = os.path.join(a.dir, "picks_" + wave + "_uni" + a.suffix + ".csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        if not len(d):
            print("  %-9s empty -- skipped" % wave)
            continue
        meta = {}
        mp = f + ".meta.json"
        if os.path.exists(mp):
            meta = json.load(open(mp))
        panels.append((wave, lab, d, meta))
        print("  %-9s %s rows | %s pairs | T %.2f-%.2f | U %.2f-%.2f"
              % (wave, format(len(d), ","), format(d.station_pair.nunique(), ","),
                 d.inst_period.min(), d.inst_period.max(),
                 d.group_velocity.min(), d.group_velocity.max()))
    if not panels:
        raise SystemExit("no pick tables in %s" % a.dir)

    n = len(panels)
    fig, axs = plt.subplots(2, n, figsize=(6.6 * n, 9.2), squeeze=False,
                            gridspec_kw={"height_ratios": [2.2, 1.0]})
    for i, (wave, lab, d, meta) in enumerate(panels):
        te = rung_edges(d["inst_period"].values)
        tc = np.sqrt(te[:-1] * te[1:])
        H, _, _ = np.histogram2d(d["inst_period"], d["group_velocity"], bins=[te, V_EDGES])
        ax = axs[0][i]
        vmax = np.percentile(H[H > 0], CLIP_PCT) if (H > 0).any() else 1.0
        pc = ax.pcolormesh(te, V_EDGES, np.ma.masked_where(H.T <= 0, H.T),
                           cmap="magma", vmin=0, vmax=vmax)
        med = d.groupby(np.round(d["inst_period"], 4))["group_velocity"].median()
        ax.plot(med.index.values, med.values, "-", color="deepskyblue", lw=1.8)
        vb = meta.get("vbounds_km_s")
        if vb:
            for y in np.asarray(vb, float):
                ax.axhline(float(y), color="cyan", ls="--", lw=1.0)
        curve = meta.get("vbounds_curve_km_s")
        if curve:
            cv = np.asarray(curve, float)
            k = min(len(cv), len(tc))
            ax.plot(tc[:k], cv[:k], "-", color="darkorange", lw=1.6, label="applied bound")
            ax.legend(loc="upper right", fontsize=8)
        if a.xscale == "log":
            ax.set_xscale("log")
            ax.set_xticks([0.3, 0.5, 1, 2, 3, 5])
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlim(te[0], te[-1])
        ax.set_ylim(max(V_EDGES[0], d.group_velocity.min() - 0.2),
                    min(V_EDGES[-1], d.group_velocity.max() + 0.2))
        sub = "%s rows | %s pairs" % (format(len(d), ","),
                                      format(d.station_pair.nunique(), ","))
        if vb:
            sub += "  | bounds %.1f-%.1f" % tuple(np.asarray(vb, float)[:2])
        if meta.get("cut_pct") is not None:
            sub += "  | cull -%.2f%%" % float(meta["cut_pct"])
        ax.set_title("%s\n%s" % (lab, sub), fontsize=10)
        ax.set_ylabel("%s velocity [km/s]" % measure.capitalize())
        cb = fig.colorbar(pc, ax=ax)
        cb.set_label("pairs per cell (p%g clip)" % CLIP_PCT)

        axr = axs[1][i]
        cnt = d.groupby(np.round(d["inst_period"], 4)).size()
        axr.semilogy(cnt.index.values, cnt.values, "o-", color="crimson", ms=3, lw=1.4)
        if a.xscale == "log":
            axr.set_xscale("log")
            axr.set_xticks([0.3, 0.5, 1, 2, 3, 5])
            axr.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axr.set_xlim(te[0], te[-1])
        axr.grid(alpha=0.3, which="both")
        axr.set_xlabel("Period [s]  (CWT scale rungs)")
        axr.set_ylabel("rays per period map")
        axr.set_title("%d period maps" % len(cnt), fontsize=9)

    fig.suptitle("%s -- exported tomography pick tables (%s velocity)"
                 % (a.title, measure), fontsize=13)
    fig.tight_layout()
    out = a.out or os.path.join(a.dir, "picks_hist2d_%s.png" % measure)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
