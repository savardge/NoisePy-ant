#!/usr/bin/env python3
"""Side-by-side period-velocity 2D distributions: INPUT PICKS vs OUTPUT MAP CELLS.

For every (network, measure, wave) one figure with the requested TWO columns --
    left  : the pick tables the inversion was given
    right : the velocity-map cell values it produced
-- and one ROW per Cd model. The picks are identical across Cd models (all three read the
same k-tree), so repeating the left column is deliberate: it puts each Cd's output directly
beside its own input.

WHAT THE COMPARISON SHOWS. The map is a smoothed, regularised projection of the picks onto a
grid, so the two distributions should share a dispersion ridge but need not match in spread:
  * map narrower than picks   -> the prior absorbed the spread (expected; the maps resolve
    ~19 effective dof, so most of the pick scatter cannot be represented)
  * map OFFSET from the picks -> the maps hold velocities the data never measured
  * map with a tail the picks lack -> amplified noise, the phase scaled/measured signature

BINNING follows the discrete-axis rule that governs these pick tables: the period axis is the
CWT scale ladder (geometric midpoints between rungs, from plot_exported_picks_hist2d) and the
velocity axis is 0.02 km/s with edges offset half a node (0.195, not 0.20) -- `group_velocity`
steps by exactly 0.01 km/s and edges landing ON a node push picks into the bin below.

NORMALISATION: each period column is normalised to sum 1, independently for picks and cells.
Without it the comparison would be dominated by there being thousands of picks and thousands
of cells with completely unrelated totals, and by the strong period dependence of both.

Usage:
  python pick_vs_cell_hist2d.py --all            # every net x measure x wave
  python pick_vs_cell_hist2d.py --net riehen --wave fund --measure group
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_exported_picks_hist2d import V_EDGES, rung_edges      # noqa: E402

EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
OUT = f"{EHM}/_inversion_comparison/pick_vs_cell_hist2d"
DX = {"riehen": "0.2", "aargau": "0.5", "hautesorne": "0.5"}
WAVES = ("fund", "overtone", "love")
MEASURES = ("group", "phase")
CDS = ("blanket", "measured", "scaled")
TITLE = {"fund": "Rayleigh fundamental", "overtone": "Rayleigh overtone",
         "love": "Love fundamental"}


def col_normalise(H):
    """Each period column -> sums to 1; empty columns stay NaN."""
    s = H.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        Hn = np.where(s > 0, H / np.where(s == 0, np.nan, s), np.nan)
    return np.where(Hn > 0, Hn, np.nan)


def median_curve(H, centres):
    """Per-period median of a column-normalised histogram (NaN where the column is empty)."""
    out = []
    for j in range(H.shape[0]):
        col = H[j]
        good = np.isfinite(col) & (col > 0)
        if not good.any():
            out.append(np.nan)
            continue
        c = np.cumsum(col[good]) / np.nansum(col[good])
        out.append(centres[good][np.searchsorted(c, 0.5)])
    return np.asarray(out, float)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", default=None)
    ap.add_argument("--wave", default=None)
    ap.add_argument("--measure", default=None)
    ap.add_argument("--k", default="k3", help="which vbounds cull tree fed the runs")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    nets = list(DX) if a.all else [a.net]
    waves = WAVES if (a.all or not a.wave) else (a.wave,)
    meas_l = MEASURES if (a.all or not a.measure) else (a.measure,)
    os.makedirs(OUT, exist_ok=True)
    vc = 0.5 * (V_EDGES[:-1] + V_EDGES[1:])

    for net in nets:
        V = f"{EHM}/{net}/tomo/1_velocity_maps"
        for meas in meas_l:
            suf = "_phase" if meas == "phase" else ""
            for wave in waves:
                pf = f"{V}/0_inputs/culled_picks_vbounds/{a.k}/picks_{wave}_uni{suf}.csv"
                if not os.path.exists(pf):
                    print("  no picks %s/%s/%s" % (net, meas, wave)); continue
                d = pd.read_csv(pf, usecols=["inst_period", "group_velocity"])
                d = d[np.isfinite(d.group_velocity)]
                if len(d) < 500:
                    print("  too few picks %s/%s/%s" % (net, meas, wave)); continue
                te = rung_edges(d["inst_period"].values)
                tc = np.sqrt(te[:-1] * te[1:])
                Hp, _, _ = np.histogram2d(d.inst_period, d.group_velocity,
                                          bins=[te, V_EDGES])
                Hp = col_normalise(Hp)
                mp = median_curve(Hp, vc)

                fig, axs = plt.subplots(len(CDS), 2, figsize=(13.6, 4.3 * len(CDS)),
                                        squeeze=False, sharex=True, sharey=True)
                any_data = False
                for i, cd in enumerate(CDS):
                    root = (f"{V}/1_production/tspws_{meas}_{cd}_dx{DX[net]}_prod3_{a.k}"
                            f"/production/{wave}")
                    Ts, Hc = [], np.zeros((len(te) - 1, len(V_EDGES) - 1))
                    ncell = []
                    for f in glob.glob(f"{root}/map_T*.npz"):
                        z = np.load(f)
                        T = float(z["period"])
                        v = z["vel"][np.isfinite(z["vel"])]
                        if v.size < 50:
                            continue
                        j = np.clip(np.searchsorted(te, T) - 1, 0, len(te) - 2)
                        h, _ = np.histogram(v, bins=V_EDGES)
                        Hc[j] += h
                        Ts.append(T); ncell.append(v.size)
                    if not Ts:
                        for c in (0, 1):
                            axs[i][c].axis("off")
                        continue
                    any_data = True
                    Hc = col_normalise(Hc)
                    mc = median_curve(Hc, vc)
                    for c, (H, med, lab, n) in enumerate((
                            (Hp, mp, "INPUT PICKS (%s, %s)" % (a.k, meas), len(d)),
                            (Hc, mc, "MAP CELLS (Cd = %s)" % cd, int(np.median(ncell))))):
                        ax = axs[i][c]
                        vmax = np.nanpercentile(H, 99) if np.isfinite(H).any() else 1
                        ax.pcolormesh(te, V_EDGES, H.T, cmap="inferno", vmin=0, vmax=vmax)
                        ax.plot(tc, med, "-", color="#39ff88", lw=1.8, label="median")
                        # the other side's median, for direct overlay comparison
                        other = mc if c == 0 else mp
                        ax.plot(tc, other, "--", color="cyan", lw=1.4, alpha=0.9,
                                label="median of the other panel")
                        # linear period axis: shows the true rung spacing, so the crowding
                        # of the CWT ladder at short period is visible rather than hidden
                        ax.set_xlim(te[0], te[-1])
                        ax.set_title("%s\n%s = %s" % (lab, "picks" if c == 0 else "cells/period",
                                                      f"{n:,}"), fontsize=9.5)
                        if c == 0:
                            ax.set_ylabel("%s velocity [km/s]\n(Cd row: %s)" % (meas, cd),
                                          fontsize=9)
                        if i == len(CDS) - 1:
                            ax.set_xlabel("period [s]  (CWT scale rungs)")
                        if i == 0 and c == 0:
                            ax.legend(fontsize=7, loc="upper right", framealpha=0.75)
                if not any_data:
                    plt.close(fig); continue
                lo = np.nanpercentile(d.group_velocity, 0.5)
                hi = np.nanpercentile(d.group_velocity, 99.5)
                axs[0][0].set_ylim(max(V_EDGES[0], lo - 0.3), min(V_EDGES[-1], hi + 0.3))
                fig.suptitle("%s — %s, %s: input picks vs output map cells\n"
                             "each period column normalised to sum 1; solid green = that "
                             "panel's median, dashed cyan = the other panel's"
                             % (net, TITLE.get(wave, wave), meas), fontsize=13,
                             fontweight="bold")
                fig.tight_layout()
                p = f"{OUT}/{net}_{meas}_{wave}_{a.k}.png"
                fig.savefig(p, dpi=125, bbox_inches="tight")
                plt.close(fig)
                print("  wrote", os.path.basename(p), flush=True)


if __name__ == "__main__":
    main()
