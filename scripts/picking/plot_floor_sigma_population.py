#!/usr/bin/env python3
"""Where do the floor-sigma picks sit in the period-velocity plane?

The substack jackknife measures REPEATABILITY, not accuracy. A pick locked onto the same
wrong arrival in every substack block has MAD -> 0 and pins at the sigma floor
(1.4826 * 0.02 = 0.0297 km/s). A measured-Cd inversion then weights it ~1/sigma^2, i.e. up
to ~500x an honest noisy pick, which blew Riehen fund maps up to 26 km/s structure and a
NEGATIVE mean velocity at T = 0.51 s.

This figure locates that population. Four panels per wave:

  (a) all picks                      the usual 2D histogram, for reference
  (b) floor-sigma picks only         where the over-trusted population actually lives
  (c) everything else                what the map would see without them
  (d) floor fraction per cell        the diagnostic: which (T, V) cells are contaminated

Binning follows the project convention. Period uses the CWT scale ladder the picks were
measured on (uneven, ~5.95% steps) with geometric midpoints as edges -- NOT a uniform grid,
which would leave empty columns. Velocity is on a 0.01 km/s node grid, so edges are offset
by half a node; edges landing ON a node push picks into the neighbouring bin and manufacture
horizontal stripes.

Usage:
  python plot_floor_sigma_population.py --picks-dir <.../inputs_tspws> --net riehen \
      [--waves fund overtone love] [--floor 0.0298] [--out FILE]
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WAVE_FILE = {"fund": "picks_fund_uni.csv", "overtone": "picks_overtone_uni.csv",
             "love": "picks_love_uni.csv"}
WAVE_LABEL = {"fund": "Rayleigh fundamental", "overtone": "Rayleigh overtone",
              "love": "Love fundamental"}


def rung_edges(T):
    """Geometric midpoints between the CWT rungs actually present."""
    u = np.unique(T)
    if len(u) < 2:
        return np.array([u[0] * 0.97, u[0] * 1.03])
    mid = np.sqrt(u[1:] * u[:-1])
    return np.concatenate(([u[0] ** 2 / mid[0]], mid, [u[-1] ** 2 / mid[-1]]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--picks-dir", required=True)
    ap.add_argument("--net", required=True)
    ap.add_argument("--waves", nargs="+", default=["fund", "overtone", "love"])
    ap.add_argument("--floor", type=float, default=0.0298,
                    help="sigma at or below this counts as floored (default just above "
                         "the 1.4826*0.02 quantum)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    waves = [w for w in a.waves if os.path.exists(os.path.join(a.picks_dir, WAVE_FILE[w]))]
    if not waves:
        raise SystemExit("no pick tables found in %s" % a.picks_dir)

    fig, axs = plt.subplots(4, len(waves), figsize=(5.0 * len(waves), 15), squeeze=False)
    # velocity node grid is 0.01 km/s -> offset the edges by half a node
    vedges = np.arange(0.195, 3.615, 0.02)

    for j, wave in enumerate(waves):
        d = pd.read_csv(os.path.join(a.picks_dir, WAVE_FILE[wave]))
        # std may already be winsorized; std_jk preserves the raw jackknife value, which is
        # what defines the floored population.
        scol = "std_jk" if "std_jk" in d.columns else "std"
        T, V, S = d["inst_period"].values, d["group_velocity"].values, d[scol].values
        floored = S <= a.floor
        tedges = rung_edges(T)

        Hall, _, _ = np.histogram2d(T, V, bins=[tedges, vedges])
        Hflr, _, _ = np.histogram2d(T[floored], V[floored], bins=[tedges, vedges])
        Hrest = Hall - Hflr
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(Hall >= 5, 100.0 * Hflr / Hall, np.nan)

        pct = 100.0 * floored.mean()
        panels = [
            (Hall, "(a) all picks  (n=%s)" % format(len(d), ","), "magma", None),
            (Hflr, "(b) FLOOR-sigma only  (%s, %.1f%%)" % (format(int(floored.sum()), ","), pct),
             "magma", None),
            (Hrest, "(c) everything else", "magma", None),
            (frac, "(d) %% of picks at the sigma floor", "inferno", (0, 60)),
        ]
        for i, (Z, title, cmap, lim) in enumerate(panels):
            ax = axs[i][j]
            if lim is None:
                pos = Z[Z > 0]
                vmax = np.percentile(pos, 99) if pos.size else 1
                pc = ax.pcolormesh(tedges, vedges, np.where(Z > 0, Z, np.nan).T,
                                   cmap=cmap, vmin=0, vmax=vmax, shading="flat")
                cb = "picks per cell (p99 clip)"
            else:
                pc = ax.pcolormesh(tedges, vedges, Z.T, cmap=cmap,
                                   vmin=lim[0], vmax=lim[1], shading="flat")
                cb = "% floored"
            plt.colorbar(pc, ax=ax, shrink=0.85, label=cb)
            ax.set_xlim(tedges[0], tedges[-1]); ax.set_ylim(0.2, 3.6)
            ax.set_xlabel("period [s]  (CWT scale rungs)")
            ax.set_ylabel("group velocity [km/s]")
            ax.set_title("%s %s\n%s" % (a.net, WAVE_LABEL[wave], title), fontsize=10)

    fig.suptitle("%s: location of the floor-sigma (perfectly repeatable) pick population "
                 "-- these are the picks a measured-Cd inversion over-trusts" % a.net,
                 y=1.0, fontsize=12)
    fig.tight_layout()
    out = a.out or os.path.join(a.picks_dir, "floor_sigma_population.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)

    for wave in waves:
        d = pd.read_csv(os.path.join(a.picks_dir, WAVE_FILE[wave]))
        scol = "std_jk" if "std_jk" in d.columns else "std"
        f = d[scol] <= a.floor
        if f.any():
            print("  %-9s floored %6s (%4.1f%%) | median V %.3f vs %.3f for the rest | "
                  "median T %.2f vs %.2f"
                  % (wave, format(int(f.sum()), ","), 100 * f.mean(),
                     d.loc[f, "group_velocity"].median(),
                     d.loc[~f, "group_velocity"].median(),
                     d.loc[f, "inst_period"].median(), d.loc[~f, "inst_period"].median()))


if __name__ == "__main__":
    main()
