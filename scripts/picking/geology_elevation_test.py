#!/usr/bin/env python3
"""Is the "Tertiary molasse faster than Mesozoic carbonate" signal a topographic artifact?

Haute-Sorne shows the Tertiary basin fill reading FASTER than the Malm/Dogger carbonates that
core the folds (+0.25 km/s, 11%, at T=3.23 s) -- backwards for surface lithology. One
candidate explanation is depth referencing rather than rock: surface-wave depth sensitivity is
measured from the SURFACE, so a cell in a 500 m valley samples 500 m deeper into the section
than a crest cell at the same period, and the basin cells would then see more of the fast
carbonate platform beneath. The molasse sits in valleys and the Malm on the ridges, so
lithology and elevation are confounded by construction.

The test: within each period, regress cell velocity on cell elevation across ALL masked
cells, then re-compare the lithology classes on the RESIDUALS. If the molasse-fast signal
survives, it is not an elevation effect; if it collapses, it is.

The elevation-velocity slope is reported too -- its sign is diagnostic on its own, since a
pure depth-referencing effect makes velocity DECREASE with elevation at every period.

Usage:
  python geology_elevation_test.py --net hautesorne
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
from scipy.interpolate import RegularGridInterpolator

from swtomotv.config import DatasetConfig, MethodConfig
from swtomotv.geometry import make_grid
from swtomotv.products.figures import build_local_dem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from velocity_by_geology import cell_units          # noqa: E402

EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
TOMO = "/Users/genevievesavard/Codes/NoisePy-ant/param_files/cluster/tomo"
DEM_CFG = {"hautesorne": f"{EHM}/hautesorne/tomo/1_velocity_maps/0_inputs/configs/"
                         "hautesorne_unified_tomo_ffv2.yaml"}
DEFAULT_RUN = {"riehen": "1_production/tspws_group_scaled_dx0.2_prod3_k3",
               "aargau": "1_production/tspws_group_scaled_dx0.5_prod3_k3",
               "hautesorne": "1_production/tspws_group_scaled_dx0.5_prod3_k3"}
CLASSES = ("Mesozoic carbonate", "Tertiary molasse", "Quaternary (soft)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", default="hautesorne")
    ap.add_argument("--wave", default="fund")
    a = ap.parse_args()
    net = a.net
    run = DEFAULT_RUN[net]
    dx = float(run.split("_dx")[1].split("_")[0])

    ds = DatasetConfig.from_yaml(glob.glob(f"{TOMO}/{net}_tspws_group_scaled_lccov.yaml")[0])
    ds.dx_km = dx
    grid = make_grid(ds.bounds, dx)
    units = cell_units(net, grid, "cover")

    # DEM on the velocity grid. build_local_dem returns h as (nx_dem, ny_dem) on its own
    # x/y axes in the SAME local km frame as the velocity grid, so a regular-grid
    # interpolation onto (grid.x, grid.y) is exact registration, not a re-projection.
    dsd = DatasetConfig.from_yaml(DEM_CFG[net])
    d = build_local_dem(dsd, MethodConfig(), grid)
    itp = RegularGridInterpolator((d["x"], d["y"]), d["h"], bounds_error=False, fill_value=None)
    XI, YI = np.meshgrid(grid.x, grid.y, indexing="ij")
    ELEV = itp(np.column_stack([XI.ravel(), YI.ravel()])).reshape(XI.shape)
    print("elevation on grid: %.0f - %.0f m (median %.0f)"
          % (np.nanmin(ELEV), np.nanmax(ELEV), np.nanmedian(ELEV)))
    ok_u = pd.notna(units)
    for c in CLASSES:
        m = ok_u & (units.astype(str) == c)
        if m.sum():
            print("   %-22s n=%5d  elevation median %.0f m (p25 %.0f, p75 %.0f)"
                  % (c, m.sum(), np.nanmedian(ELEV[m]),
                     np.nanpercentile(ELEV[m], 25), np.nanpercentile(ELEV[m], 75)))

    rows = []
    root = f"{EHM}/{net}/tomo/1_velocity_maps/{run}/production/{a.wave}"
    for f in sorted(glob.glob(f"{root}/map_T*.npz")):
        z = np.load(f)
        V = np.where(z["mask"].astype(bool), z["vel"], np.nan)
        good = np.isfinite(V) & np.isfinite(ELEV) & ok_u
        if good.sum() < 200:
            continue
        v, e = V[good], ELEV[good]
        # linear V(elev) over ALL masked cells, then residuals
        b, c0 = np.polyfit(e, v, 1)
        res = v - (b * e + c0)
        lab = units[good].astype(str)
        r = dict(T=float(z["period"]), slope_km_s_per_km=b * 1000.0,
                 r=float(np.corrcoef(e, v)[0, 1]))
        for cl in CLASSES:
            m = lab == cl
            if m.sum() >= 25:
                r["raw_" + cl] = float(np.median(v[m]))
                r["res_" + cl] = float(np.median(res[m]))
                r["n_" + cl] = int(m.sum())
        rows.append(r)
    D = pd.DataFrame(rows).sort_values("T")
    out = f"{EHM}/{net}/tomo/1_velocity_maps/{run}/geology"
    D.to_csv(f"{out}/elevation_test_{a.wave}.csv", index=False)

    have = [c for c in ("Tertiary molasse", "Mesozoic carbonate") if "raw_" + c in D]
    if len(have) == 2:
        D["raw_diff"] = D["raw_Tertiary molasse"] - D["raw_Mesozoic carbonate"]
        D["res_diff"] = D["res_Tertiary molasse"] - D["res_Mesozoic carbonate"]
        print("\n  T      V-elev slope   corr    molasse-carbonate: RAW    after elev removed"
              "    surviving")
        for _, x in D.iterrows():
            if x["T"] not in (0.51, 0.81, 1.21, 1.61, 2.03, 2.56, 3.23, 4.06, 5.12, 6.45):
                continue
            keep = 100 * x["res_diff"] / x["raw_diff"] if abs(x["raw_diff"]) > 1e-9 else np.nan
            print("  %-6.2f %+8.3f      %+.2f        %+.3f                %+.3f          %5.0f%%"
                  % (x["T"], x["slope_km_s_per_km"], x["r"], x["raw_diff"], x["res_diff"], keep))
        print("\n  median over all periods: raw %+.3f -> residual %+.3f km/s  (%.0f%% survives)"
              % (D.raw_diff.median(), D.res_diff.median(),
                 100 * D.res_diff.median() / D.raw_diff.median()))

    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    axs[0].plot(D["T"], D.slope_km_s_per_km, "o-", color="k", lw=2)
    axs[0].axhline(0, color="0.6", lw=1)
    axs[0].set_xscale("log"); axs[0].set_xlabel("period [s]")
    axs[0].set_ylabel("dV/d(elevation)  [km/s per km]")
    axs[0].set_title("velocity-elevation slope\nnegative = higher ground is slower", fontsize=10)
    for cl, col in zip(CLASSES, ("tab:blue", "tab:red", "tab:green")):
        if "raw_" + cl in D:
            axs[1].plot(D["T"], D["raw_" + cl], "o-", color=col, lw=2, label=cl)
            axs[2].plot(D["T"], D["res_" + cl], "o-", color=col, lw=2, label=cl)
    axs[1].set_title("RAW median velocity by lithology", fontsize=10)
    axs[1].set_ylabel("group velocity [km/s]")
    axs[2].set_title("RESIDUAL after removing the linear V(elevation) trend\n"
                     "if the classes converge here, the signal was topographic", fontsize=10)
    axs[2].set_ylabel("velocity residual [km/s]")
    axs[2].axhline(0, color="0.6", lw=1)
    for ax in axs[1:]:
        ax.set_xscale("log"); ax.set_xlabel("period [s]"); ax.legend(fontsize=8)
    for ax in axs:
        ax.set_xticks([0.3, 0.5, 1, 2, 3, 5])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(alpha=0.3)
    fig.suptitle("%s %s: is the lithology contrast an elevation artifact?" % (net, a.wave),
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{out}/elevation_test_{a.wave}.png", dpi=135, bbox_inches="tight")
    print("\n  wrote %s/elevation_test_%s.{png,csv}" % (out, a.wave))


if __name__ == "__main__":
    main()
