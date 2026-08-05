#!/usr/bin/env python3
"""Is the Malm anomalously SLOW because it is fold-damaged, rather than the basin fast?

Haute-Sorne shows Tertiary basin fill reading faster than the Malm/Dogger carbonates that core
the folds (+0.25 km/s at T=3.23 s). The elevation-confounding explanation was tested and
rejected -- 86-103% of the effect survives removal of the V(elevation) trend at the periods
where it is strongest, and the two classes differ by only 57 m in median elevation.

This tests the remaining structural explanation: intensely fractured and karstified limestone
in the fold cores is mechanically much slower than intact limestone, while the basin fill is
undeformed. That predicts, WITHIN the carbonate class alone:

    velocity should RISE with distance from a mapped fold axis / fault,

and it predicts the molasse-carbonate gap should shrink when the comparison uses only
carbonate cells far from any mapped structure.

Distances are to the GK500 tectonic line layers already used for the styled maps: anticline
axes (`LI_Axes_de_struct`) and faults/thrusts (`LI_Accident_tecto`), densified to ~100 m and
queried with a KD-tree in the grid's own local-km frame.

Usage:
  python geology_structure_test.py --net hautesorne
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from swtomotv.config import DatasetConfig
from swtomotv.geometry import make_grid, ll2xy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from velocity_by_geology import cell_units          # noqa: E402

EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
TOMO = "/Users/genevievesavard/Codes/NoisePy-ant/param_files/cluster/tomo"
GK = ("/Users/genevievesavard/Data/hautesorne/GIS/GK-500-V-4/GK500_V1_4_FR/Shapes_WGS84")
DEFAULT_RUN = {"riehen": "1_production/tspws_group_scaled_dx0.2_prod3_k3",
               "aargau": "1_production/tspws_group_scaled_dx0.5_prod3_k3",
               "hautesorne": "1_production/tspws_group_scaled_dx0.5_prod3_k3"}


def line_points(path, origin, where=None, step_km=0.1):
    """All vertices of a line layer, densified to ~step_km, in local km."""
    g = gpd.read_file(path).to_crs(4326)
    if where:
        g = g.query(where, engine="python")
    pts = []
    for geom in g.geometry:
        if geom is None:
            continue
        for ln in getattr(geom, "geoms", [geom]):
            n = max(2, int(ln.length * 111.0 / step_km))
            for f in np.linspace(0, 1, n):
                p = ln.interpolate(f, normalized=True)
                pts.append((p.y, p.x))
    if not pts:
        return np.empty((0, 2))
    lat, lon = np.array(pts).T
    x, y = ll2xy(lat, lon, *origin)
    return np.column_stack([x, y])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", default="hautesorne")
    ap.add_argument("--wave", default="fund")
    a = ap.parse_args()
    net, run = a.net, DEFAULT_RUN[a.net]
    dx = float(run.split("_dx")[1].split("_")[0])
    ds = DatasetConfig.from_yaml(glob.glob(f"{TOMO}/{net}_tspws_group_scaled_lccov.yaml")[0])
    ds.dx_km = dx
    grid = make_grid(ds.bounds, dx)
    units = cell_units(net, grid, "cover")
    XI, YI = np.meshgrid(grid.x, grid.y, indexing="ij")
    cells = np.column_stack([XI.ravel(), YI.ravel()])

    layers = {
        "fold axis": (f"{GK}/LI_Axes_de_struct_wgs84.shp", None),
        "fault": (f"{GK}/LI_Accident_tecto_wgs84.shp", "Type.str.contains('Faille')"),
        "thrust": (f"{GK}/LI_Accident_tecto_wgs84.shp", "Type.str.contains('Chevauchement')"),
    }
    dist = {}
    for name, (p, w) in layers.items():
        P = line_points(p, grid.origin, w)
        if len(P) < 5:
            print("  %s: no segments" % name); continue
        dist[name] = cKDTree(P).query(cells)[0].reshape(XI.shape)
        print("  %-10s %6d densified pts | cell distance median %.1f km, max %.1f"
              % (name, len(P), np.median(dist[name]), dist[name].max()))
    dist["any structure"] = np.min(np.stack(list(dist.values())), axis=0)

    ok_u = pd.notna(units)
    carb = ok_u & (units.astype(str) == "Mesozoic carbonate")
    mol = ok_u & (units.astype(str) == "Tertiary molasse")
    print("\n  distance to nearest FOLD AXIS, by class [km]:")
    for nm, m in (("carbonate", carb), ("molasse", mol)):
        print("     %-10s median %.2f  p25 %.2f  p75 %.2f"
              % (nm, np.median(dist["fold axis"][m]), np.percentile(dist["fold axis"][m], 25),
                 np.percentile(dist["fold axis"][m], 75)))

    root = f"{EHM}/{net}/tomo/1_velocity_maps/{run}/production/{a.wave}"
    key = "fold axis"
    edges = [0, 0.5, 1.0, 1.5, 2.5, 99]
    rows = []
    for f in sorted(glob.glob(f"{root}/map_T*.npz")):
        z = np.load(f)
        V = np.where(z["mask"].astype(bool), z["vel"], np.nan)
        ok = np.isfinite(V)
        r = dict(T=float(z["period"]))
        cm = ok & carb
        if cm.sum() >= 100:
            dd, vv = dist[key][cm], V[cm]
            r["slope_per_km"] = float(np.polyfit(dd, vv, 1)[0])
            r["r"] = float(np.corrcoef(dd, vv)[0, 1])
            for i in range(len(edges) - 1):
                b = (dd >= edges[i]) & (dd < edges[i + 1])
                if b.sum() >= 25:
                    r["carb_%g_%g" % (edges[i], edges[i + 1])] = float(np.median(vv[b]))
        mm = ok & mol
        if mm.sum() >= 25:
            r["molasse"] = float(np.median(V[mm]))
        rows.append(r)
    D = pd.DataFrame(rows).sort_values("T")
    out = f"{EHM}/{net}/tomo/1_velocity_maps/{run}/geology"
    D.to_csv(f"{out}/structure_test_{a.wave}.csv", index=False)

    far = "carb_2.5_99"
    near = "carb_0_0.5"
    print("\n  WITHIN CARBONATE: does velocity rise away from the fold axes?")
    print("  T      dV/d(dist) km/s per km    r      near(<0.5km)  far(>2.5km)  far-near  "
          "| molasse  molasse-FAR")
    for _, x in D.iterrows():
        if x["T"] not in (0.81, 1.21, 1.61, 2.03, 2.56, 3.23, 4.06, 5.12):
            continue
        n_, f_ = x.get(near, np.nan), x.get(far, np.nan)
        mo = x.get("molasse", np.nan)
        print("  %-6.2f %+10.3f            %+.2f    %s  %s  %s   | %s  %s"
              % (x["T"], x.get("slope_per_km", np.nan), x.get("r", np.nan),
                 "%.3f" % n_ if np.isfinite(n_) else "  --  ",
                 "%.3f" % f_ if np.isfinite(f_) else "  --  ",
                 "%+.3f" % (f_ - n_) if np.isfinite(f_ * n_) else "  --  ",
                 "%.3f" % mo if np.isfinite(mo) else "  --  ",
                 "%+.3f" % (mo - f_) if np.isfinite(mo * f_) else "  --  "))
    sl = D.slope_per_km.dropna()
    print("\n  median dV/d(distance from fold axis) = %+.3f km/s per km  (positive supports "
          "fold damage)" % sl.median())

    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2))
    axs[0].plot(D["T"], D.slope_per_km, "o-", color="k", lw=2)
    axs[0].axhline(0, color="0.6", lw=1)
    axs[0].set_ylabel("dV/d(distance from fold axis) [km/s per km]")
    axs[0].set_title("within Mesozoic carbonate only\npositive = intact rock away from folds "
                     "is faster", fontsize=10)
    cols = [c for c in D.columns if c.startswith("carb_")]
    for c, col in zip(cols, plt.cm.viridis(np.linspace(0, 0.9, len(cols)))):
        lo, hi = c.split("_")[1:]
        axs[1].plot(D["T"], D[c], "-", color=col, lw=2,
                    label="carbonate %s-%s km from axis" % (lo, hi))
    if "molasse" in D:
        axs[1].plot(D["T"], D["molasse"], "--", color="tab:red", lw=2.5, label="Tertiary molasse")
    axs[1].set_ylabel("group velocity [km/s]")
    axs[1].set_title("carbonate by distance from fold axis, vs basin fill", fontsize=10)
    axs[1].legend(fontsize=8)
    for ax in axs:
        ax.set_xscale("log"); ax.set_xlabel("period [s]")
        ax.set_xticks([0.3, 0.5, 1, 2, 3, 5])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(alpha=0.3)
    fig.suptitle("%s %s: is the Malm slow because it is fold-damaged?" % (net, a.wave),
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{out}/structure_test_{a.wave}.png", dpi=135, bbox_inches="tight")
    print("  wrote %s/structure_test_%s.{png,csv}" % (out, a.wave))


if __name__ == "__main__":
    main()
