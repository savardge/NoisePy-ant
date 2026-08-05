#!/usr/bin/env python3
"""Distribution of velocity-map cell values per period, grouped by surface geology unit.

Assigns every resolution-masked map cell to a swisstopo GK500 tectonic unit
(`PY_Surfaces_base_wgs84.shp`, attribute `LEG_TEC_2` by default) by point-in-polygon on the
cell CENTRE, then plots median +/- IQR velocity against period, one curve per unit.

Motivating question (Haute-Sorne): the Tertiary basins read FASTER than the Malm-cored folds
of the Jura interne, which is backwards for surface lithology. Note the comparison is
intrinsically mixed-depth -- GK500 maps the SURFACE, while at T=2 s the wave samples roughly
the top kilometre -- so a mismatch is not automatically an artifact. A thin Tertiary cover
over the same Mesozoic platform can be fast at depth despite a slow cap.

TWO TRAPS, both hit earlier in this project:
  * `Grid.vec_to_map` returns (nx, ny) -- x along axis 0. Pairing it with
    `meshgrid(indexing="xy")` transposes the map SILENTLY on a square grid and swaps
    east-west for north-south. Coordinates here are built with indexing="ij".
  * Cell lon/lat come from `swtomotv.geometry.xy2ll`, the exact inverse of the ll2xy used to
    build the grid -- NOT a linear interpolation across the bounds, which drifts.

Usage:
  python velocity_by_geology.py --net hautesorne --run tspws_group_scaled_dx0.5_prod2_k3
  python velocity_by_geology.py --all
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swtomotv.config import DatasetConfig
from swtomotv.geometry import make_grid, xy2ll

EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
TOMO = "/Users/genevievesavard/Codes/NoisePy-ant/param_files/cluster/tomo"
GEOL = ("/Users/genevievesavard/Data/swisstopo/GK500_V1_1/GK500_V1_1_FR/Shapes_WGS84/"
        "PY_Surfaces_base_wgs84.shp")
WAVES = ("fund", "overtone", "love")
TITLES = {"fund": "Rayleigh fundamental", "overtone": "Rayleigh overtone",
          "love": "Love fundamental"}
# run values are paths RELATIVE to tomo/1_velocity_maps/ (2026-08-05 workflow layout)
DEFAULT_RUN = {"riehen": "1_production/tspws_group_scaled_dx0.2_prod3_k3",
               "aargau": "1_production/tspws_group_scaled_dx0.5_prod3_k3",
               "hautesorne": "1_production/tspws_group_scaled_dx0.5_prod3_k3"}



# Synthetic level "cover": LEG_GEOL aggregated into competence classes. Needed because
# neither tectonic level separates basin FILL from basin SETTING -- GK500's LEG_TEC_2 puts
# the whole Delemont basin inside "Jura interne", since tectonically it lies within the
# folded Jura. Only the lithostratigraphic legend distinguishes the Cenozoic fill from the
# Malm/Dogger carbonates that core the folds, which is the contrast a velocity map should see.
COVER = [
    ("Quaternary (soft)", ("alluvion", "loess", "limons", "moraine", "eboulis", "tourbe",
                           "cailloutis", "tassement", "glissement", "depots", "cone",
                           "glaciaire", "fluvioglac")),
    ("Tertiary molasse",  ("chattien", "aquitanien", "burdigalien", "langhien", "serravalien",
                           "usm", "osm", "omm", "siderolithique", "oligocene", "miocene")),
    ("Mesozoic carbonate", ("malm", "dogger", "lias", "muschelkalk", "keuper", "trias",
                            "cretace", "buntsandstein", "jurassique")),
    ("Crystalline basement", ("granite", "gneiss", "cristallin", "varisque", "socle")),
]


def to_cover(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    t = str(v).lower()
    for name, keys in COVER:
        if any(k in t for k in keys):
            return name
    return "other"


def cell_units(net, grid, level):
    """(nx, ny) array of unit labels for every cell centre, via point-in-polygon."""
    XI, YI = np.meshgrid(grid.x, grid.y, indexing="ij")   # (nx, ny) -- matches vec_to_map
    # xy2ll returns (lat, lon), NOT (lon, lat). Unpacking it the other way silently produces
    # "longitudes" of 47.x, the bbox read returns zero polygons, and every cell comes back
    # unassigned rather than raising.
    lat, lon = xy2ll(XI.ravel(), YI.ravel(), *grid.origin)
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs=4326)
    pad = 0.05
    poly = gpd.read_file(GEOL, bbox=(lon.min() - pad, lat.min() - pad,
                                     lon.max() + pad, lat.max() + pad)).to_crs(4326)
    if not len(poly):
        raise RuntimeError("no GK500 polygons in %.3f-%.3f lon, %.3f-%.3f lat -- check the "
                           "cell lon/lat, xy2ll returns (lat, lon)"
                           % (lon.min(), lon.max(), lat.min(), lat.max()))
    src = "LEG_GEOL" if level == "cover" else level
    j = gpd.sjoin(pts, poly[[src, "geometry"]], how="left", predicate="within")
    # a point on a shared edge can match two polygons; keep the first
    j = j[~j.index.duplicated(keep="first")]
    col = j[src]
    if level == "cover":
        col = col.map(to_cover)
    lab = col.to_numpy().reshape(XI.shape)
    hit = pd.notna(lab).mean()
    print("   cell->unit match rate %.1f%%" % (100 * hit))
    if hit < 0.5:
        raise RuntimeError("only %.0f%% of cells matched a polygon -- geometry mismatch"
                           % (100 * hit))
    return lab


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--level", default="LEG_TEC_2")
    ap.add_argument("--min-cells", type=int, default=25,
                    help="drop a unit at a period if it has fewer cells than this")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    nets = list(DEFAULT_RUN) if a.all else [a.net]

    for net in nets:
        run = a.run if (a.run and not a.all) else DEFAULT_RUN[net]
        cfg = glob.glob(f"{TOMO}/{net}_tspws_group_scaled_lccov.yaml")
        if not cfg:
            print("no config for %s" % net); continue
        ds = DatasetConfig.from_yaml(cfg[0])
        dx = float(run.split("_dx")[1].split("_")[0])
        ds.dx_km = dx
        grid = make_grid(ds.bounds, dx)
        units = cell_units(net, grid, a.level)
        root = f"{EHM}/{net}/tomo/1_velocity_maps/{run}/production"
        print("\n=== %s  %s  (%s) ===" % (net, run, a.level))
        ok_u = pd.notna(units)
        uu, cc = np.unique(units[ok_u].astype(str), return_counts=True)
        for u, c in sorted(zip(uu, cc), key=lambda t: -t[1]):
            print("   %6d cells  %s" % (c, u))

        rows = []
        for wave in WAVES:
            for f in sorted(glob.glob(f"{root}/{wave}/map_T*.npz")):
                z = np.load(f)
                V = np.where(z["mask"].astype(bool), z["vel"], np.nan)
                if V.shape != units.shape:
                    V = V.T                      # defensive; should not trigger
                ok = np.isfinite(V)
                for u in uu:
                    m = ok & ok_u & (units.astype(str) == u)
                    if m.sum() < a.min_cells:
                        continue
                    v = V[m]
                    rows.append(dict(net=net, wave=wave, T=float(z["period"]), unit=u,
                                     n=int(m.sum()), med=float(np.median(v)),
                                     q25=float(np.percentile(v, 25)),
                                     q75=float(np.percentile(v, 75))))
        if not rows:
            print("   no data"); continue
        d = pd.DataFrame(rows)
        out = f"{EHM}/{net}/tomo/1_velocity_maps/{run}/geology"
        os.makedirs(out, exist_ok=True)
        d.to_csv(f"{out}/velocity_by_{a.level}.csv", index=False)

        # ---- figure: median +/- IQR vs period, one curve per unit, one panel per wave
        order = [u for u, _ in sorted(zip(uu, cc), key=lambda t: -t[1])]
        cols = dict(zip(order, plt.cm.tab10.colors))
        fig, axs = plt.subplots(1, len(WAVES), figsize=(6.2 * len(WAVES), 5.4), squeeze=False)
        for j, wave in enumerate(WAVES):
            ax = axs[0][j]
            g = d[d.wave == wave]
            for u in order:
                s = g[g.unit == u].sort_values("T")
                if len(s) < 3:
                    continue
                ax.plot(s["T"], s["med"], "-", color=cols[u], lw=2.2,
                        label="%s (n~%d)" % (u[:34], int(s["n"].median())))
                ax.fill_between(s["T"], s["q25"], s["q75"], color=cols[u], alpha=0.16, lw=0)
            ax.set_xscale("log")
            ax.set_xticks([0.3, 0.5, 1, 2, 3, 5])
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.set_xlabel("period [s]"); ax.set_ylabel("group velocity [km/s]")
            ax.set_title("%s  %s" % (net, TITLES.get(wave, wave)), fontsize=11)
            ax.grid(alpha=0.3); ax.legend(fontsize=7.5, loc="best")
        fig.suptitle("%s: map-cell velocity by surface geology (%s)  --  %s\n"
                     "line = median over cells, band = IQR; masked cells only. "
                     "GK500 maps the SURFACE, the waves sample ~the top km."
                     % (net, a.level, run), fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(f"{out}/velocity_by_{a.level}.png", dpi=135, bbox_inches="tight")
        plt.close(fig)

        # ---- companion: where the units actually are, so the assignment is checkable
        fig2, ax = plt.subplots(figsize=(7.4, 6.4))
        code = np.full(units.shape, -1, int)
        for i, u in enumerate(order):
            code[ok_u & (units.astype(str) == u)] = i
        XI, YI = np.meshgrid(grid.x, grid.y, indexing="ij")
        ax.pcolormesh(XI, YI, np.where(code >= 0, code, np.nan), cmap="tab10",
                      vmin=-0.5, vmax=9.5, shading="auto")
        for i, u in enumerate(order[:10]):
            ax.plot([], [], "s", color=plt.cm.tab10.colors[i], label=u[:40])
        ax.set_aspect("equal"); ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]")
        ax.set_title("%s: cell -> %s assignment (cell centres)" % (net, a.level), fontsize=11)
        ax.legend(fontsize=7.5, loc="upper left")
        fig2.tight_layout()
        fig2.savefig(f"{out}/unit_map_{a.level}.png", dpi=135, bbox_inches="tight")
        plt.close(fig2)
        print("   wrote %s/velocity_by_%s.{png,csv} + unit_map" % (out, a.level))


if __name__ == "__main__":
    main()
