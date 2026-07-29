#!/usr/bin/env python3
"""Styled velocity-map figures for ALL networks (riehen / aargau / hautesorne):
hillshade terrain underlay + semi-transparent velocity cells + GK500 tectonic
overlays (faults, thrusts, anticline axes) + deep-well markers (>1 km, no
labels). Optionally rivers + town names (hautesorne convention).

Reuses the swtomotv figure helpers (build_hillshade / load_tecto / draw_tecto /
draw_layer) with per-network dem/tecto overrides, so the swtomotv dataset YAMLs
do not need editing.

Usage:  PYTHONPATH=~/Codes/NoisePy-ant:~/Codes/swtomotv/src \
        python3 styled_maps.py [net ...]        (default: all three)

Output: <production_run_dir>/styled_maps/<wave>/map_T*.png  -- the styled figures live
INSIDE the production run directory they were made from (sibling of production/), so
provenance is unambiguous. Never write to a shared Projects/<net>/tomo/styled_maps
(the old layout mixed runs and was deprecated 2026-07-25).
"""
import os
import sys
import glob

import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

from swtomotv.config import DatasetConfig, MethodConfig
from swtomotv.geometry import make_grid, ll2xy
from swtomotv.products._shared import imshow_extent
from swtomotv.products.figures import build_hillshade, load_tecto, draw_tecto, draw_layer
from noisepy.lv95 import wgs84_to_lv95

EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
GK500_V11 = "/Users/genevievesavard/Data/swisstopo/GK500_V1_1/GK500_V1_1_FR/Shapes_WGS84"
GK500_V14 = ("/Users/genevievesavard/Data/hautesorne/GIS/GK-500-V-4/GK500_V1_4_FR/"
             "Shapes_WGS84")
TLM = "/Users/genevievesavard/Data/swisstopo/swisstlmregio_2022_2056.shp/swissTLMRegio_Product_LV95"

# deep wells > 1 km (lat, lon) -- markers only, no labels (declutter).
# riehen/aargau: grid_vs_postprocess.WELLS_GT1KM (swisstopo deep_wells.csv);
# hautesorne: GVL-1 (user shapefile); the swisstopo >500 m layer has no other
# >1 km well inside the map bounds.
WELLS = {
    "riehen": [(47.585413, 7.595614), (47.577748, 7.603832), (47.587100, 7.649485),
               (47.593696, 7.657156), (47.509691, 7.604882)],
    "aargau": [(47.565033, 8.227163), (47.504507, 8.189936), (47.589033, 8.205224),
               (47.539828, 8.031539), (47.369472, 8.148685), (47.563788, 8.458407),
               (47.565144, 8.453530)],
    "hautesorne": [(47.33307609, 7.22020453)],
}

NETS = {
    "riehen": dict(
        yaml=f"{EHM}/riehen/tomo/1_velocity_maps/inputs/riehen_unified_tomo_200m_prod.yaml",
        group_root=f"{EHM}/riehen/tomo/1_velocity_maps/production/"
                   "production_2026-07-24_uni_group_dx0.2/production",
        dem=dict(kind="geotiff", path=f"{EHM}/riehen/tomo/dem/output_SRTMGL1.tif",
                 crs="EPSG:4326"),
        tecto=[dict(path=f"{GK500_V14}/LI_Accident_tecto_wgs84.shp",
                    kind="fault", where="Type.str.contains('Faille')"),
               dict(path=f"{GK500_V14}/LI_Accident_tecto_wgs84.shp",
                    kind="thrust", where="Type.str.contains('Chevauchement')"),
               dict(path=f"{GK500_V14}/LI_Axes_de_struct_wgs84.shp",
                    kind="anticline")],
        rivers=False, towns=[]),
    "aargau": dict(
        yaml=f"{EHM}/aargau/tomo/1_velocity_maps/inputs/aargau_unified_tomo_500m_prod.yaml",
        group_root=f"{EHM}/aargau/tomo/1_velocity_maps/production/"
                   "production_2026-07-24_uni_group_dx0.5/production",
        dem=dict(kind="geotiff", path=f"{EHM}/aargau/tomo/dem/N47E008.hgt",
                 crs="EPSG:4326"),
        tecto=[dict(path=f"{GK500_V14}/LI_Accident_tecto_wgs84.shp",
                    kind="fault", where="Type.str.contains('Faille')"),
               dict(path=f"{GK500_V14}/LI_Accident_tecto_wgs84.shp",
                    kind="thrust", where="Type.str.contains('Chevauchement')"),
               dict(path=f"{GK500_V14}/LI_Axes_de_struct_wgs84.shp",
                    kind="anticline")],
        rivers=False, towns=[]),
    "hautesorne": dict(
        yaml=f"{EHM}/hautesorne/tomo/1_velocity_maps/inputs/hautesorne_unified_tomo_ffv2.yaml",
        group_root=f"{EHM}/hautesorne/tomo/1_velocity_maps/production/"
                   "production_2026-07-25_uni_group_ffv2_dx0.5/production",
        dem=None,   # YAML already carries the local arcgrid DEM
        tecto=None,  # YAML already carries the GK500 v1.4 layers
        rivers=True,
        towns=["Delémont", "Bassecourt", "Glovelier", "Courfaivre", "Courtételle",
               "Undervelier", "Saulcy", "Boécourt", "Develier", "Courrendlin",
               "Saignelégier", "Moutier", "Porrentruy"]),
}
TITLES = {"fund": "Rayleigh fundamental", "overtone": "Rayleigh overtone",
          "love": "Love fundamental", "love_ot": "Love overtone"}


def load_water(ds, grid):
    b = ds.bounds
    g = gpd.read_file(f"{TLM}/Hydrography/WGS84/swissTLMRegio_FlowingWater_wgs84.shp"
                      ).cx[b[2]:b[3], b[0]:b[1]]
    g = g[g["KLASSE"] <= 7]
    segs = []
    for _, row in g.iterrows():
        for ln in getattr(row.geometry, "geoms", [row.geometry]):
            xy = np.asarray(ln.coords)
            x, y = ll2xy(xy[:, 1], xy[:, 0], *grid.origin)
            segs.append((x, y, int(row["KLASSE"])))
    return segs


def load_towns(ds, grid, names):
    b = ds.bounds
    g = gpd.read_file(f"{TLM}/Names/swissTLMRegio_NamedLocation.shp").to_crs(4326)
    g = g.cx[b[2]:b[3], b[0]:b[1]]
    g = g[g["NAMN1"].isin(names)].drop_duplicates("NAMN1")
    out = []
    for _, row in g.iterrows():
        x, y = ll2xy(np.array([row.geometry.y]), np.array([row.geometry.x]), *grid.origin)
        out.append((row["NAMN1"], float(x[0]), float(y[0])))
    return out


def draw_extras(ax, ext, water, towns, wells_xy):
    for x, y, kl in water:
        ax.plot(x, y, color="#3070c0", lw=1.4 if kl <= 5 else 0.8, alpha=0.85,
                zorder=3, solid_capstyle="round")
    for name, x, y in towns:
        if not (ext[0] <= x <= ext[1] and ext[2] <= y <= ext[3]):
            continue
        ax.plot(x, y, "o", ms=4, mfc="w", mec="k", mew=0.8, zorder=6)
        ax.annotate(name, (x, y), xytext=(4, 4), textcoords="offset points",
                    fontsize=7, style="italic", zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="w")])
    for x, y in wells_xy:
        if ext[0] <= x <= ext[1] and ext[2] <= y <= ext[3]:
            ax.plot(x, y, marker="D", ms=7, mfc="none", mec="k", mew=1.6, zorder=7)
            ax.plot(x, y, marker="D", ms=7, mfc="none", mec="yellow", mew=0.7, zorder=8)


def run_net(net):
    cfg = NETS[net]
    ds = DatasetConfig.from_yaml(cfg["yaml"])
    if cfg["dem"] is not None:
        ds.dem = cfg["dem"]
    if cfg["tecto"] is not None:
        ds.tecto_layers = cfg["tecto"]
    method = MethodConfig()
    grid = make_grid(ds.bounds, ds.dx_km)
    ext = imshow_extent(grid)
    try:
        hs, hs_ext = build_hillshade(ds, method, grid)
    except Exception as e:
        print(f"{net}: no hillshade ({e})")
        hs, hs_ext = None, None
    tecto = load_tecto(ds, grid)
    water = load_water(ds, grid) if cfg["rivers"] else []
    towns = load_towns(ds, grid, cfg["towns"]) if cfg["towns"] else []
    wl = WELLS.get(net, [])
    wx, wy = (ll2xy(np.array([w[0] for w in wl]), np.array([w[1] for w in wl]),
                    *grid.origin) if wl else (np.array([]), np.array([])))
    wells_xy = list(zip(wx, wy))
    print(f"{net}: {sum(len(v) for v in tecto.values())} tecto segs, "
          f"{len(water)} rivers, {len(towns)} towns, {len(wells_xy)} wells")

    # regla de mapas 2026-07-25: ejes en LV95 km. El dibujo queda en km locales de la
    # grilla (ll2xy); se re-etiquetan los ticks sumando el origen LV95 (error por
    # convergencia de meridiano <~100 m sobre estas cajas, sub-pixel a dpi 140).
    E0, N0 = wgs84_to_lv95(grid.origin[1], grid.origin[0])
    E0_km, N0_km = float(E0) / 1e3, float(N0) / 1e3

    # styled figures viven DENTRO del run de produccion (hermano de production/):
    # asi nunca se mezclan mapas de runs distintos en una carpeta compartida
    outbase = os.path.join(os.path.dirname(os.path.normpath(cfg["group_root"])),
                           "styled_maps")
    print(f"{net}: styled maps -> {outbase}")
    for wdir in sorted(glob.glob(os.path.join(cfg["group_root"], "*"))):
        wave = os.path.basename(wdir)
        if wave not in TITLES:
            continue
        outdir = os.path.join(outbase, wave)
        os.makedirs(outdir, exist_ok=True)
        files = sorted(glob.glob(os.path.join(wdir, "map_T*.npz")))
        for f in files:
            d = np.load(f)
            T = float(d["period"])
            V = np.where(d["mask"].astype(bool), d["vel"], np.nan)
            if not np.isfinite(V).any():
                continue
            lo, hi = np.nanpercentile(V, [2, 98])
            fig, ax = plt.subplots(figsize=(8.6, 6.2))
            draw_layer(ax, V, "jet_r", "group velocity (km/s)", hs, hs_ext, ext,
                       vmin=lo, vmax=hi)
            draw_tecto(ax, tecto, legend=True)
            draw_extras(ax, ext, water, towns, wells_xy)
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
            ax.set_aspect("equal")
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{v + E0_km:.0f}"))
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{v + N0_km:.0f}"))
            ax.set_xlabel("E (km LV95)"); ax.set_ylabel("N (km LV95)")
            ax.set_title(f"{net}  {TITLES[wave]}  T = {T:g} s\n"
                         f"N={int(d['N'])}  var_red={float(d['var_red']):.2f}  "
                         f"(diamonds = wells >1 km)", fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"map_T{T:.1f}.png"), dpi=140)
            plt.close(fig)
        print(f"  {net}/{wave}: {len(files)} maps -> {outdir}")


if __name__ == "__main__":
    for net in (sys.argv[1:] or ("riehen", "aargau", "hautesorne")):
        run_net(net)
