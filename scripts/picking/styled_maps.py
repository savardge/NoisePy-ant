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
import pandas as pd
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

# Paths follow the 2026-08-05 workflow layout of tomo/1_velocity_maps/:
#   0_inputs/{configs,exported_picks_tspws,culled_picks_vbounds,culled_picks_hist2d}
#   1_production/   (current: _prod3_k3)   2_superseded/...   3_diagnostics/
# group_root defaults point at the CURRENT production run; --run-root overrides per call.
NETS = {
    "riehen": dict(
        yaml=f"{EHM}/riehen/tomo/1_velocity_maps/0_inputs/configs/"
             "riehen_unified_tomo_200m_prod.yaml",
        group_root=f"{EHM}/riehen/tomo/1_velocity_maps/1_production/"
                   "tspws_group_scaled_dx0.2_prod3_k3/production",
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
        yaml=f"{EHM}/aargau/tomo/1_velocity_maps/0_inputs/configs/"
             "aargau_unified_tomo_500m_prod.yaml",
        group_root=f"{EHM}/aargau/tomo/1_velocity_maps/1_production/"
                   "tspws_group_scaled_dx0.5_prod3_k3/production",
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
        yaml=f"{EHM}/hautesorne/tomo/1_velocity_maps/0_inputs/configs/"
             "hautesorne_unified_tomo_ffv2.yaml",
        group_root=f"{EHM}/hautesorne/tomo/1_velocity_maps/1_production/"
                   "tspws_group_scaled_dx0.5_prod3_k3/production",
        dem=None,   # YAML already carries the local arcgrid DEM
        tecto=None,  # YAML already carries the GK500 v1.4 layers
        rivers=True,
        towns=["Delémont", "Bassecourt", "Glovelier", "Courfaivre", "Courtételle",
               "Undervelier", "Saulcy", "Boécourt", "Develier", "Courrendlin",
               "Saignelégier", "Moutier", "Porrentruy"]),
}
TITLES = {"fund": "Rayleigh fundamental", "overtone": "Rayleigh overtone",
          "love": "Love fundamental", "love_ot": "Love overtone"}

# Bottom-strip histogram: what the inversion was GIVEN vs what it PRODUCED.
# A model distribution much narrower than the picks means the prior absorbed the spread;
# one that is offset means the map holds velocities the data never measured.
C_PICKS = "#1b7837"   # input pairwise picks
C_CELLS = "#762a83"   # velocity-map cells
HIST_ALPHA = 0.50


def load_picks_table(picks_dir, wave, measure):
    """The pick table this run was built from. Group and phase tables share the column
    names -- `group_velocity` holds PHASE velocity in the _phase files (see the sidecar
    .meta.json, key velocity_column_is_misnamed), so the caller must label by `measure`,
    never by the column name."""
    if not picks_dir:
        return None
    f = os.path.join(picks_dir,
                     "picks_%s_uni%s.csv" % (wave, "_phase" if measure == "phase" else ""))
    if not os.path.exists(f):
        print(f"    (no pick table {os.path.basename(f)})")
        return None
    d = pd.read_csv(f, usecols=["inst_period", "group_velocity"])
    return d[np.isfinite(d.group_velocity)]


def picks_at_period(dpick, T, tol=0.02):
    if dpick is None or not len(dpick):
        return np.array([])
    return dpick.group_velocity.values[np.abs(dpick.inst_period.values - T) <= tol]


def draw_vdist_strip(axh, vcells, vpicks, measure):
    """Two overlaid, semi-transparent histograms on a shared bin grid: the input picks
    and the map cells. Densities, not counts -- there are ~3k cells against anywhere from
    40 to 8000 picks, so raw counts would compare nothing."""
    parts = [a for a in (vcells, vpicks) if a.size]
    if not parts:
        axh.axis("off")
        return
    allv = np.concatenate(parts)
    lo, hi = np.nanpercentile(allv, [0.5, 99.5])
    if not np.isfinite(lo) or hi <= lo:
        lo, hi = float(np.nanmin(allv)), float(np.nanmax(allv)) + 1e-6
    pad = 0.04 * (hi - lo)
    bins = np.linspace(lo - pad, hi + pad, 56)
    for v, c, lab in ((vpicks, C_PICKS, "input pairwise picks"),
                      (vcells, C_CELLS, "velocity-map cells")):
        if not v.size:
            continue
        axh.hist(v, bins=bins, density=True, color=c, alpha=HIST_ALPHA,
                 label="%s  (n=%d, med %.2f)" % (lab, v.size, np.median(v)))
        axh.hist(v, bins=bins, density=True, histtype="step", color=c, lw=1.3)
        axh.axvline(np.median(v), color=c, lw=1.6, ls="--", alpha=0.95)
    axh.set_xlim(bins[0], bins[-1])
    axh.set_xlabel("%s velocity (km/s)" % measure)
    axh.set_ylabel("density")
    axh.legend(fontsize=6.8, loc="upper right", framealpha=0.85)
    axh.grid(alpha=0.25)
    axh.tick_params(labelsize=8)


def fill_speckle_holes(V, d, min_neighbours=6):
    """Fill single-cell gaps that are surrounded by shown cells, from `vel_full`.

    `--mask-mode res` keeps cells whose res_diag clears `--res-drop-q` (0.10), a RELATIVE
    quantile: it always discards the worst 10% of cells regardless of whether their
    resolution is actually poor. res_diag is speckly cell to cell -- it is a matrix diagonal,
    sensitive to local ray geometry -- so the discarded tenth is scattered, not contiguous,
    and interior cells drop out at random. Measured on Aargau phase/scaled k2, T=5.75 s:
    71 such holes, res_diag 0.0118 against a 0.0148 threshold while their neighbours sit at
    0.027. A cell ringed by better-resolved cells is not meaningfully less resolved, and the
    inversion did produce a value for it (`vel_full`, 2.65-3.88 km/s there, all plausible).

    Only holes with >= min_neighbours of their 8 neighbours already shown are filled, so this
    cannot grow the imaged footprint outward -- it closes interior speckle and nothing else.
    """
    if "vel_full" not in d.files:
        return V
    from scipy import ndimage
    Vf = d["vel_full"]
    K = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    V = V.copy()
    for _ in range(3):          # converges by ~3; a cluster closes from its edges inward
        shown = np.isfinite(V)
        nb = ndimage.convolve(shown.astype(np.uint8), K, mode="constant", cval=0)
        hole = (~shown) & (nb >= min_neighbours) & np.isfinite(Vf)
        if not hole.any():
            break
        V[hole] = Vf[hole]
    # Some interior gaps survive because vel_full is NaN there too -- the inversion returned
    # no value, not merely a low-resolution one (94 of 95 remaining holes at Aargau phase
    # T=6.09 s). Those are left as gaps; filling them would be inventing data.
    return V


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


def run_net(net, picks_dir=None, measure=None):
    cfg = NETS[net]
    # `measure` decides which pick table pairs with this run and how the axes are labelled.
    # Derive it from the run directory when not given: tspws_<measure>_<cd>_dx<dx>[_tag].
    if measure is None:
        base = os.path.basename(os.path.dirname(os.path.normpath(cfg["group_root"])))
        measure = "phase" if "_phase_" in base or base.startswith("tspws_phase") else "group"
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
        dpick = load_picks_table(picks_dir, wave, measure)
        files = sorted(glob.glob(os.path.join(wdir, "map_T*.npz")))
        for f in files:
            d = np.load(f)
            T = float(d["period"])
            # The --vplaus "plausibility" veil is DISABLED (2026-08-04, user decision):
            # inversion output must not be masked on the basis of a histogram of the input
            # picks. Where a run was made with it on, `vel` has the veiled cells stripped and
            # `vel_hidden` holds them, so merging the two here restores the full map without
            # re-inverting. On Aargau the veil removed 111-144 cells per period, 100% of them
            # "Plate-forme mesozoique epivarisque" in the north -- flagged purely for being
            # fast (2.74-3.53 km/s), which thin Mesozoic over crystalline basement should be.
            V = np.where(d["mask"].astype(bool), d["vel"], np.nan)
            if "vel_hidden" in d.files:
                Vh = d["vel_hidden"]
                V = np.where(np.isfinite(V), V, np.where(np.isfinite(Vh), Vh, np.nan))
            V = fill_speckle_holes(V, d)
            if not np.isfinite(V).any():
                continue
            lo, hi = np.nanpercentile(V, [2, 98])
            # map on top, velocity-distribution strip underneath
            fig = plt.figure(figsize=(8.6, 7.9))
            gsf = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.30], hspace=0.30)
            ax = fig.add_subplot(gsf[0])
            axh = fig.add_subplot(gsf[1])
            draw_layer(ax, V, "inferno", "%s velocity (km/s)" % measure, hs, hs_ext, ext,
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
            ax.set_title(f"{net}  {TITLES[wave]}  {measure}  T = {T:g} s\n"
                         f"N={int(d['N'])}  var_red={float(d['var_red']):.2f}  "
                         f"(diamonds = wells >1 km)", fontsize=9)
            draw_vdist_strip(axh, V[np.isfinite(V)].ravel(),
                             picks_at_period(dpick, T), measure)
            fig.savefig(os.path.join(outdir, f"map_T{T:.1f}.png"), dpi=140,
                        bbox_inches="tight")
            plt.close(fig)
        print(f"  {net}/{wave}: {len(files)} maps -> {outdir}")


if __name__ == "__main__":
    # --run-root <net>=<path> overrides the hardcoded production dir for that network, so
    # a new run (e.g. the cluster ts-PWS maps, which live in tspws_<measure>_<cd>_dx<dx>/)
    # can be styled without editing this file. Repeatable.
    # --picks-dir <dir>  the exported pick tables this run was built from; they feed the
    #                    bottom velocity-distribution strip. Without it the strip shows the
    #                    map cells alone.
    # --measure group|phase  overrides the guess taken from the run directory name.
    args = list(sys.argv[1:])
    overrides = {}
    keep = []
    picks_dir = None
    measure = None
    i = 0
    while i < len(args):
        if args[i] == "--run-root" and i + 1 < len(args):
            net_, _, path_ = args[i + 1].partition("=")
            overrides[net_] = path_
            i += 2
        elif args[i] == "--picks-dir" and i + 1 < len(args):
            picks_dir = args[i + 1]
            i += 2
        elif args[i] == "--measure" and i + 1 < len(args):
            measure = args[i + 1]
            i += 2
        else:
            keep.append(args[i])
            i += 1
    for net in (keep or list(overrides) or ("riehen", "aargau", "hautesorne")):
        if net in overrides:
            NETS[net]["group_root"] = overrides[net]
        run_net(net, picks_dir=picks_dir, measure=measure)
