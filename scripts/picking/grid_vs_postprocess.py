"""Post-processing + diagnostic figures for the full-grid Vs inversion (grid_vs_inversion.py).

From volume_fund.npz + volume_fundot.npz per network, produces, on an SRTM hillshade with a
semi-transparent velocity mesh on top:
  * vs_map_<ws>.png       depth-slice posterior-median Vs maps
  * unc_map_<ws>.png      depth-slice Vs uncertainty (68% half-width) maps  [spatial uncertainty]
  * fund_vs_fundot_diff.png  depth-slice (Vs_fundot - Vs_fund) maps + RMS(depth) curve
  * qc_map.png            chi_fund, chi_overtone, n_layers, |diff| depth-mean
  * xsection_<ws>.png     transect Vs + uncertainty + (fundot-fund) cross-sections

Runs in an env with rasterio + matplotlib (e.g. /opt/anaconda3/bin/python). Reads only the
volume npz (cell lon/lat are stored there); no bayesbay/BayHunter deps.

Example:
  /opt/anaconda3/bin/python grid_vs_postprocess.py --net riehen --axis x --line 15
  /opt/anaconda3/bin/python grid_vs_postprocess.py --net aargau --axis y --line 15
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dem_hillshade import hillshade_for_bbox, dem_for_net

DEPTHS = [0.3, 0.6, 1.0, 1.5, 2.5]


def node_coords(cells, lonlat):
    """Fit affine lon(ix,iy), lat(ix,iy) from cell centres; return full-grid lon2d,lat2d and
    the grid shape. The tomo grid is regular in (ix,iy), so lon/lat is ~affine in the indices."""
    ix = cells[:, 0].astype(float); iy = cells[:, 1].astype(float)
    A = np.column_stack([np.ones_like(ix), ix, iy])
    clon, _, _, _ = np.linalg.lstsq(A, lonlat[:, 0], rcond=None)
    clat, _, _, _ = np.linalg.lstsq(A, lonlat[:, 1], rcond=None)
    nx = int(cells[:, 0].max()) + 1; ny = int(cells[:, 1].max()) + 1
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    B = np.column_stack([np.ones(gx.size), gx.ravel(), gy.ravel()])
    lon2d = (B @ clon).reshape(nx, ny)
    lat2d = (B @ clat).reshape(nx, ny)
    return lon2d, lat2d, nx, ny


def grid_field(cells, values, nx, ny):
    """Scatter per-cell values onto a (nx,ny) array, NaN where no cell."""
    g = np.full((nx, ny), np.nan)
    for (ix, iy), v in zip(cells, values):
        g[int(ix), int(iy)] = v
    return g


def depth_index(depth, z):
    return int(np.argmin(np.abs(z - depth)))


def bbox_of(lonlat, pad=0.005):
    lo0, lo1 = lonlat[:, 0].min(), lonlat[:, 0].max()
    la0, la1 = lonlat[:, 1].min(), lonlat[:, 1].max()
    return lo0 - pad, lo1 + pad, la0 - pad, la1 + pad


def _basemap(ax, hs, extent):
    ax.imshow(hs, extent=extent, cmap="gray", origin="upper", aspect="auto", zorder=0)


def map_panels(net, cells, lonlat, z, field3d, depths, hs, extent, title, cmap, cbar_label,
               out, vlim=None, symmetric=False, per_panel=True):
    """field3d: (ncell, ndepth). One map panel per depth, Vs mesh (alpha) over hillshade."""
    lon2d, lat2d, nx, ny = node_coords(cells, lonlat)
    n = len(depths)
    fig, axs = plt.subplots(1, n, figsize=(3.6 * n, 4.2), squeeze=False)
    axs = axs.ravel()
    for a, dpth in zip(axs, depths):
        k = depth_index(dpth, z)
        g = grid_field(cells, field3d[:, k], nx, ny)
        _basemap(a, hs, extent)
        if per_panel and vlim is None:
            finite = g[np.isfinite(g)]
            if symmetric:
                m = np.nanpercentile(np.abs(finite), 96) if finite.size else 1
                vmin, vmax = -m, m
            else:
                vmin = np.nanpercentile(finite, 4) if finite.size else 0
                vmax = np.nanpercentile(finite, 96) if finite.size else 1
        else:
            vmin, vmax = vlim
        pc = a.pcolormesh(lon2d, lat2d, g, cmap=cmap, vmin=vmin, vmax=vmax,
                          alpha=0.65, shading="nearest", zorder=1)
        a.set_title(f"{dpth:g} km depth", fontsize=10)
        a.set_xlim(extent[0], extent[1]); a.set_ylim(extent[2], extent[3])
        a.set_xlabel("lon"); a.tick_params(labelsize=7)
        plt.colorbar(pc, ax=a, fraction=0.046, pad=0.04).set_label(cbar_label, fontsize=8)
    axs[0].set_ylabel("lat")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print("wrote", out)


def cross_section(net, axis, line, cells, lonlat, z, vf, uf, vd, out, title):
    """Transect at ix=line (axis y) or iy=line (axis x): Vs, uncertainty, (fundot-fund) rows."""
    fixed = 0 if axis == "y" else 1     # axis y -> vary iy at fixed ix; axis x -> vary ix at fixed iy
    vary = 1 - fixed
    sel = [i for i in range(len(cells)) if cells[i, fixed] == line]
    if not sel:
        print(f"no cells on {axis}-line {line}"); return
    sel.sort(key=lambda i: cells[i, vary])
    along = np.array([lonlat[i, 1] if axis == "y" else lonlat[i, 0] for i in sel])
    Vs = np.array([vf[i] for i in sel]).T       # (ndepth, ncol)
    Un = np.array([uf[i] for i in sel]).T
    Df = np.array([vd[i] for i in sel]).T
    o = np.argsort(along); along = along[o]; Vs = Vs[:, o]; Un = Un[:, o]; Df = Df[:, o]
    lbl = "lat (S->N)" if axis == "y" else "lon (W->E)"
    fig, axs = plt.subplots(3, 1, figsize=(max(8, 0.7 * len(sel)), 10), sharex=True)
    for a, D, cmap, lab, sym in (
        (axs[0], Vs, "RdYlBu", "median Vs [km/s]", False),
        (axs[1], Un, "viridis", "Vs 68% half-width [km/s]", False),
        (axs[2], Df, "RdBu_r", "Vs(fund+ot) - Vs(fund) [km/s]", True)):
        if sym:
            m = np.nanpercentile(np.abs(D[np.isfinite(D)]), 96); vmin, vmax = -m, m
        else:
            vmin = np.nanpercentile(D[np.isfinite(D)], 3); vmax = np.nanpercentile(D[np.isfinite(D)], 97)
        pc = a.pcolormesh(along, z, D, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
        a.invert_yaxis(); a.set_ylabel("depth [km]")
        plt.colorbar(pc, ax=a, pad=0.01).set_label(lab, fontsize=9)
    axs[2].set_xlabel(lbl)
    axs[0].set_title(title)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print("wrote", out)


GK500 = ("/Users/genevievesavard/Library/CloudStorage/OneDrive-LumidasInc/Switzerland/swisstopo/"
         "GK500_V1_1/GK500_V1_1_FR/Shapes_WGS84")
SLICE_DEPTHS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def _trim_label(griddir):
    """Infer the period-trim criterion label from the grid dir name for figure titles."""
    b = os.path.basename(os.path.normpath(griddir)).lower()
    for k in ("physical", "combined", "tomographic"):
        if k in b:
            return f"{k}-trim"
    return "no-trim" if "grid" == b or b.startswith("grid_2") else "combined-trim"


def geol_overlays(net):
    """(faults, axes) GeoDataFrames in WGS84 from the GK500 cut per network, or (None,None)."""
    try:
        import geopandas as gpd
    except Exception as e:
        print(f"  (no geopandas: {e}); skipping fault/axes overlays"); return None, None
    f = f"{GK500}/cut_{net}/LI_Accident_tecto_wgs84_{net}.shp"
    a = f"{GK500}/cut_{net}/LI_Axes_de_struct_wgs84_{net}.shp"
    faults = gpd.read_file(f) if os.path.exists(f) else None
    axes = gpd.read_file(a) if os.path.exists(a) else None
    return faults, axes


def _plot_geol(ax, faults, axes):
    """Overlay GK500 faults (Faille) / thrusts (Chevauchement) and anticline axes."""
    from matplotlib.lines import Line2D
    handles = []
    if faults is not None and len(faults):
        for typ, sub in faults.groupby("Type"):
            thrust = "Chevauchement" in typ
            ls = "-" if "certain" in typ else (0, (4, 2))
            col = "saddlebrown" if thrust else "black"
            sub.plot(ax=ax, color=col, linewidth=1.4 if thrust else 0.8, linestyle=ls, zorder=4)
        handles += [Line2D([0], [0], color="black", lw=0.9, label="fault"),
                    Line2D([0], [0], color="saddlebrown", lw=1.6, label="thrust")]
    if axes is not None and len(axes):
        axes.plot(ax=ax, color="red", linewidth=2.2, linestyle="-.", zorder=5)
        handles += [Line2D([0], [0], color="red", lw=2.2, ls="-.", label="anticline axis")]
    if handles:
        ax.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.85)


def depth_slice_maps(net, griddir, vs_med, cells, lonlat, z, hs, extent, depths):
    """One map figure per depth: SRTM hillshade + semi-transparent Vs mesh + faults + anticlines."""
    lon2d, lat2d, nx, ny = node_coords(cells, lonlat)
    faults, axes = geol_overlays(net)
    outdir = os.path.join(griddir, "depth_slices")
    os.makedirs(outdir, exist_ok=True)
    for d in depths:
        k = depth_index(d, z)
        g = grid_field(cells, vs_med[:, k], nx, ny)
        finite = g[np.isfinite(g)]
        vmin, vmax = (np.nanpercentile(finite, 4), np.nanpercentile(finite, 96)) if finite.size else (0, 1)
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.imshow(hs, extent=extent, cmap="gray", origin="upper", aspect="auto", zorder=0)
        pc = ax.pcolormesh(lon2d, lat2d, g, cmap="RdYlBu", vmin=vmin, vmax=vmax, alpha=0.6,
                           shading="nearest", zorder=1)
        plt.colorbar(pc, ax=ax, fraction=0.046, pad=0.04).set_label("Vs [km/s]")
        _plot_geol(ax, faults, axes)
        ax.set(xlim=(extent[0], extent[1]), ylim=(extent[2], extent[3]), xlabel="lon", ylabel="lat",
               title=f"{net.capitalize()} — median Vs at {d:g} km depth (fund+overtone, {_trim_label(griddir)})")
        out = os.path.join(outdir, f"vs_depth_{d:g}km.png")
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
        print("wrote", out)


BORDERS_SHP = ("/Users/genevievesavard/Library/CloudStorage/OneDrive-LumidasInc/Switzerland/"
               "swisstopo/borders/Shapes_WGS84/swissBOUNDARIES3D_1_4_TLM_LANDESGEBIET_wgs84.shp")
# deep boreholes (>1 km) near each array: name, lat, lon, depth_m (swisstopo deep_wells.csv)
WELLS_GT1KM = {
    "riehen": [("Basel-1", 47.585413, 7.595614, 5009), ("Otterbach-2", 47.577748, 7.603832, 2745),
               ("Riehen-1", 47.587100, 7.649485, 1547), ("Riehen-2", 47.593696, 7.657156, 1247),
               ("Reinach-1", 47.509691, 7.604882, 1793)],
    "aargau": [("Boettstein", 47.565033, 8.227163, 1501), ("Riniken", 47.504507, 8.189936, 1800),
               ("Leuggern", 47.589033, 8.205224, 1689), ("Kaisten", 47.539828, 8.031539, 1306),
               ("Schafisheim", 47.369472, 8.148685, 2006), ("Weiach-1", 47.563788, 8.458407, 2482),
               ("Weiach-2", 47.565144, 8.453530, 2013)],
}


def _load_borders():
    try:
        import geopandas as gpd
        return gpd.read_file(BORDERS_SHP).boundary        # national border lines (CH/DE/...)
    except Exception as e:
        print(f"  (no borders: {e})"); return None


_HYDRO = ("/Users/genevievesavard/Library/CloudStorage/OneDrive-LumidasInc/Switzerland/swisstopo/"
          "swisstlmregio_2022_2056.shp/swissTLMRegio_Product_LV95/Hydrography/WGS84")
_WATER_CACHE = {}


def _load_water():
    """(major rivers KLASSE<=5 incl. the Rhine, lakes) as WGS84 GeoDataFrames, cached."""
    if "w" not in _WATER_CACHE:
        try:
            import geopandas as gpd
            fw = gpd.read_file(f"{_HYDRO}/swissTLMRegio_FlowingWater_wgs84.shp")
            rivers = fw[fw["KLASSE"] <= 5]                 # Rhine, Wiese, Birs, canals -- no streams
            lakes = gpd.read_file(f"{_HYDRO}/swissTLMRegio_Lake_wgs84.shp")
            _WATER_CACHE["w"] = (rivers, lakes)
        except Exception as e:
            print(f"  (no hydrography: {e})"); _WATER_CACHE["w"] = (None, None)
    return _WATER_CACHE["w"]


def _plot_water(ax, water, extent):
    rivers, lakes = water
    bb = (extent[0], extent[1], extent[2], extent[3])
    if lakes is not None:
        sub = lakes.cx[bb[0]:bb[1], bb[2]:bb[3]]
        if len(sub):
            sub.plot(ax=ax, facecolor="#7fb8e0", edgecolor="#2b6ca3", linewidth=0.4, alpha=0.9, zorder=7)
    if rivers is not None:
        sub = rivers.cx[bb[0]:bb[1], bb[2]:bb[3]]
        if len(sub):
            sub.plot(ax=ax, color="#1f6fb2", linewidth=1.4, zorder=7)


def _plot_extras(ax, faults, axes, borders, wells, extent, water=None):
    """Overlay faults/anticlines + rivers/lakes + country borders + deep-well markers."""
    _plot_geol(ax, faults, axes)
    if water is not None:
        _plot_water(ax, water, extent)
    if borders is not None:
        borders.plot(ax=ax, color="black", linewidth=1.6, linestyle=(0, (6, 2)), zorder=6)
    for name, lat, lon, dep in wells:
        if extent[0] <= lon <= extent[1] and extent[2] <= lat <= extent[3]:
            ax.plot(lon, lat, marker="*", ms=14, mfc="white", mec="black", mew=1.2, zorder=8)
            ax.annotate(name, (lon, lat), xytext=(4, 4), textcoords="offset points",
                        fontsize=8, fontweight="bold", color="black", zorder=8,
                        path_effects=[])


def depth_vs_unc_maps(net, griddir, depths):
    """Paired Vs (left) + 68%-half-width uncertainty (right) maps per depth, on the SRTM
    hillshade with faults, anticline axes, country borders, and >1 km well markers."""
    V = np.load(os.path.join(griddir, "volume_fundot.npz"))
    z = V["depth"]; cells = V["cells"]; lonlat = V["lonlat"]
    vs = V["vs_median"]; unc = 0.5 * (V["vs_p84"] - V["vs_p16"])
    lon2d, lat2d, nx, ny = node_coords(cells, lonlat)
    lo0, lo1, la0, la1 = bbox_of(lonlat)
    hs, extent = hillshade_for_bbox(dem_for_net(net), lo0, lo1, la0, la1)
    faults, axes = geol_overlays(net)
    borders = _load_borders()
    water = _load_water()
    wells = WELLS_GT1KM.get(net, [])
    kk = [depth_index(d, z) for d in depths]
    umax = np.nanpercentile(unc[:, kk], 96)               # shared uncertainty scale across depths
    outdir = os.path.join(griddir, "depth_vs_unc")
    os.makedirs(outdir, exist_ok=True)
    for d in depths:
        k = depth_index(d, z)
        gV = grid_field(cells, vs[:, k], nx, ny)
        gU = grid_field(cells, unc[:, k], nx, ny)
        fig, (aL, aR) = plt.subplots(1, 2, figsize=(15.5, 7.2), sharex=True, sharey=True)
        fin = gV[np.isfinite(gV)]
        vlo, vhi = (np.nanpercentile(fin, 4), np.nanpercentile(fin, 96)) if fin.size else (0, 1)
        for ax, g, cmap, lab, vlim in ((aL, gV, "RdYlBu", "Vs [km/s]", (vlo, vhi)),
                                       (aR, gU, "viridis", "Vs 68% half-width [km/s]", (0, umax))):
            ax.imshow(hs, extent=extent, cmap="gray", origin="upper", aspect="auto", zorder=0)
            pc = ax.pcolormesh(lon2d, lat2d, g, cmap=cmap, vmin=vlim[0], vmax=vlim[1], alpha=0.65,
                               shading="nearest", zorder=1)
            plt.colorbar(pc, ax=ax, fraction=0.046, pad=0.04).set_label(lab)
            _plot_extras(ax, faults, axes, borders, wells, extent, water)
            ax.set(xlim=(extent[0], extent[1]), ylim=(extent[2], extent[3]), xlabel="lon")
        aL.set_ylabel("lat")
        aL.set_title("median Vs", fontsize=11); aR.set_title("uncertainty (68% half-width)", fontsize=11)
        fig.suptitle(f"{net.capitalize()} — {d:g} km depth (fund+overtone, {_trim_label(griddir)})",
                     fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out = os.path.join(outdir, f"vs_unc_depth_{d:04.2f}km.png")
        fig.savefig(out, dpi=145); plt.close(fig)
        print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", required=True)
    ap.add_argument("--griddir", default=None)
    ap.add_argument("--axis", default="x", choices=("x", "y"))
    ap.add_argument("--line", type=int, default=15)
    ap.add_argument("--depths", default=None, help="comma km list; default 0.3,0.6,1,1.5,2.5")
    ap.add_argument("--slice-depths", default=None,
                    help="comma km list for the fault/anticline depth-slice maps; default 0.5..3.0")
    ap.add_argument("--primary", default="fundot",
                    help="waveset for the main Vs/uncertainty/QC/section maps (volume_<primary>.npz)")
    ap.add_argument("--compare", default="fund,fundot",
                    help="A,B wavesets for the difference map + effect-vs-depth (Vs_B - Vs_A); "
                         "e.g. fundlove,fundotlove = overtone value WITH Love; "
                         "fund,fundlove = Love value")
    args = ap.parse_args()
    griddir = args.griddir or f"/Users/genevievesavard/Codes/extract_higher_modes/Projects/{args.net}/tomo/2_vs_depth_inversion/_archive/grid"
    depths = [float(x) for x in args.depths.split(",")] if args.depths else DEPTHS
    wa, wb = [w.strip() for w in args.compare.split(",")][:2]
    LBL = {"fund": "fund", "fundot": "fund+ot", "love": "Love", "fundlove": "fund+Love",
           "fundotlove": "fund+ot+Love"}
    la, lb, lp = LBL.get(wa, wa), LBL.get(wb, wb), LBL.get(args.primary, args.primary)

    Vfo = np.load(os.path.join(griddir, f"volume_{args.primary}.npz"))
    Vf = np.load(os.path.join(griddir, f"volume_{wa}.npz"))
    Vb = np.load(os.path.join(griddir, f"volume_{wb}.npz"))
    z = Vfo["depth"]
    cells_fo = Vfo["cells"]; lonlat = Vfo["lonlat"]
    lo0, lo1, la0, la1 = bbox_of(lonlat)
    hs, extent = hillshade_for_bbox(dem_for_net(args.net), lo0, lo1, la0, la1)
    NET = args.net.capitalize()

    # --- Vs maps (primary waveset) ---
    map_panels(args.net, cells_fo, lonlat, z, Vfo["vs_median"], depths, hs, extent,
               f"{NET} — median Vs ({lp}) on SRTM hillshade", "RdYlBu",
               "Vs [km/s]", os.path.join(griddir, f"vs_map_{args.primary}.png"))
    # --- uncertainty maps (primary waveset) ---
    unc_fo = 0.5 * (Vfo["vs_p84"] - Vfo["vs_p16"])
    map_panels(args.net, cells_fo, lonlat, z, unc_fo, depths, hs, extent,
               f"{NET} — Vs uncertainty (68% half-width, {lp})", "viridis",
               "Δu [km/s]", os.path.join(griddir, f"unc_map_{args.primary}.png"), per_panel=True)

    # --- A/B difference (Vs_B - Vs_A, match cells present in both) ---
    key = lambda c: (int(c[0]), int(c[1]))
    idx_a = {key(c): i for i, c in enumerate(Vf["cells"])}
    common = [(i, idx_a[key(c)]) for i, c in enumerate(Vb["cells"]) if key(c) in idx_a]
    ci = np.array([a for a, _ in common]); fi = np.array([b for _, b in common])
    diff = Vb["vs_median"][ci] - Vf["vs_median"][fi]      # (ncommon, ndepth)  Vs_B - Vs_A
    tag = f"{wa}_vs_{wb}"
    map_panels(args.net, Vb["cells"][ci], Vb["lonlat"][ci], z, diff, depths, hs, extent,
               f"{NET} — {lb} minus {la}: Vs({lb}) - Vs({la})", "RdBu_r",
               "ΔVs [km/s]", os.path.join(griddir, f"diff_{tag}.png"), symmetric=True)

    # RMS(depth) of the effect + mean uncertainty change (B - A)
    rms = np.sqrt(np.nanmean(diff ** 2, axis=0))
    unc_a = 0.5 * (Vf["vs_p84"] - Vf["vs_p16"])
    unc_b = 0.5 * (Vb["vs_p84"] - Vb["vs_p16"])
    du = np.nanmean(unc_b[ci] - unc_a[fi], axis=0)
    fig, ax = plt.subplots(1, 2, figsize=(9, 5))
    ax[0].plot(rms, z, "b-"); ax[0].invert_yaxis(); ax[0].set(xlabel="RMS ΔVs [km/s]",
              ylabel="depth [km]", title=f"{lb} vs {la}: |ΔVs| vs depth")
    ax[1].plot(du, z, "r-"); ax[1].axvline(0, color="k", lw=0.5); ax[1].invert_yaxis()
    ax[1].set(xlabel="mean Δ(68% half-width) [km/s]", title=f"uncertainty change\n({lb} − {la})")
    fig.suptitle(f"{NET} — what {lb} adds over {la} ({len(ci)} cells)")
    fig.tight_layout(); fig.savefig(os.path.join(griddir, f"effect_depth_{tag}.png"), dpi=140)
    plt.close(fig); print(f"wrote effect_depth_{tag}.png")

    # --- QC maps ---
    lon2d, lat2d, nx, ny = node_coords(cells_fo, lonlat)
    dmean = np.nanmean(np.abs(diff), axis=1)
    panels = [("chi_fund", Vfo["chi_fund"], "magma", (0, 3)),
              ("chi_overtone", Vfo["chi_overtone"], "magma", (0, 6))]
    if "chi_love" in Vfo.files and np.isfinite(Vfo["chi_love"]).any():
        panels.append(("chi_love", Vfo["chi_love"], "magma", (0, 6)))
    panels += [("mean n_layers", Vfo["n_layers"], "cividis", None),
               (f"|Vs({lb})-Vs({la})| depth-mean", None, "OrRd", None)]
    fig, axs = plt.subplots(1, len(panels), figsize=(4.25 * len(panels), 4.2))
    for a, (lab, vals, cmap, vlim) in zip(axs, panels):
        _basemap(a, hs, extent)
        if lab.startswith("|Vs"):
            g = grid_field(Vb["cells"][ci], dmean, nx, ny)
        else:
            g = grid_field(cells_fo, vals, nx, ny)
        vmin, vmax = vlim if vlim else (np.nanpercentile(g[np.isfinite(g)], 4),
                                        np.nanpercentile(g[np.isfinite(g)], 96))
        pc = a.pcolormesh(lon2d, lat2d, g, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.7,
                          shading="nearest")
        a.set_title(lab, fontsize=9); a.set_xlim(extent[0], extent[1]); a.set_ylim(extent[2], extent[3])
        a.tick_params(labelsize=7); plt.colorbar(pc, ax=a, fraction=0.046, pad=0.04)
    fig.suptitle(f"{NET} — inversion QC / data fit")
    fig.tight_layout(); fig.savefig(os.path.join(griddir, "qc_map.png"), dpi=140)
    plt.close(fig); print("wrote qc_map.png")

    # --- cross-sections (fund+overtone Vs, uncertainty, overtone difference) ---
    diff_full = np.full_like(Vfo["vs_median"], np.nan)
    diff_full[ci] = diff
    cross_section(args.net, args.axis, args.line, cells_fo, lonlat, z,
                  Vfo["vs_median"], unc_fo, diff_full,
                  os.path.join(griddir, "xsection_fundot.png"),
                  f"{NET} {args.axis}-transect (line {args.line}) — Vs, uncertainty, overtone effect")

    # --- per-depth (500 m) Vs maps on hillshade with GK500 faults + anticline axes ---
    sdepths = [float(x) for x in args.slice_depths.split(",")] if args.slice_depths else SLICE_DEPTHS
    depth_slice_maps(args.net, griddir, Vfo["vs_median"], cells_fo, lonlat, z, hs, extent, sdepths)

    # --- paired Vs + uncertainty maps every 250 m to max depth (hillshade + faults + anticlines
    #     + rivers + country border + deep wells) ---
    vudepths = [round(x, 2) for x in np.arange(0.25, float(z.max()) + 1e-6, 0.25)]
    depth_vs_unc_maps(args.net, griddir, vudepths)
    print("done.")


if __name__ == "__main__":
    main()
