#!/usr/bin/env python
"""Six-panel Vs depth-slice figure at constant ELEVATION (m a.s.l.), one shared colour scale.

Reproduces the manuscript-style layout (reference figure, 2026-09-09): 3 rows x 2 columns of
map panels, each a slice of the 3-D tension-spline Vs field (volume_<ws>_gridded.npz) at a
fixed elevation, drawn over the DEM hillshade with GK500 faults / fold axes, the A-A', B-B',
C-C' section traces, town labels and the GVL-1 well. ONE colour scale for every panel.

Elevation slices vs depth slices: the gridded volume is indexed by depth BELOW SURFACE, so a
constant-elevation map is built per column -- the DEM is sampled at each grid node and the
column is interpolated at depth = (elev - z_asl). Where that depth falls above the surface or
below the reliable window the node is blank.

Usage (das-ambient-noise env; only numpy/matplotlib/pyproj via noisepy.lv95):
    PYTHONPATH=~/Codes/NoisePy-ant /opt/anaconda3/envs/das-ambient-noise/bin/python \
        vs_elevation_slices.py --griddir <arm dir> [--net hautesorne] [--waveset fund] \
        [--elev -500,-1000,-1500,-2000,-3000,-4500] [--vlims 1.5,4.0] [--cmap roma]

Writes <griddir>/manuscript_elevation_slices.png + .pdf (vector copy for print / slides).
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

E = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from noisepy.lv95 import wgs84_to_lv95, lv95_to_wgs84, extent_lv95_km   # noqa: E402
from noisepy.colormaps import get_cmap                                   # noqa: E402
from section_figure import SECTIONS                                      # noqa: E402
from well_vs_qc import WELLS                                             # noqa: E402

# towns drawn on the hautesorne panels (lon, lat); same set as the reference figure
TOWNS = {
    "hautesorne": [("Porrentruy", 7.0755, 47.4153), ("St-Ursanne", 7.1540, 47.3647),
                   ("Delémont", 7.3440, 47.3640), ("Moutier", 7.3700, 47.2790),
                   ("Saignelégier", 6.9960, 47.2560)],
    "riehen": [], "aargau": [],
}
# section traces: (key, start label, end label), endpoints come from section_figure.SECTIONS
TRACES = {"hautesorne": [("AA", "A", "A'"), ("BB", "B", "B'"), ("CC", "C", "C'")],
          "riehen": [], "aargau": []}


def bilinear(elev, extent, lon, lat):
    """Sample the DEM (row 0 = north) at lon/lat arrays; nan outside."""
    lo0, lo1, la0, la1 = extent
    ny, nx = elev.shape
    fx = (np.asarray(lon) - lo0) / (lo1 - lo0) * (nx - 1)
    fy = (la1 - np.asarray(lat)) / (la1 - la0) * (ny - 1)
    ok = (fx >= 0) & (fx < nx - 1) & (fy >= 0) & (fy < ny - 1)
    x0 = np.clip(np.floor(fx).astype(int), 0, nx - 2)
    y0 = np.clip(np.floor(fy).astype(int), 0, ny - 2)
    tx, ty = np.clip(fx - x0, 0, 1), np.clip(fy - y0, 0, 1)
    z = (elev[y0, x0] * (1 - tx) * (1 - ty) + elev[y0, x0 + 1] * tx * (1 - ty)
         + elev[y0 + 1, x0] * (1 - tx) * ty + elev[y0 + 1, x0 + 1] * tx * ty)
    return np.where(ok, z, np.nan)


def hillshade(elev, extent):
    ny, nx = elev.shape
    latm = 0.5 * (extent[2] + extent[3])
    dy = (extent[3] - extent[2]) / max(ny, 1) * 111_000.0
    dx = (extent[1] - extent[0]) / max(nx, 1) * 111_000.0 * np.cos(np.deg2rad(latm))
    z = np.nan_to_num(elev, nan=np.nanmin(elev))
    return LightSource(azdeg=315, altdeg=45).hillshade(z, vert_exag=3.0, dx=dx, dy=dy)


def draw_tecto(ax, gk, lw=0.7):
    """GK500 flat layout (verts lon/lat, offsets, kindcode): 0/1 = fault/thrust, 2 = fold axis."""
    if gk is None:
        return
    verts, offs, kc = gk["verts"], gk["offsets"], gk["kindcode"]
    for i in range(len(kc)):
        xy = verts[offs[i]:offs[i + 1]]
        x, y = wgs84_to_lv95(xy[:, 0], xy[:, 1])
        if kc[i] == 2:
            ax.plot(x / 1e3, y / 1e3, color="royalblue", lw=lw, ls="-.", zorder=3)
        else:
            ax.plot(x / 1e3, y / 1e3, color="darkred", lw=lw, zorder=3)


def slice_at_elevation(vs, z_km, elev_m, z_asl_m):
    """Interpolate every column of vs (nz, ny, nx) at depth (elev - z_asl); nan where outside.

    z_km is uniform, so the fractional index is direct; nan below the reliable window is
    carried by np.nan in vs itself (a nan neighbour blanks the node, as it should).
    """
    dep = (elev_m - z_asl_m) / 1000.0                       # (ny, nx) depth below surface [km]
    dz = z_km[1] - z_km[0]
    f = (dep - z_km[0]) / dz
    ok = np.isfinite(f) & (f >= 0) & (f <= len(z_km) - 1)
    f = np.where(ok, f, 0.0)
    k0 = np.clip(np.floor(f).astype(int), 0, len(z_km) - 2)
    t = f - k0
    ny, nx = dep.shape
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    out = vs[k0, jj, ii] * (1 - t) + vs[k0 + 1, jj, ii] * t
    return np.where(ok, out, np.nan), dep


def _uncertainty_slice(cellv, zasl, X, Y, vs_slice, mode, by_depth=False):
    """Posterior half-width of the cells at elevation zasl, linearly gridded, clipped to the Vs footprint."""
    from scipy.interpolate import griddata
    z = cellv["z"]
    dep = np.full(len(cellv["ground"]), float(zasl)) if by_depth else (cellv["ground"] - zasl) / 1000.0
    k = np.clip(np.round((dep - z[0]) / (z[1] - z[0])).astype(int), 0, len(z) - 1)
    idx = np.arange(len(k))
    ok = np.isfinite(dep) & (dep >= z[0]) & (dep <= z[-1]) & cellv["rel"][idx, k]
    val = cellv["hw"][idx, k]
    if mode == "rel":
        val = 100.0 * val / cellv["med"][idx, k]
    pts = np.column_stack([cellv["E"][ok], cellv["N"][ok]])
    out = griddata(pts, val[ok], (X, Y), method="linear")
    return np.where(np.isfinite(vs_slice), out, np.nan)


def _reach_depth(tree, res_floor, cellv):
    """Per-cell depth reach [km below surface]: longest period with map res_diag >= res_floor in `tree`,
    then the depth above which 80 % of that cell's own kernel lies (disba, Vp/Vs 1.73, Brocher rho).
    Measure (phase/group) and wave (rayleigh/love) are inferred from the tree path: 'phase' in the path ->
    dc/dVs, else dU/dVs; a path ending in /love -> Love, else Rayleigh."""
    import glob
    from disba import PhaseSensitivity, GroupSensitivity
    meas = "phase" if "phase" in tree else "group"
    wave = "love" if tree.rstrip("/").endswith("love") else "rayleigh"
    Sens = PhaseSensitivity if meas == "phase" else GroupSensitivity
    cells = cellv["cells"]
    Ts, R = [], []
    for f in sorted(glob.glob(os.path.join(tree, "map_T*.npz"))):
        d = np.load(f, allow_pickle=True)
        rd = np.where(np.asarray(d["mask"], bool), np.asarray(d["res_diag"], float), np.nan)
        Ts.append(float(d["period"])); R.append(rd[cells[:, 0], cells[:, 1]])
    Ts, R = np.array(Ts), np.array(R)
    z = cellv["z"]
    zz = np.arange(0.0, z[-1] + 1e-6, 0.1)
    zmid = zz[:-1] + 0.05
    reach = np.full(len(cells), np.nan)
    for c in range(len(cells)):
        usable = np.flatnonzero(R[:, c] >= res_floor)
        if usable.size == 0 or not np.isfinite(cellv["med"][c]).any():
            continue
        Tstar = Ts[usable].max()
        col = np.interp(zmid, z, np.nan_to_num(cellv["med"][c], nan=np.nanmean(cellv["med"][c])))
        vs_ = np.r_[col, col[-1]]
        vp = 1.73 * vs_
        rho = 1.6612 * vp - 0.4721 * vp**2 + 0.0671 * vp**3 - 0.0043 * vp**4 + 0.000106 * vp**5
        try:
            k = np.abs(np.asarray(Sens(np.r_[np.full(len(col), 0.1), 0.0], vp, vs_, rho)(
                float(Tstar), mode=0, wave=wave, parameter="velocity_s").kernel)[:-1])
            reach[c] = zmid[np.searchsorted(np.cumsum(k) / k.sum(), 0.8)]
        except Exception:
            pass
    print(f"reach cap from {os.path.basename(os.path.dirname(os.path.dirname(tree)))}/{wave} {meas} (res_diag >= {res_floor:g}): "
          f"T* median {np.nanmedian([Ts[np.flatnonzero(R[:, c] >= res_floor)].max() for c in range(len(cells)) if (R[:, c] >= res_floor).any()]):.2f} s, "
          f"depth reach median {np.nanmedian(reach):.1f} km (p10 {np.nanpercentile(reach, 10):.1f}, p90 {np.nanpercentile(reach, 90):.1f})")
    return reach


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--griddir", required=True)
    ap.add_argument("--net", default="hautesorne", choices=("riehen", "aargau", "hautesorne"))
    ap.add_argument("--waveset", default="fund")
    ap.add_argument("--elev", default="-500,-1000,-1500,-2000,-3000,-4500",
                    help="slice elevations in m a.s.l., comma-separated (6 = 3x2 layout)")
    ap.add_argument("--depth", default=None,
                    help="slice at constant DEPTH BELOW SURFACE instead of elevation: km, comma-separated, "
                         "e.g. 1,2,3,4,5,6 (the model bottom is 6 km, so -6000 m a.s.l. would be empty)")
    ap.add_argument("--vlims", default="1.5,4.0", help="ONE Vs scale for every panel [km/s]")
    ap.add_argument("--vcenter", type=float, default=None,
                    help="Vs [km/s] that the colormap's WHITE/lightest entry should mark (e.g. the Vs at the "
                         "basement top). The colormap is resampled so its lightest entry sits at 0.5 and a "
                         "two-slope norm maps vmin..vcenter..vmax onto 0..0.5..1")
    ap.add_argument("--unc-max", type=float, default=None,
                    help="blank nodes whose posterior half-width (p84-p16)/2 exceeds this [km/s]; read from "
                         "the cell volume like --uncertainty")
    ap.add_argument("--cmap", default="roma")
    ap.add_argument("--alpha", type=float, default=0.72)
    ap.add_argument("--margin-km", type=float, default=1.0,
                    help="map frame = gridded-field bounding box + this margin")
    ap.add_argument("--anomaly", action="store_true",
                    help="plot each slice as the Vs anomaly in %% relative to THAT slice's mean "
                         "(100*(Vs/mean-1)); the slice mean is printed in the elevation box")
    ap.add_argument("--alims", type=float, default=20.0,
                    help="symmetric colour limit for --anomaly [%%]; one value for every panel "
                         "and every arm (20 clips ~1%% of nodes across the vs_prod3 arms)")
    ap.add_argument("--uncertainty", choices=("abs", "rel"), default=None,
                    help="draw the POSTERIOR HALF-WIDTH (p84-p16)/2 of the cell inversions instead of Vs: "
                         "'abs' in km/s, 'rel' in %% of the median Vs. Read from volume_<ws>.npz (cells), "
                         "sliced at the same elevations, linearly interpolated onto the spline grid and "
                         "clipped to the Vs footprint so the panels overlay the Vs figures exactly")
    ap.add_argument("--ulims", default=None,
                    help="colour limits for --uncertainty 'lo,hi' (default 0,0.3 km/s for abs, 0,12 %% for rel)")
    ap.add_argument("--reach-from", default=None,
                    help="OPTIONAL resolution-based depth cap: path of the PHASE map tree (…/production/fund) "
                         "that fed the arm. Per cell, the longest period whose map res_diag >= --reach-res is "
                         "converted to a depth reach (depth above which 80%% of the disba dc/dVs kernel of that "
                         "cell's own median profile lies); nodes deeper than that are blanked. The stock "
                         "reliable_mask (chain agreement + lambda/3 floor) passes ~96%% of hautesorne cells to the "
                         "6 km model bottom, so it never bites at depth -- this cap does")
    ap.add_argument("--reach-res", type=float, default=0.02,
                    help="res_diag floor for --reach-from (0.02 keeps phase periods to ~4.8 s at hautesorne)")
    ap.add_argument("--fs", type=float, default=14.0,
                    help="base font size [pt]; ticks/labels/annotations scale from it. "
                         "14 gives slide- and print-legible text at the 13x15 in canvas")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    fs = a.fs
    plt.rcParams.update({"font.size": fs})

    gf = os.path.join(a.griddir, f"volume_{a.waveset}_gridded.npz")
    g = np.load(gf, allow_pickle=True)
    z, xk, yk, vs = (np.asarray(g["depth"], float), np.asarray(g["x_km"], float),
                     np.asarray(g["y_km"], float), np.asarray(g["vs"], float))
    print(f"{os.path.basename(gf)}: vs {vs.shape} (depth, y, x), depth {z[0]:g}-{z[-1]:g} km")

    dem = np.load(f"{E}/{a.net}/tomo/2_vs_depth_inversion/fig_assets_{a.net}_dem.npz")
    elev, extent = dem["elev"].astype(float), dem["extent"]
    cellv = None
    if a.uncertainty or a.reach_from or a.unc_max:
        cv = np.load(os.path.join(a.griddir, f"volume_{a.waveset}.npz"), allow_pickle=True)
        cE, cN = wgs84_to_lv95(cv["lonlat"][:, 0], cv["lonlat"][:, 1])
        cellv = dict(E=cE / 1e3, N=cN / 1e3, z=np.asarray(cv["depth"], float), med=np.asarray(cv["vs_median"], float),
                     hw=0.5 * (np.asarray(cv["vs_p84"], float) - np.asarray(cv["vs_p16"], float)),
                     rel=np.asarray(cv["reliable_mask"], bool), cells=cv["cells"],
                     ground=bilinear(elev, extent, cv["lonlat"][:, 0], cv["lonlat"][:, 1]))
        print(f"cells: {len(cE)} from volume_{a.waveset}.npz (posterior half-width median "
              f"{np.nanmedian(cellv['hw'][cellv['rel']]):.3f} km/s over reliable samples)")
    reach_km = None
    if a.reach_from:
        reach_km = _reach_depth(a.reach_from, a.reach_res, cellv)
    gk_path = f"{E}/{a.net}/tomo/2_vs_depth_inversion/fig_assets_{a.net}_gk500.npz"
    gk = np.load(gk_path, allow_pickle=True) if os.path.exists(gk_path) else None
    hs = hillshade(elev, extent)
    ext_km = extent_lv95_km(extent)

    # DEM elevation at every gridded node (grid is LV95 km -> lon/lat -> bilinear sample)
    X, Y = np.meshgrid(xk, yk)
    lon, lat = lv95_to_wgs84(X.ravel() * 1e3, Y.ravel() * 1e3)
    node_elev = bilinear(elev, extent, lon, lat).reshape(X.shape)
    inside = np.isfinite(vs).any(axis=0)
    print(f"surface elevation over the model: {np.nanmin(node_elev[inside]):.0f}-"
          f"{np.nanmax(node_elev[inside]):.0f} m a.s.l.")

    # map frame: bounding box of the gridded field + margin
    jj, ii = np.where(inside)
    x0, x1 = xk[ii.min()] - a.margin_km, xk[ii.max()] + a.margin_km
    y0, y1 = yk[jj.min()] - a.margin_km, yk[jj.max()] + a.margin_km

    elevs = [float(s) for s in (a.depth if a.depth else a.elev).split(",")]
    by_depth = a.depth is not None
    vmin, vmax = (float(s) for s in a.vlims.split(","))
    if a.anomaly:
        vmin, vmax = -a.alims, a.alims
    if a.uncertainty:
        vmin, vmax = ((float(t) for t in a.ulims.split(",")) if a.ulims
                      else ((0.0, 0.3) if a.uncertainty == "abs" else (0.0, 12.0)))
    if a.uncertainty and a.cmap == "roma":
        a.cmap = "magma_r"                      # sequential scale for a width, not a level
    cmap = get_cmap(a.cmap)
    norm = None
    if a.vcenter is not None and not a.uncertainty:
        from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
        tab = cmap(np.linspace(0, 1, 256))[:, :3]
        pw = int(np.argmax(tab.min(1))) / 255.0                # position of the lightest entry
        u = np.linspace(0, 1, 256)
        src = np.where(u <= 0.5, u / 0.5 * pw, pw + (u - 0.5) / 0.5 * (1 - pw))
        cmap = LinearSegmentedColormap.from_list(f"{a.cmap}_c{a.vcenter:g}", cmap(src))
        norm = TwoSlopeNorm(vmin=vmin, vcenter=a.vcenter, vmax=vmax)
        print(f"colormap white moved from {vmin + pw * (vmax - vmin):.2f} to {a.vcenter:g} km/s")
    nrow, ncol = int(np.ceil(len(elevs) / 2)), 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(13.5, 4.4 * nrow + 1.2),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    arm = os.path.basename(os.path.normpath(a.griddir))

    towns = [(nm, *[c / 1e3 for c in wgs84_to_lv95(lo, la)]) for nm, lo, la in TOWNS[a.net]]
    wells = [(nm, *[c / 1e3 for c in wgs84_to_lv95(lo, la)])
             for nm, la, lo, _dep in WELLS.get(a.net, [])]

    pc = None
    for p, (ax, zasl) in enumerate(zip(axes, elevs)):
        if by_depth:
            kz = int(np.argmin(np.abs(z - zasl)))
            sl = vs[kz].copy()
            dep = np.full(sl.shape, float(z[kz]))
        else:
            sl, dep = slice_at_elevation(vs, z, node_elev, zasl)
        if a.unc_max is not None:
            hwg = _uncertainty_slice(cellv, zasl, X, Y, sl, "abs", by_depth=by_depth)
            drop = np.isfinite(sl) & (hwg > a.unc_max)
            print(f"  {zasl:+6.0f}{' km' if by_depth else ' m'}: uncertainty mask (half-width > {a.unc_max:g} km/s) blanks "
                  f"{100 * drop.sum() / max(np.isfinite(sl).sum(), 1):.0f}% of drawn nodes")
            sl = np.where(drop, np.nan, sl)
        if reach_km is not None:
            from scipy.interpolate import griddata as _gd
            okc = np.isfinite(reach_km)
            pts = np.column_stack([cellv["E"][okc], cellv["N"][okc]])
            cap = _gd(pts, reach_km[okc], (X, Y), method="linear")
            cap = np.where(np.isfinite(cap), cap, _gd(pts, reach_km[okc], (X, Y), method="nearest"))
            drop = np.isfinite(sl) & (dep > cap)
            keep = np.isfinite(sl) & ~drop
            print(f"  {zasl:+6.0f}{' km' if by_depth else ' m'}: reach cap blanks {100 * drop.sum() / max(np.isfinite(sl).sum(), 1):.0f}% of drawn nodes "
                  f"(Vs mean blanked {np.nanmean(sl[drop]) if drop.any() else float('nan'):.2f} vs kept {np.nanmean(sl[keep]):.2f} km/s; "
                  f"frac > 3.8: blanked {np.mean(sl[drop] > 3.8) if drop.any() else 0:.2f} vs kept {np.mean(sl[keep] > 3.8):.2f})")
            sl = np.where(drop, np.nan, sl)
        if a.uncertainty:
            sl = _uncertainty_slice(cellv, zasl, X, Y, sl, a.uncertainty, by_depth=by_depth)
        slice_mean = float(np.nanmean(sl[inside]))
        if a.anomaly:
            sl = 100.0 * (sl / slice_mean - 1.0)
        n_ok, n_in = int(np.isfinite(sl[inside]).sum()), int(inside.sum())
        d_in = dep[inside & np.isfinite(sl)]
        if n_ok == 0:
            print(f"  {zasl:+6.0f} {'km depth' if by_depth else 'm a.s.l.'}: nothing left to draw (fully blanked) -- empty panel")
        else:
            fin = sl[np.isfinite(sl)]
            print(f"  {zasl:+6.0f} {'km depth' if by_depth else 'm a.s.l.'}: {100 * n_ok / n_in:5.1f}% of model nodes drawn "
                  f"(depth below surface {d_in.min():.2f}-{d_in.max():.2f} km); "
                  f"mean {slice_mean:.2f} km/s, plotted {fin.min():.2f}-{fin.max():.2f}, "
                  f"{100 * np.mean((fin < vmin) | (fin > vmax)):.1f}% outside the colour range")
        ax.imshow(hs, extent=ext_km, cmap="gray", origin="upper", zorder=0,
                  vmin=0.0, vmax=1.3)
        pc = ax.pcolormesh(xk, yk, sl, cmap=cmap, alpha=a.alpha, shading="nearest", zorder=1,
                           **(dict(norm=norm) if norm is not None else dict(vmin=vmin, vmax=vmax)))
        draw_tecto(ax, gk)
        # section traces (verified LV95 endpoints from section_figure.SECTIONS)
        for key, l0, l1 in TRACES[a.net]:
            _, _, _tag, P, Q = SECTIONS[key]
            px, py = np.array([P[0], Q[0]]) / 1e3, np.array([P[1], Q[1]]) / 1e3
            ax.plot(px, py, "k-", lw=1.4, zorder=5)
            for lab, xx, yy in ((l0, px[0], py[0]), (l1, px[1], py[1])):
                ax.annotate(lab, (xx, yy), xytext=(3, 3), textcoords="offset points",
                            fontsize=fs, fontweight="bold", zorder=6)
        for nm, tx, ty in towns:
            ax.plot(tx, ty, "o", mfc="w", mec="k", ms=6, zorder=5)
            ax.annotate(nm, (tx, ty), xytext=(5, -12), textcoords="offset points",
                        fontsize=fs - 3, style="italic", zorder=6)
        for nm, wx, wy in wells:
            ax.plot(wx, wy, "*", mfc="red", mec="k", ms=15, zorder=6)
            ax.annotate(nm, (wx, wy), xytext=(7, -4), textcoords="offset points",
                        fontsize=fs - 3, fontweight="bold", zorder=6)
        box = f"{zasl:g} km depth" if by_depth else f"{zasl:.0f} m a.s.l."
        if a.anomaly:
            box += f"\nmean {slice_mean:.2f} km/s"
        if a.uncertainty:
            box += f"\nmedian {slice_mean:.2f} " + ("km/s" if a.uncertainty == "abs" else "%")
        ax.text(0.03, 0.04, box, transform=ax.transAxes, fontsize=fs + 1,
                fontweight="bold", va="bottom", ha="left", zorder=7,
                bbox=dict(fc="w", ec="k", lw=0.8, pad=4))
        ax.text(-0.02, 1.02, f"{'abcdefgh'[p]})", transform=ax.transAxes, fontsize=fs + 6,
                fontweight="bold", va="bottom", ha="right")
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=fs - 2)
        ax.locator_params(axis="both", nbins=6)
    for ax in axes[len(elevs):]:
        ax.set_visible(False)
    for ax in axes[-ncol:]:
        ax.set_xlabel("Easting [km, LV95]", fontsize=fs)
    for ax in axes[::ncol]:
        ax.set_ylabel("Northing [km, LV95]", fontsize=fs)

    fig.subplots_adjust(left=0.07, right=0.985, top=0.965, bottom=0.115, wspace=0.08, hspace=0.17)
    cax = fig.add_axes([0.28, 0.050, 0.44, 0.016])
    cb = fig.colorbar(pc, cax=cax, orientation="horizontal")
    cb.set_label(("Vs posterior half-width (p84-p16)/2 [km/s]" if a.uncertainty == "abs"
                  else "Vs posterior half-width [% of Vs]" if a.uncertainty == "rel"
                  else "Vs anomaly [% of slice mean]" if a.anomaly else "Vs [km/s]"), fontsize=fs + 1)
    cb.ax.tick_params(labelsize=fs - 1)
    unit = "%" if (a.anomaly or a.uncertainty == "rel") else "km/s"
    fig.text(0.99, 0.005, f"{arm}  ({a.waveset}, {a.cmap}, {vmin:g} to {vmax:g} {unit} shared)",
             ha="right", va="bottom", fontsize=fs - 5, color="0.45")
    out = a.out or os.path.join(a.griddir, "manuscript_depth_slices.png" if by_depth else "manuscript_elevation_slices.png")
    fig.savefig(out, dpi=200)
    fig.savefig(os.path.splitext(out)[0] + ".pdf")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
