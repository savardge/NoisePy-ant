#!/usr/bin/env python
"""Vs model figures from a production volume: 500 m depth-slice maps + 1 km-spaced sections.

Design (user spec 2026-07-22):
  MAPS   one figure per depth 0.5..6.0 km step 0.5: Vs mesh (alpha) over DEM hillshade,
         GK500 faults/fold-axes, stations, wells, city labels, scalebar. Color limits are
         per-depth and SHARED across arms via vlims.json written by the first (reference) run,
         so reference-vs-otexempt maps are directly comparable.
  SECTIONS every 1 km along easting (fixed northing / iy) and northing (fixed ix):
         each cell's 1-D profile is snapped to its local DEM surface elevation and the section
         is plotted against ELEVATION (m a.s.l.) -- Y[k,j] = elev_j - 1000*z_k (2-D pcolormesh).
         An exaggerated topographic profile rides on top (own panel, shared x); geophones within
         200 m of the profile are surface triangles; wells within 200 m are vertical lines to
         their true vertical depth below surface with a name label on top; a small inset map
         shows the section line on the hillshade.

Assets from prep_fig_assets.py (DEM + GK500 as plain npz) -- this script needs only
numpy/matplotlib and runs in the bayhunter env:
  PYTHONPATH=~/Codes/NoisePy-ant /opt/anaconda3/envs/bayhunter/bin/python vs_model_figures.py \
      --griddir .../production_2026-07-17_hybrid_recipe --net riehen
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

E = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from well_vs_qc import WELLS                                     # noqa: E402
from noisepy.lv95 import wgs84_to_lv95, extent_lv95_km          # noqa: E402

MAP_DEPTHS = np.arange(0.5, 6.01, 0.5)
NEAR_KM = 0.2                                                    # stations/wells within 200 m
CITIES = {
    "riehen": [("Basel", 7.588, 47.560), ("Riehen", 7.651, 47.583),
               ("Lörrach", 7.664, 47.615), ("Weil a. Rhein", 7.620, 47.593)],
    "aargau": [("Brugg", 8.208, 47.481), ("Baden", 8.306, 47.473),
               ("Waldshut", 8.214, 47.623), ("Frick", 8.024, 47.507)],
}
SECT_STEP_KM = 1.0


# ------------------------------------------------------------------ small helpers
def bilinear(elev, extent, lon, lat):
    """Sample DEM (row 0 = top/north) at lon/lat arrays."""
    lo0, lo1, la0, la1 = extent
    ny, nx = elev.shape
    fx = (np.asarray(lon) - lo0) / (lo1 - lo0) * (nx - 1)
    fy = (la1 - np.asarray(lat)) / (la1 - la0) * (ny - 1)
    x0 = np.clip(np.floor(fx).astype(int), 0, nx - 2)
    y0 = np.clip(np.floor(fy).astype(int), 0, ny - 2)
    tx, ty = np.clip(fx - x0, 0, 1), np.clip(fy - y0, 0, 1)
    z = (elev[y0, x0] * (1 - tx) * (1 - ty) + elev[y0, x0 + 1] * tx * (1 - ty)
         + elev[y0 + 1, x0] * (1 - tx) * ty + elev[y0 + 1, x0 + 1] * tx * ty)
    return z


def km_xy(lon, lat, lat0=None):
    """LV95 (EPSG:2056) E/N en km — regla fija 2026-07-25: todos los mapas en LV95,
    aspecto igual. lat0 se conserva por compatibilidad de firma (sin uso)."""
    E_, N_ = wgs84_to_lv95(lon, lat)
    return E_ / 1e3, N_ / 1e3


def dist_to_polyline_km(px, py, qx, qy):
    """Min distance from points (px,py) to the polyline (qx,qy), all in km."""
    d = np.full(np.shape(px), np.inf)
    for a in range(len(qx) - 1):
        ax, ay, bx, by = qx[a], qy[a], qx[a + 1], qy[a + 1]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = np.clip(((px - ax) * dx + (py - ay) * dy) / max(L2, 1e-12), 0, 1)
        d = np.minimum(d, np.hypot(px - (ax + t * dx), py - (ay + t * dy)))
    return d


def _hillshade(elev, extent):
    ny, nx = elev.shape
    latm = 0.5 * (extent[2] + extent[3])
    dy = (extent[3] - extent[2]) / max(ny, 1) * 111_000.0
    dx = (extent[1] - extent[0]) / max(nx, 1) * 111_000.0 * np.cos(np.deg2rad(latm))
    z = np.nan_to_num(elev, nan=np.nanmin(elev))
    return LightSource(azdeg=315, altdeg=45).hillshade(z, vert_exag=3.0, dx=dx, dy=dy)


def _tecto(ax, gk, lw=1.0, zorder=3):
    """GK500 lines from the flat (verts, offsets, kindcode) layout: 0/1=fault/thrust, 2=axis."""
    verts, offs, kc = gk["verts"], gk["offsets"], gk["kindcode"]
    for i in range(len(kc)):
        xy = verts[offs[i]:offs[i + 1]]
        gx_, gy_ = km_xy(xy[:, 0], xy[:, 1])
        if kc[i] == 2:
            ax.plot(gx_, gy_, color="royalblue", lw=lw, ls="-.", zorder=zorder)
        else:
            ax.plot(gx_, gy_, color="darkred", lw=lw, zorder=zorder)


def load_zrel(griddir, cells):
    """Per-cell z_reliable_max [km] from cells/*.npz, aligned to `cells` order (nan if absent).

    THE RING FIX (user report 2026-07-23): below a cell's data reach the trans-D posterior
    parsimoniously EXTENDS the last constrained velocity downward, so short-reach (rim) cells
    paint an unphysical slow ring at depth. Measured on the riehen reference: cells beyond
    their z_reliable_max are 0.3-0.8 km/s slower than reached cells at the same depth, deficit
    growing downward; at 2 km 56% of cells are beyond reach. Everything below z_reliable_max
    (chain-agreement AND wavelength floor, the wells-validated product of vs_reliability) is
    therefore MASKED in maps and sections -- prior fill is not a measurement.
    """
    out = np.full(len(cells), np.nan)
    for i, (cix, ciy) in enumerate(cells):
        fp = os.path.join(griddir, "cells", f"cell_{int(cix)}_{int(ciy)}_fundotlove.npz")
        if os.path.exists(fp):
            r = np.load(fp, allow_pickle=True)
            if "z_reliable_max" in r.files:
                out[i] = float(r["z_reliable_max"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--griddir", required=True)
    ap.add_argument("--net", required=True, choices=("riehen", "aargau"))
    ap.add_argument("--vlims-from", default=None,
                    help="reference griddir whose figures/vlims.json fixes the color scale "
                         "(default: use/create this griddir's own)")
    ap.add_argument("--unmasked", action="store_true",
                    help="disable the below-reach reliability mask (artifact demo only)")
    a = ap.parse_args()
    net = a.net
    v = np.load(os.path.join(a.griddir, "volume_fundotlove.npz"), allow_pickle=True)
    cells, lonlat, z = v["cells"], v["lonlat"], np.asarray(v["depth"], float)
    vs = np.asarray(v["vs_median"], float)                       # (ncell, ndepth)
    if not a.unmasked:
        zrel = load_zrel(a.griddir, cells)
        n_masked = 0
        for i in range(len(cells)):
            if np.isfinite(zrel[i]):
                m = z > zrel[i] + 1e-9
                n_masked += int(np.isfinite(vs[i, m]).sum())
                vs[i, m] = np.nan
        print(f"reliability mask: NaN below per-cell z_reliable_max "
              f"({n_masked} samples masked across {len(cells)} cells)")
    dem = np.load(f"{E}/{net}/tomo/2_vs_depth_inversion/fig_assets_{net}_dem.npz")
    elev, extent = dem["elev"].astype(float), dem["extent"]
    gk = np.load(f"{E}/{net}/tomo/2_vs_depth_inversion/fig_assets_{net}_gk500.npz",
                 allow_pickle=True)
    st = np.genfromtxt(f"{E}/{net}/tomo/1_velocity_maps/inputs/stations.csv",
                       delimiter=",", names=True)
    hs = _hillshade(elev, extent)
    # optional Geo2Riehen seismic horizons (riehen; prep_fig_assets --stage geo2riehen)
    hz_path = f"{E}/{net}/tomo/2_vs_depth_inversion/fig_assets_{net}_horizons.npz"
    horizons = np.load(hz_path) if os.path.exists(hz_path) else None
    figdir = os.path.join(a.griddir, "figures")
    os.makedirs(os.path.join(figdir, "maps"), exist_ok=True)
    os.makedirs(os.path.join(figdir, "sections"), exist_ok=True)
    arm = os.path.basename(os.path.normpath(a.griddir))
    lat0 = float(np.mean(lonlat[:, 1]))

    # shared color limits (reference writes, comparison arms reuse)
    vl_path = os.path.join((a.vlims_from or a.griddir), "figures", "vlims.json")
    if os.path.exists(vl_path):
        vlims = {float(k): tuple(vv) for k, vv in json.load(open(vl_path)).items()}
        print(f"vlims from {vl_path}")
    else:
        vlims = {}
        for d in MAP_DEPTHS:
            k = int(np.argmin(np.abs(z - d)))
            col = vs[:, k][np.isfinite(vs[:, k])]
            vlims[float(d)] = (float(np.percentile(col, 2)), float(np.percentile(col, 98)))
        os.makedirs(os.path.dirname(vl_path), exist_ok=True)
        json.dump({str(k): vv for k, vv in vlims.items()}, open(vl_path, "w"))
        print(f"vlims written -> {vl_path}")

    # affine node grid (regular in ix/iy)
    ix, iy = cells[:, 0].astype(float), cells[:, 1].astype(float)
    A = np.column_stack([np.ones_like(ix), ix, iy])
    clon = np.linalg.lstsq(A, lonlat[:, 0], rcond=None)[0]
    clat = np.linalg.lstsq(A, lonlat[:, 1], rcond=None)[0]
    nx, ny = int(ix.max()) + 1, int(iy.max()) + 1
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    B = np.column_stack([np.ones(gx.size), gx.ravel(), gy.ravel()])
    lon2d, lat2d = (B @ clon).reshape(nx, ny), (B @ clat).reshape(nx, ny)
    e2d, n2d = km_xy(lon2d, lat2d)                               # LV95 km (regla de mapas)
    extent_km = extent_lv95_km(extent)
    stx, sty = km_xy(st["longitude"], st["latitude"])
    dx_km = abs(clon[1]) * 111.0 * np.cos(np.deg2rad(lat0))      # grid step along ix [km]
    dy_km = abs(clat[2]) * 111.0                                 # grid step along iy [km]

    wells = [(nm, la, lo, dep) for nm, la, lo, dep in WELLS[net]]

    # ------------------------------------------------------------------ A. depth-slice maps
    for d in MAP_DEPTHS:
        k = int(np.argmin(np.abs(z - d)))
        g = np.full((nx, ny), np.nan)
        for (cix, ciy), val in zip(cells, vs[:, k]):
            g[int(cix), int(ciy)] = val
        vmin, vmax = vlims[float(d)]
        fig, ax = plt.subplots(figsize=(7.6, 7.2))
        ax.imshow(hs, extent=extent_km, cmap="gray", origin="upper", zorder=0)
        pc = ax.pcolormesh(e2d, n2d, g, cmap="RdYlBu", vmin=vmin, vmax=vmax,
                           alpha=0.68, shading="nearest", zorder=1)
        _tecto(ax, gk)
        ax.plot(stx, sty, ".", color="k", ms=1.6, alpha=0.55, zorder=4)
        for nm, la, lo, dep in wells:
            wx, wy = km_xy(lo, la)
            ax.plot(wx, wy, "s", mfc="k", mec="w", ms=7, zorder=5)
            ax.annotate(nm, (wx, wy), xytext=(4, 4), textcoords="offset points",
                        fontsize=7.5, fontweight="bold", zorder=6,
                        bbox=dict(fc="w", alpha=0.65, ec="none", pad=1))
        for nm, lo, la in CITIES[net]:
            cx, cy = km_xy(lo, la)
            ax.plot(cx, cy, "o", mfc="w", mec="k", ms=4, zorder=5)
            ax.annotate(nm, (cx, cy), xytext=(4, -8), textcoords="offset points",
                        fontsize=7, style="italic", zorder=6)
        # 2 km scalebar, bottom left (ejes ya en km, largo literal)
        sb_x = extent_km[0] + 0.08 * (extent_km[1] - extent_km[0])
        sb_y = extent_km[2] + 0.05 * (extent_km[3] - extent_km[2])
        ax.plot([sb_x, sb_x + 2.0], [sb_y, sb_y], "k-", lw=3, zorder=6)
        ax.annotate("2 km", (sb_x + 1.0, sb_y), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8, zorder=6)
        ax.set_xlim(extent_km[0], extent_km[1]); ax.set_ylim(extent_km[2], extent_km[3])
        ax.set_aspect("equal")
        ax.set_xlabel("E [km LV95]"); ax.set_ylabel("N [km LV95]")
        ax.set_title(f"{net} Vs median at {d:g} km depth — {arm}", fontsize=11)
        plt.colorbar(pc, ax=ax, fraction=0.04, pad=0.02).set_label("Vs [km/s]")
        out = os.path.join(figdir, "maps", f"vs_map_z{d:03.1f}km.png")
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"maps: {len(MAP_DEPTHS)} -> {figdir}/maps/")

    # ------------------------------------------------------------------ B. sections
    st_elev = bilinear(elev, extent, st["longitude"], st["latitude"])
    global_vmin = min(vv[0] for vv in vlims.values())
    global_vmax = max(vv[1] for vv in vlims.values())

    def one_section(axis, line_idx):
        """axis='EW': fixed iy, along ix. axis='NS': fixed ix, along iy."""
        if axis == "EW":
            sel = np.where(cells[:, 1] == line_idx)[0]
            order = np.argsort(cells[sel, 0])
        else:
            sel = np.where(cells[:, 0] == line_idx)[0]
            order = np.argsort(cells[sel, 1])
        sel = sel[order]
        if len(sel) < 5:
            return False
        plon, plat = lonlat[sel, 0], lonlat[sel, 1]
        px, py = km_xy(plon, plat, lat0)
        along = np.concatenate([[0], np.cumsum(np.hypot(np.diff(px), np.diff(py)))])
        surf = bilinear(elev, extent, plon, plat)                # m a.s.l. per column
        Vs = vs[sel].T                                           # (ndepth, ncol)
        X = np.tile(along, (len(z), 1))
        Y = surf[None, :] - 1000.0 * z[:, None]                  # m a.s.l.

        fig = plt.figure(figsize=(max(9, 0.55 * len(sel)) + 2.2, 7.4))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 4.6], hspace=0.06)
        ax_t = fig.add_subplot(gs[0])
        ax = fig.add_subplot(gs[1], sharex=ax_t)

        # fine topo profile (50 m sampling), exaggerated by the short top panel
        f_along = np.linspace(0, along[-1], max(int(along[-1] / 0.05), 2))
        f_lon = np.interp(f_along, along, plon); f_lat = np.interp(f_along, along, plat)
        f_topo = bilinear(elev, extent, f_lon, f_lat)
        ax_t.fill_between(f_along, f_topo, f_topo.min() - 10, color="0.75", lw=0)
        ax_t.plot(f_along, f_topo, "k-", lw=0.9)
        ax_t.set_ylabel("m a.s.l.", fontsize=8)
        ax_t.tick_params(labelbottom=False, labelsize=7)
        ax_t.set_title(f"{net} — {axis} section "
                       f"{'lat' if axis == 'EW' else 'lon'}="
                       f"{np.mean(plat if axis == 'EW' else plon):.4f} — {arm}", fontsize=10)

        pc = ax.pcolormesh(X, Y, Vs, cmap="RdYlBu", vmin=global_vmin, vmax=global_vmax,
                           shading="nearest")
        ax.plot(along, surf, "k-", lw=1.0)                       # surface
        # geophones within 200 m
        dsta = dist_to_polyline_km(stx, sty, px, py)
        near = dsta <= NEAR_KM
        if near.any():
            # along-position = along-coordinate of the nearest profile vertex (adequate at
            # the 0.5-1 km column spacing; stations are <=200 m off the line by selection)
            vi_near = [int(np.argmin(np.hypot(px - stx[i], py - sty[i])))
                       for i in np.where(near)[0]]
            ax.plot(along[vi_near], st_elev[near] + 15, "v", mfc="lime", mec="k",
                    ms=6, zorder=6, label=f"geophones ≤{int(NEAR_KM*1000)} m")
            ax_t.plot(along[vi_near], st_elev[near], "v", mfc="lime", mec="k", ms=5, zorder=6)
        # wells within 200 m: vertical line to TVD below surface, label on top
        for nm, wla, wlo, wdep in wells:
            wx, wy = km_xy(wlo, wla, lat0)
            if dist_to_polyline_km(np.array([wx]), np.array([wy]), px, py)[0] > NEAR_KM:
                continue
            wa = along[int(np.argmin(np.hypot(px - wx, py - wy)))]
            wel = float(bilinear(elev, extent, np.array([wlo]), np.array([wla]))[0])
            ax.plot([wa, wa], [wel, wel - wdep], color="k", lw=2.2, zorder=7)
            ax.annotate(nm, (wa, wel), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=8.5, fontweight="bold", zorder=8,
                        bbox=dict(fc="w", alpha=0.75, ec="k", lw=0.4, pad=1.5))
        # Geo2Riehen seismic horizons: points within 200 m of the profile, projected to the
        # along-coordinate and linearly interpolated (drawn only across the span where points
        # exist -- no extrapolation beyond the constrained reach of the 3D survey).
        if horizons is not None:
            hv, ho = horizons["verts"], horizons["offsets"]
            hstyle = {"TKris": ("m", "-"), "TMusch": ("darkgreen", "--"),
                      "Base Mesozoic": ("saddlebrown", ":")}
            for hi, hname in enumerate(horizons["names"]):
                pts3 = hv[ho[hi]:ho[hi + 1]]
                hx, hy = km_xy(pts3[:, 0], pts3[:, 1], lat0)
                dh = dist_to_polyline_km(hx, hy, px, py)
                m = dh <= NEAR_KM
                if m.sum() < 2:
                    continue
                # project: along-coordinate of the nearest profile vertex per point, then bin
                # to a fine along-grid and take the median z per bin (robust to the swath of
                # 3D-survey points collapsing onto one profile coordinate)
                ai = np.array([along[int(np.argmin(np.hypot(px - hx[j], py - hy[j])))]
                               for j in np.where(m)[0]])
                zi = pts3[m, 2].astype(float)
                order = np.argsort(ai)
                ai, zi = ai[order], zi[order]
                bins = np.arange(ai.min(), ai.max() + 0.26, 0.25)
                bc, bz = [], []
                for b0, b1 in zip(bins[:-1], bins[1:]):
                    sel_b = (ai >= b0) & (ai < b1)
                    if sel_b.any():
                        bc.append(0.5 * (b0 + b1)); bz.append(np.median(zi[sel_b]))
                if len(bc) < 2:
                    continue
                key = next((k for k in hstyle if str(hname).startswith(k)), "TKris")
                col, ls = hstyle[key]
                ax.plot(bc, bz, color=col, ls=ls, lw=2.0, zorder=6,
                        label=str(hname).split(" (")[0] + " (Geo2Riehen)")
        ax.set_xlabel("distance along profile [km]")
        ax.set_ylabel("elevation [m a.s.l.]")
        ax.set_xlim(0, along[-1])
        ax.set_ylim(np.nanmin(Y) - 100, np.nanmax(surf) + 150)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7, loc="lower left")
        plt.colorbar(pc, ax=ax, pad=0.01, fraction=0.035).set_label("Vs median [km/s]")

        # inset map with the section line (LV95 km, aspecto igual)
        axi = fig.add_axes([0.795, 0.70, 0.185, 0.24])
        axi.imshow(hs, extent=extent_km, cmap="gray", origin="upper")
        nkx, nky = km_xy(lonlat[:, 0], lonlat[:, 1])
        axi.plot(nkx, nky, ".", color="0.55", ms=0.8)
        axi.plot(px, py, "r-", lw=2)
        axi.set_aspect("equal")
        axi.set_xticks([]); axi.set_yticks([])
        for sp in axi.spines.values():
            sp.set_linewidth(1.4)

        tag = f"iy{line_idx:02d}" if axis == "EW" else f"ix{line_idx:02d}"
        out = os.path.join(figdir, "sections", f"sect_{axis}_{tag}.png")
        fig.savefig(out, dpi=145, bbox_inches="tight"); plt.close(fig)
        return True

    step_iy = max(1, int(round(SECT_STEP_KM / dy_km)))
    step_ix = max(1, int(round(SECT_STEP_KM / dx_km)))
    n_ew = sum(one_section("EW", l) for l in range(0, ny, step_iy))
    n_ns = sum(one_section("NS", l) for l in range(0, nx, step_ix))
    print(f"sections: {n_ew} EW + {n_ns} NS -> {figdir}/sections/ "
          f"(steps iy={step_iy}, ix={step_ix}; grid d={dx_km:.2f}/{dy_km:.2f} km)")


if __name__ == "__main__":
    main()
