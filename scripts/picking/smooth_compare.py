"""Compare 5 smoothing/interpolation approaches for the final 1-D-cell Vs volumes
(volume_fundot.npz of each vs_inversion run), working in the LV95 Swiss grid (EPSG:2056).

Approaches (per depth level, on a regular LV95 grid; all masked beyond 1.5 correlation
lengths from the nearest data cell so none of them extrapolates wildly):

  1. linear      -- Delaunay linear (barycentric) interpolation. No smoothing, no
                    uncertainty. The "what the raw model looks like" baseline.
  2. nearest_gauss -- nearest-neighbour gridding followed by a 2-D Gaussian convolution
                    (sigma = L/3) in grid space. The classic quick-look smoothing used in
                    many tomography papers; uncertainty-blind.
  3. idw_unc     -- inverse-distance weighting (power 2) combined with inverse-variance
                    weights: w_i = 1/(d_i^2 + eps^2) * 1/sigma_i^2. Local (k nearest
                    neighbours); noisy cells are down-weighted.
  4. gauss_unc   -- uncertainty-weighted Gaussian kernel regression (Nadaraya-Watson):
                    w_i = exp(-d_i^2 / 2h^2) / sigma_i^2 with h = L/3. Smooth global
                    weighting; the simplest principled use of the posterior widths.
  5. krige_unc   -- heteroscedastic ordinary kriging (GP regression, Matern-3/2,
                    length scale = tomographic correlation length L), each cell's
                    (68% half-width)^2 as observation noise + a nugget floor. The
                    geostatistical reference method; also returns its own interpolation
                    std, which is used to veil unreliable areas.

Uncertainty per cell: sigma = 0.5*(vs_p84 - vs_p16). Approaches 2-5 get a light vertical
Gaussian coupling (sigma_z = 0.15 km) so the levels form a coherent 3-D volume; the linear
baseline is left untouched.

Outputs per run, in <griddir>/smoothing_comparison/:
  volume_<approach>.npz                   -- (E2d, N2d [m LV95], depth [km], vs; +std for kriging)
  compare_depth_<d>km.png                 -- raw cells + the 5 approaches side by side
  maps_<approach>.png                     -- one approach at 6 depths
  profile_NS_<i>_<approach+compare>.png   -- N-S vertical sections, equal aspect (km = km)
  profile_EW_<i>_...                      -- E-W vertical sections, equal aspect

Run (base env):
  /opt/anaconda3/bin/python smooth_compare.py --net riehen \
      --griddir .../riehen/tomo/vs_inversion/grid_physical_200m
  /opt/anaconda3/bin/python smooth_compare.py --all      # every run of both networks
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.linalg import cho_factor, cho_solve
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.spatial import cKDTree, Delaunay
from scipy.spatial.distance import cdist

PROJROOT = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
RUNS = {
    "riehen": ["grid", "grid_combined", "grid_combined_200m", "grid_physical_200m"],
    "aargau": ["grid", "grid_combined", "grid_combined_500m", "grid_physical_500m"],
}
LENGTH_KM = {"riehen": 1.5, "aargau": 3.0}   # tomographic correlation length per net
MASK_L = 1.5                                  # mask beyond MASK_L * L from nearest cell
DZ = 0.1                                      # km, resampled depth step for the volumes
VSMOOTH_KM = 0.15                             # vertical Gaussian coupling
MAP_DEPTHS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]  # km
UNC_SHADE = 0.2                               # km/s, kriging-std veil threshold
CMAP = "RdYlBu"

APPROACHES = ["linear", "nearest_gauss", "idw_unc", "gauss_unc", "krige_unc"]
LABELS = {
    "linear": "linear (Delaunay, no smoothing)",
    "nearest_gauss": "nearest + Gaussian conv. (unweighted)",
    "idw_unc": "IDW $\\times$ 1/$\\sigma^2$ (unc.-weighted)",
    "gauss_unc": "Gaussian kernel $\\times$ 1/$\\sigma^2$ (unc.-weighted)",
    "krige_unc": "ordinary kriging, heterosc. noise (unc.-weighted)",
}

_TR = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)


def ll_to_lv95_km(lon, lat):
    e, n = _TR.transform(np.asarray(lon), np.asarray(lat))
    return np.asarray(e) / 1e3, np.asarray(n) / 1e3


def load_run(net, griddir):
    """Load volume_fundot.npz -> dict with LV95-km cell coords, vs, sigma, depth."""
    V = np.load(os.path.join(griddir, "volume_fundot.npz"))
    lonlat, z = V["lonlat"], V["depth"]
    vs = V["vs_median"]
    sd = 0.5 * (V["vs_p84"] - V["vs_p16"])
    xd, yd = ll_to_lv95_km(lonlat[:, 0], lonlat[:, 1])
    # regular prediction grid: half the median cell spacing, capped at ~150 nodes/side
    tree = cKDTree(np.column_stack([xd, yd]))
    dnn = np.median(tree.query(np.column_stack([xd, yd]), k=2)[0][:, 1])
    pad = 0.5 * dnn
    ex = (xd.min() - pad, xd.max() + pad)
    ey = (yd.min() - pad, yd.max() + pad)
    dg = max(dnn / 2.0, (max(ex[1] - ex[0], ey[1] - ey[0])) / 150.0)
    gx = np.arange(ex[0], ex[1] + dg / 2, dg)
    gy = np.arange(ey[0], ey[1] + dg / 2, dg)
    E2d, N2d = np.meshgrid(gx, gy, indexing="ij")           # (nx, ny)
    dmin = tree.query(np.column_stack([E2d.ravel(), N2d.ravel()]))[0].reshape(E2d.shape)
    zz = np.round(np.arange(0.0, float(z.max()) + 1e-6, DZ), 3)
    jz = [int(np.argmin(np.abs(z - d))) for d in zz]
    stations = None
    stf = os.path.join(PROJROOT, net, "tomo", "stations.csv")
    if os.path.exists(stf):
        s = np.genfromtxt(stf, delimiter=",", names=True, dtype=None, encoding="utf-8")
        stations = ll_to_lv95_km(s["longitude"], s["latitude"])
    return dict(net=net, griddir=griddir, xd=xd, yd=yd, vs=vs[:, jz], sd=sd[:, jz],
                depth=zz, gx=gx, gy=gy, E2d=E2d, N2d=N2d, dmin=dmin, dnn=dnn,
                L=LENGTH_KM[net], stations=stations,
                lon=lonlat[:, 0].copy(), lat=lonlat[:, 1].copy())


# ---------------------------------------------------------------- LV95 overlays (Swisstopo)
import grid_vs_postprocess as G

_OV_CACHE = {}


def _to_km_gdf(gdf):
    """Reproject a WGS84 GeoDataFrame to LV95 and scale metres -> km (so it plots on km axes)."""
    g = gdf.to_crs(2056).copy()
    g["geometry"] = g.geometry.scale(xfact=1e-3, yfact=1e-3, origin=(0, 0))
    return g


def load_overlays(net, lon, lat):
    """Swisstopo faults / anticline axes / rivers / lakes / national border reprojected to LV95 km,
    plus an SRTM hillshade warped to LV95 km (pcolormesh mesh). Cached per net; returns a dict with
    None entries where a layer or geopandas/rasterio is unavailable."""
    key = net
    if key in _OV_CACHE:
        return _OV_CACHE[key]
    ov = dict(faults=None, axes=None, rivers=None, lakes=None, borders=None, hs=None)
    try:
        faults, axes_ = G.geol_overlays(net)
        ov["faults"] = _to_km_gdf(faults) if faults is not None else None
        ov["axes"] = _to_km_gdf(axes_) if axes_ is not None else None
    except Exception as e:
        print("  (fault/anticline overlay skipped:", e, ")")
    try:
        rivers, lakes = G._load_water()
        ov["rivers"] = _to_km_gdf(rivers) if rivers is not None else None
        ov["lakes"] = _to_km_gdf(lakes) if lakes is not None else None
    except Exception as e:
        print("  (hydrography overlay skipped:", e, ")")
    try:
        b = G._load_borders()
        ov["borders"] = b.to_crs(2056).scale(xfact=1e-3, yfact=1e-3, origin=(0, 0)) \
            if b is not None else None
    except Exception as e:
        print("  (border overlay skipped:", e, ")")
    try:
        from dem_hillshade import hillshade_for_bbox, dem_for_net
        lo0, lo1 = float(np.min(lon)), float(np.max(lon))
        la0, la1 = float(np.min(lat)), float(np.max(lat))
        hs, ext = hillshade_for_bbox(dem_for_net(net), lo0, lo1, la0, la1)
        ny, nx = hs.shape
        lons = np.linspace(ext[0], ext[1], nx)
        lats = np.linspace(ext[3], ext[2], ny)                # row 0 = north (lat max)
        LON, LAT = np.meshgrid(lons, lats)
        Xk, Yk = ll_to_lv95_km(LON.ravel(), LAT.ravel())
        ov["hs"] = (Xk.reshape(hs.shape), Yk.reshape(hs.shape), hs)
    except Exception as e:
        print("  (hillshade skipped:", e, ")")
    _OV_CACHE[key] = ov
    return ov


def _draw_basemap(ax, ov):
    """SRTM hillshade in LV95 km (grayscale, behind everything)."""
    if ov.get("hs") is not None:
        Xk, Yk, H = ov["hs"]
        ax.pcolormesh(Xk, Yk, H, cmap="gray", shading="nearest", zorder=0, rasterized=True)


def _draw_overlays(ax, ov, extent_km):
    """Faults / thrusts / anticline axes + rivers + lakes + national border, drawn on top."""
    bb = extent_km
    f = ov.get("faults")
    if f is not None and len(f):
        for typ, sub in f.groupby("Type"):
            thrust = "Chevauchement" in typ
            ls = "-" if "certain" in typ else (0, (4, 2))
            sub.plot(ax=ax, color="saddlebrown" if thrust else "black",
                     linewidth=1.3 if thrust else 0.7, linestyle=ls, zorder=4)
    a = ov.get("axes")
    if a is not None and len(a):
        a.plot(ax=ax, color="red", linewidth=1.8, linestyle="-.", zorder=5)
    lakes = ov.get("lakes")
    if lakes is not None:
        sub = lakes.cx[bb[0]:bb[1], bb[2]:bb[3]]
        if len(sub):
            sub.plot(ax=ax, facecolor="#7fb8e0", edgecolor="#2b6ca3", linewidth=0.4,
                     alpha=0.9, zorder=6)
    rivers = ov.get("rivers")
    if rivers is not None:
        sub = rivers.cx[bb[0]:bb[1], bb[2]:bb[3]]
        if len(sub):
            sub.plot(ax=ax, color="#1f6fb2", linewidth=1.1, zorder=7)
    b = ov.get("borders")
    if b is not None:
        b.plot(ax=ax, color="black", linewidth=1.3, linestyle=(0, (6, 2)), zorder=6)


# ------------------------------------------------- depth-resolution surface z_res(x, y)
# The 1-D inversions run to 6 km everywhere, but the data do not constrain that deep:
# (a) each cell's dispersion curve stops at some max period (short at the network edges), and
# (b) beyond a per-network cap T_cap even the existing long-period picks are unreliable --
#     group velocities measured on paths shorter than ~2 wavelengths are biased LOW (near-field/
#     interference), which is what carves the spurious model-wide LVZ at 4-5 km depth.
# z_res = lambda/2 at the longest *reliable* period of the cell's own curves, both waves.
MIN_FARFIELD_LAMBDA = 2.0     # a pick is far-field if distance >= 2 wavelengths
MIN_FARFIELD_PICKS = 30       # T_cap = longest T with at least this many far-field picks

_TCAP_CACHE = {}


def _period_cap(net, wave):
    """Longest period whose global pick pool still has >= MIN_FARFIELD_PICKS picks measured at
    >= MIN_FARFIELD_LAMBDA wavelengths of interstation distance (lambda from the period's
    median group velocity)."""
    key = (net, wave)
    if key in _TCAP_CACHE:
        return _TCAP_CACHE[key]
    import pandas as pd
    f = os.path.join(PROJROOT, net, "tomo", f"picks_{wave}.csv")
    tcap = np.inf
    if os.path.exists(f):
        p = pd.read_csv(f)
        ok_T = []
        for T, s in p.groupby("inst_period"):
            lam = s["group_velocity"].median() * T
            if (s["distance"] >= MIN_FARFIELD_LAMBDA * lam).sum() >= MIN_FARFIELD_PICKS:
                ok_T.append(T)
        tcap = max(ok_T) if ok_T else np.inf
    _TCAP_CACHE[key] = tcap
    print(f"  [{net}] far-field period cap {wave}: {tcap:g} s")
    return tcap


def load_zres(R):
    """Per-cell reliable depth z_res = lambda/2 at min(cell T_max, T_cap) per wave (max over
    waves), interpolated onto the map grid. Returns (zres_grid, zres_cells) or (None, None)."""
    import glob as _glob
    net = R["net"]
    files = _glob.glob(os.path.join(R["griddir"], "cells", "cell_*_fundot.npz"))
    if not files:
        print("  (no cells/ dir; skipping depth-resolution flagging)")
        return None, None
    pts, zr = [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        best = 0.0
        for wv in ("fund", "overtone"):
            kT = f"obsT_{wv}"
            if kT not in d.files or d[kT].size == 0:
                continue
            T, U = d[kT], d[f"obs_{wv}"]
            tc = _period_cap(net, wv)
            ok = T <= tc
            if not ok.any():
                continue
            i = int(np.argmax(T[ok]))
            lam = 1.1 * U[ok][i] * T[ok][i]         # phase vel ~ 1.1 x group vel
            best = max(best, 0.5 * lam)
        if best > 0:
            lo, la = d["cell_lonlat"]
            pts.append((lo, la)); zr.append(best)
    if not pts:
        return None, None
    pts = np.asarray(pts); zr = np.asarray(zr)
    xz, yz = ll_to_lv95_km(pts[:, 0], pts[:, 1])
    # smooth L/3 Gaussian-kernel interpolation of z_res onto the map grid
    tree = cKDTree(np.column_stack([xz, yz]))
    h = R["L"] / 3.0
    Q = np.column_stack([R["E2d"].ravel(), R["N2d"].ravel()])
    zg = np.full(len(Q), np.nan)
    for q, idx in enumerate(tree.query_ball_point(Q, r=3.0 * h)):
        if not idx:
            continue
        idx = np.asarray(idx)
        d2 = np.sum((np.column_stack([xz, yz])[idx] - Q[q]) ** 2, axis=1)
        w = np.exp(-0.5 * d2 / h ** 2)
        zg[q] = np.sum(w * zr[idx]) / w.sum()
    return zg.reshape(R["E2d"].shape), (xz, yz, zr)


RES_VEIL_KW = dict(colors="none", hatches=["////"], zorder=3)


def _veil_below_zres(ax, along, z, zres_line):
    """Hatch the part of a vertical section below the local resolution depth + draw the line."""
    if zres_line is None:
        return
    zl = np.asarray(zres_line, float)
    zl_fill = np.where(np.isfinite(zl), zl, 0.0)       # unknown resolution -> fully veiled
    A, Z = np.meshgrid(along, z)
    bad = Z > zl_fill[None, :]
    ax.contourf(A, Z, bad.astype(float), levels=[0.5, 1.5], **RES_VEIL_KW)
    ax.pcolormesh(along, z, np.where(bad, 1.0, np.nan), cmap=_gray_cmap(), alpha=0.45,
                  shading="nearest", zorder=3)
    ax.plot(along, zl, "k--", lw=1.4, zorder=6)        # NaN gaps break the line naturally


def _veil_map_zres(ax, R, zres_grid, depth_km):
    """Gray veil on a depth map wherever the slice is below the local resolution depth."""
    if zres_grid is None:
        return
    bad = np.where(~(zres_grid >= depth_km), 1.0, np.nan)   # NaN z_res counts as unresolved
    ax.pcolormesh(R["E2d"], R["N2d"], bad, cmap=_gray_cmap(), alpha=0.55,
                  shading="nearest", zorder=3)


def _gray_cmap():
    from matplotlib.colors import ListedColormap
    return ListedColormap(["#b4b4b4"])


# ---------------------------------------------------------------- per-level engines
def _finite(xd, yd, vd, sd):
    ok = np.isfinite(vd) & np.isfinite(sd) & (sd > 0)
    return xd[ok], yd[ok], vd[ok], sd[ok]


def level_linear(R, vd, sd):
    xd, yd, vd, sd = _finite(R["xd"], R["yd"], vd, sd)
    f = LinearNDInterpolator(Delaunay(np.column_stack([xd, yd])), vd)
    return f(R["E2d"], R["N2d"]), None


def level_nearest_gauss(R, vd, sd):
    xd, yd, vd, sd = _finite(R["xd"], R["yd"], vd, sd)
    f = NearestNDInterpolator(np.column_stack([xd, yd]), vd)
    g = f(R["E2d"], R["N2d"])
    dg = R["gx"][1] - R["gx"][0]
    return gaussian_filter(g, sigma=(R["L"] / 3.0) / dg, mode="nearest"), None


def level_idw_unc(R, vd, sd, power=2.0, k=24):
    xd, yd, vd, sd = _finite(R["xd"], R["yd"], vd, sd)
    tree = cKDTree(np.column_stack([xd, yd]))
    k = min(k, len(vd))
    Q = np.column_stack([R["E2d"].ravel(), R["N2d"].ravel()])
    d, i = tree.query(Q, k=k)
    eps = 0.5 * R["dnn"]                                   # softening: no bull's-eyes on nodes
    w = 1.0 / (d ** power + eps ** power) / (sd[i] ** 2)
    m = np.sum(w * vd[i], axis=1) / np.sum(w, axis=1)
    return m.reshape(R["E2d"].shape), None


def level_gauss_unc(R, vd, sd):
    xd, yd, vd, sd = _finite(R["xd"], R["yd"], vd, sd)
    h = R["L"] / 3.0
    tree = cKDTree(np.column_stack([xd, yd]))
    Q = np.column_stack([R["E2d"].ravel(), R["N2d"].ravel()])
    m = np.full(len(Q), np.nan)
    groups = tree.query_ball_point(Q, r=3.0 * h)
    P = np.column_stack([xd, yd])
    for q, idx in enumerate(groups):
        if not idx:
            continue
        idx = np.asarray(idx)
        d2 = np.sum((P[idx] - Q[q]) ** 2, axis=1)
        w = np.exp(-0.5 * d2 / h ** 2) / sd[idx] ** 2
        sw = w.sum()
        if sw > 0:
            m[q] = np.sum(w * vd[idx]) / sw
    return m.reshape(R["E2d"].shape), None


def _matern32(u):
    a = np.sqrt(3.0) * u
    return (1.0 + a) * np.exp(-a)


def level_krige_unc(R, vd, sd, nugget_frac=0.4):
    """Heteroscedastic ordinary kriging, Matern-3/2, as in smooth_maps.py."""
    xd, yd, vd, sd = _finite(R["xd"], R["yd"], vd, sd)
    L = R["L"]
    w = 1.0 / sd ** 2
    mu = float(np.sum(w * vd) / np.sum(w))
    r = vd - mu
    amp2 = max(float(np.var(r)), 1e-4)
    sd = np.sqrt(sd ** 2 + (nugget_frac ** 2) * amp2)
    P = np.column_stack([xd, yd])
    K = amp2 * _matern32(cdist(P, P) / L) + np.diag(sd ** 2) + 1e-6 * np.eye(len(xd))
    c = cho_factor(K, lower=True, check_finite=False)
    alpha = cho_solve(c, r, check_finite=False)
    Q = np.column_stack([R["E2d"].ravel(), R["N2d"].ravel()])
    Ks = amp2 * _matern32(cdist(Q, P) / L)
    mean = mu + Ks @ alpha
    V = cho_solve(c, Ks.T, check_finite=False)
    var = np.maximum(amp2 - np.einsum("ij,ji->i", Ks, V), 0.0)
    return mean.reshape(R["E2d"].shape), np.sqrt(var).reshape(R["E2d"].shape)


ENGINES = {"linear": level_linear, "nearest_gauss": level_nearest_gauss,
           "idw_unc": level_idw_unc, "gauss_unc": level_gauss_unc,
           "krige_unc": level_krige_unc}


def build_volumes(R):
    """Run every approach on every depth level -> {name: (VS3d, SD3d|None)} + save npz."""
    nx, ny, nz = R["E2d"].shape[0], R["E2d"].shape[1], len(R["depth"])
    cover = R["dmin"] <= MASK_L * R["L"]
    outdir = os.path.join(R["griddir"], "smoothing_comparison")
    os.makedirs(outdir, exist_ok=True)
    vols = {}
    for name in APPROACHES:
        VS = np.full((nx, ny, nz), np.nan)
        SD = np.full((nx, ny, nz), np.nan) if name == "krige_unc" else None
        for k in range(nz):
            m, s = ENGINES[name](R, R["vs"][:, k], R["sd"][:, k])
            VS[:, :, k] = m
            if SD is not None:
                SD[:, :, k] = s
        if name != "linear":                      # vertical coupling (not the raw baseline)
            good = np.isfinite(VS)
            VS = gaussian_filter1d(np.nan_to_num(VS), sigma=VSMOOTH_KM / DZ, axis=2)
            VS[~good] = np.nan
            if SD is not None:
                SD = gaussian_filter1d(np.nan_to_num(SD), sigma=VSMOOTH_KM / DZ, axis=2)
                SD[~good] = np.nan
        VS[~cover] = np.nan
        if SD is not None:
            SD[~cover] = np.nan
        vols[name] = (VS, SD)
        kw = dict(E2d=R["E2d"] * 1e3, N2d=R["N2d"] * 1e3, depth=R["depth"], vs=VS,
                  approach=name, length_km=R["L"], net=R["net"])
        if SD is not None:
            kw["std"] = SD
        np.savez_compressed(os.path.join(outdir, f"volume_{name}.npz"), **kw)
        print(f"  [{R['net']}/{os.path.basename(R['griddir'])}] {name} done")
    return vols, outdir


# ------------------------------------------------------------------------- figures
def _decorate_map(ax, R):
    if R["stations"] is not None:
        ax.plot(R["stations"][0], R["stations"][1], "v", ms=3, mfc="k", mec="w",
                mew=0.3, ls="none", zorder=5)
    ax.set_aspect("equal")
    ax.set_xlim(R["gx"][0], R["gx"][-1])
    ax.set_ylim(R["gy"][0], R["gy"][-1])


def _veil(ax, X, Y, std_field, thresh=UNC_SHADE):
    from matplotlib.colors import ListedColormap
    if std_field is None:
        return
    ax.pcolormesh(X, Y, np.where(std_field > thresh, 1.0, np.nan),
                  cmap=ListedColormap(["#c8c8c8"]), alpha=0.55, shading="gouraud", zorder=3)


def fig_compare_depth(R, vols, outdir, d, zres=None):
    k = int(np.argmin(np.abs(R["depth"] - d)))
    raw = R["vs"][:, k]
    fin = raw[np.isfinite(raw)]
    vlo, vhi = np.nanpercentile(fin, 2), np.nanpercentile(fin, 98)
    fig, axs = plt.subplots(2, 3, figsize=(16, 10.5), sharex=True, sharey=True)
    axs = axs.ravel()
    sc = axs[0].scatter(R["xd"], R["yd"], c=raw, s=14, cmap=CMAP, vmin=vlo, vmax=vhi,
                        marker="s", lw=0)
    axs[0].set_title("raw inversion cells", fontsize=10)
    _decorate_map(axs[0], R)
    for a, name in zip(axs[1:], APPROACHES):
        VS, SD = vols[name]
        a.pcolormesh(R["E2d"], R["N2d"], VS[:, :, k], cmap=CMAP, vmin=vlo, vmax=vhi,
                     shading="gouraud")
        if name == "krige_unc":
            _veil(a, R["E2d"], R["N2d"], SD[:, :, k])
        _veil_map_zres(a, R, zres, d)
        a.set_title(LABELS[name], fontsize=10)
        _decorate_map(a, R)
    for a in axs[3:]:
        a.set_xlabel("E LV95 [km]")
    for a in axs[::3]:
        a.set_ylabel("N LV95 [km]")
    cb = fig.colorbar(sc, ax=axs.tolist(), fraction=0.025, pad=0.02)
    cb.set_label("Vs [km/s]")
    fig.suptitle(f"{R['net'].capitalize()} — {os.path.basename(R['griddir'])} — depth "
                 f"{d:g} km (fund+overtone; mask > {MASK_L:g}L from data; "
                 f"gray veil = below data resolution depth or kriging std > "
                 f"{UNC_SHADE:g} km/s)", fontsize=13)
    out = os.path.join(outdir, f"compare_depth_{d:04.2f}km.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig); print("  wrote", out)


def fig_maps_approach(R, vols, outdir, name, zres=None):
    VS, SD = vols[name]
    kk = [int(np.argmin(np.abs(R["depth"] - d))) for d in MAP_DEPTHS]
    fig, axs = plt.subplots(2, 3, figsize=(16, 10.5), sharex=True, sharey=True)
    for a, d, k in zip(axs.ravel(), MAP_DEPTHS, kk):
        g = VS[:, :, k]
        fin = g[np.isfinite(g)]
        if fin.size == 0:
            a.set_axis_off(); continue
        pc = a.pcolormesh(R["E2d"], R["N2d"], g, cmap=CMAP, shading="gouraud",
                          vmin=np.nanpercentile(fin, 2), vmax=np.nanpercentile(fin, 98))
        if name == "krige_unc":
            _veil(a, R["E2d"], R["N2d"], SD[:, :, k])
        _veil_map_zres(a, R, zres, d)
        fig.colorbar(pc, ax=a, fraction=0.046, pad=0.03).set_label("Vs [km/s]")
        a.set_title(f"{d:g} km", fontsize=11)
        _decorate_map(a, R)
    for a in axs[1]:
        a.set_xlabel("E LV95 [km]")
    for a in axs[:, 0]:
        a.set_ylabel("N LV95 [km]")
    fig.suptitle(f"{R['net'].capitalize()} — {os.path.basename(R['griddir'])} — "
                 f"{LABELS[name]}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(outdir, f"maps_{name}.png")
    fig.savefig(out, dpi=140); plt.close(fig); print("  wrote", out)


def _profile_lines(R):
    """3 N-S lines (constant E) + 3 E-W lines (constant N) through the cell cloud."""
    qs = (25, 50, 75)
    return ([("NS", i + 1, float(np.percentile(R["xd"], q))) for i, q in enumerate(qs)] +
            [("EW", i + 1, float(np.percentile(R["yd"], q))) for i, q in enumerate(qs)])


def fig_profile(R, ov, vols, outdir, kind, idx, coord, zres=None):
    """Per approach: vertical section (left) + mid-depth map (right) with the profile line drawn,
    the Vs slice semi-transparent over a Swisstopo hillshade with faults/anticlines/rivers on top.
    Both columns equal km:km aspect. Sections and maps carry SEPARATE colour scales/colorbars
    (the mid-depth slice spans a narrower Vs range than the full-depth section)."""
    z = R["depth"]
    kmid = int(np.argmin(np.abs(z - 0.5 * z[-1])))          # mid-depth slice for the map
    extent_km = (R["gx"][0], R["gx"][-1], R["gy"][0], R["gy"][-1])
    if kind == "NS":
        j = int(np.argmin(np.abs(R["gx"] - coord)))
        along = R["gy"]
        get = lambda VS: VS[j, :, :].T                     # (nz, ny)
        xlab, where = "N LV95 [km] (S $\\rightarrow$ N)", f"E = {coord*1e3:.0f} m"
        line_xy = ([coord, coord], [R["gy"][0], R["gy"][-1]])
        zres_line = zres[j, :] if zres is not None else None
    else:
        j = int(np.argmin(np.abs(R["gy"] - coord)))
        along = R["gx"]
        get = lambda VS: VS[:, j, :].T                     # (nz, nx)
        xlab, where = "E LV95 [km] (W $\\rightarrow$ E)", f"N = {coord*1e3:.0f} m"
        line_xy = ([R["gx"][0], R["gx"][-1]], [coord, coord])
        zres_line = zres[:, j] if zres is not None else None
    allfin = np.concatenate([get(vols[n][0])[np.isfinite(get(vols[n][0]))]
                             for n in APPROACHES])
    if allfin.size == 0:
        return
    vlo, vhi = np.nanpercentile(allfin, 2), np.nanpercentile(allfin, 98)     # section scale
    mapfin = np.concatenate([vols[n][0][:, :, kmid][np.isfinite(vols[n][0][:, :, kmid])]
                             for n in APPROACHES])
    mlo, mhi = np.nanpercentile(mapfin, 2), np.nanpercentile(mapfin, 98)     # map scale
    n = len(APPROACHES)
    fig = plt.figure(figsize=(17, 2.7 * n), layout="constrained")
    gs = fig.add_gridspec(n, 2, width_ratios=[2.3, 1])
    pc_sec = pc_map = None
    left_axes, right_axes = [], []
    aL = aR = None
    for i, name in enumerate(APPROACHES):
        # -- left: vertical section (full-depth colour scale)
        aL = fig.add_subplot(gs[i, 0]); left_axes.append(aL)
        D = get(vols[name][0])
        pc_sec = aL.pcolormesh(along, z, D, cmap=CMAP, vmin=vlo, vmax=vhi, shading="gouraud")
        if name == "krige_unc" and vols[name][1] is not None:
            _veil(aL, along, z, get(vols[name][1]))
        _veil_below_zres(aL, along, z, zres_line)
        aL.invert_yaxis()
        aL.set_aspect("equal")                             # km depth == km horizontal
        aL.set_ylabel("z [km]")
        aL.text(0.008, 0.94, LABELS[name], transform=aL.transAxes, fontsize=9,
                va="top", bbox=dict(fc="w", alpha=0.8, ec="none"))
        if i < n - 1:
            aL.tick_params(labelbottom=False)
        # -- right: mid-depth map, hillshade + semi-transparent Vs + geology, with the trace
        aR = fig.add_subplot(gs[i, 1]); right_axes.append(aR)
        VS, SD = vols[name]
        _draw_basemap(aR, ov)
        pc_map = aR.pcolormesh(R["E2d"], R["N2d"], VS[:, :, kmid], cmap=CMAP,
                               vmin=mlo, vmax=mhi, alpha=0.72, shading="gouraud", zorder=1)
        if name == "krige_unc" and SD is not None:
            _veil(aR, R["E2d"], R["N2d"], SD[:, :, kmid])
        _veil_map_zres(aR, R, zres, z[kmid])
        _draw_overlays(aR, ov, extent_km)
        if R["stations"] is not None:
            aR.plot(R["stations"][0], R["stations"][1], "v", ms=2.5, mfc="k", mec="w",
                    mew=0.3, ls="none", zorder=8)
        aR.plot(*line_xy, "-", color="k", lw=2.4, zorder=9)
        aR.plot(*line_xy, "--", color="w", lw=1.1, zorder=10)
        aR.set_aspect("equal")
        aR.set_xlim(R["gx"][0], R["gx"][-1]); aR.set_ylim(R["gy"][0], R["gy"][-1])
        aR.set_ylabel("N [km]", fontsize=8); aR.tick_params(labelsize=8)
        if i == 0:
            aR.set_title(f"map @ {z[kmid]:g} km depth", fontsize=9)
        if i < n - 1:
            aR.tick_params(labelbottom=False)
    aL.set_xlabel(xlab)
    aR.set_xlabel("E LV95 [km]", fontsize=8)
    cb1 = fig.colorbar(pc_sec, ax=left_axes, fraction=0.02, pad=0.01)
    cb1.set_label("section Vs [km/s]")
    cb2 = fig.colorbar(pc_map, ax=right_axes, fraction=0.05, pad=0.02)
    cb2.set_label(f"map Vs @ {z[kmid]:g} km [km/s]")
    fig.suptitle(f"{R['net'].capitalize()} — {os.path.basename(R['griddir'])} — "
                 f"{kind} profile {idx} ({where}); equal aspect", fontsize=13)
    out = os.path.join(outdir, f"profile_{kind}_{idx}.png")
    fig.savefig(out, dpi=140); plt.close(fig); print("  wrote", out)


def process_run(net, griddir):
    print(f"== {net} / {os.path.basename(griddir)} ==")
    R = load_run(net, griddir)
    vols, outdir = build_volumes(R)
    ov = load_overlays(net, R["lon"], R["lat"])
    zres, zres_cells = load_zres(R)
    if zres is not None:
        np.savez_compressed(os.path.join(outdir, "zres.npz"),
                            E2d=R["E2d"] * 1e3, N2d=R["N2d"] * 1e3, zres=zres,
                            cell_x_km=zres_cells[0], cell_y_km=zres_cells[1],
                            cell_zres=zres_cells[2])
        print(f"  z_res: median {np.nanmedian(zres):.1f} km, "
              f"p10 {np.nanpercentile(zres,10):.1f}, p90 {np.nanpercentile(zres,90):.1f} km")
    for d in MAP_DEPTHS:
        fig_compare_depth(R, vols, outdir, d, zres)
    for name in APPROACHES:
        fig_maps_approach(R, vols, outdir, name, zres)
    for kind, idx, coord in _profile_lines(R):
        fig_profile(R, ov, vols, outdir, kind, idx, coord, zres)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", choices=tuple(RUNS))
    ap.add_argument("--griddir")
    ap.add_argument("--all", action="store_true", help="process every run of both networks")
    args = ap.parse_args()
    if args.all:
        for net, runs in RUNS.items():
            for run in runs:
                process_run(net, os.path.join(PROJROOT, net, "tomo", "vs_inversion", run))
    else:
        if not (args.net and args.griddir):
            ap.error("--net and --griddir required unless --all")
        process_run(args.net, args.griddir)
    print("done.")


if __name__ == "__main__":
    main()
