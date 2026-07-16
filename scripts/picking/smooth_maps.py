"""Uncertainty-weighted 3-D smoothing/interpolation of a Vs grid volume by Gaussian-process
regression (kriging), and re-rendering of all the maps + cross-sections from the smoothed field.

Why kriging (GP regression): it is the geostatistical standard for interpolating scattered
seismic velocity estimates -- the best linear unbiased predictor, it ingests each cell's
posterior uncertainty directly as heteroscedastic observation noise (so well-constrained cells
are honoured and noisy ones are down-weighted), it honours an anisotropic correlation length, and
it returns its OWN interpolation-uncertainty field (unlike a plain tension spline, which gives a
point estimate only). See e.g. Szwillus et al. (2019, JGR) geostatistical crustal-velocity
analysis and Bayesian eikonal tomography with Gaussian processes.

Method: per depth level, ordinary kriging of the posterior-median Vs with a Matern-3/2 covariance,
length scale = the tomographic correlation length (Riehen 1.5 km, Aargau 3.0 km -- i.e. we do not
claim structure finer than the tomography resolves), amplitude = the per-level signal variance,
observation noise = the cell's (68% half-width)^2. A light vertical Gaussian smoothing couples the
depth levels into a coherent 3-D volume. The kriging standard deviation is the interpolated Vs
uncertainty (small near well-constrained cells, growing into gaps). Predictions are masked beyond
~1.5 correlation lengths from data to avoid unconstrained extrapolation.

Run (base env: rasterio + geopandas + scipy):
  /opt/anaconda3/bin/python smooth_maps.py --net riehen \
    --griddir .../riehen/tomo/vs_inversion/grid_physical_200m --length 1.5
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d

import grid_vs_postprocess as G
from dem_hillshade import hillshade_for_bbox, dem_for_net

LENGTH_KM = {"riehen": 1.5, "aargau": 3.0}       # tomographic correlation length per net


def _ll_to_km(lon, lat, lon0, lat0):
    x = (lon - lon0) * 111.0 * np.cos(np.deg2rad(lat0))
    y = (lat - lat0) * 111.0
    return x, y


def _matern32(u):
    a = np.sqrt(3.0) * u
    return (1.0 + a) * np.exp(-a)


def krige_slice(xd, yd, vd, sd, xg, yg, L, nugget_frac=0.4):
    """Ordinary kriging (GP regression) of one depth level.
    xd,yd,vd,sd: data coords [km], values, 1-sigma. xg,yg: grid coords [km] (flat).
    nugget_frac: extra noise floor = nugget_frac*signal-std added in quadrature to each cell's
    sigma, so a lone tightly-constrained outlier cannot produce a bull's-eye.
    Returns (mean_grid, std_grid, dmin_grid): kriged Vs, kriging std, dist to nearest datum."""
    ok = np.isfinite(vd) & np.isfinite(sd) & (sd > 0)
    xd, yd, vd, sd = xd[ok], yd[ok], vd[ok], sd[ok]
    w = 1.0 / sd ** 2
    mu = float(np.sum(w * vd) / np.sum(w))                 # inverse-variance prior mean
    r = vd - mu
    amp2 = max(float(np.var(r)), 1e-4)                     # signal variance
    sd = np.sqrt(sd ** 2 + (nugget_frac ** 2) * amp2)      # + nugget floor
    P = np.column_stack([xd, yd])
    K = amp2 * _matern32(cdist(P, P) / L) + np.diag(sd ** 2) + 1e-6 * np.eye(len(xd))
    c = cho_factor(K, lower=True, check_finite=False)
    alpha = cho_solve(c, r, check_finite=False)
    Q = np.column_stack([xg, yg])
    Dg = cdist(Q, P)                                       # ngrid x ndata
    Ks = amp2 * _matern32(Dg / L)
    mean = mu + Ks @ alpha
    V = cho_solve(c, Ks.T, check_finite=False)             # ndata x ngrid
    var = np.maximum(amp2 - np.einsum("ij,ji->i", Ks, V), 0.0)
    return mean, np.sqrt(var), Dg.min(axis=1)


def smooth_volume(net, griddir, length_km=None, dz=0.1, vsmooth=0.25, mask_L=1.5, out=None):
    """Krige every depth level onto the full grid; return a smoothed 3-D volume dict + save npz."""
    L = length_km if length_km is not None else LENGTH_KM.get(net, 1.5)
    V = np.load(os.path.join(griddir, "volume_fundot.npz"))
    cells, lonlat, z = V["cells"], V["lonlat"], V["depth"]
    vs = V["vs_median"]; unc = 0.5 * (V["vs_p84"] - V["vs_p16"])
    lon2d, lat2d, nx, ny = G.node_coords(cells, lonlat)
    lon0, lat0 = float(lonlat[:, 0].mean()), float(lonlat[:, 1].mean())
    xd, yd = _ll_to_km(lonlat[:, 0], lonlat[:, 1], lon0, lat0)
    xg, yg = _ll_to_km(lon2d.ravel(), lat2d.ravel(), lon0, lat0)

    zz = np.round(np.arange(0.0, float(z.max()) + 1e-6, dz), 3)
    VS = np.full((nx, ny, len(zz)), np.nan)
    SD = np.full((nx, ny, len(zz)), np.nan)
    dmin = None
    for kk, d in enumerate(zz):
        j = int(np.argmin(np.abs(z - d)))
        m, s, dm = krige_slice(xd, yd, vs[:, j], unc[:, j], xg, yg, L)
        VS[:, :, kk] = m.reshape(nx, ny)
        SD[:, :, kk] = s.reshape(nx, ny)
        dmin = dm.reshape(nx, ny)
    # light vertical smoothing -> coherent 3-D field
    good = np.isfinite(VS)
    VS = gaussian_filter1d(np.nan_to_num(VS), sigma=max(vsmooth / dz, 0.3), axis=2)
    SD = gaussian_filter1d(np.nan_to_num(SD), sigma=max(vsmooth / dz, 0.3), axis=2)
    VS[~good] = np.nan; SD[~good] = np.nan
    # mask cells farther than mask_L correlation lengths from any datum (no wild extrapolation)
    cover = dmin <= mask_L * L
    VS[~cover] = np.nan; SD[~cover] = np.nan
    out = out or os.path.join(griddir, "volume_smoothed.npz")
    np.savez_compressed(out, lon2d=lon2d, lat2d=lat2d, depth=zz, vs=VS, std=SD,
                        length_km=L, net=net)
    print(f"wrote {out}  (grid {nx}x{ny}, {len(zz)} depths, L={L} km)")
    return dict(lon2d=lon2d, lat2d=lat2d, depth=zz, vs=VS, std=SD, L=L, griddir=griddir, net=net)


def _overlays(net):
    return (G.geol_overlays(net), G._load_borders(), G._load_water(), G.WELLS_GT1KM.get(net, []))


UNC_SHADE = 0.2      # km/s: veil areas whose kriging std exceeds this


def _shade_unreliable(ax, lon2d, lat2d, std_field, thresh=UNC_SHADE):
    """Semi-transparent gray veil over areas with kriging std > thresh [km/s]."""
    from matplotlib.colors import ListedColormap
    veil = np.where(std_field > thresh, 1.0, np.nan)
    ax.pcolormesh(lon2d, lat2d, veil, cmap=ListedColormap(["#c8c8c8"]), alpha=0.55,
                  shading="gouraud", zorder=3)


def _panel(ax, lon2d, lat2d, field, cmap, vlim, hs, extent, geol, borders, water, wells, lab,
           std_field=None):
    ax.imshow(hs, extent=extent, cmap="gray", origin="upper", aspect="auto", zorder=0)
    pc = ax.pcolormesh(lon2d, lat2d, field, cmap=cmap, vmin=vlim[0], vmax=vlim[1], alpha=0.72,
                       shading="gouraud", zorder=1)
    plt.colorbar(pc, ax=ax, fraction=0.046, pad=0.04).set_label(lab)
    if std_field is not None:
        _shade_unreliable(ax, lon2d, lat2d, std_field)
    G._plot_extras(ax, geol[0], geol[1], borders, wells, extent, water)
    ax.set(xlim=(extent[0], extent[1]), ylim=(extent[2], extent[3]), xlabel="lon")


def make_maps(sm, depths):
    net, griddir = sm["net"], sm["griddir"]
    lon2d, lat2d, z, VS, SD, L = (sm["lon2d"], sm["lat2d"], sm["depth"], sm["vs"], sm["std"], sm["L"])
    lo0, lo1 = np.nanmin(lon2d), np.nanmax(lon2d)
    la0, la1 = np.nanmin(lat2d), np.nanmax(lat2d)
    hs, extent = hillshade_for_bbox(dem_for_net(net), lo0, lo1, la0, la1)
    geol, borders, water, wells = _overlays(net)
    tag = f"{G._trim_label(griddir)}, kriged (GP, L={L:g} km)"
    kk = [int(np.argmin(np.abs(z - d))) for d in depths]
    umax = np.nanpercentile(SD[:, :, kk], 96)
    outdir = os.path.join(griddir, "smoothed"); os.makedirs(outdir, exist_ok=True)
    for d in depths:
        k = int(np.argmin(np.abs(z - d)))
        gV, gU = VS[:, :, k], SD[:, :, k]
        fin = gV[np.isfinite(gV)]
        vlo, vhi = (np.nanpercentile(fin, 3), np.nanpercentile(fin, 97)) if fin.size else (0, 1)
        fig, (aL, aR) = plt.subplots(1, 2, figsize=(15.5, 7.2), sharex=True, sharey=True)
        _panel(aL, lon2d, lat2d, gV, "RdYlBu", (vlo, vhi), hs, extent, geol, borders, water, wells,
               "Vs [km/s]", std_field=gU)
        _panel(aR, lon2d, lat2d, gU, "viridis", (0, umax), hs, extent, geol, borders, water, wells,
               "kriging std [km/s]")
        aL.set_ylabel("lat")
        aL.set_title(f"median Vs (kriged; gray veil = std > {UNC_SHADE:g} km/s)", fontsize=11)
        aR.set_title("interpolation uncertainty", fontsize=11)
        fig.suptitle(f"{net.capitalize()} — {d:g} km depth (fund+overtone, {tag})", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out = os.path.join(outdir, f"smooth_vs_unc_depth_{d:04.2f}km.png")
        fig.savefig(out, dpi=145); plt.close(fig); print("wrote", out)


def make_xsection(sm, axis, line):
    """Smoothed Vs + uncertainty cross-section along a grid row/column of the kriged volume."""
    net, griddir = sm["net"], sm["griddir"]
    lon2d, lat2d, z, VS, SD = sm["lon2d"], sm["lat2d"], sm["depth"], sm["vs"], sm["std"]
    if axis == "x":
        along, Vs, Un = lon2d[:, line], VS[:, line, :].T, SD[:, line, :].T
        xlab = "lon (W->E)"
    else:
        along, Vs, Un = lat2d[line, :], VS[line, :, :].T, SD[line, :, :].T
        xlab = "lat (S->N)"
    o = np.argsort(along); along = along[o]; Vs = Vs[:, o]; Un = Un[:, o]
    fig, axs = plt.subplots(2, 1, figsize=(max(8, 0.02 * len(along)), 8), sharex=True)
    for a, D, cmap, lab in ((axs[0], Vs, "RdYlBu", "median Vs [km/s]"),
                            (axs[1], Un, "viridis", "kriging std [km/s]")):
        fin = D[np.isfinite(D)]
        pc = a.pcolormesh(along, z, D, cmap=cmap, shading="gouraud",
                          vmin=np.nanpercentile(fin, 3), vmax=np.nanpercentile(fin, 97))
        a.invert_yaxis(); a.set_ylabel("depth [km]")
        plt.colorbar(pc, ax=a, pad=0.01).set_label(lab)
    # gray veil on the Vs section where kriging std exceeds the reliability threshold
    from matplotlib.colors import ListedColormap
    axs[0].pcolormesh(along, z, np.where(Un > UNC_SHADE, 1.0, np.nan),
                      cmap=ListedColormap(["#c8c8c8"]), alpha=0.55, shading="gouraud", zorder=3)
    axs[1].set_xlabel(xlab)
    axs[0].set_title(f"{net.capitalize()} {axis}-transect (line {line}) — kriged Vs + uncertainty "
                     f"({G._trim_label(griddir)}; veil = std > {UNC_SHADE:g} km/s)")
    fig.tight_layout()
    out = os.path.join(griddir, "smoothed", f"smooth_xsection_{axis}{line}.png")
    fig.savefig(out, dpi=145); plt.close(fig); print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", required=True, choices=("riehen", "aargau"))
    ap.add_argument("--griddir", required=True)
    ap.add_argument("--length", type=float, default=None, help="correlation length km (default per net)")
    ap.add_argument("--axis", default="x", choices=("x", "y"))
    ap.add_argument("--line", type=int, default=None)
    args = ap.parse_args()
    sm = smooth_volume(args.net, args.griddir, args.length)
    depths = [round(x, 2) for x in np.arange(0.25, float(sm["depth"].max()) + 1e-6, 0.25)]
    make_maps(sm, depths)
    line = args.line if args.line is not None else (sm["vs"].shape[1 if args.axis == "x" else 0] // 2)
    make_xsection(sm, args.axis, line)
    print("done.")


if __name__ == "__main__":
    main()
