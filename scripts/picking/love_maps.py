"""Geographic per-period Love group-velocity + uncertainty maps from run_production output.

For each map_T*.npz under {output_root}/production/{wave}/, draw a two-panel figure:
  LEFT  : group velocity (RdYlBu), semi-transparent over the SRTM hillshade, with
          tectonic faults/axes, high-res Swiss national borders, high-res rivers/lakes,
          and >1 km deep-well markers.
  RIGHT : the associated posterior uncertainty (converted to velocity units, km/s),
          same overlays and mask.

Reuses the map furniture in grid_vs_postprocess.py (borders/water/wells/faults) and the
hillshade in dem_hillshade.py. Cell centres come from the swtomotv grid (bounds+dx in the
dataset YAML) via geometry.xy2ll, so the mesh registers exactly with the velocity array.

Run (base or das-ambient-noise env with rasterio + geopandas):
  PYTHONPATH=~/Codes/NoisePy-ant python love_maps.py \
      --config <swtomotv_love_yaml> --wave love [--net riehen] [--pmin 0.4 --pmax 4.0]

Writes {output_root}/production/{wave}/geo_maps/love_vg_unc_T*.png (one per period)
plus a contact sheet love_vg_grid.png.
"""
import argparse
import glob
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yaml

import grid_vs_postprocess as G
from dem_hillshade import hillshade_for_bbox, dem_for_net

R_EARTH_KM = 6371.0


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _ll2xy(lat, lon, olat, olon):
    """swtomotv.geometry.ll2xy replica (midpoint-latitude x scaling; parity-exact)."""
    x = R_EARTH_KM * np.cos((lat + olat) / 2 * np.pi / 180) * (lon - olon) * np.pi / 180
    y = R_EARTH_KM * (lat - olat) * np.pi / 180
    return x, y


def _xy2ll(x, y, olat, olon):
    lat = olat + np.asarray(y, float) / R_EARTH_KM * 180 / np.pi
    lon = olon + np.asarray(x, float) / (R_EARTH_KM * np.cos((lat + olat) / 2 * np.pi / 180)) * 180 / np.pi
    return lon, lat


def cell_lonlat(cfg):
    """(lon2d, lat2d) of cell centres, shape (nx, ny) matching npz 'vel' (make_grid replica)."""
    min_lat, max_lat, min_lon, max_lon = cfg["bounds"]
    dx = float(cfg["dx_km"])
    xmax, ymax = _ll2xy(max_lat, max_lon, min_lat, min_lon)
    x = np.arange(0.0, np.ceil(xmax) + dx / 2, dx)
    y = np.arange(0.0, np.ceil(ymax) + dx / 2, dx)
    xc, yc = x + dx / 2.0, y + dx / 2.0
    XC, YC = np.meshgrid(xc, yc, indexing="ij")
    lon2d, lat2d = _xy2ll(XC, YC, min_lat, min_lon)
    return lon2d, lat2d


def station_lonlat(cfg):
    import pandas as pd
    s = pd.read_csv(cfg["station_file"])
    return s.longitude.values, s.latitude.values


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="swtomotv Love dataset YAML")
    ap.add_argument("--wave", default="love")
    ap.add_argument("--net", default="riehen", help="key for hillshade/overlays (dem_for_net, WELLS)")
    ap.add_argument("--pmin", type=float, default=0.4)
    ap.add_argument("--pmax", type=float, default=4.0)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    dx_km = float(cfg["dx_km"])
    proddir = os.path.join(cfg["output_root"], "production", args.wave)
    files = sorted(glob.glob(os.path.join(proddir, "map_T*.npz")),
                   key=lambda f: float(os.path.basename(f)[5:-4]))
    files = [f for f in files
             if args.pmin <= float(os.path.basename(f)[5:-4]) <= args.pmax]
    if not files:
        raise SystemExit(f"no map_T*.npz in {proddir} within T {args.pmin}-{args.pmax} s")

    lon2d, lat2d = cell_lonlat(cfg)
    lo0, lo1 = float(lon2d.min()), float(lon2d.max())
    la0, la1 = float(lat2d.min()), float(lat2d.max())
    hs, extent = hillshade_for_bbox(dem_for_net(args.net), lo0, lo1, la0, la1)
    geol = G.geol_overlays(args.net)
    borders = G._load_borders()
    water = G._load_water()
    wells = G.WELLS_GT1KM.get(args.net, [])
    slon, slat = station_lonlat(cfg)

    outdir = os.path.join(proddir, "geo_maps")
    os.makedirs(outdir, exist_ok=True)

    # shared velocity + uncertainty colour ranges across periods (robust percentiles)
    vals, uncs = [], []
    for f in files:
        z = np.load(f)
        V = z["vel"]
        show = np.isfinite(V)
        vals.append(V[show])
        du = (V ** 2) * z["unc_s"]                  # slowness std -> velocity std, dv = v^2 ds
        uncs.append(du[show])
    allv = np.concatenate(vals); allu = np.concatenate([u[np.isfinite(u)] for u in uncs])
    vlo, vhi = np.nanpercentile(allv, [4, 96])
    umax = np.nanpercentile(allu, 92)

    def panel(ax, field, cmap, vlim, lab):
        ax.imshow(hs, extent=extent, cmap="gray", origin="upper", aspect="auto", zorder=0)
        pc = ax.pcolormesh(lon2d, lat2d, field, cmap=cmap, vmin=vlim[0], vmax=vlim[1],
                           alpha=0.72, shading="nearest", zorder=1)
        plt.colorbar(pc, ax=ax, fraction=0.046, pad=0.04).set_label(lab)
        ax.plot(slon, slat, "^", ms=2.6, mfc="0.15", mec="white", mew=0.25, zorder=5)
        G._plot_extras(ax, geol[0], geol[1], borders, wells, extent, water)
        ax.set(xlim=(extent[0], extent[1]), ylim=(extent[2], extent[3]))
        ax.set_aspect(1.0 / np.cos(np.deg2rad(0.5 * (la0 + la1))))

    grid_thumbs = []
    for f in files:
        z = np.load(f)
        T = float(z["period"]); V = z["vel"]
        dv = np.where(np.isfinite(V), (V ** 2) * z["unc_s"], np.nan)
        N = int(z["N"]); vr = float(z["var_red"]); cov = int(z["coverage"])
        fig, (aL, aR) = plt.subplots(1, 2, figsize=(15.5, 7.0), sharex=True, sharey=True)
        panel(aL, V, "RdYlBu", (vlo, vhi), "Love group velocity [km/s]")
        panel(aR, dv, "viridis", (0, umax), "posterior velocity uncertainty [km/s]")
        aL.set(xlabel="Longitude", ylabel="Latitude")
        aR.set(xlabel="Longitude")
        aL.set_title(f"Love $V_g$   T = {T:.1f} s", fontsize=12, fontweight="bold")
        aR.set_title(f"uncertainty   (N={N} rays, var_red={vr:.2f}, {cov} cells)", fontsize=11)
        fig.suptitle(f"{args.net.capitalize()} Love-wave group velocity — "
                     f"{dx_km*1000:.0f} m grid,  T = {T:.1f} s", fontsize=13, y=0.98)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out = os.path.join(outdir, f"love_vg_unc_T{T:.1f}.png")
        fig.savefig(out, dpi=140); plt.close(fig)
        grid_thumbs.append((T, V))
        print("wrote", out)

    # contact sheet (velocity only)
    n = len(grid_thumbs); nc = min(6, n); nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(3.1 * nc, 3.0 * nr), squeeze=False)
    for ax, (T, V) in zip(axs.ravel(), grid_thumbs):
        ax.imshow(hs, extent=extent, cmap="gray", origin="upper", aspect="auto", zorder=0)
        ax.pcolormesh(lon2d, lat2d, V, cmap="RdYlBu", vmin=vlo, vmax=vhi, alpha=0.75,
                      shading="nearest", zorder=1)
        if borders is not None:
            borders.plot(ax=ax, color="k", lw=0.8, linestyle="--", zorder=6)
        ax.set(xlim=(extent[0], extent[1]), ylim=(extent[2], extent[3]),
               xticks=[], yticks=[])
        ax.set_aspect(1.0 / np.cos(np.deg2rad(0.5 * (la0 + la1))))
        ax.set_title(f"T={T:.1f}s", fontsize=9)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"{args.net.capitalize()} Love $V_g$ ({dx_km*1000:.0f} m grid), "
                 f"shared scale {vlo:.2f}-{vhi:.2f} km/s", y=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "love_vg_grid.png"), dpi=120); plt.close(fig)
    print("wrote", os.path.join(outdir, "love_vg_grid.png"))
    print(f"\n{len(files)} periods -> {outdir}")


if __name__ == "__main__":
    main()
