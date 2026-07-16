"""Geographic per-period group-velocity + uncertainty + resolution maps for ANY wave.

Generalization of love_maps.py to the unified tomography (swtomotv-output-uni/production/
{fund,overtone,love}). For each map_T*.npz, draw a three-panel figure:

  LEFT   : group velocity (RdYlBu), semi-transparent over the SRTM hillshade
  MIDDLE : posterior uncertainty (--unc-mode, see the CAVEAT below)
  RIGHT  : resolution-matrix diagonal res_diag (plasma) -- the display mask keeps cells above
           the run's res_thresh (worst --res-drop-q quantile of covered cells is hidden), so this
           panel says WHERE the velocity panel is actually resolved rather than prior-dominated.

CAVEAT (--unc-mode): swtomotv is Tarantola-Valette in SLOWNESS, so the native posterior is unc_s
[s/km]; velocity uncertainty needs the Jacobian, dv = v^2 * unc_s. That v^2 makes fast regions look
uncertain even when they are better constrained. Verified on Riehen Love T=1.6 s: unc_s is flat
across the flexure (W 0.0178 vs E 0.0169 s/km -- the east is slightly BETTER, matching its higher
res_diag 0.069 vs 0.049), yet dv doubles W->E (0.031 -> 0.061 km/s) purely because v^2 goes
1.77 -> 3.57. So:
  dv        (default) = the honest error bar ON THE PLOTTED km/s VALUE -- use for reading the map,
                        but do NOT read it as a constraint map; that is what the res_diag panel is.
  rel       = dv / v [%]  -- fractional error, partly de-Jacobianed; good middle ground.
  slowness  = unc_s [s/km] -- the native inverted parameter; the cleanest "how well constrained".

All panels carry the same furniture: SRTM hillshade, GK500 faults / Jura thrusts / anticline axes,
high-res Swiss national borders, swissTLMRegio rivers+lakes, >1 km deep-well markers, stations.

Colour ranges are shared across periods per wave (robust percentiles) so figures are comparable;
the resolution panel is per-figure autoscaled (res_diag falls with period).

Run (base anaconda env: rasterio + geopandas + yaml):
  PYTHONPATH=~/Codes/NoisePy-ant python tomo_maps.py --config <swtomotv_yaml> --wave love \
      [--net riehen] [--pmin 0.4 --pmax 4.0] [--no-contact-sheet]

Writes {output_root}/production/{wave}/geo_maps/{wave}_T*.png (one per period) + a contact sheet.
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
WAVE_LABEL = {"fund": "Rayleigh fundamental", "overtone": "Rayleigh 1st overtone",
              "love": "Love fundamental"}


def _ll2xy(lat, lon, olat, olon):
    """swtomotv.geometry.ll2xy replica (midpoint-latitude x scaling; parity-exact)."""
    x = R_EARTH_KM * np.cos((lat + olat) / 2 * np.pi / 180) * (lon - olon) * np.pi / 180
    y = R_EARTH_KM * (lat - olat) * np.pi / 180
    return x, y


def _xy2ll(x, y, olat, olon):
    lat = olat + np.asarray(y, float) / R_EARTH_KM * 180 / np.pi
    lon = olon + np.asarray(x, float) / (R_EARTH_KM * np.cos((lat + olat) / 2 * np.pi / 180)) \
        * 180 / np.pi
    return lon, lat


def cell_lonlat(cfg):
    """(lon2d, lat2d) of cell centres, shape (nx, ny) matching npz 'vel' (make_grid replica)."""
    min_lat, max_lat, min_lon, max_lon = cfg["bounds"]
    dx = float(cfg["dx_km"])
    xmax, ymax = _ll2xy(max_lat, max_lon, min_lat, min_lon)
    x = np.arange(0.0, np.ceil(xmax) + dx / 2, dx)
    y = np.arange(0.0, np.ceil(ymax) + dx / 2, dx)
    XC, YC = np.meshgrid(x + dx / 2.0, y + dx / 2.0, indexing="ij")
    return _xy2ll(XC, YC, min_lat, min_lon)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="swtomotv dataset YAML")
    ap.add_argument("--wave", required=True, help="wave key = production subdir (fund/overtone/love)")
    ap.add_argument("--net", default=None, help="riehen|aargau for hillshade/overlays (default: "
                                                "inferred from the YAML name)")
    ap.add_argument("--pmin", type=float, default=0.3)
    ap.add_argument("--pmax", type=float, default=5.0)
    ap.add_argument("--no-contact-sheet", action="store_true")
    ap.add_argument("--unc-mode", default="dv", choices=("dv", "rel", "slowness"),
                    help="uncertainty panel: dv = v^2*unc_s [km/s] (error bar on the plotted "
                         "value, default); rel = dv/v [%%]; slowness = unc_s [s/km] (native, "
                         "Jacobian-free). See the module CAVEAT.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    net = args.net or ("riehen" if "riehen" in cfg["name"] else "aargau")
    dx_km = float(cfg["dx_km"])
    proddir = os.path.join(cfg["output_root"], "production", args.wave)
    files = sorted(glob.glob(os.path.join(proddir, "map_T*.npz")),
                   key=lambda f: float(os.path.basename(f)[5:-4]))
    files = [f for f in files if args.pmin <= float(os.path.basename(f)[5:-4]) <= args.pmax]
    if not files:
        raise SystemExit(f"no map_T*.npz in {proddir} within T {args.pmin}-{args.pmax} s")

    lon2d, lat2d = cell_lonlat(cfg)
    lo0, lo1 = float(lon2d.min()), float(lon2d.max())
    la0, la1 = float(lat2d.min()), float(lat2d.max())
    hs, extent = hillshade_for_bbox(dem_for_net(net), lo0, lo1, la0, la1)
    geol = G.geol_overlays(net)
    borders, water = G._load_borders(), G._load_water()
    wells = G.WELLS_GT1KM.get(net, [])
    import pandas as pd
    sta = pd.read_csv(cfg["station_file"])
    slon, slat = sta.longitude.values, sta.latitude.values

    outdir = os.path.join(proddir, "geo_maps")
    os.makedirs(outdir, exist_ok=True)

    def unc_field(z, V):
        """Uncertainty per --unc-mode (see the module CAVEAT on the v^2 Jacobian)."""
        if args.unc_mode == "slowness":
            return z["unc_s"]
        if args.unc_mode == "rel":
            return 100.0 * V * z["unc_s"]              # dv/v [%] = v*unc_s
        return (V ** 2) * z["unc_s"]                   # dv [km/s]

    ULAB = {"dv": "posterior velocity uncertainty $\\delta v$ [km/s]",
            "rel": "relative uncertainty $\\delta v/v$ [%]",
            "slowness": "posterior slowness uncertainty [s/km]"}[args.unc_mode]

    # shared velocity + uncertainty scales across periods (robust percentiles)
    vals, uncs = [], []
    for f in files:
        z = np.load(f)
        V = z["vel"]; m = np.isfinite(V)
        vals.append(V[m]); uncs.append(unc_field(z, V)[m])
    allv = np.concatenate(vals)
    allu = np.concatenate([u[np.isfinite(u)] for u in uncs])
    vlo, vhi = np.nanpercentile(allv, [4, 96])
    umax = np.nanpercentile(allu, 92)
    wlab = WAVE_LABEL.get(args.wave, args.wave)

    def panel(ax, field, cmap, vlim, lab):
        ax.imshow(hs, extent=extent, cmap="gray", origin="upper", aspect="auto", zorder=0)
        pc = ax.pcolormesh(lon2d, lat2d, field, cmap=cmap, vmin=vlim[0], vmax=vlim[1],
                           alpha=0.72, shading="nearest", zorder=1)
        plt.colorbar(pc, ax=ax, fraction=0.046, pad=0.04).set_label(lab)
        ax.plot(slon, slat, "^", ms=2.6, mfc="0.15", mec="white", mew=0.25, zorder=5)
        G._plot_extras(ax, geol[0], geol[1], borders, wells, extent, water)
        ax.set(xlim=(extent[0], extent[1]), ylim=(extent[2], extent[3]))
        ax.set_aspect(1.0 / np.cos(np.deg2rad(0.5 * (la0 + la1))))

    thumbs = []
    for f in files:
        z = np.load(f)
        T = float(z["period"]); V = z["vel"]
        show = np.isfinite(V)
        dv = np.where(show, unc_field(z, V), np.nan)
        R = np.where(show, z["res_diag"], np.nan)          # masked like the velocity panel
        N, vr, cov = int(z["N"]), float(z["var_red"]), int(z["coverage"])
        chi2 = float(z["chi2_red"]); rthr = float(z["res_thresh"])
        rhi = np.nanpercentile(R, 98) if np.isfinite(R).any() else 1.0

        fig, axs = plt.subplots(1, 3, figsize=(22, 7.0), sharex=True, sharey=True)
        panel(axs[0], V, "RdYlBu", (vlo, vhi), f"{wlab} $V_g$ [km/s]")
        panel(axs[1], dv, "viridis", (0, umax), ULAB)
        panel(axs[2], R, "plasma", (rthr, rhi), "resolution (res_diag)")
        axs[0].set(xlabel="Longitude", ylabel="Latitude")
        axs[1].set_xlabel("Longitude"); axs[2].set_xlabel("Longitude")
        axs[0].set_title(f"$V_g$   T = {T:.1f} s", fontsize=12, fontweight="bold")
        _uh = {"dv": " — $v^2$-scaled, read resolution panel for constraint",
               "rel": "", "slowness": " — native, Jacobian-free"}[args.unc_mode]
        axs[1].set_title(f"uncertainty ({args.unc_mode}){_uh}\n(N={N} rays, var_red={vr:.2f})",
                         fontsize=10)
        axs[2].set_title(f"resolution   (shown ≥ {rthr:.3f}, {cov} cells, χ²={chi2:.1f})",
                         fontsize=11)
        fig.suptitle(f"{net.capitalize()} — {wlab} group velocity, unified picks — "
                     f"{dx_km*1000:.0f} m grid,  T = {T:.1f} s", fontsize=13, y=0.98)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out = os.path.join(outdir, f"{args.wave}_T{T:.1f}.png")
        fig.savefig(out, dpi=140); plt.close(fig)
        thumbs.append((T, V))
        print("wrote", out)

    if not args.no_contact_sheet and thumbs:
        n = len(thumbs); nc = min(6, n); nr = int(np.ceil(n / nc))
        fig, axs = plt.subplots(nr, nc, figsize=(3.1 * nc, 3.0 * nr), squeeze=False)
        for ax, (T, V) in zip(axs.ravel(), thumbs):
            ax.imshow(hs, extent=extent, cmap="gray", origin="upper", aspect="auto", zorder=0)
            ax.pcolormesh(lon2d, lat2d, V, cmap="RdYlBu", vmin=vlo, vmax=vhi, alpha=0.75,
                          shading="nearest", zorder=1)
            if borders is not None:
                borders.plot(ax=ax, color="k", lw=0.8, linestyle="--", zorder=6)
            ax.set(xlim=(extent[0], extent[1]), ylim=(extent[2], extent[3]), xticks=[], yticks=[])
            ax.set_aspect(1.0 / np.cos(np.deg2rad(0.5 * (la0 + la1))))
            ax.set_title(f"T={T:.1f}s", fontsize=9)
        for ax in axs.ravel()[n:]:
            ax.axis("off")
        fig.suptitle(f"{net.capitalize()} {wlab} $V_g$ ({dx_km*1000:.0f} m grid), "
                     f"shared scale {vlo:.2f}-{vhi:.2f} km/s", y=1.0, fontsize=13)
        fig.tight_layout()
        sheet = os.path.join(outdir, f"{args.wave}_contact_sheet.png")
        fig.savefig(sheet, dpi=120); plt.close(fig)
        print("wrote", sheet)
    print(f"\n{len(files)} periods -> {outdir}")


if __name__ == "__main__":
    main()
