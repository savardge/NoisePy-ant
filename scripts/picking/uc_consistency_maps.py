"""Map where group and phase disagree: U/c per cell, per period, on DEM + GK500.

Normal dispersion with dc/dT > 0 forces U < c, so U/c > 1 is kinematically impossible and marks
a cell/period where the group and phase measurements contradict each other. Mapping it
separates the two failure modes, which need opposite fixes:

  PERIOD-localised  (all cells bad below some T)  -> trim the band
  SPATIALLY-localised (a subset of cells bad across a broad band) -> cull those cells/paths

Two figures per (net, wave):
  uc_panels_<net>_<wave>.png   grid of maps, one per period, U/c on a diverging scale at 1.0
  uc_summary_<net>_<wave>.png  per-cell fraction of periods violating, over the same basemap

  python uc_consistency_maps.py --indir <uc_maps dir> --outdir <figdir> [--net aargau]
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vs_model_figures import _hillshade, _tecto, km_xy, E, CITIES   # noqa: E402
from well_vs_qc import WELLS                                        # noqa: E402
from noisepy.lv95 import extent_lv95_km                             # noqa: E402


def basemap(ax, net, dem, gk, extent_km, hs, wells=True, labels=True):
    ax.imshow(hs, extent=extent_km, cmap="gray", origin="upper", zorder=0)
    _tecto(ax, gk, lw=0.7)
    if wells:
        for nm, la, lo, _ in WELLS.get(net, []):
            wx, wy = km_xy(lo, la)
            ax.plot(wx, wy, "s", mfc="k", mec="w", ms=4, zorder=6)
            if labels:
                ax.annotate(nm, (wx, wy), xytext=(3, 3), textcoords="offset points",
                            fontsize=6, fontweight="bold", zorder=7,
                            bbox=dict(fc="w", alpha=0.6, ec="none", pad=0.8))
    ax.set_xlim(extent_km[0], extent_km[1])
    ax.set_ylim(extent_km[2], extent_km[3])
    ax.set_aspect("equal")


def load_assets(net):
    dem = np.load(f"{E}/{net}/tomo/2_vs_depth_inversion/fig_assets_{net}_dem.npz")
    elev, extent = dem["elev"].astype(float), dem["extent"]
    gk_path = f"{E}/{net}/tomo/2_vs_depth_inversion/fig_assets_{net}_gk500.npz"
    gk = np.load(gk_path, allow_pickle=True) if os.path.exists(gk_path) else None
    return elev, extent, gk, _hillshade(elev, extent), extent_lv95_km(extent)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--indir", required=True, help="dir of uc_<net>_<wave>.npz")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--net", default=None, help="restrict to one network")
    ap.add_argument("--tmax", type=float, default=3.6,
                    help="only map periods up to this (violations vanish above it)")
    ap.add_argument("--max-panels", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    for f in sorted(glob.glob(os.path.join(a.indir, "uc_*.npz"))):
        stem = os.path.basename(f)[3:-4]              # <net>_<wave>
        net, wave = stem.split("_", 1)
        if a.net and net != a.net:
            continue
        z = np.load(f, allow_pickle=True)
        cells, lonlat = z["cells"], z["lonlat"]
        T, ratio = np.asarray(z["T"], float), np.asarray(z["ratio"], float)
        elev, extent, gk, hs, extent_km = load_assets(net)
        cx, cy = km_xy(lonlat[:, 0], lonlat[:, 1])
        nx, ny = cells[:, 0].max() + 1, cells[:, 1].max() + 1

        def grid(vals):
            g = np.full((nx, ny), np.nan)
            for (ix, iy), v in zip(cells, vals):
                g[int(ix), int(iy)] = v
            return g

        # cell mesh in LV95 km, from the affine node grid
        gx = np.full((nx, ny), np.nan); gy = np.full((nx, ny), np.nan)
        for (ix, iy), x, y in zip(cells, cx, cy):
            gx[int(ix), int(iy)] = x; gy[int(ix), int(iy)] = y
        dx = np.nanmedian(np.diff(np.nanmean(gx, axis=1)))
        dy = np.nanmedian(np.diff(np.nanmean(gy, axis=0)))
        X = np.nanmean(gx, axis=1); Y = np.nanmean(gy, axis=0)
        X = np.where(np.isfinite(X), X, np.nanmin(X) + dx * np.arange(nx))
        Y = np.where(np.isfinite(Y), Y, np.nanmin(Y) + dy * np.arange(ny))
        Xe = np.append(X - dx / 2, X[-1] + dx / 2)
        Ye = np.append(Y - dy / 2, Y[-1] + dy / 2)

        # ---------- panel grid, one map per period ----------
        sel = np.where(T <= a.tmax)[0]
        if len(sel) > a.max_panels:
            sel = sel[np.linspace(0, len(sel) - 1, a.max_panels).astype(int)]
        ncol = 4
        nrow = int(np.ceil(len(sel) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.3 * nrow),
                                 squeeze=False)
        norm = TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=1.5)
        pc = None
        for k, ax in zip(sel, axes.ravel()):
            basemap(ax, net, elev, gk, extent_km, hs, wells=True, labels=False)
            pc = ax.pcolormesh(Xe, Ye, grid(ratio[:, k]).T, cmap="RdBu_r", norm=norm,
                               alpha=0.8, shading="flat", zorder=2)
            frac = float(np.nanmean(ratio[:, k] > 1))
            ax.set_title(f"T = {T[k]:.2f} s   U>c in {100*frac:.0f}% of cells", fontsize=8.5)
            ax.set_xticks([]); ax.set_yticks([])
        for ax in axes.ravel()[len(sel):]:
            ax.axis("off")
        if pc is not None:
            cb = fig.colorbar(pc, ax=axes, fraction=0.02, pad=0.01,
                              extend="both", ticks=[0.5, 0.75, 1.0, 1.25, 1.5])
            cb.set_label("U / c   (>1 is kinematically impossible)")
        fig.suptitle(f"{net} — {wave} fundamental: group/phase consistency by period\n"
                     f"red = U>c (group and phase contradict); blue = physical",
                     fontsize=11)
        out = os.path.join(a.outdir, f"uc_panels_{net}_{wave}.png")
        fig.savefig(out, dpi=135, bbox_inches="tight"); plt.close(fig)
        print(f"wrote {out}  ({len(sel)} periods)")

        # ---------- summary: per-cell fraction of periods violating ----------
        band = T <= a.tmax
        frac_cell = np.nanmean(ratio[:, band] > 1, axis=1)
        fig, ax = plt.subplots(figsize=(7.4, 7.0))
        basemap(ax, net, elev, gk, extent_km, hs)
        pc = ax.pcolormesh(Xe, Ye, grid(frac_cell).T, cmap="inferno_r", vmin=0, vmax=1,
                           alpha=0.82, shading="flat", zorder=2)
        for nm, lo, la in CITIES.get(net, []):
            mx, my = km_xy(lo, la)
            ax.plot(mx, my, "o", mfc="w", mec="k", ms=4, zorder=6)
            ax.annotate(nm, (mx, my), xytext=(4, -8), textcoords="offset points",
                        fontsize=7, style="italic", zorder=7)
        ax.set_xlabel("E [km LV95]"); ax.set_ylabel("N [km LV95]")
        ax.set_title(f"{net} — {wave} fundamental\n"
                     f"fraction of periods (T <= {a.tmax:g} s) with U > c", fontsize=11)
        plt.colorbar(pc, ax=ax, fraction=0.04, pad=0.02).set_label("fraction of periods U>c")
        out = os.path.join(a.outdir, f"uc_summary_{net}_{wave}.png")
        fig.savefig(out, dpi=145, bbox_inches="tight"); plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
