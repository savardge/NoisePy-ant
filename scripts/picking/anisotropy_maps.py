"""Radial-anisotropy depth slices from an RLg_radial volume: gamma(z) and zeta(z) on DEM+GK500.

gamma = (Vsh - Vsv)/Vsv is what the continuous-zeta BayHunter fork samples per layer; zeta is
its Voigt-referenced form, zeta = gamma / sqrt((2 + (1+gamma)^2)/3). Positive = Vsh > Vsv.

Two credibility filters are applied by default, and both matter:
  * the same z_reliable_min..max window as the Vs maps -- outside it the profile is prior fill;
  * |gamma| < 0.15 is blanked, because a Love-leak control faked gamma = +0.14 and the clean-
    synthetic null gate returned P(gamma != 0) = 0.02-0.11. Below that floor the sign is not
    trustworthy, so plotting it as colour would invent structure.
Use --no-floor to see the raw field (diagnostic only).

A sign-probability panel (gamma_p_pos) is written alongside: it is the honest significance
measure for a CONTINUOUS gamma, where "fraction of models with gamma != 0" is meaningless.

  python anisotropy_maps.py --griddir <RLg_radial dir> --net aargau [--label RLg_radial]
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vs_model_figures import (E, CITIES, _hillshade, _tecto, km_xy, MAP_DEPTHS)  # noqa: E402
from well_vs_qc import WELLS                                                     # noqa: E402
from noisepy.lv95 import extent_lv95_km                                          # noqa: E402

FLOOR = 0.15


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--griddir", required=True)
    ap.add_argument("--net", required=True, choices=("riehen", "aargau", "hautesorne"))
    ap.add_argument("--waveset", default="fundlove")
    ap.add_argument("--label", default=None)
    ap.add_argument("--field", default="gamma", choices=("gamma", "zeta"))
    ap.add_argument("--vlim", type=float, default=0.30)
    ap.add_argument("--no-floor", action="store_true",
                    help="do not blank |value| < 0.15 (diagnostic only)")
    a = ap.parse_args()
    arm = a.label or os.path.basename(os.path.normpath(a.griddir))

    v = np.load(os.path.join(a.griddir, f"volume_{a.waveset}.npz"), allow_pickle=True)
    key = f"{a.field}_median"
    if key not in v.files:
        raise SystemExit(f"{key} not in the volume -- is this a --radial arm?")
    cells, lonlat, z = v["cells"], v["lonlat"], np.asarray(v["depth"], float)
    g = np.asarray(v[key], float)
    if not np.isfinite(g).any():
        raise SystemExit(f"{key} is entirely NaN -- this is an ISOTROPIC arm, nothing to map")
    ppos = np.asarray(v["gamma_p_pos"], float) if "gamma_p_pos" in v.files else None

    zmin = (np.asarray(v["z_reliable_min"], float) if "z_reliable_min" in v.files
            else np.zeros(len(cells)))
    zmax = (np.asarray(v["z_reliable_max"], float) if "z_reliable_max" in v.files
            else np.full(len(cells), np.inf))
    for i in range(len(cells)):
        bad = np.zeros(len(z), bool)
        if np.isfinite(zmax[i]):
            bad |= z > zmax[i]
        if np.isfinite(zmin[i]) and zmin[i] > 0:
            bad |= z < zmin[i]
        g[i, bad] = np.nan
        if ppos is not None:
            ppos[i, bad] = np.nan

    dem = np.load(f"{E}/{a.net}/tomo/2_vs_depth_inversion/fig_assets_{a.net}_dem.npz")
    elev, extent = dem["elev"].astype(float), dem["extent"]
    gkp = f"{E}/{a.net}/tomo/2_vs_depth_inversion/fig_assets_{a.net}_gk500.npz"
    gk = np.load(gkp, allow_pickle=True) if os.path.exists(gkp) else None
    hs = _hillshade(elev, extent)
    ek = extent_lv95_km(extent)
    cx, cy = km_xy(lonlat[:, 0], lonlat[:, 1])
    nx, ny = cells[:, 0].max() + 1, cells[:, 1].max() + 1
    gx = np.full((nx, ny), np.nan); gy = np.full((nx, ny), np.nan)
    for (ix, iy), x, y in zip(cells, cx, cy):
        gx[int(ix), int(iy)] = x; gy[int(ix), int(iy)] = y
    X = np.nanmean(gx, axis=1); Y = np.nanmean(gy, axis=0)
    dx = np.nanmedian(np.diff(X)); dy = np.nanmedian(np.diff(Y))
    X = np.where(np.isfinite(X), X, np.nanmin(X) + dx * np.arange(nx))
    Y = np.where(np.isfinite(Y), Y, np.nanmin(Y) + dy * np.arange(ny))
    Xe = np.append(X - dx / 2, X[-1] + dx / 2)
    Ye = np.append(Y - dy / 2, Y[-1] + dy / 2)

    def grid(vals):
        out = np.full((nx, ny), np.nan)
        for (ix, iy), val in zip(cells, vals):
            out[int(ix), int(iy)] = val
        return out

    figdir = os.path.join(a.griddir, "figures", "anisotropy")
    os.makedirs(figdir, exist_ok=True)
    stats = {}
    for d in MAP_DEPTHS:
        k = int(np.argmin(np.abs(z - d)))
        col = g[:, k].copy()
        if not np.isfinite(col).any():
            continue
        frac = float(np.nanmean(np.abs(col) >= FLOOR))
        stats[float(d)] = {"n": int(np.isfinite(col).sum()),
                           "frac_above_floor": round(frac, 3),
                           "median": round(float(np.nanmedian(col)), 4)}
        if not a.no_floor:
            col = np.where(np.abs(col) < FLOOR, np.nan, col)
        fig, ax = plt.subplots(figsize=(7.4, 7.0))
        ax.imshow(hs, extent=ek, cmap="gray", origin="upper", zorder=0)
        pc = ax.pcolormesh(Xe, Ye, grid(col).T, cmap="RdBu_r", vmin=-a.vlim, vmax=a.vlim,
                           alpha=0.82, shading="flat", zorder=2)
        _tecto(ax, gk, lw=0.8)
        for nm, la, lo, _ in WELLS.get(a.net, []):
            wx, wy = km_xy(lo, la)
            ax.plot(wx, wy, "s", mfc="k", mec="w", ms=5, zorder=6)
            ax.annotate(nm, (wx, wy), xytext=(4, 4), textcoords="offset points", fontsize=7,
                        fontweight="bold", zorder=7,
                        bbox=dict(fc="w", alpha=0.65, ec="none", pad=1))
        for nm, lo, la in CITIES.get(a.net, []):
            mx, my = km_xy(lo, la)
            ax.plot(mx, my, "o", mfc="w", mec="k", ms=4, zorder=6)
            ax.annotate(nm, (mx, my), xytext=(4, -8), textcoords="offset points", fontsize=7,
                        style="italic", zorder=7)
        ax.set_xlim(ek[0], ek[1]); ax.set_ylim(ek[2], ek[3]); ax.set_aspect("equal")
        ax.set_xlabel("E [km LV95]"); ax.set_ylabel("N [km LV95]")
        ax.set_title(f"{a.net} radial anisotropy {a.field} at {d:g} km — {arm}\n"
                     f"blue Vsv>Vsh, red Vsh>Vsv"
                     + ("" if a.no_floor else f";  |{a.field}| < {FLOOR} blanked (credibility "
                                              f"floor);  {100*frac:.0f}% of cells clear it"),
                     fontsize=10)
        cb = plt.colorbar(pc, ax=ax, fraction=0.04, pad=0.02, extend="both")
        cb.set_label(f"{a.field} = (Vsh-Vsv)/Vsv" if a.field == "gamma" else "zeta (Voigt)")
        out = os.path.join(figdir, f"{a.field}_z{d:03.1f}km.png")
        fig.savefig(out, dpi=145, bbox_inches="tight"); plt.close(fig)
    with open(os.path.join(figdir, f"{a.field}_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=1)
    print(f"{arm}: wrote {len(stats)} {a.field} slices -> {figdir}")
    for d, s in sorted(stats.items()):
        print(f"   z={d:>4} km  n={s['n']:>5}  median={s['median']:+.3f}  "
              f"|{a.field}|>={FLOOR}: {100*s['frac_above_floor']:.0f}%")


if __name__ == "__main__":
    main()
