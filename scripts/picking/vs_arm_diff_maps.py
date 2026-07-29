#!/usr/bin/env python
"""Per-depth difference maps between two production arms: dVs = arm B - arm A (median).

One figure per depth (0.5..6.0 km step 0.5), symmetric colormap over the hillshade, with the
cells whose INPUTS changed (recomputed in B) outlined -- differences outside those outlines are
exactly zero by construction (byte-copied cells), which is itself the visual control.

  PYTHONPATH=~/Codes/NoisePy-ant .../bayhunter/bin/python vs_arm_diff_maps.py \
      --a <reference griddir> --b <comparison griddir> --net riehen --label otexempt-minus-ref
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

E = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vs_model_figures import MAP_DEPTHS, _hillshade, _tecto, load_zrel, km_xy  # noqa: E402
from noisepy.lv95 import extent_lv95_km                                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="reference griddir")
    ap.add_argument("--b", required=True, help="comparison griddir (figures go here)")
    ap.add_argument("--net", required=True, choices=("riehen", "aargau"))
    ap.add_argument("--label", default="B-A")
    args = ap.parse_args()

    va = np.load(os.path.join(args.a, "volume_fundotlove.npz"), allow_pickle=True)
    vb = np.load(os.path.join(args.b, "volume_fundotlove.npz"), allow_pickle=True)
    z = np.asarray(va["depth"], float)
    A = {tuple(c): i for i, c in enumerate(va["cells"])}
    B = {tuple(c): i for i, c in enumerate(vb["cells"])}
    common = sorted(set(A) & set(B))
    cells = np.array(common)
    lonlat = np.array([va["lonlat"][A[c]] for c in common])
    d3 = np.array([vb["vs_median"][B[c]] - va["vs_median"][A[c]] for c in common])
    # mask below the data reach of EITHER arm: a difference between two prior-fills is noise
    za = load_zrel(args.a, cells)
    zb = load_zrel(args.b, cells)
    zmin = np.fmin(za, zb)
    for i in range(len(cells)):
        if np.isfinite(zmin[i]):
            d3[i, z > zmin[i] + 1e-9] = np.nan
    changed = np.array([not np.allclose(np.nan_to_num(vb["vs_median"][B[c]]),
                                        np.nan_to_num(va["vs_median"][A[c]])) for c in common])
    print(f"{args.net}: {len(common)} common cells, {changed.sum()} differ "
          f"(recomputed inputs); max |dVs| = {np.nanmax(np.abs(d3)):.3f} km/s")

    dem = np.load(f"{E}/{args.net}/tomo/2_vs_depth_inversion/fig_assets_{args.net}_dem.npz")
    hs = _hillshade(dem["elev"].astype(float), dem["extent"])
    gk = np.load(f"{E}/{args.net}/tomo/2_vs_depth_inversion/fig_assets_{args.net}_gk500.npz")
    extent = dem["extent"]

    ix, iy = cells[:, 0].astype(float), cells[:, 1].astype(float)
    M = np.column_stack([np.ones_like(ix), ix, iy])
    clon = np.linalg.lstsq(M, lonlat[:, 0], rcond=None)[0]
    clat = np.linalg.lstsq(M, lonlat[:, 1], rcond=None)[0]
    nx, ny = int(ix.max()) + 1, int(iy.max()) + 1
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    Bm = np.column_stack([np.ones(gx.size), gx.ravel(), gy.ravel()])
    lon2d, lat2d = (Bm @ clon).reshape(nx, ny), (Bm @ clat).reshape(nx, ny)
    e2d, n2d = km_xy(lon2d, lat2d)                   # LV95 km (regla de mapas 2026-07-25)
    extent_km = extent_lv95_km(extent)
    chx, chy = km_xy(lonlat[changed, 0], lonlat[changed, 1])

    outdir = os.path.join(args.b, "figures", "diff_maps")
    os.makedirs(outdir, exist_ok=True)
    vmax = max(0.05, float(np.nanpercentile(np.abs(d3[changed]), 98))) if changed.any() else 0.05
    for dpth in MAP_DEPTHS:
        k = int(np.argmin(np.abs(z - dpth)))
        g = np.full((nx, ny), np.nan)
        for (cx, cy), val in zip(cells, d3[:, k]):
            g[int(cx), int(cy)] = val
        fig, ax = plt.subplots(figsize=(7.6, 7.2))
        ax.imshow(hs, extent=extent_km, cmap="gray", origin="upper", zorder=0)
        pc = ax.pcolormesh(e2d, n2d, g, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           alpha=0.75, shading="nearest", zorder=1)
        _tecto(ax, gk, lw=0.7)
        ax.plot(chx, chy, "s", mfc="none", mec="k", ms=3.2, mew=0.5, zorder=4)
        ax.set_xlim(extent_km[0], extent_km[1]); ax.set_ylim(extent_km[2], extent_km[3])
        ax.set_aspect("equal")
        ax.set_xlabel("E [km LV95]"); ax.set_ylabel("N [km LV95]")
        ax.set_title(f"{args.net} dVs ({args.label}) at {dpth:g} km depth\n"
                     f"outlined = recomputed cells; elsewhere identical by construction",
                     fontsize=10)
        plt.colorbar(pc, ax=ax, fraction=0.04, pad=0.02).set_label("dVs [km/s]")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"dvs_map_z{dpth:03.1f}km.png"), dpi=150)
        plt.close(fig)
    print(f"diff maps: {len(MAP_DEPTHS)} -> {outdir}")


if __name__ == "__main__":
    main()
