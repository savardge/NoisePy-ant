"""Map the posterior Vp/Vs from a free-Vp/Vs arm, and test whether it tracks geology.

Prediction (user, 2026-08-18): the fitted ratio should be HIGH in the sedimentary basins
(Delemont, Ajoie) and LOW where bedrock is shallow (the Jura ridges). If instead it is
spatially random, the ratio is absorbing per-cell noise rather than resolving anything.

Reads the per-cell npz directly (the volume does not carry vpvs_post). Writes a map on
DEM + GK500 and prints Spearman(vpvs, elevation) as the basin/ridge test -- expected NEGATIVE
(low ground = basin = high ratio).

  python vpvs_map.py --cells <free arm cells dir> --net hautesorne --out <dir>
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vs_model_figures import E, CITIES, _hillshade, _tecto, km_xy, bilinear   # noqa: E402
from well_vs_qc import WELLS                                                  # noqa: E402
from noisepy.lv95 import extent_lv95_km                                       # noqa: E402


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30 or np.unique(a[m]).size < 2 or np.unique(b[m]).size < 2:
        return np.nan, int(m.sum())
    ra = np.argsort(np.argsort(a[m])).astype(float); rb = np.argsort(np.argsort(b[m])).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum())), int(m.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cells", required=True)
    ap.add_argument("--net", required=True, choices=("riehen", "aargau", "hautesorne"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="free Vp/Vs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # read the compact per-cell extract (vpvs_extract.py, run where the cells live), or fall
    # back to scanning a raw cells/ tree
    if a.cells.endswith(".npz"):
        zz = np.load(a.cells, allow_pickle=True)
        cells, lonlat = zz["cells"], zz["lonlat"]
        med = np.asarray(zz["vpvs_med"], float)
        width = np.asarray(zz["vpvs_p84"], float) - np.asarray(zz["vpvs_p16"], float)
        cd = np.asarray(zz["chain_disagree"], float)
    else:
        cells, lonlat, med, width, cd = [], [], [], [], []
        for f in sorted(glob.glob(os.path.join(a.cells, "cell_*.npz"))):
            try:
                z = np.load(f, allow_pickle=True)
                vp = np.asarray(z["vpvs_post"], float).ravel()
            except Exception:
                continue
            if vp.size < 10 or vp.max() - vp.min() < 1e-6:
                continue                                    # a FIXED-ratio cell, not this arm
            cells.append([int(v) for v in z["cell_ixiy"]])
            lonlat.append([float(v) for v in z["cell_lonlat"]])
            med.append(np.median(vp)); width.append(np.percentile(vp, 84) - np.percentile(vp, 16))
            cd.append(float(z["chain_disagree"]))
        cells = np.array(cells); lonlat = np.array(lonlat)
        med = np.array(med); width = np.array(width); cd = np.array(cd)
    n = len(med)
    print(f"{a.net} {a.label}: {n} cells with a sampled Vp/Vs")
    if n < 20:
        return

    dem = np.load(f"{E}/{a.net}/tomo/2_vs_depth_inversion/fig_assets_{a.net}_dem.npz")
    elev, extent = dem["elev"].astype(float), dem["extent"]
    gkp = f"{E}/{a.net}/tomo/2_vs_depth_inversion/fig_assets_{a.net}_gk500.npz"
    gk = np.load(gkp, allow_pickle=True) if os.path.exists(gkp) else None
    ev = bilinear(elev, extent, lonlat[:, 0], lonlat[:, 1])

    # ---- the geological test --------------------------------------------------------------
    r_el, m = spearman(med, ev)
    r_w, _ = spearman(width, ev)
    print(f"  Spearman(Vp/Vs median, elevation) = {r_el:+.3f}  (n={m})   "
          f"<- prediction: NEGATIVE (basins low & high ratio; ridges high & low ratio)")
    print(f"  Spearman(Vp/Vs 16-84 width, elevation) = {r_w:+.3f}")
    # tercile split by elevation
    q1, q3 = np.nanpercentile(ev, [33, 67])
    for lab, sel in (("lowest third (basins)", ev <= q1), ("middle third", (ev > q1) & (ev < q3)),
                     ("highest third (ridges)", ev >= q3)):
        print(f"    {lab:<24} elev {np.nanmedian(ev[sel]):5.0f} m   Vp/Vs median {np.median(med[sel]):.3f}"
              f"   16-84 width {np.median(width[sel]):.3f}   n={sel.sum()}")

    # ---- map -----------------------------------------------------------------------------
    hs = _hillshade(elev, extent); ek = extent_lv95_km(extent)
    cx, cy = km_xy(lonlat[:, 0], lonlat[:, 1])
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, val, ttl, cmap, vmin, vmax, cbl in (
            (axes[0], med, "posterior median Vp/Vs", "viridis", 1.8, 3.2, "Vp/Vs"),
            (axes[1], width, "posterior 16-84% width of Vp/Vs", "magma", 0.3, 1.5, "width")):
        ax.imshow(hs, extent=ek, cmap="gray", origin="upper", zorder=0)
        sc = ax.scatter(cx, cy, c=val, s=22, marker="s", cmap=cmap, vmin=vmin, vmax=vmax,
                        alpha=0.9, linewidths=0, zorder=2)
        _tecto(ax, gk, lw=0.8)
        for nm, la, lo_, _ in WELLS.get(a.net, []):
            wx, wy = km_xy(lo_, la)
            ax.plot(wx, wy, "s", mfc="k", mec="w", ms=6, zorder=6)
            ax.annotate(nm, (wx, wy), xytext=(4, 4), textcoords="offset points", fontsize=8,
                        fontweight="bold", zorder=7, bbox=dict(fc="w", alpha=0.65, ec="none", pad=1))
        for nm, lo_, la in CITIES.get(a.net, []):
            mx, my = km_xy(lo_, la)
            ax.plot(mx, my, "o", mfc="w", mec="k", ms=4, zorder=6)
            ax.annotate(nm, (mx, my), xytext=(4, -9), textcoords="offset points", fontsize=8,
                        style="italic", zorder=7)
        ax.set_xlim(ek[0], ek[1]); ax.set_ylim(ek[2], ek[3]); ax.set_aspect("equal")
        ax.set_xlabel("E [km LV95]"); ax.set_ylabel("N [km LV95]")
        ax.set_title(f"{a.net} — {ttl}  ({n} cells)", fontsize=11)
        plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02).set_label(cbl)
    fig.suptitle(f"{a.label}: Spearman(Vp/Vs, elevation) = {r_el:+.2f}   "
                 f"(prediction: negative = basins high, ridges low)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(a.out, "vpvs_map.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
