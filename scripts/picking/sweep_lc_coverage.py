#!/usr/bin/env python3
"""What LC does the ray coverage actually demand, per period?

Calibration for `run_production.py --lc-mode coverage`, which picks the smallest LC that
still bridges the gaps between adequately-sampled cells. Two knobs decide everything and
neither has a derived value, so sweep them and read LC(T) off the data:

  --minrays  rays through a cell before it counts as CONSTRAINED. A cell clipped by one ray
             is not; at dx 0.5 km nearly every interior cell is clipped, so a low threshold
             saturates and returns the floor at every period, measuring nothing.
  --q        percentile of the interior distance-to-nearest-constrained-cell taken as LC.
             q=90 accepts 10% of interior cells sitting further than one correlation length
             from adequate coverage; higher q is the conservative choice against exactly the
             prior-dominated patterns this is meant to prevent.

Builds the kernel per period (no inversion), thresholds G_sum -- the ray-density map, NOT
`mask`, which build_G already binarises at min_density -- then distance-transforms from the
constrained cells over the FILLED coverage footprint.

Writes a CSV and one LC(T) figure per wave with a curve per (minrays, q).

Usage:
  python sweep_lc_coverage.py --config <tomo yaml> --wave fund \
      [--minrays 20 100 300 1000] [--q 90 98] [--out DIR]
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

from swtomotv.config import DatasetConfig, MethodConfig
from swtomotv.geometry import make_grid
from swtomotv.data.picks import build_cache, load_cache, available_periods
from swtomotv.kernel import build_G


def _hull_mask(binary):
    """Convex hull of the True cells, as a boolean mask of the same shape.

    The imaging footprint. Using fill_holes(hit) instead makes the measurement
    self-referential -- the domain becomes the constrained set, so the distance from a
    domain cell to a constrained cell is ~0 by construction and every threshold returns
    the floor. The hull is the area the map CLAIMS to cover, which is what an unconstrained
    cell has to be measured against.
    """
    idx = np.argwhere(binary)
    if len(idx) < 3:
        return binary.copy()
    try:
        from scipy.spatial import ConvexHull, Delaunay
        hull = Delaunay(idx[ConvexHull(idx).vertices])
    except Exception:
        return binary.copy()
    ii, jj = np.meshgrid(np.arange(binary.shape[0]), np.arange(binary.shape[1]),
                         indexing="ij")
    pts = np.column_stack([ii.ravel(), jj.ravel()])
    return (hull.find_simplex(pts) >= 0).reshape(binary.shape)


def lc_from_coverage(gsum, dx, frac, q):
    """(LC_km, pct_interior_constrained, gap_p50) for one period."""
    g = np.where(np.isfinite(np.asarray(gsum, float)), np.asarray(gsum, float), 0.0)
    nz = g[g > 0]
    ref = float(np.median(nz)) if nz.size else 0.0
    hit = g >= frac * ref          # relative to this period's own median density
    if not hit.any():
        return np.nan, 0.0, np.nan
    domain = _hull_mask(g > 0)          # everything the map claims, not just the covered part
    if not domain.any():
        return np.nan, 0.0, np.nan
    d_km = ndimage.distance_transform_edt(~hit)[domain] * float(dx)
    if not d_km.size:
        return np.nan, 0.0, np.nan
    return (float(np.percentile(d_km, q)), 100.0 * float(hit[domain].mean()),
            float(np.percentile(d_km, 50)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--wave", default="fund")
    ap.add_argument("--frac", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--q", type=float, nargs="+", default=[90, 98])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ds = DatasetConfig.from_yaml(a.config)
    method = MethodConfig()
    grid = make_grid(ds.bounds, ds.dx_km)
    build_cache(ds, grid, a.wave, force=False)
    periods = available_periods(ds, a.wave)

    rows = []
    for T in periods:
        z = load_cache(ds, a.wave, T)
        if len(z["tau"]) < 40:
            continue
        _, _, G_sum = build_G(ds, method, grid, a.wave, T, use_cache=True)
        for mr in a.frac:
            for q in a.q:
                lc, pct, p50 = lc_from_coverage(G_sum, ds.dx_km, mr, q)
                rows.append(dict(period=round(float(T), 3), n_rays=int(len(z["tau"])),
                                 frac=mr, q=q, pct_constrained=round(pct, 2),
                                 gap_p50_km=round(p50, 3) if np.isfinite(p50) else np.nan,
                                 lc_km=round(lc, 3) if np.isfinite(lc) else np.nan))
        print("  T=%-7g rays=%-6d %s" % (T, len(z["tau"]),
              "  ".join("f%g/q%g:%s" % (r["frac"], r["q"],
                        ("%.2f" % r["lc_km"]) if np.isfinite(r["lc_km"]) else "n/a")
                        for r in rows[-len(a.frac) * len(a.q):])), flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    out = a.out or os.path.dirname(os.path.abspath(a.config))
    os.makedirs(out, exist_ok=True)
    stem = os.path.join(out, "lccov_sweep_%s_%s" % (ds.name, a.wave))
    df.to_csv(stem + ".csv", index=False)

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for mr in a.frac:
        for q in a.q:
            g = df[(df.frac == mr) & (df.q == q)].sort_values("period")
            axs[0].plot(g.period, g.lc_km, "o-", ms=3, lw=1.2,
                        label="frac=%g, q=%g" % (mr, q))
            axs[1].plot(g.period, g.pct_constrained, "o-", ms=3, lw=1.2,
                        label="frac=%g" % mr)
    axs[0].set_xlabel("period [s]"); axs[0].set_ylabel("coverage-required LC [km]")
    axs[0].set_title("%s %s: LC demanded by the ray coverage" % (ds.name, a.wave), fontsize=10)
    axs[1].set_xlabel("period [s]"); axs[1].set_ylabel("% of interior cells constrained")
    axs[1].set_title("how much of the interior clears the ray threshold", fontsize=10)
    for ax in axs:
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle("%s %s (dx %.2f km): calibration for --lc-mode coverage"
                 % (ds.name, a.wave, ds.dx_km), y=1.0)
    fig.tight_layout(); fig.savefig(stem + ".png", dpi=130); plt.close(fig)
    print("wrote %s.csv and %s.png" % (os.path.basename(stem), os.path.basename(stem)))


if __name__ == "__main__":
    main()
