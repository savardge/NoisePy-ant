"""Does the U>c inconsistency track OVERTONE excitation (and geology)?

Hypothesis (user, 2026-08-16): where higher modes are strongly excited -- valley basins,
anticline ridges, fault zones -- overtone energy leaks into the fundamental FTAN pick. Overtones
are faster, so the fundamental GROUP velocity is biased high, which is exactly what U/c > 1
reports. If true, the violation should track overtone strength spatially rather than fall in a
period band.

Per cell we compare the violation fraction against:
  ot_cov    fraction of overtone periods actually measured there (mask=True)  -- excitation proxy
  ot_res    mean overtone resolution diagonal                                 -- ray density
  ot_ratio  mean overtone/fundamental velocity ratio                          -- mode separation
  fund_res  the same for the fundamental                                      -- coverage CONTROL
  elev, relief  DEM elevation and local relief (valley vs ridge)
  d_fault   distance to the nearest GK500 fault/thrust (where the asset exists)

fund_res is the control that matters: overtone coverage correlates with general data quality, so
a correlation with ot_cov that is matched by fund_res says "well-sampled cells", not "overtone".

  python uc_overtone_correlation.py --indir <uc_maps> [--net riehen]
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vs_model_figures import E, km_xy, bilinear                      # noqa: E402

PROD = {"riehen": "tspws_group_blanket_dx0.2_prod3_k3",
        "aargau": "tspws_group_blanket_dx0.5_prod3_k3",
        "hautesorne": "tspws_group_blanket_dx0.5_prod3_k3"}
# band where each (net, wave) actually violates -- outside it there is nothing to correlate
BAND = {("aargau", "fund"): (0.0, 1.05), ("aargau", "love"): (0.0, 1.5),
        ("riehen", "fund"): (0.0, 99.), ("riehen", "love"): (1.4, 3.0),
        ("hautesorne", "fund"): (0.0, 1.05), ("hautesorne", "love"): (0.9, 2.6)}


def cell_wave_stats(net, wave, cells):
    """Per-cell (coverage fraction, mean res_diag, mean velocity) for one wave's maps."""
    d = f"{E}/{net}/tomo/1_velocity_maps/1_production/{PROD[net]}/production/{wave}"
    fs = sorted(glob.glob(os.path.join(d, "map_T*.npz")))
    if not fs:
        return None
    nsee = np.zeros(len(cells)); nres = np.zeros(len(cells)); nvel = np.zeros(len(cells))
    ntot = 0
    for f in fs:
        z = np.load(f)
        m, r, v = z["mask"], z["res_diag"], z["vel"]
        ntot += 1
        for i, (ix, iy) in enumerate(cells):
            ix, iy = int(ix), int(iy)
            if ix < m.shape[0] and iy < m.shape[1] and bool(m[ix, iy]):
                nsee[i] += 1
                nres[i] += float(r[ix, iy])
                nvel[i] += float(v[ix, iy])
    cov = nsee / max(ntot, 1)
    res = np.where(nsee > 0, nres / np.maximum(nsee, 1), np.nan)
    vel = np.where(nsee > 0, nvel / np.maximum(nsee, 1), np.nan)
    return cov, res, vel


def spearman(a, b):
    """Rank correlation, NaN-safe (avoids assuming linearity)."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 30:
        return np.nan, int(ok.sum())
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return (float((ra * rb).sum() / den) if den > 0 else np.nan), int(ok.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--indir", required=True)
    ap.add_argument("--net", default=None)
    a = ap.parse_args()

    for f in sorted(glob.glob(os.path.join(a.indir, "uc_*.npz"))):
        net, wave = os.path.basename(f)[3:-4].split("_", 1)
        if a.net and net != a.net:
            continue
        z = np.load(f, allow_pickle=True)
        cells, lonlat = z["cells"], z["lonlat"]
        T, ratio = np.asarray(z["T"], float), np.asarray(z["ratio"], float)
        lo, hi = BAND.get((net, wave), (0.0, 99.0))
        band = (T >= lo) & (T <= hi)
        if band.sum() < 2:
            continue
        viol = np.nanmean(ratio[:, band] > 1, axis=1)
        if not np.isfinite(viol).any() or np.nanmax(viol) == 0:
            print(f"\n=== {net}/{wave}: no violations in band {lo}-{hi} s, skipped")
            continue

        ot = cell_wave_stats(net, "overtone", cells)
        fu = cell_wave_stats(net, "fund", cells)
        print(f"\n=== {net}/{wave}   band {lo}-{hi} s   "
              f"mean violation fraction {np.nanmean(viol):.2f}")
        rows = []
        if ot is not None:
            rows += [("overtone coverage", ot[0]), ("overtone res_diag", ot[1])]
        if ot is not None and fu is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                rows += [("overtone/fund vel ratio", ot[2] / fu[2])]
        if fu is not None:
            rows += [("fund res_diag  [CONTROL]", fu[1]), ("fund coverage  [CONTROL]", fu[0])]

        dem = np.load(f"{E}/{net}/tomo/2_vs_depth_inversion/fig_assets_{net}_dem.npz")
        elev, extent = dem["elev"].astype(float), dem["extent"]
        ev = bilinear(elev, extent, lonlat[:, 0], lonlat[:, 1])
        rows.append(("elevation", ev))
        # local relief: spread of DEM within ~1 km, a valley/ridge discriminator
        rel = np.full(len(cells), np.nan)
        for i, (lo_, la_) in enumerate(lonlat):
            dl = 0.01
            pts = bilinear(elev, extent,
                           np.array([lo_ - dl, lo_ + dl, lo_, lo_]),
                           np.array([la_, la_, la_ - dl, la_ + dl]))
            rel[i] = np.nanmax(pts) - np.nanmin(pts)
        rows.append(("local relief (~1 km)", rel))

        gkp = f"{E}/{net}/tomo/2_vs_depth_inversion/fig_assets_{net}_gk500.npz"
        if os.path.exists(gkp):
            gk = np.load(gkp, allow_pickle=True)
            verts, offs, kc = gk["verts"], gk["offsets"], gk["kindcode"]
            fx, fy = [], []
            for i in range(len(kc)):
                if kc[i] == 2:                      # fold axes are not faults
                    continue
                xy = verts[offs[i]:offs[i + 1]]
                x_, y_ = km_xy(xy[:, 0], xy[:, 1])
                fx.append(x_); fy.append(y_)
            if fx:
                fx = np.concatenate(fx); fy = np.concatenate(fy)
                cx, cy = km_xy(lonlat[:, 0], lonlat[:, 1])
                dmin = np.array([np.min(np.hypot(fx - x, fy - y)) for x, y in zip(cx, cy)])
                rows.append(("distance to fault [km]", dmin))

        print(f"    {'metric':<28}{'Spearman r':>12}{'n':>8}")
        for name, val in rows:
            r, n = spearman(viol, np.asarray(val, float))
            flag = ""
            if np.isfinite(r) and abs(r) >= 0.3:
                flag = "  <<<" if "CONTROL" not in name else "  (control also strong)"
            print(f"    {name:<28}{r:>12.3f}{n:>8}{flag}")


if __name__ == "__main__":
    main()
