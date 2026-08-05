#!/usr/bin/env python3
"""Quantify how the inversion choices (vbounds k, Cd model) change the velocity maps.

Compares, per period, every combination of
    cull      k=2 | k=3                (one-sided vbounds upper bound)
    Cd model  blanket | measured | scaled
    measure   group | phase
    wave      fund | overtone | love
on four axes that matter for interpretation:

1. ANOMALY AMPLITUDE  -- spatial spread of the relative anomaly a = 100*(V-med)/med over the
   imaged cells, as both std and IQR [%]. "How strong is the lateral structure."

2. ANOMALY SIZE       -- the 1/e decorrelation length of the anomaly field [km], from its
   radially-averaged 2D autocorrelation. Computed FFT-wise with a mask-normalised estimator
   (num = |FFT(a·m)|^2, den = |FFT(m)|^2) so the irregular imaged footprint does not bias it.
   "How big are the features."

3. GEOLOGY CORRELATION -- eta^2 of a one-way ANOVA of the anomaly against the surface-geology
   class (`velocity_by_geology.COVER`: Mesozoic carbonate / Tertiary molasse / Quaternary /
   crystalline), i.e. the fraction of lateral variance explained by mapped geology. Reported
   with the carbonate-minus-molasse contrast, the specific Haute-Sorne question.

4. PERIOD CONTINUITY  -- per CELL, the second difference of V along the period ladder
   (nearly uniform in log T), normalised by that cell's mean velocity, then aggregated over
   cells. Low = the cell's dispersion curve is smooth; high = the map jitters period to
   period, which is a hallmark of an under-constrained inversion rather than structure.

Metric choices worth knowing:
  * Everything is computed on the RELATIVE anomaly, not raw velocity, so the strong dispersion
    trend does not dominate the spatial statistics.
  * eta^2 needs classes to coexist at the same period; classes with < MIN_CLASS cells are
    dropped for that period, and eta^2 is set NaN if fewer than 2 classes survive.
  * Continuity uses each run's OWN period set (k2 and k3 do not always retain the same rungs).

Usage:
  python compare_inversion_choices.py --all
  python compare_inversion_choices.py --net hautesorne --band 1 2
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

from swtomotv.config import DatasetConfig
from swtomotv.geometry import make_grid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from velocity_by_geology import cell_units          # noqa: E402

EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
TOMO = "/Users/genevievesavard/Codes/NoisePy-ant/param_files/cluster/tomo"
DX = {"riehen": "0.2", "aargau": "0.5", "hautesorne": "0.5"}
WAVES = ("fund", "overtone", "love")
MEASURES = ("group", "phase")
CDS = ("blanket", "measured", "scaled")
KS = ("k2", "k3")
MIN_CLASS = 25
MIN_CELLS = 100


def anomaly(V):
    """Relative anomaly [%] over finite cells; NaN elsewhere."""
    m = np.isfinite(V)
    if m.sum() < MIN_CELLS:
        return None, m
    med = np.median(V[m])
    if not np.isfinite(med) or med == 0:
        return None, m
    return np.where(m, 100.0 * (V - med) / med, np.nan), m


def decorrelation_km(a, mask, dx):
    """1/e length of the radially-averaged autocorrelation of the anomaly field.

    Mask-normalised: dividing the field autocorrelation by the MASK autocorrelation removes
    the footprint's own shape, which would otherwise set the apparent feature size.
    """
    if mask.sum() < MIN_CELLS:
        return np.nan
    f = np.where(mask, a - np.nanmean(a[mask]), 0.0)
    m = mask.astype(float)
    F = np.fft.fft2(f)
    M = np.fft.fft2(m)
    num = np.real(np.fft.ifft2(F * np.conj(F)))
    den = np.real(np.fft.ifft2(M * np.conj(M)))
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(den > 0.5, num / np.where(den == 0, np.nan, den), np.nan)
    c = np.fft.fftshift(c)
    c0 = c[c.shape[0] // 2, c.shape[1] // 2]
    if not np.isfinite(c0) or c0 <= 0:
        return np.nan
    c = c / c0
    ny, nx = c.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(yy - ny // 2, xx - nx // 2) * dx
    rmax = min(ny, nx) // 2 * dx
    edges = np.arange(0, rmax, dx)
    prof = []
    for i in range(len(edges) - 1):
        sel = (r >= edges[i]) & (r < edges[i + 1])
        prof.append(np.nanmean(c[sel]) if sel.any() else np.nan)
    prof = np.asarray(prof, float)
    below = np.where(prof < 1.0 / np.e)[0]
    if not len(below):
        return np.nan
    i = below[0]
    if i == 0:
        return float(edges[0])
    # linear interpolation between the bracketing radial bins
    p0, p1 = prof[i - 1], prof[i]
    if not np.isfinite(p0) or p1 == p0:
        return float(edges[i])
    frac = (p0 - 1.0 / np.e) / (p0 - p1)
    return float(edges[i - 1] + frac * dx)


def eta2_geology(a, mask, units, ok_u):
    """(eta^2, per-class medians) of the anomaly against surface-geology class."""
    sel = mask & ok_u
    if sel.sum() < MIN_CELLS:
        return np.nan, {}
    vals = a[sel]
    lab = units[sel].astype(str)
    meds, groups = {}, []
    for u in np.unique(lab):
        v = vals[lab == u]
        if v.size >= MIN_CLASS:
            groups.append(v)
            meds[u] = float(np.median(v))
    if len(groups) < 2:
        return np.nan, meds
    allv = np.concatenate(groups)
    grand = allv.mean()
    ss_tot = float(((allv - grand) ** 2).sum())
    ss_btw = float(sum(g.size * (g.mean() - grand) ** 2 for g in groups))
    return (ss_btw / ss_tot if ss_tot > 0 else np.nan), meds


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--band", type=float, nargs=2, default=[1.0, 2.0],
                    help="period band [s] for the focused geology comparison")
    ap.add_argument("--out", default=f"{EHM}/_inversion_comparison")
    a = ap.parse_args()
    nets = list(DX) if a.all else [a.net]
    os.makedirs(a.out, exist_ok=True)

    rows, cont_rows, cell_rows = [], [], []
    for net in nets:
        dx = float(DX[net])
        ds = DatasetConfig.from_yaml(f"{TOMO}/{net}_tspws_group_scaled_lccov.yaml")
        ds.dx_km = dx
        grid = make_grid(ds.bounds, dx)
        units = cell_units(net, grid, "cover")
        ok_u = pd.notna(units)
        print("\n=== %s (dx=%.1f, %d cells) ===" % (net, dx, grid.ncell), flush=True)

        for k in KS:
            for meas in MEASURES:
                for cd in CDS:
                    run = f"1_production/tspws_{meas}_{cd}_dx{DX[net]}_prod3_{k}"
                    root = f"{EHM}/{net}/tomo/1_velocity_maps/{run}/production"
                    for wave in WAVES:
                        files = sorted(glob.glob(f"{root}/{wave}/map_T*.npz"),
                                       key=lambda p: float(np.load(p)["period"]))
                        if not files:
                            continue
                        Ts, stack = [], []
                        for f in files:
                            z = np.load(f)
                            V = np.where(z["mask"].astype(bool), z["vel"], np.nan)
                            an, m = anomaly(V)
                            T = float(z["period"])
                            Ts.append(T)
                            stack.append(V)
                            if an is None:
                                continue
                            e2, meds = eta2_geology(an, m, units, ok_u)
                            rec = dict(net=net, k=k, measure=meas, cd=cd, wave=wave, T=T,
                                       n_cells=int(m.sum()), N=int(z["N"]),
                                       var_red=float(z["var_red"]),
                                       amp_std=float(np.std(an[m])),
                                       amp_iqr=float(np.subtract(
                                           *np.percentile(an[m], [75, 25]))),
                                       size_km=decorrelation_km(an, m, dx),
                                       eta2=e2)
                            for cl, mv in meds.items():
                                rec["geo_" + cl] = mv
                            rows.append(rec)
                        # ---- continuity: per-cell second difference along the ladder
                        if len(stack) >= 5:
                            S = np.stack(stack)                       # (nT, nx, ny)
                            d2 = S[:-2] - 2 * S[1:-1] + S[2:]
                            good = np.isfinite(d2).sum(axis=0) >= 3
                            with np.errstate(invalid="ignore"):
                                mean_v = np.nanmean(S, axis=0)
                                rough = 100.0 * np.nanmedian(np.abs(d2), axis=0) / mean_v
                            rough = np.where(good & np.isfinite(rough), rough, np.nan)
                            if np.isfinite(rough).sum() >= MIN_CELLS:
                                cont_rows.append(dict(
                                    net=net, k=k, measure=meas, cd=cd, wave=wave,
                                    n_periods=len(stack),
                                    rough_med=float(np.nanmedian(rough)),
                                    rough_p90=float(np.nanpercentile(rough, 90)),
                                    frac_rough_gt5=float(np.nanmean(rough > 5.0))))
                                cell_rows.append(dict(net=net, k=k, measure=meas, cd=cd,
                                                      wave=wave, rough=rough))
                    print("  %s %s %-8s done" % (k, meas, cd), flush=True)

    D = pd.DataFrame(rows)
    C = pd.DataFrame(cont_rows)
    D.to_csv(f"{a.out}/per_period_metrics.csv", index=False)
    C.to_csv(f"{a.out}/continuity_metrics.csv", index=False)
    np.savez_compressed(f"{a.out}/cell_roughness.npz",
                        **{f"{r['net']}|{r['k']}|{r['measure']}|{r['cd']}|{r['wave']}":
                           r["rough"] for r in cell_rows})
    print("\nwrote %s/per_period_metrics.csv  (%d rows)" % (a.out, len(D)))
    print("wrote %s/continuity_metrics.csv (%d rows)" % (a.out, len(C)))


if __name__ == "__main__":
    main()
