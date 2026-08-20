#!/usr/bin/env python3
"""Riehen model selection: posterior Vs(z) per waveset combo vs the MICHEL (2016) sonic-derived
Vs model at Basel-1 / Otterbach-2.

This is the decisive test for which input combination should anchor the new manuscript. A
threshold-crossing metric could not decide it -- the ranking flips with the chosen threshold
(at 2.6 km/s group-only lands within 0.08 km of the Basel-1 basement; at 2.8 km/s it is 3.4 km
too deep, because group-only profiles creep through 2.6 then plateau). The Michel log fixes the
VELOCITY SCALE, which turns the comparison into a curve-vs-curve misfit and removes that
degree of freedom.

Reference: `Vsmodel_well_Basel1_Otterbach_Michel2016.csv` (Vs [m/s], depth [m]), a blocky
staircase derived from the Basel-1 / Otterbach-2 sonic logs. On T7blue -- NOT the T7Shield
path still hardcoded in well_vs_qc.py, which is stale.

METRICS, all restricted to the depth range the log actually covers and to each run's own
`z_reliable_max` (comparing where the posterior is prior-filled would score the prior):
  rmse       root-mean-square Vs difference [km/s]
  bias       median (model - log); + means the inversion is FAST
  r          Pearson correlation of the two Vs(z) curves (shape agreement, scale-free)
  z_bas      depth where the model first reaches the log's own basement velocity, vs the known
             basement depth for that well -- the log supplies the velocity, the well the depth,
             so neither is a free parameter.

Usage:
  python riehen_michel_compare.py
  python riehen_michel_compare.py --zmax 4.0
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
MICHEL = ("/Volumes/T7blue/riehen-data/well-data/"
          "Vsmodel_well_Basel1_Otterbach_Michel2016.csv")
TESTS = f"{EHM}/riehen/tomo/2_vs_depth_inversion/tests"
# well -> (cell, known crystalline basement depth [km] from the well)
WELLS = {"Basel-1": ("23_47", 2.426), "Otterbach-2": ("26_43", 2.650)}
COMBOS = ["R0g", "R0gR1g", "R0gL0g", "L0g", "R0gR0p", "L0gL0p", "R0pL0p", "R0gL0gR0pL0p"]
PHASE = {"R0gR0p", "L0gL0p", "R0pL0p", "R0gL0gR0pL0p"}
CCOL = {c: plt.cm.tab10(i / 10) for i, c in enumerate(COMBOS)}


def michel():
    d = pd.read_csv(MICHEL)
    d.columns = [c.strip() for c in d.columns]
    return d["depth"].values / 1000.0, d["Vs"].values / 1000.0     # km, km/s


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--zmin", type=float, default=0.3,
                    help="skip the shallowest section: the inversion's minimum layer and the "
                         "0.5 s period floor cannot resolve the thin Tertiary cover, so "
                         "scoring there measures the prior, not the data")
    ap.add_argument("--zmax", type=float, default=None,
                    help="default: each run's own z_reliable_max, capped by the log's extent")
    a = ap.parse_args()

    zl, vl = michel()
    rows = []
    fig, axs = plt.subplots(1, len(WELLS), figsize=(7.2 * len(WELLS), 8.4), squeeze=False)
    for k, (well, (cell, zbas)) in enumerate(WELLS.items()):
        ax = axs[0][k]
        ax.step(vl, zl, where="post", color="k", lw=2.4, label="Michel (2016) log", zorder=5)
        ax.axhline(zbas, color="0.35", ls="--", lw=1.4)
        ax.text(0.62, zbas - 0.06, f"basement {zbas:.3f} km", fontsize=8, color="0.35")
        # the log's own basement velocity: median of the log below the known basement depth
        v_bas = float(np.median(vl[zl >= zbas]))
        for combo in COMBOS:
            f = f"{TESTS}/test_2026-08-06_waveset_combos/{well}_cell_{cell}/{combo}/bayhunter_result.npz"
            if not os.path.exists(f):
                continue
            z = np.load(f, allow_pickle=True)
            d, v = z["depth"], z["vs_median"]
            zrel = float(z["z_reliable_max"]) if "z_reliable_max" in z else d.max()
            zhi = a.zmax if a.zmax else min(zrel, zl.max())
            m = (d >= a.zmin) & (d <= zhi)
            if m.sum() < 5:
                continue
            vref = np.interp(d[m], zl, vl)          # log resampled onto the model grid
            dv = v[m] - vref
            i = np.argmax(v >= v_bas)
            zb = float(d[i]) if (v >= v_bas).any() else np.nan
            rows.append(dict(well=well, combo=combo, phase=combo in PHASE,
                             z_used=round(zhi, 2), n=int(m.sum()),
                             rmse=round(float(np.sqrt(np.mean(dv ** 2))), 3),
                             bias=round(float(np.median(dv)), 3),
                             r=round(float(np.corrcoef(v[m], vref)[0, 1]), 3),
                             z_bas=round(zb, 2) if np.isfinite(zb) else np.nan,
                             z_bas_err=round(zb - zbas, 2) if np.isfinite(zb) else np.nan))
            ax.plot(v, d, "-", color=CCOL[combo], lw=1.5,
                    alpha=0.95 if combo in PHASE else 0.5,
                    ls="-" if combo in PHASE else "--", label=combo)
        ax.set_ylim(min(4.5, zl.max()), 0); ax.set_xlim(0.5, 3.9)
        ax.set_xlabel("Vs [km/s]"); ax.grid(alpha=0.3)
        ax.set_title(f"{well}  (cell {cell})\nsolid = phase-bearing, dashed = group-only",
                     fontsize=10)
        if k == 0:
            ax.set_ylabel("depth [km]"); ax.legend(fontsize=7.5, loc="lower left")
    fig.suptitle("Riehen: posterior Vs vs the Michel (2016) sonic-derived log",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = f"{TESTS}/test_2026-08-06_waveset_combos"
    p = os.path.join(out, "michel_log_comparison.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)

    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(out, "michel_log_comparison.csv"), index=False)
    for well in WELLS:
        s = D[D.well == well].sort_values("rmse")
        print(f"\n=== {well} — vs Michel log, {a.zmin} km to z_reliable_max ===")
        print(s[["combo", "phase", "z_used", "rmse", "bias", "r", "z_bas", "z_bas_err"]]
              .to_string(index=False))
    print("\n=== mean over both wells (lower rmse better; |z_bas_err| = basement depth error) ===")
    g = D.groupby(["combo", "phase"])[["rmse", "bias", "r", "z_bas_err"]].mean().round(3)
    print(g.sort_values("rmse").to_string())
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
