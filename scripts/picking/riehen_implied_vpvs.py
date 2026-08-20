#!/usr/bin/env python3
"""Riehen: IMPLIED Vp/Vs from the Michel (2016) Vp log and our inverted Vs.

WHY THIS AND NOT A Vs COMPARISON. The Michel (2016) "Vs log" is not a Vs measurement: it is
the sonic Vp divided by an ASSUMED ratio -- 1.90 through the Mesozoic cover and 1.75 in the
basement (verified from Michel2016_gpdc.model; the switch is at 2310 m). Scoring our models
against that Vs therefore rewards agreement with an assumption. Worse, the direction matters:
fracturing and clay minerals lower Vs more than Vp, i.e. they RAISE Vp/Vs, so if the true
ratio exceeds 1.90/1.75 the log's Vs is too FAST and a model that looks "too slow" may be
right.

So compare in the quantity each side actually constrains:

    Vp   <- measured by the sonic log
    Vs   <- constrained by the surface-wave dispersion (this is what we invert)
    Vp/Vs = Vp_log(z) / Vs_model(z)      <- the implied, testable physical quantity

and ask whether the implied ratio is geologically plausible rather than whether it equals the
assumed one. Reference bands used here (crystalline/sedimentary rock, fluid-saturated):

    1.65-1.80  intact crystalline basement
    1.80-2.00  normal Mesozoic carbonate/marl cover
    2.00-2.30  fractured and/or clay-bearing  -- the manuscript's interpretation
    > 2.40     implausible for these lithologies: points at a Vs that is too low

CAVEAT, stated because it cuts the other way: our BayHunter runs FIX Vp/Vs = 1.73 internally,
so we assume a ratio too, just in the opposite inferential direction. Surface waves constrain
Vs far more strongly than Vp, so the effect on the recovered Vs is modest -- but the implied
ratios below are not free of that assumption either.

Usage:
  python riehen_implied_vpvs.py
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
MODEL = "/Volumes/T7blue/riehen-data/well-data/Michel2016_gpdc.model"
TESTS = f"{EHM}/riehen/tomo/2_vs_depth_inversion/tests/test_2026-08-06_waveset_combos"
WELLS = {"Basel-1": ("23_47", 2.426), "Otterbach-2": ("26_43", 2.650)}
COMBOS = ["R0g", "R0gR1g", "R0gL0g", "L0g", "R0gR0p", "L0gL0p", "R0pL0p", "R0gL0gR0pL0p"]
PHASE = {"R0gR0p", "L0gL0p", "R0pL0p", "R0gL0gR0pL0p"}
BANDS = [(1.65, 1.80, "intact crystalline", "#4575b4"),
         (1.80, 2.00, "normal cover", "#91bfdb"),
         (2.00, 2.30, "fractured / clay-bearing", "#fee090"),
         (2.30, 2.60, "strongly altered", "#fc8d59")]


def michel_vp():
    """(depth_km edges, Vp km/s, Vs_assumed km/s, assumed ratio) as a staircase."""
    rows = [l.split() for l in open(MODEL).read().split("\n")[1:] if l.strip()]
    th = np.array([float(r[0]) for r in rows]) / 1000.0
    vp = np.array([float(r[1]) for r in rows]) / 1000.0
    vs = np.array([float(r[2]) for r in rows]) / 1000.0
    top = np.concatenate([[0.0], np.cumsum(th)[:-1]])
    return top, vp, vs


def step_at(top, val, z):
    """Staircase lookup: value of the layer containing depth z."""
    i = np.searchsorted(top, z, side="right") - 1
    return val[np.clip(i, 0, len(val) - 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--zmin", type=float, default=0.3)
    a = ap.parse_args()
    top, vp, vs_assumed = michel_vp()

    rows = []
    fig, axs = plt.subplots(1, 2 * len(WELLS), figsize=(7.0 * len(WELLS), 8.4), sharey=True)
    for k, (well, (cell, zbas)) in enumerate(WELLS.items()):
        axv, axr = axs[2 * k], axs[2 * k + 1]
        # Vp panel: the measurement
        zz = np.linspace(0, 4.5, 900)
        axv.step(step_at(top, vp, zz), zz, where="post", color="k", lw=2.2,
                 label="Vp (sonic, measured)")
        axv.step(step_at(top, vs_assumed, zz), zz, where="post", color="0.45", lw=1.8,
                 ls="--", label="Vs = Vp/1.90 or /1.75 (ASSUMED)")
        # ratio panel: plausibility bands
        for lo, hi, lab, col in BANDS:
            axr.axvspan(lo, hi, color=col, alpha=0.35)
            axr.text(0.5 * (lo + hi), 0.06, lab, rotation=90, fontsize=6.5,
                     ha="center", va="bottom", color="0.25")
        axr.step(step_at(top, vp / vs_assumed, zz), zz, where="post", color="0.45",
                 lw=1.8, ls="--", label="assumed (1.90 / 1.75)")
        axr.axhline(zbas, color="0.3", ls=":", lw=1.2)

        for combo in COMBOS:
            f = f"{TESTS}/{well}_cell_{cell}/{combo}/bayhunter_result.npz"
            if not os.path.exists(f):
                continue
            z = np.load(f, allow_pickle=True)
            d, v = z["depth"], z["vs_median"]
            zrel = float(z["z_reliable_max"]) if "z_reliable_max" in z else d.max()
            m = (d >= a.zmin) & (d <= min(zrel, top[-1], 4.5)) & (v > 0)
            if m.sum() < 5:
                continue
            ratio = step_at(top, vp, d[m]) / v[m]
            col = plt.cm.tab10(COMBOS.index(combo) / 10)
            ls = "-" if combo in PHASE else "--"
            axv.plot(v[m], d[m], ls, color=col, lw=1.4, alpha=0.9)
            axr.plot(ratio, d[m], ls, color=col, lw=1.5, alpha=0.9, label=combo)
            cov = d[m] < zbas
            bas = d[m] >= zbas
            rows.append(dict(
                well=well, combo=combo, phase=combo in PHASE,
                vpvs_cover=round(float(np.median(ratio[cov])), 2) if cov.any() else np.nan,
                vpvs_basement=round(float(np.median(ratio[bas])), 2) if bas.any() else np.nan,
                vpvs_all=round(float(np.median(ratio)), 2),
                frac_gt_2p3=round(float(np.mean(ratio > 2.3)), 2),
                frac_gt_2p6=round(float(np.mean(ratio > 2.6)), 2)))
        axv.set_ylim(4.5, 0); axv.set_xlim(0.5, 6.2); axv.grid(alpha=0.3)
        axv.set_xlabel("velocity [km/s]"); axv.set_title(f"{well}: log Vp & our Vs", fontsize=10)
        axr.set_ylim(4.5, 0); axr.set_xlim(1.4, 3.2); axr.grid(alpha=0.3)
        axr.set_xlabel("implied Vp/Vs = Vp$_{log}$ / Vs$_{model}$")
        axr.set_title(f"{well}: implied ratio", fontsize=10)
        if k == 0:
            axv.set_ylabel("depth [km]"); axv.legend(fontsize=7, loc="lower left")
            axr.legend(fontsize=6.5, loc="lower right", ncol=2)
    fig.suptitle("Riehen: implied Vp/Vs from the measured sonic Vp and our inverted Vs\n"
                 "solid = phase-bearing, dashed = group-only; grey dashed = the log's own "
                 "ASSUMED ratio",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(TESTS, "implied_vpvs.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)

    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(TESTS, "implied_vpvs.csv"), index=False)
    for well in WELLS:
        s = D[D.well == well].sort_values("vpvs_all")
        print(f"\n=== {well} — implied Vp/Vs (log assumes 1.90 cover / 1.75 basement) ===")
        print(s[["combo", "phase", "vpvs_cover", "vpvs_basement", "vpvs_all",
                 "frac_gt_2p3", "frac_gt_2p6"]].to_string(index=False))
    print("\n=== mean over both wells ===")
    print(D.groupby(["combo", "phase"])[["vpvs_cover", "vpvs_basement", "vpvs_all",
                                         "frac_gt_2p3"]].mean().round(2)
          .sort_values("vpvs_all").to_string())
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
