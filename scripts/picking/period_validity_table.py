#!/usr/bin/env python3
"""Decision table for choosing VALID PERIOD RANGES before the 1D Vs depth inversion.

One row per (network, measure, wave, period) with every diagnostic bearing on "should this
period enter the Vs inversion", so the choice is made against numbers rather than by eye.
This does NOT decide anything -- it assembles the evidence.

COLUMNS

  n_rays        picks entering that period map. Below ~40 the map is skipped upstream; below
                a few hundred the map is essentially the prior.
  var_red       variance reduction of the map fit. <= 0 means the map fits worse than a
                constant -- those periods should not be inverted for depth.
  cell_cov      fraction of grid cells with a value (the imaged footprint at that period).
  amp_pct       lateral anomaly amplitude [% std]. Near-zero = no lateral information.
  rough_pct     per-cell |2nd difference| of V(T) [% of cell mean], median over cells. High
                = the cell dispersion curves jitter period to period, which a 1D inversion
                will try to fit with layering.
  eta2_geol     fraction of lateral variance explained by mapped surface geology.
  lambda_km     v_med * T -- the wavelength.
  z_sens_km     lambda/3, the standard rule-of-thumb sensitivity depth for the Rayleigh
                fundamental. INDICATIVE ONLY: it is wrong for overtones (they sample
                deeper for the same period) and only approximate for Love. Use it to see
                which depth range a period band buys you, not as a kernel.
  issues        known issues from the 2026-08-03..05 audit (see the per-net README ledger).

ISSUES  (column `issues`; NOT named `flags` -- that shadows pandas
         DataFrame.flags and `d.flags` silently returns the attribute)
  PHASE_PROXY_SIGMA  phase under scaled/measured Cd: proxy sigma inflates amplitude, halves
                     period continuity, loses geology correlation. Prefer phase_blanket.
  NEG_VARRED         var_red <= 0 at this period.
  THIN               n_rays < 300.
  SHORT_T_PATHBIAS   T < 1 s, where the graben-side velocity was shown to scale with path
                     length (Riehen: x3 from 0-3 km to >10 km paths) -- an apparent lateral
                     contrast that is really a path-length artifact.
  LOW_AMP            anomaly amplitude < 3%: little lateral information to invert.
  ROUGH              per-cell roughness > 3%.

Usage:
  python period_validity_table.py                       # writes CSV + per-net figures
  python period_validity_table.py --measure group       # one measure only
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
CMP = f"{EHM}/_inversion_comparison"
OUT = f"{EHM}/_period_validity"
DX = {"riehen": "0.2", "aargau": "0.5", "hautesorne": "0.5"}
# the recommended Cd per measure (see each net's 1_velocity_maps/README.md)
CD = {"group": "scaled", "phase": "blanket"}
WAVES = ("fund", "overtone", "love")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--k", default="k3")
    ap.add_argument("--measure", default=None)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    met = pd.read_csv(f"{CMP}/per_period_metrics.csv")
    cont = pd.read_csv(f"{CMP}/continuity_metrics.csv")
    measures = (a.measure,) if a.measure else ("group", "phase")

    rows = []
    for net in DX:
        for meas in measures:
            cd = CD[meas]
            for wave in WAVES:
                root = (f"{EHM}/{net}/tomo/1_velocity_maps/1_production/"
                        f"tspws_{meas}_{cd}_dx{DX[net]}_prod3_{a.k}/production/{wave}")
                files = glob.glob(f"{root}/map_T*.npz")
                if not files:
                    continue
                rr = cont[(cont.net == net) & (cont.measure == meas) &
                          (cont.cd == cd) & (cont.wave == wave)]
                rough = float(rr.rough_med.iloc[0]) if len(rr) else np.nan
                for f in sorted(files, key=lambda p: float(np.load(p)["period"])):
                    z = np.load(f)
                    T = float(z["period"])
                    V = z["vel"]
                    ok = np.isfinite(V)
                    if ok.sum() < 50:
                        continue
                    vmed = float(np.median(V[ok]))
                    m = met[(met.net == net) & (met.measure == meas) & (met.cd == cd) &
                            (met.wave == wave) & (np.abs(met["T"] - T) < 1e-6) &
                            (met.k == a.k)]
                    amp = float(m.amp_std.iloc[0]) if len(m) else np.nan
                    eta = float(m.eta2.iloc[0]) if len(m) else np.nan
                    size = float(m.size_km.iloc[0]) if len(m) else np.nan
                    n = int(z["N"]); vr = float(z["var_red"])
                    lam = vmed * T
                    flags = []
                    if meas == "phase" and cd in ("scaled", "measured"):
                        flags.append("PHASE_PROXY_SIGMA")
                    if vr <= 0:
                        flags.append("NEG_VARRED")
                    if n < 300:
                        flags.append("THIN")
                    if T < 1.0:
                        flags.append("SHORT_T_PATHBIAS")
                    if np.isfinite(amp) and amp < 3.0:
                        flags.append("LOW_AMP")
                    if np.isfinite(rough) and rough > 3.0:
                        flags.append("ROUGH")
                    rows.append(dict(
                        net=net, measure=meas, cd=cd, wave=wave, k=a.k, T=round(T, 3),
                        n_rays=n, var_red=round(vr, 3),
                        cell_cov=round(ok.sum() / V.size, 3),
                        v_med=round(vmed, 3), amp_pct=round(amp, 2) if np.isfinite(amp) else np.nan,
                        size_km=round(size, 2) if np.isfinite(size) else np.nan,
                        rough_pct=round(rough, 2) if np.isfinite(rough) else np.nan,
                        eta2_geol=round(eta, 3) if np.isfinite(eta) else np.nan,
                        lambda_km=round(lam, 2), z_sens_km=round(lam / 3.0, 2),
                        issues="|".join(flags)))
    D = pd.DataFrame(rows).sort_values(["net", "measure", "wave", "T"])
    p = f"{a.out}/period_validity_{a.k}.csv"
    D.to_csv(p, index=False)
    print("wrote %s  (%d rows)" % (p, len(D)))

    # a blank decision sheet: one row per (net, measure, wave) for the user to fill in
    dec = (D.groupby(["net", "measure", "wave"])
             .agg(T_available_min=("T", "min"), T_available_max=("T", "max"),
                  n_periods=("T", "size"))
             .reset_index())
    dec["T_valid_min"] = ""
    dec["T_valid_max"] = ""
    dec["reason"] = ""
    dp = f"{a.out}/period_ranges_DECISIONS.csv"
    if not os.path.exists(dp):
        dec.to_csv(dp, index=False)
        print("wrote %s  (blank -- fill T_valid_min/max)" % dp)
    else:
        print("kept existing %s (not overwritten)" % dp)

    for net in D.net.unique():
        g = D[D.net == net]
        fig, axs = plt.subplots(4, len(WAVES), figsize=(6.0 * len(WAVES), 13), squeeze=False)
        for j, wave in enumerate(WAVES):
            for i, (col, lab, logy) in enumerate((
                    ("n_rays", "rays per period map", True),
                    ("var_red", "var_red (<=0 unusable)", False),
                    ("amp_pct", "anomaly amplitude [%]", False),
                    ("z_sens_km", "indicative depth lambda/3 [km]", False))):
                ax = axs[i][j]
                for meas, c in (("group", "tab:blue"), ("phase", "tab:red")):
                    s = g[(g.wave == wave) & (g.measure == meas)].sort_values("T")
                    if not len(s):
                        continue
                    ax.plot(s["T"], s[col], "o-", color=c, ms=3, lw=1.6,
                            label="%s (%s)" % (meas, CD[meas]))
                if logy:
                    ax.set_yscale("log")
                if col == "var_red":
                    ax.axhline(0, color="k", lw=1)
                ax.grid(alpha=0.3)
                if i == 0:
                    ax.set_title("%s — %s" % (net, wave), fontsize=11)
                    ax.legend(fontsize=8)
                if j == 0:
                    ax.set_ylabel(lab, fontsize=9.5)
                if i == 3:
                    ax.set_xlabel("period [s]")
        fig.suptitle("%s: period-validity diagnostics for the Vs depth inversion "
                     "(prod3 %s, group=%s / phase=%s)"
                     % (net, a.k, CD["group"], CD["phase"]), fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(f"{a.out}/period_validity_{net}_{a.k}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    print("wrote %d per-net figures" % D.net.nunique())


if __name__ == "__main__":
    main()
