"""Score the quality-selective restack pilot: picker metrics per stack tree.

Trees (built by regime_select_restack.py, picked by dispersion_unified with DISP_STACK=linear):
    all   = every window (max stacking depth)   top50 = best half by ellipticity   bot50 = worst half

The two comparisons that matter:
    top50 vs bot50 -- SAME window count, so any difference is PURE selection quality
    top50 vs all   -- does selecting good windows beat simply stacking more of them?

Metrics, flagged by whether they are circular w.r.t. selecting on ellipticity:
    ref_mad_G_LR0   |c_phase - VSG c_ref| MAD for G_LR0 phase picks   [NON-CIRCULAR, primary --
                    the reference is an external network-derived standard; LOWER = better]
    ref_med_G_LR0   median signed residual vs the reference           [NON-CIRCULAR; |.| lower better]
    rf_coinc_band   frac of TT argmax picks with |dU_rayfund|<=0.15, T 1.3-1.7 s, graben-W only
                    [NON-CIRCULAR; LOWER = less R-fund-on-TT contamination]
    yield_group/phase  picks with snr>=5 and finite velocity          [NON-CIRCULAR; HIGHER = better]
    snr_G_LR0/snr_TT   median narrowband SNR                          [semi-independent; HIGHER better]
    xmode_fund_T1.5 median xmode_amp of G_LR0 picks at T>1.5 s        [PARTLY CIRCULAR -- both this
                    and the ellipticity selector probe Z-R coherence; report, don't rely on]

Usage: python regime_pilot_score.py --net {riehen,aargau}
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOTS = {"riehen": "/Users/genevievesavard/Codes/extract_higher_modes/Projects/riehen",
         "aargau": "/Users/genevievesavard/Codes/extract_higher_modes/Projects/aargau"}
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--net", required=True, choices=list(ROOTS))
args = ap.parse_args()
PROJ = ROOTS[args.net]
ROOT = os.path.join(PROJ, "regime_pilot")

ref = np.loadtxt(os.path.join(PROJ, "vsg_modesep", "ref_fundamental_phase.txt"))
pairs = pd.read_csv(os.path.join(ROOT, "pairs_selected.csv"))
west = set(pairs.loc[pairs.stratum == "W", "pair"]) if (pairs.stratum == "W").any() else None

TREES = ["all", "top50", "bot50"]
rows = []
for tree in TREES:
    files = glob.glob(os.path.join(ROOT, f"picks_{tree}", "*", "*_unified.csv"))
    if not files:
        continue
    df = pd.concat([pd.read_csv(f).assign(pair=os.path.basename(f)[:-12]) for f in files],
                   ignore_index=True)
    am = df[df.pick_method == "argmax"]
    # --- primary NON-CIRCULAR: agreement with the external VSG phase reference ---
    g0p = am[(am.component == "G_LR0") & np.isfinite(am.phase_velocity)
             & (am.ratio_d_lambda >= 2.0)]
    cref = np.interp(g0p.nominal_period, ref[:, 0], ref[:, 1], left=np.nan, right=np.nan)
    resid = (g0p.phase_velocity - cref).dropna() if len(g0p) else pd.Series(dtype=float)
    med = float(np.median(resid)) if len(resid) else np.nan
    mad = float(np.median(np.abs(resid - med))) if len(resid) else np.nan
    # --- NON-CIRCULAR: graben Love contamination in the diagnosed band ---
    tt = am[am.component == "TT"]
    band = tt[(tt.nominal_period >= 1.3) & (tt.nominal_period <= 1.7) & np.isfinite(tt.dU_rayfund)]
    if west is not None:
        band = band[band.pair.isin(west)]
    ok = df[(df.snr_nbG >= 5)]
    rows.append({
        "tree": tree, "n_pairs": df.pair.nunique(),
        "ref_mad_G_LR0": mad, "ref_med_G_LR0": med, "n_ref": len(resid),
        "rf_coinc_band": float((band.dU_rayfund.abs() <= 0.15).mean()) if len(band) else np.nan,
        "n_band": len(band),
        "yield_group": int(np.isfinite(ok.group_velocity).sum()),
        "yield_phase": int(np.isfinite(ok.phase_velocity).sum()),
        "snr_G_LR0": float(am[am.component == "G_LR0"].snr_nbG.median()),
        "snr_TT": float(tt.snr_nbG.median()),
        "xmode_fund_T1.5": float(am[(am.component == "G_LR0")
                                    & (am.nominal_period > 1.5)].xmode_amp.median()),
    })
res = pd.DataFrame(rows).set_index("tree")

lines = [f"=== quality-selective restack pilot: {args.net} ===",
         f"(selection metric: ellipticity; top50/bot50 have EQUAL window counts)", "",
         res.to_string(), ""]
if {"top50", "bot50", "all"} <= set(res.index):
    t, b, a = res.loc["top50"], res.loc["bot50"], res.loc["all"]
    lines.append("--- top50 vs bot50 (pure selection effect, equal depth) ---")
    lines.append(f"  ref MAD vs VSG    : {t.ref_mad_G_LR0:.4f} vs {b.ref_mad_G_LR0:.4f}  "
                 f"({'top better' if t.ref_mad_G_LR0 < b.ref_mad_G_LR0 else 'bot better'}, "
                 f"{100*(b.ref_mad_G_LR0-t.ref_mad_G_LR0)/b.ref_mad_G_LR0:+.1f}%)")
    lines.append(f"  rf_coinc (graben) : {t.rf_coinc_band:.3f} vs {b.rf_coinc_band:.3f}")
    lines.append(f"  yield group       : {t.yield_group:,} vs {b.yield_group:,}")
    lines.append(f"  snr G_LR0         : {t.snr_G_LR0:.2f} vs {b.snr_G_LR0:.2f}")
    lines.append("--- top50 vs all (selection vs stacking depth) ---")
    lines.append(f"  ref MAD vs VSG    : {t.ref_mad_G_LR0:.4f} vs {a.ref_mad_G_LR0:.4f}  "
                 f"({100*(a.ref_mad_G_LR0-t.ref_mad_G_LR0)/a.ref_mad_G_LR0:+.1f}%)")
    lines.append(f"  yield group       : {t.yield_group:,} vs {a.yield_group:,} "
                 f"({100*(t.yield_group-a.yield_group)/a.yield_group:+.1f}%)")
    gates = {"ref MAD: top < bot (selection works)": t.ref_mad_G_LR0 < b.ref_mad_G_LR0,
             "ref MAD: top <= all (beats depth)": t.ref_mad_G_LR0 <= a.ref_mad_G_LR0,
             "yield: top >= 0.9x all": t.yield_group >= 0.9 * a.yield_group}
    lines.append("")
    lines.append("VERDICT: " + ("PASS" if all(gates.values()) else "MIXED/FAIL"))
    for k, v in gates.items():
        lines.append(f"   [{'Y' if v else 'n'}] {k}")
report = "\n".join(lines)
print(report)
with open(os.path.join(ROOT, "regime_pilot_report.txt"), "w") as f:
    f.write(report + "\n")

metrics = ["ref_mad_G_LR0", "rf_coinc_band", "yield_group", "yield_phase",
           "snr_G_LR0", "snr_TT", "xmode_fund_T1.5"]
fig, axs = plt.subplots(1, len(metrics), figsize=(2.9 * len(metrics), 4.2))
cols = {"all": "0.4", "top50": "tab:green", "bot50": "tab:red"}
for ax, mname in zip(axs, metrics):
    v = res[mname]
    ax.bar(range(len(v)), v.values, color=[cols.get(t, "C0") for t in v.index])
    ax.set_xticks(range(len(v)))
    ax.set_xticklabels(v.index, rotation=45, ha="right", fontsize=9)
    ax.set_title(mname, fontsize=10)
fig.suptitle(f"{args.net}: quality-selective restack (green=best-half, red=worst-half, grey=all)",
             y=1.0)
fig.tight_layout()
fp = os.path.join(ROOT, "regime_pilot_score.png")
fig.savefig(fp, dpi=120, bbox_inches="tight")
print(f"\nwrote {fp}")
