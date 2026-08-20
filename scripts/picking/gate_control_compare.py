#!/usr/bin/env python
"""Is the mode gate's effect on Vs real, or just sampler noise?

Uses an exact internal control: cells the gate left untouched re-ran on IDENTICAL data, so
their gated-vs-ungated difference is the sampler's own run-to-run floor. Cells that lost
samples must beat that floor. This is what makes the comparison falsifiable -- without it a
0.2 km/s shift is indistinguishable from MCMC scatter.
"""
import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr

E = "/Users/genevievesavard/Codes/extract_higher_modes"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--base", default="L0g")
    ap.add_argument("--gated", default="L0g_modegate")
    ap.add_argument("--waveset", default="love")
    a = ap.parse_args()

    B = f"{E}/Projects/{a.net}/tomo/2_vs_depth_inversion/vs_prod3"
    va = np.load(f"{B}/{a.base}/volume_{a.waveset}.npz", allow_pickle=True)
    vb = np.load(f"{B}/{a.gated}/volume_{a.waveset}.npz", allow_pickle=True)
    drops = {(int(r["ix"]), int(r["iy"])): int(r["dropped"])
             for r in csv.DictReader(
                 open(f"{E}/Projects/_gate_eval/gatedrops_{a.net}_love.csv"))}

    ka = {tuple(c): i for i, c in enumerate(va["cells"])}
    kb = {tuple(c): i for i, c in enumerate(vb["cells"])}
    common = [c for c in sorted(set(ka) & set(kb)) if c in drops]
    ia = np.array([ka[c] for c in common]); ib = np.array([kb[c] for c in common])
    nd = np.array([drops[c] for c in common])
    z = va["depth"]
    d = vb["vs_median"][ib] - va["vs_median"][ia]
    mx = np.nanmax(np.abs(d), axis=1)

    ctl, sig = nd == 0, nd > 0
    print(f"\n=== {a.net}: {a.gated} vs {a.base} ===")
    print(f"paired {len(common)}  control(0 drops) {ctl.sum()}  gated(>0) {sig.sum()}  "
          f"total dropped {nd.sum()}")
    c, s = mx[ctl][np.isfinite(mx[ctl])], mx[sig][np.isfinite(mx[sig])]
    if c.size and s.size:
        u = mannwhitneyu(s, c, alternative="greater")
        print(f"max|dVs| control median {np.median(c):.4f} p90 {np.percentile(c,90):.4f}")
        print(f"max|dVs| gated   median {np.median(s):.4f} p90 {np.percentile(s,90):.4f}")
        print(f"  ratio {np.median(s)/max(np.median(c),1e-9):.0f}x   p={u.pvalue:.3g}")
    g = np.isfinite(mx)
    r = spearmanr(nd[g], mx[g])
    print(f"drops vs max|dVs|: rho={r.statistic:+.3f} p={r.pvalue:.3g}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    bins = np.linspace(0, np.nanpercentile(mx, 99), 45)
    if ctl.any():
        ax[0].hist(mx[ctl], bins=bins, alpha=.6, density=True, label=f"0 drops (n={ctl.sum()})")
    ax[0].hist(mx[sig], bins=bins, alpha=.6, density=True, label=f">0 drops (n={sig.sum()})")
    ax[0].set(xlabel="max |ΔVs| over depth (km/s)", ylabel="density",
              title="Gate effect vs MCMC noise floor")
    ax[0].legend()

    k = np.clip(nd, 0, 6)
    ax[1].boxplot([mx[(k == i) & np.isfinite(mx)] for i in range(7)],
                  tick_labels=["0","1","2","3","4","5","6+"], showfliers=False)
    ax[1].set(xlabel="periods dropped by gate", ylabel="max |ΔVs| (km/s)",
              title=f"Dose–response (ρ={r.statistic:+.2f})")

    if ctl.any():
        ax[2].plot(np.nanmedian(d[ctl], 0), z, lw=2, label="0 drops (control)")
    ax[2].plot(np.nanmedian(d[sig], 0), z, lw=2, label=">0 drops")
    ax[2].axvline(0, color="k", lw=.8); ax[2].invert_yaxis()
    ax[2].set(xlabel="median ΔVs (km/s)", ylabel="depth (km)", title="Where the gate moves Vs")
    ax[2].legend(); ax[2].grid(alpha=.3)

    fig.suptitle(f"{a.net} {a.waveset} — mode gate vs zero-drop control "
                 f"({nd.sum()} samples dropped over {sig.sum()} cells)", fontsize=12)
    fig.tight_layout()
    o = f"{B}/{a.gated}/gate_effect_vs_control.png"
    fig.savefig(o, dpi=140); print("wrote", o)


main()
