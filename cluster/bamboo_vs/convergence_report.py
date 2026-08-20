"""Convergence summary per (net, cfg) for the vs_prod3 campaign.

Gate on chain_disagree(Vs), NOT on logL. The calibration work behind this campaign showed the
"chains in the best logL basin" count is misleading when the noise sigma is free: chains split
into basins that differ only in sigma while their Vs(z) is effectively identical, so logL says
1/24 where the Vs posterior is fine. chain_disagree measures spread of the per-chain Vs(z)
medians, which is the thing we actually care about.

Also reports n_chains_used (must be 24 -- anything less means chain files went missing, the
signature of the duplicate-array collision) and the per-wave misfit.

  python convergence_report.py [runs_dir]
"""
import glob
import os
import sys

import numpy as np

CFGS = ["R0g", "R0p", "L0g", "L0p", "RLg_radial", "RLg_iso"]
NETS = ["riehen", "aargau", "hautesorne"]


def main():
    runs = sys.argv[1] if len(sys.argv) > 1 else \
        "/srv/beegfs/scratch/users/s/savardg/vs_prod3/runs"
    print(f"{'net/cfg':<24}{'n':>7}{'chain_disagree med/p90':>25}{'frac>0.3':>10}"
          f"{'nchains<24':>12}{'gamma|>.15':>12}")
    for net in NETS:
        for cfg in CFGS:
            fs = sorted(glob.glob(os.path.join(runs, net, cfg, "cells", "cell_*.npz")))
            if not fs:
                continue
            cd, nc, gpos = [], [], []
            for f in fs:
                try:
                    with np.load(f, allow_pickle=True) as z:
                        cd.append(float(z["chain_disagree"]) if "chain_disagree" in z.files
                                  else np.nan)
                        nc.append(float(z["n_chains_used"]) if "n_chains_used" in z.files
                                  else np.nan)
                        if cfg == "RLg_radial" and "gamma_median" in z.files:
                            g = np.asarray(z["gamma_median"], float)
                            gpos.append(float(np.nanmax(np.abs(g))) if g.size else np.nan)
                except Exception:
                    continue
            cd = np.array(cd, float); nc = np.array(nc, float)
            cdf = cd[np.isfinite(cd)]
            if not len(cdf):
                continue
            frac_bad = float((cdf > 0.3).mean())
            few = int((nc[np.isfinite(nc)] < 24).sum())
            gstr = ""
            if gpos:
                g = np.array(gpos, float)
                g = g[np.isfinite(g)]
                # |gamma| below ~0.15 is not trustworthy: a Love-leak control faked +0.14 and
                # the clean-synthetic null gate gave P(gamma!=0)=0.02-0.11
                gstr = f"{float((g > 0.15).mean()):.2f}"
            print(f"{net + '/' + cfg:<24}{len(cdf):>7}"
                  f"{np.median(cdf):>13.3f}/{np.percentile(cdf, 90):<11.3f}"
                  f"{frac_bad:>10.2f}{few:>12}{gstr:>12}")
    print("\nchain_disagree: spread of per-chain Vs(z) medians; >0.3 = chains disagree on Vs.")
    print("nchains<24 must be 0 -- any shortfall means missing chain files (collision).")
    print("gamma|>.15 = fraction of radial cells whose |gamma| clears the credibility floor.")


if __name__ == "__main__":
    main()
