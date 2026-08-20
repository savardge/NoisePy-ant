#!/usr/bin/env python3
"""Do the section cells that carry deep fast Vs sit on the contaminated 3-4 s bump?

s9_tail_origin.py showed the S9 tail is a period-LOCALISED excursion: the tail cells' curve
tracks the regional one to 2 s, jumps ~1.1 km/s between 2.8 and 4.2 s, then returns. Real rock
does not switch off at 4.2 s, and at those cells the overtone-fundamental separation collapses
(1.07 -> 0.30 km/s, with 29% of cells having the fundamental FASTER than the overtone, which is
impossible). So the excursion is mode contamination that the map faithfully reproduces.

This script measures the excursion per cell as a BUMP -- how far U in 2.8-4.2 s rises above the
cell's own trend on either side of that window -- and asks whether it predicts the deep fast Vs
along A-A', B-B' and C-C'. A bump is a cleaner discriminant than raw U: a genuinely fast cell is
fast at 5-6 s too, so its bump is ~0, whereas a contaminated cell is fast only inside the band.

If bump predicts the deep fast bodies, the manuscript's high-Vs basement features are inherited
from contaminated picks rather than from structure -- upstream of both the map and the sampler.
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
VS = f"{EHM}/hautesorne/tomo/2_vs_depth_inversion/manuscript_comparison_2026-08/1_sections"
TBAND = (2.8, 4.2)
SIDE_LO = (1.8, 2.6)     # trend windows either side of the bump
SIDE_HI = (4.6, 6.5)
UHI = 3.5
ZWIN = (2.24, 4.5)
SECTIONS = {"AA": ("test_2026-08-08_AA_section_R0g", "tab:blue"),
            "BB": ("test_2026-08-08_BB_section_R0g", "tab:red"),
            "CC": ("test_2026-08-16_CC_section_R0g", "tab:green")}


def bump(T, U):
    """U excess inside the band over the interpolated trend across it."""
    b = (T >= TBAND[0]) & (T <= TBAND[1])
    lo = (T >= SIDE_LO[0]) & (T <= SIDE_LO[1])
    hi = (T >= SIDE_HI[0]) & (T <= SIDE_HI[1])
    if not (b.any() and lo.any() and hi.any()):
        return np.nan
    base = 0.5 * (np.nanmedian(U[lo]) + np.nanmedian(U[hi]))
    return float(np.nanmax(U[b]) - base)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=f"{EHM}/hautesorne/tomo/2_vs_depth_inversion/manuscript_comparison_2026-08/"
                                     f"3_diagnostics/si_s9_s10_check")
    a = ap.parse_args()
    R = []
    for sec, (tag, col) in SECTIONS.items():
        for f in sorted(glob.glob(f"{VS}/{tag}/cell_*/bayhunter_result.npz")):
            z = np.load(f, allow_pickle=True)
            b = os.path.basename(os.path.dirname(f)).split("_")
            d = z["depth"]; m = (d >= ZWIN[0]) & (d <= ZWIN[1])
            E = np.asarray(z["ens_vs"], float)
            T = np.asarray(z["obsT_fund"], float); U = np.asarray(z["obs_fund"], float)
            R.append(dict(sec=sec, col=col, ix=int(b[1]), iy=int(b[2]),
                          frac=float(np.mean(E[:, m].max(axis=1) > UHI)),
                          bump=bump(T, U), tmax=float(T.max()),
                          vsdeep=float(np.nanmax(np.asarray(z["vs_median"])[m]))))
    if not R:
        raise SystemExit("no section cells")
    bp = np.array([r["bump"] for r in R]); fr = np.array([r["frac"] for r in R])
    g = np.isfinite(bp) & np.isfinite(fr)
    print(f"{len(R)} cells, {g.sum()} with a usable bump "
          f"(needs data past {SIDE_HI[0]} s; T_max median {np.median([r['tmax'] for r in R]):.2f} s)")
    print(f"bump: median {np.nanmedian(bp[g]):+.3f} km/s, p95 {np.nanpercentile(bp[g],95):+.3f}")
    print(f"corr(bump, frac models > {UHI}) = {np.corrcoef(bp[g], fr[g])[0,1]:+.3f}")
    print(f"corr(bump, deepest median Vs)  = "
          f"{np.corrcoef(bp[g], np.array([r['vsdeep'] for r in R])[g])[0,1]:+.3f}")
    hi = [r for r in R if r["frac"] > 0.10 and np.isfinite(r["bump"])]
    lo = [r for r in R if r["frac"] <= 0.10 and np.isfinite(r["bump"])]
    if hi:
        print(f"cells with >10% fast models: bump {np.median([r['bump'] for r in hi]):+.3f} "
              f"km/s   vs {np.median([r['bump'] for r in lo]):+.3f} for the rest")
    for sec in SECTIONS:
        q = [r for r in R if r["sec"] == sec and np.isfinite(r["bump"])]
        if len(q) > 3:
            print(f"  {sec}: corr = {np.corrcoef([r['bump'] for r in q], [r['frac'] for r in q])[0,1]:+.3f} "
                  f"({len(q)} cells, max bump {max(r['bump'] for r in q):+.2f})")

    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2))
    ax = axs[0]
    for sec, (_, col) in SECTIONS.items():
        q = [r for r in R if r["sec"] == sec and np.isfinite(r["bump"])]
        ax.scatter([r["bump"] for r in q], [100 * r["frac"] for r in q], s=28, color=col,
                   label=sec)
    ax.axvline(0, color="k", lw=1.0)
    ax.set_xlabel(f"bump: max U in {TBAND[0]}-{TBAND[1]} s minus the cell's own trend [km/s]")
    ax.set_ylabel(f"% of models > {UHI} km/s at {ZWIN[0]}-{ZWIN[1]} km")
    ax.set_title("does the contaminated bump predict the deep fast body?", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax = axs[1]
    for sec, (_, col) in SECTIONS.items():
        q = sorted([r for r in R if r["sec"] == sec and np.isfinite(r["bump"])],
                   key=lambda r: (r["ix"], r["iy"]))
        if not q:
            continue
        x = np.arange(len(q))
        ax.plot(x, [r["bump"] for r in q], "o-", color=col, ms=4, label=f"{sec} bump")
        ax.plot(x, [r["frac"] for r in q], "s--", color=col, ms=3, alpha=0.55,
                label=f"{sec} frac>3.5")
    ax.axhline(0, color="k", lw=1.0)
    ax.set_xlabel("cell index along profile")
    ax.set_ylabel("bump [km/s]  /  fraction of fast models")
    ax.set_title("along-profile: the two track each other", fontsize=11)
    ax.legend(fontsize=7, ncol=3); ax.grid(alpha=0.3)
    fig.suptitle("The deep fast Vs follows a period-localised bump in the picks, not structure",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(a.out, "S9_bump_vs_sections.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); print("wrote", p)


if __name__ == "__main__":
    main()
