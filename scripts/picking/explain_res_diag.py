#!/usr/bin/env python3
"""What res_diag is, why it collapses at long period, and what trimming on it does.

DEFINITION (swtomotv/inversion.py::tv_two_step). Each period's group-velocity map is a linear
Bayesian inversion of travel times tau for cell slowness m:

    m_post = m_prior + (G' Cd^-1 G + CM^-1)^-1 G' Cd^-1 (tau - G m_prior)

with G the ray kernel (km), Cd the data covariance and CM the exponential prior covariance. The
RESOLUTION MATRIX is

    R = (G' Cd^-1 G + CM^-1)^-1 (G' Cd^-1 G)   =   H^-1 A

and `res_diag` is its diagonal, stored per cell in each map_T*.npz. It answers one question per
cell: what fraction of this cell's posterior value came from the DATA rather than from the
prior? R_ii = 1 means the cell is fully determined by rays crossing it; R_ii = 0 means the map
is simply showing the prior back. tr(R) = sum(res_diag) is the effective number of parameters
the data actually constrains.

WHY IT MATTERS HERE. `unc_s` -- the posterior slowness standard deviation, the other quantity
in the npz -- does NOT carry this information. It is bounded above by the prior width, so as the
data stop constraining a cell, unc_s quietly relaxes toward the prior std instead of blowing up.
A cell that is 2% data and 98% prior can therefore report a perfectly respectable uncertainty.
Since the Vs inversion weights each period by its sigma, an unconstrained long-period map cell
enters the depth inversion with the SAME authority as a well-resolved short-period one. That is
the failure the trim addresses, and it is invisible to any check based on sigma or on misfit.

Outputs a three-panel figure and the numbers behind the T <= 5.12 s cut used for the trimmed arm.
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
MAPS = (f"{EHM}/hautesorne/tomo/1_velocity_maps/1_production/"
        f"tspws_group_blanket_dx0.5_prod3_k2/production/fund")
R_FRAC = 0.05          # the threshold used for the trimmed arm
GVL = [(41, 21), (41, 22), (42, 21), (42, 22)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=f"{EHM}/hautesorne/tomo/2_vs_depth_inversion/manuscript_comparison_2026-08/3_diagnostics")
    a = ap.parse_args()
    T, RD, US, MK, VEL = [], [], [], [], []
    for f in sorted(glob.glob(f"{MAPS}/map_T*.npz")):
        z = np.load(f, allow_pickle=True)
        T.append(float(z["period"])); RD.append(z["res_diag"]); US.append(z["unc_s"])
        MK.append(z["mask"].astype(bool)); VEL.append(z["vel"])
    T = np.array(T); RD = np.stack(RD); US = np.stack(US); MK = np.stack(MK)
    cov = MK.any(axis=0)
    print(f"{len(T)} periods, {cov.sum()} covered cells\n")

    trR = np.array([RD[i][cov].sum() for i in range(len(T))])
    med = np.array([np.median(RD[i][cov]) for i in range(len(T))])
    mx = np.array([RD[i][cov].max() for i in range(len(T))])
    umed = np.array([np.median(US[i][cov]) for i in range(len(T))])
    frac_ok = np.array([np.mean(RD[i][cov] >= R_FRAC) for i in range(len(T))])

    print(" T[s]   tr(R)   median res_diag   max    cells>=0.05   median unc_s")
    for i in range(len(T)):
        print(f"{T[i]:5.2f}  {trR[i]:6.1f}   {med[i]:12.4f}  {mx[i]:6.3f}   "
              f"{100*frac_ok[i]:8.1f}%   {umed[i]:.5f}")

    # where the T <= 5.12 s cut comes from: the GVL-1 cells' own reliable band
    print(f"\nthe cut used for the trimmed arm (res_diag >= {R_FRAC} at the GVL-1 cells):")
    for ix, iy in GVL:
        ok = [T[i] for i in range(len(T)) if MK[i][ix, iy] and RD[i][ix, iy] >= R_FRAC]
        print(f"  cell ({ix},{iy}): {min(ok):.2f} - {max(ok):.2f} s  ({len(ok)} periods kept "
              f"of {int(MK[:, ix, iy].sum())})")
    lastT = max(max(T[i] for i in range(len(T))
                    if MK[i][ix, iy] and RD[i][ix, iy] >= R_FRAC) for ix, iy in GVL)
    print(f"  -> longest period surviving at any GVL-1 cell: {lastT:.2f} s")
    kept = T <= lastT
    print(f"  -> the trimmed arm keeps {kept.sum()} of {len(T)} periods; the {(~kept).sum()} "
          f"dropped carry tr(R) {trR[~kept].min():.1f}-{trR[~kept].max():.1f} "
          f"of {cov.sum()} cells")

    fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.0))
    ax = axs[0]
    ax.plot(T, med, "o-", color="tab:red", lw=2, label="median over covered cells")
    ax.plot(T, mx, "^--", color="tab:red", lw=1, ms=4, alpha=0.7, label="best cell")
    ax.axhline(R_FRAC, color="k", ls="--", lw=1.4, label=f"trim threshold {R_FRAC}")
    ax.axvspan(lastT, T.max(), color="tab:red", alpha=0.10)
    ax.set_yscale("log"); ax.set_xlabel("period [s]"); ax.set_ylabel("res_diag")
    ax.set_title("how much of each cell comes from the DATA\n(1 = fully resolved, 0 = prior)",
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    ax = axs[1]
    ax.plot(T, trR, "o-", color="tab:blue", lw=2)
    ax.axvspan(lastT, T.max(), color="tab:red", alpha=0.10)
    ax.set_xlabel("period [s]")
    ax.set_ylabel(f"tr(R) — effective resolved parameters (of {cov.sum()} cells)")
    ax.set_title("the whole map's information content\ncollapses with period", fontsize=10)
    ax.grid(alpha=0.3)

    ax = axs[2]
    a1 = ax.plot(T, med / med.max(), "o-", color="tab:red", lw=2,
                 label="res_diag (normalised)")
    ax.set_xlabel("period [s]"); ax.set_ylabel("res_diag / max", color="tab:red")
    ax.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax.twinx()
    a2 = ax2.plot(T, umed, "s-", color="tab:green", lw=2, label="unc_s (the reported sigma)")
    ax2.set_ylabel("median unc_s [s/km]", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    ax.axvspan(lastT, T.max(), color="tab:red", alpha=0.10)
    ax.set_title("resolution collapses; the reported sigma does not\n"
                 "— so the Vs inversion cannot tell", fontsize=10)
    ax.legend(a1 + a2, [l.get_label() for l in a1 + a2], fontsize=8, loc="center left")
    ax.grid(alpha=0.3)
    print(f"\nover {T.min():.2f}-{T.max():.2f} s: median res_diag goes {med.max():.4f} -> "
          f"{med.min():.5f} (tr(R) {trR.max():.0f} -> {trR.min():.1f} of {cov.sum()} cells) "
          f"while median unc_s changes only {umed.max()/umed.min():.2f}x")
    fig.suptitle("res_diag — the resolution-matrix diagonal, and why the long periods were cut",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(a.out, "res_diag_explained.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); print("wrote", p)


if __name__ == "__main__":
    main()
