#!/usr/bin/env python3
"""Summary of the continuous-zeta RADIAL combo runs vs their ISOTROPIC counterparts.

For each well and each Love+Rayleigh combo (R0gL0g, R0pL0p, R0gL0gR0pL0p), compares the
2026-08-07 radial CZ run against the 2026-08-06 isotropic run of the SAME combo:

  * gamma(z) posterior (median + 68/95% bands) with the sensitivity caveat: gamma is only
    meaningful where BOTH Love and Rayleigh kernels live (Riehen: ~<2 km; see
    combo_sensitivity.py in the isotropic test dir)
  * Vs(z): radial (Vsv) vs isotropic median -- does letting zeta float move Vs?
  * per-target misfits: does zeta absorb the Love-vs-Rayleigh data conflict?
  * convergence: chain_disagree(Vs) radial vs isotropic (the calibrated gate)

Outputs: per-well radial_vs_iso.png + one radial_summary.csv beside the well dirs.

Interpretation guard (riehen-love-r0-contamination): a Love-leak CONTROL faked gamma=+0.14,
and the null gate on clean synthetics gave P(gamma!=0)=0.02-0.11 -- treat |gamma| below
~0.15 with skepticism, and gamma outside the mutual Love+Rayleigh sensitivity depth as
prior noise.
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
COMBOS = ["R0gL0g", "R0pL0p", "R0gL0gR0pL0p"]
CCOL = {"R0gL0g": "tab:blue", "R0pL0p": "tab:orange", "R0gL0gR0pL0p": "tab:green"}
TARGETS = ["fund", "love", "fund_phase", "love_phase"]


_ZFALLBACK = {"riehen": 2.4, "aargau": 2.6, "hautesorne": 5.6}


def _zcut(iso_wdir, net, combo):
    """Union-kernel reach for this combo, from combo_sensitivity.csv beside the iso runs."""
    f = os.path.join(os.path.dirname(iso_wdir), "combo_sensitivity.csv")
    if os.path.exists(f):
        t = pd.read_csv(f)
        t = t[t.combo == combo]
        if len(t) and np.isfinite(t.z_sens_max.iloc[0]):
            return float(t.z_sens_max.median())
    return _ZFALLBACK.get(net, 2.6)


def parse_log(log):
    txt = open(log).read()
    m = re.search(r"=== BayHunter \(subprocess\) ===.*?misfit \{(.*?)\}", txt, re.S)
    return ({k: float(v) for k, v in re.findall(r"'([a-z_]+)'\)?: ([\d.]+)", m.group(1))}
            if m else {})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--radial-tag", default="test_2026-08-07_radial_cz_combos")
    ap.add_argument("--iso-tag", default="test_2026-08-06_waveset_combos")
    a = ap.parse_args()

    rwells = sorted(glob.glob(f"{EHM}/*/tomo/2_vs_depth_inversion/tests/{a.radial_tag}/*_cell_*"))
    rows = []
    for wdir in rwells:
        net = wdir.split("/Projects/")[1].split("/")[0]
        wellbase = os.path.basename(wdir)
        well = wellbase.rsplit("_cell_", 1)[0]
        iso_wdir = wdir.replace(a.radial_tag, a.iso_tag)

        fig, axs = plt.subplots(1, 4, figsize=(17.5, 6.2),
                                gridspec_kw={"width_ratios": [1.15, 1, 1, 1]})
        drew = False
        for ci, combo in enumerate(COMBOS):
            rnpz = os.path.join(wdir, combo, "bayhunter_result.npz")
            inpz = os.path.join(iso_wdir, combo, "bayhunter_result.npz")
            if not os.path.exists(rnpz):
                continue
            drew = True
            zr = np.load(rnpz, allow_pickle=True)
            zi = np.load(inpz, allow_pickle=True) if os.path.exists(inpz) else None
            d = zr["depth"]

            # gamma(z)
            ax = axs[0]
            ax.plot(zr["gamma_median"], d, "-", color=CCOL[combo], lw=2.2, label=combo)
            ax.fill_betweenx(d, zr["gamma_p16"], zr["gamma_p84"], color=CCOL[combo], alpha=0.18)

            # Vs radial vs isotropic
            ax = axs[1 + ci]
            ax.plot(zr["vs_median"], d, "-", color="crimson", lw=2.2, label="radial (Vsv)")
            ax.fill_betweenx(d, zr["vs_p16"], zr["vs_p84"], color="crimson", alpha=0.15)
            if zi is not None:
                ax.plot(zi["vs_median"], d, "-", color="0.25", lw=2.0, label="isotropic")
                ax.fill_betweenx(d, zi["vs_p16"], zi["vs_p84"], color="0.25", alpha=0.12)
            ax.set_ylim(float(d.max()), 0); ax.grid(alpha=0.3)
            ax.set_xlabel("Vs [km/s]")
            ax.set_title(combo, fontsize=10)
            if ci == 0:
                ax.legend(fontsize=8)

            mis_r = parse_log(os.path.join(wdir, combo, "run.log"))
            mis_i = parse_log(os.path.join(iso_wdir, combo, "run.log")) if os.path.exists(
                os.path.join(iso_wdir, combo, "run.log")) else {}
            r = dict(net=net, well=well, combo=combo,
                     chain_disagree_radial=round(float(zr["chain_disagree"]), 3),
                     chain_disagree_iso=(round(float(zi["chain_disagree"]), 3)
                                         if zi is not None else np.nan),
                     frac_ok_radial=float(zr["frac_chains_ok"]),
                     frac_ok_iso=(float(zi["frac_chains_ok"]) if zi is not None else np.nan),
                     dvs_radial_iso=(round(float(np.mean(np.abs(
                         zr["vs_median"] - zi["vs_median"]))), 3) if zi is not None else np.nan))
            # gamma band = the combo's own union-kernel reach, read from the isotropic
            # tree's combo_sensitivity.csv when present. Hardcoding it per network was wrong:
            # HS Love reaches ~5.6 km (8.6 s curves) vs Riehen ~2.4 km, and a fixed 2.6 km
            # would have discarded most of the GVL-1 resolved range.
            zcut = _zcut(iso_wdir, net, combo)
            m = d <= zcut
            r["gamma_med_shallow"] = round(float(np.nanmedian(zr["gamma_median"][m])), 3)
            r["gamma_p16_shallow"] = round(float(np.nanmedian(zr["gamma_p16"][m])), 3)
            r["gamma_p84_shallow"] = round(float(np.nanmedian(zr["gamma_p84"][m])), 3)
            for t in TARGETS:
                if t in mis_r:
                    r[f"mis_{t}_radial"] = round(mis_r[t], 2)
                if t in mis_i:
                    r[f"mis_{t}_iso"] = round(mis_i[t], 2)
            rows.append(r)

        if not drew:
            plt.close(fig)
            continue
        ax = axs[0]
        ax.axvline(0, color="k", lw=1)
        zcut = _zcut(iso_wdir, net, "R0gL0g")
        ax.axhspan(zcut, float(d.max()), color="0.5", alpha=0.18)
        ax.text(0.02, 0.98, f"below {zcut:g} km:\noutside mutual\nL+R sensitivity",
                transform=ax.transAxes, fontsize=8, va="top", color="0.35")
        ax.set_ylim(float(d.max()), 0); ax.grid(alpha=0.3)
        ax.set_xlabel("gamma = (Vsh-Vsv)/Vsv"); ax.set_ylabel("depth [km]")
        ax.set_title("radial anisotropy posterior\n(band = 68%)", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        fig.suptitle(f"{net} — {well}: continuous-zeta radial vs isotropic "
                     f"(BayHunter, 24 chains, prior zeta [-0.35,0.35])",
                     fontsize=12.5, fontweight="bold")
        fig.tight_layout()
        p = os.path.join(wdir, "radial_vs_iso.png")
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("wrote", p)

    D = pd.DataFrame(rows)
    if len(D):
        csv = os.path.join(os.path.dirname(rwells[0]), "radial_summary.csv")
        D.to_csv(csv, index=False)
        print("wrote %s  (%d rows)" % (csv, len(D)))


if __name__ == "__main__":
    main()
