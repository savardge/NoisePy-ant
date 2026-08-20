#!/usr/bin/env python3
"""GVL-1: posterior Vs(z) per waveset combo vs the WELL STRATIGRAPHY.

IMPORTANT — what the well actually provides. GVL-1 has NO sonic / Vp / Vs log in the data
tree; the only borehole information is a stratigraphic column (group and formation tops,
GVL-1_Stratigraphy_v2.xlsx). So this is NOT a velocity-vs-velocity correlation. Two things
can honestly be tested against tops:

  A. INTERFACE ALIGNMENT — do the peaks of |dVs/dz| fall at mapped group boundaries?
     Scored against a null: the same number of peaks placed at random depths (10k draws),
     giving a p-value for "closer to the tops than chance".
  B. LAYER VELOCITIES — mean Vs within each stratigraphic group, and whether the sequence
     increases with depth as the lithology implies (carbonate/evaporite over clastics over
     Permo-Carboniferous over crystalline basement).

Depth convention: the spreadsheet tops are MEASURED depth; the inversion depth grid is TVD
below surface. swisstopo gives 4041.5 m MD / 4005.9 m TVD for Glovelier-1, so MD is converted
to TVD by 0.9912 (tops move 9-20 m shallower -- below the 50 m depth grid, so no result
changes). Ground elevation is 494.2 m a.s.l.; the DEM interpolates 505.5 m at this point.

Usage:
  python gvl1_stratigraphy_compare.py                       # 6 km first pass
  python gvl1_stratigraphy_compare.py --iso-tag ..._z8 --radial-tag ..._z8
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
STRAT = "/Users/genevievesavard/Data/hautesorne/stratigraphy/GVL-1_Stratigraphy_v2.xlsx"
COMBOS = ["R0gL0g", "R0pL0p", "R0gL0gR0pL0p"]
CCOL = {"R0gL0g": "tab:blue", "R0pL0p": "tab:orange", "R0gL0gR0pL0p": "tab:green"}
# the boundaries a surface-wave inversion could plausibly see (major impedance steps)
KEY_TOPS = ["Muschelkalk", "Buntsandstein", "Permo-Carboniferous", "Crystalline Basement"]


# swisstopo Glovelier-1: 4041.5 m MD / 4005.9 m TVD -> the hole is near-vertical (0.9%
# difference over 4 km). The spreadsheet tops are MEASURED depth; the inversion depth axis is
# TVD, so convert. The shift is 9-20 m over the section of interest -- below the 50 m depth
# grid, so no conclusion moves, but the axes should still mean the same thing.
MD_TO_TVD = 4005.9 / 4041.5
GROUND_ELEV_M = 494.2          # swisstopo; the DEM interpolates 505.5 m here (+11.3 m)


def load_strat():
    g = pd.read_excel(STRAT, sheet_name="Groups")
    g = g.rename(columns={"MD Top [m]": "top_m", "MD Base [m]": "base_m"})
    g["top_km"] = g.top_m * MD_TO_TVD / 1000.0
    g["base_km"] = g.base_m * MD_TO_TVD / 1000.0
    return g


def grad_peaks(d, vs, zmax, min_prom=0.02):
    """Depths of local maxima of |dVs/dz|, with a minimum prominence in km/s per km."""
    g = np.abs(np.gradient(vs, d))
    pk = []
    for i in range(1, len(g) - 1):
        if d[i] > zmax:
            break
        if g[i] >= g[i - 1] and g[i] > g[i + 1] and g[i] > min_prom:
            pk.append((d[i], g[i]))
    return pk


def align_score(peaks, tops, rng, zmax, n_null=10000, topn=4):
    """Median distance from each top to its nearest STRONG gradient peak, + null p-value.

    Only the `topn` largest-|dVs/dz| peaks are used. Using every local maximum makes the
    test vacuous: a trans-D posterior median carries ~15-22 small peaks in a 6 km box, so a
    RANDOM peak already sits ~0.1 km from any given top and observed ~ null by construction.
    The strongest peaks are also the only ones a surface-wave inversion could claim as
    interfaces."""
    if not peaks:
        return np.nan, np.nan
    peaks = sorted(peaks, key=lambda t: -t[1])[:topn]
    pz = np.array([p[0] for p in peaks])
    obs = float(np.median([np.min(np.abs(pz - t)) for t in tops]))
    null = np.empty(n_null)
    for k in range(n_null):
        rp = rng.uniform(0, zmax, size=len(pz))
        null[k] = np.median([np.min(np.abs(rp - t)) for t in tops])
    return obs, float(np.mean(null <= obs))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--iso-tag", default="test_2026-08-07_gvl1_iso_combos")
    ap.add_argument("--radial-tag", default="test_2026-08-07_gvl1_radial_combos")
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    S = load_strat()
    tops = S.top_km.values[1:]                    # skip the 0-m surface
    key = S[S.Group.isin(KEY_TOPS)].top_km.values

    rows, prof = [], {}
    for tag, mode in ((a.iso_tag, "iso"), (a.radial_tag, "radial")):
        for wdir in sorted(glob.glob(f"{EHM}/hautesorne/tomo/2_vs_depth_inversion/tests/{tag}/GVL1_cell_*")):
            cell = os.path.basename(wdir).replace("GVL1_cell_", "")
            for combo in COMBOS:
                f = os.path.join(wdir, combo, "bayhunter_result.npz")
                if not os.path.exists(f):
                    continue
                z = np.load(f, allow_pickle=True)
                d, vs = z["depth"], z["vs_median"]
                prof.setdefault((mode, combo), []).append(vs)
                zmax = float(d.max())
                pk = grad_peaks(d, vs, zmax)
                obs, p = align_score(pk, key, rng, zmax)
                r = dict(mode=mode, cell=cell, combo=combo, n_peaks=len(pk),
                         z_strongest_km=round(float(max(pk, key=lambda t: t[1])[0]), 2) if pk else np.nan,
                         med_dist_km=round(obs, 3) if np.isfinite(obs) else np.nan,
                         p_vs_random=round(p, 3) if np.isfinite(p) else np.nan)
                for _, g in S.iterrows():
                    m = (d >= g.top_km) & (d < min(g.base_km, zmax))
                    if m.sum() >= 2:
                        r[f"Vs_{g.Group}"] = round(float(np.mean(vs[m])), 3)
                rows.append(r)
    D = pd.DataFrame(rows)
    outdir = f"{EHM}/hautesorne/tomo/2_vs_depth_inversion/tests/{a.iso_tag}"
    D.to_csv(os.path.join(outdir, "gvl1_stratigraphy_compare.csv"), index=False)

    # ---- figure: stratigraphic column + combo profiles (cell-averaged) ----
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 8.2), sharey=True,
                            gridspec_kw={"width_ratios": [0.5, 1.5, 1.5]})
    ax = axs[0]
    for _, g in S.iterrows():
        ax.axhspan(g.top_km, g.base_km, color=g["Hex Color"], alpha=0.85)
        if g.base_km - g.top_km > 0.12:
            ax.text(0.5, (g.top_km + g.base_km) / 2, g.Group, ha="center", va="center",
                    fontsize=7.5, rotation=0)
    ax.set_xticks([]); ax.set_ylabel("depth [km]")
    ax.set_title("GVL-1 stratigraphy\n(groups; NO velocity log exists)", fontsize=9.5)
    ax.set_xlim(0, 1)

    # depth axis follows the RUNS, not a constant: the 8 km re-runs were being clipped to
    # 6 km by a hardcoded limit, hiding a quarter of the profile.
    any_npz = glob.glob(f"{EHM}/hautesorne/tomo/2_vs_depth_inversion/tests/"
                        f"{a.iso_tag}/GVL1_cell_*/*/bayhunter_result.npz")
    zmax_fig = float(np.load(any_npz[0], allow_pickle=True)["depth"].max())
    for k, mode in enumerate(("iso", "radial")):
        ax = axs[1 + k]
        for g_ in S.itertuples():          # faint lithology bands behind the profiles
            ax.axhspan(g_.top_km, min(g_.base_km, zmax_fig), color=g_._6, alpha=0.13, zorder=0)
        for t in key:
            ax.axhline(t, color="0.35", ls="--", lw=1.1, zorder=1)
        for combo in COMBOS:
            P = prof.get((mode, combo))
            if not P:
                continue
            M = np.vstack(P)
            d = np.load(glob.glob(f"{EHM}/hautesorne/tomo/2_vs_depth_inversion/tests/"
                                  f"{a.iso_tag if mode=='iso' else a.radial_tag}/GVL1_cell_*/"
                                  f"{combo}/bayhunter_result.npz")[0],
                        allow_pickle=True)["depth"]
            ax.plot(M.mean(axis=0), d, "-", color=CCOL[combo], lw=2.2, label=combo)
            ax.fill_betweenx(d, M.min(axis=0), M.max(axis=0), color=CCOL[combo], alpha=0.15)
        ax.set_ylim(zmax_fig, 0); ax.grid(alpha=0.3, axis="x")
        ax.set_xlabel("Vs [km/s]")
        ax.set_title(f"{mode} — mean of 5 cells (band = cell spread)\n"
                     "dashed = Muschelkalk / Buntsandstein / Permo-Carb / Basement tops",
                     fontsize=9.5)
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("GVL-1: posterior Vs(z) by input combination vs well stratigraphy",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(outdir, "gvl1_vs_stratigraphy.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)

    print("\n=== interface alignment (|dVs/dz| peaks vs the 4 key tops) ===")
    print(D.groupby(["mode", "combo"])[["n_peaks", "med_dist_km", "p_vs_random"]]
          .mean().round(3).to_string())
    print("\n=== mean Vs per stratigraphic group (km/s, averaged over the 5 cells) ===")
    vcols = [c for c in D.columns if c.startswith("Vs_")]
    print(D.groupby(["mode", "combo"])[vcols].mean().round(2).T.to_string())
    print("\nwrote", os.path.join(outdir, "gvl1_stratigraphy_compare.csv"))


if __name__ == "__main__":
    main()
