#!/usr/bin/env python3
"""GVL-1 interface PROBABILITY P(z) vs the well stratigraphy — the manuscript's own statistic.

The earlier test (gvl1_stratigraphy_compare.py) located peaks of |dVs/dz| on the posterior
MEDIAN profile. That is a weaker statistic than the one the manuscript uses in its Fig 6b:
the interface probability accumulated over the whole trans-D ENSEMBLE, i.e. the histogram of
per-model layer boundaries. A median profile has already smoothed away boundaries whose depth
varies between models, so it under-reports exactly the structure the ensemble statistic is
designed to expose. This script computes the ensemble statistic and tests it properly.

P(z) = histogram of `iface_depths` (every layer boundary of every posterior model, pooled
over the cells), normalised to a probability per depth bin.

TWO TESTS, because "the peaks look like they line up" is not a result:

  A. ENRICHMENT. mean P(z) within +-TOL of a mapped top, divided by mean P(z) elsewhere.
     >1 means boundaries concentrate at the tops. Significance from a CIRCULAR-SHIFT null:
     the whole set of tops is shifted by a random offset and wrapped in depth, 10k times.
     Shifting preserves the SPACING of the tops, so the null keeps the fact that real
     stratigraphic contacts are irregularly spaced -- a uniform-random null would not, and
     would make alignment look easier than it is.

  B. PEAK DISTANCE. distance from each top to the nearest peak of P(z), against the same null.

Both are reported per combo, so the group-only vs phase-bearing question is answered on the
same footing as the velocity comparison.

Requires runs made with --save-ensemble (the default npz keeps only percentiles).

Usage:
  python gvl1_interface_probability.py --tag test_2026-08-07_gvl1_iso_combos_ens
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
COMBOS = ["R0g", "R0gL0g", "R0pL0p", "R0gL0gR0pL0p"]
CCOL = {"R0g": "tab:purple", "R0gL0g": "tab:blue", "R0pL0p": "tab:orange",
        "R0gL0gR0pL0p": "tab:green"}
KEY_TOPS = ["Muschelkalk", "Buntsandstein", "Permo-Carboniferous", "Crystalline Basement"]
DZ = 0.05          # depth bin for P(z) [km]
TOL = 0.15         # a boundary counts as "at" a top within +-TOL [km]


STRAT_CSV = ("/Users/genevievesavard/Codes/extract_higher_modes/Projects/hautesorne/tomo/"
             "2_vs_depth_inversion/gvl1_stratigraphy_groups.csv")


def _read_strat():
    """Groups sheet as a DataFrame with top_km/base_km.

    Reads the cached CSV when present. The xlsx needs openpyxl, which is installed in the
    das-ambient-noise env but NOT in bayesbay_dev -- and this script needs disba, which is
    only in bayesbay_dev. The cache is what lets one script satisfy both.
    """
    import os
    if os.path.exists(STRAT_CSV):
        g = pd.read_csv(STRAT_CSV)
    else:
        g = pd.read_excel(STRAT, sheet_name="Groups")
    g = g.rename(columns={"MD Top [m]": "top_m", "MD Base [m]": "base_m"})
    g["top_km"] = g.top_m / 1000.0
    g["base_km"] = g.base_m / 1000.0
    return g


def strat():
    return _read_strat()


def enrichment(P, centres, tops, tol):
    near = np.zeros(len(centres), bool)
    for t in tops:
        near |= np.abs(centres - t) <= tol
    if not near.any() or near.all():
        return np.nan
    return float(P[near].mean() / P[~near].mean())


def peak_dist(P, centres, tops, topn=6):
    idx = [i for i in range(1, len(P) - 1) if P[i] >= P[i - 1] and P[i] > P[i + 1]]
    if not idx:
        return np.nan
    idx = sorted(idx, key=lambda i: -P[i])[:topn]
    pz = centres[idx]
    return float(np.median([np.min(np.abs(pz - t)) for t in tops]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="test_2026-08-07_gvl1_iso_combos_ens")
    ap.add_argument("--n-null", type=int, default=10000)
    ap.add_argument("--zmin", type=float, default=0.5,
                    help="restrict P(z) AND the null shifts to depths below this. The "
                         "near-surface peak (0.1-0.3 km) dominates P(z) and is not one of the "
                         "four key tops; leaving it in lets a SHIFTED top-set land on it and "
                         "score a high enrichment the real tops (shallowest 1.07 km) can never "
                         "reach, biasing the null upward and the test toward a false negative.")
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    S = strat()
    key = S[S.Group.isin(KEY_TOPS)].top_km.values
    root = f"{EHM}/hautesorne/tomo/2_vs_depth_inversion/tests/{a.tag}"

    rows, curves = [], {}
    for combo in COMBOS:
        files = sorted(glob.glob(f"{root}/GVL1_cell_*/{combo}/bayhunter_result.npz"))
        ifc, nmod, ncell = [], 0, 0
        zmax = 8.0
        for f in files:
            z = np.load(f, allow_pickle=True)
            if "iface_depths" not in z:
                continue
            ifc.append(np.asarray(z["iface_depths"], float))
            nmod += int(z["n_models"]); ncell += 1
            zmax = float(z["depth"].max())
        if not ifc:
            print(f"  {combo}: no ensemble npz found (need --save-ensemble)"); continue
        ifc = np.concatenate(ifc)
        ifc = ifc[ifc >= a.zmin]
        edges = np.arange(a.zmin, zmax + DZ, DZ)
        centres = 0.5 * (edges[:-1] + edges[1:])
        H, _ = np.histogram(ifc, bins=edges)
        P = H / H.sum()
        curves[combo] = (centres, P)

        obs_e = enrichment(P, centres, key, TOL)
        obs_d = peak_dist(P, centres, key)
        ne = np.empty(a.n_null); nd = np.empty(a.n_null)
        for k in range(a.n_null):
            # circular shift keeps the irregular SPACING of the real tops
            span = zmax - a.zmin
            sh = a.zmin + ((key - a.zmin + rng.uniform(0, span)) % span)
            ne[k] = enrichment(P, centres, sh, TOL)
            nd[k] = peak_dist(P, centres, sh)
        rows.append(dict(combo=combo, n_cells=ncell, n_models=nmod, n_ifaces=len(ifc),
                         enrichment=round(obs_e, 3),
                         p_enrich=round(float(np.mean(ne >= obs_e)), 4),
                         peak_dist_km=round(obs_d, 3),
                         p_peakdist=round(float(np.mean(nd <= obs_d)), 4)))
        print(f"  {combo:<14} enrich {obs_e:.2f} (p={np.mean(ne>=obs_e):.3f})   "
              f"peak-dist {obs_d:.3f} km (p={np.mean(nd<=obs_d):.3f})   "
              f"{len(ifc):,} boundaries from {ncell} cells")

    if not rows:
        raise SystemExit("nothing to analyse")
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(root, "interface_probability.csv"), index=False)

    fig, axs = plt.subplots(1, 1 + len(COMBOS), figsize=(4.2 + 3.6 * len(COMBOS), 8),
                            sharey=True,
                            gridspec_kw={"width_ratios": [0.45] + [1] * len(COMBOS)})
    ax = axs[0]
    for _, g in S.iterrows():
        ax.axhspan(g.top_km, min(g.base_km, 8.0), color=g["Hex Color"], alpha=0.85)
        if g.base_km - g.top_km > 0.2:
            ax.text(0.5, (g.top_km + min(g.base_km, 8.0)) / 2, g.Group, ha="center",
                    va="center", fontsize=7)
    ax.set_xticks([]); ax.set_xlim(0, 1); ax.set_ylabel("depth [km]")
    ax.set_title("GVL-1 groups", fontsize=9.5)
    for i, combo in enumerate(COMBOS):
        ax = axs[1 + i]
        if combo not in curves:
            ax.axis("off"); continue
        c, P = curves[combo]
        ax.fill_betweenx(c, 0, P, color=CCOL[combo], alpha=0.55)
        ax.plot(P, c, "-", color=CCOL[combo], lw=1.6)
        for t in key:
            ax.axhline(t, color="0.3", ls="--", lw=1.1)
            ax.axhspan(t - TOL, t + TOL, color="0.5", alpha=0.12)
        r = D[D.combo == combo].iloc[0]
        ax.set_title(f"{combo}\nenrichment {r.enrichment:.2f} (p={r.p_enrich:.3f})",
                     fontsize=9.5)
        ax.set_xlabel("interface probability / bin")
        ax.grid(alpha=0.3)
    axs[0].set_ylim(8.0, 0)
    [ax.set_ylim(8.0, a.zmin) for ax in axs[1:]]
    fig.suptitle("GVL-1: ensemble interface probability vs stratigraphic tops "
                 "(dashed = Muschelkalk / Buntsandstein / Permo-Carb / Basement)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(root, "interface_probability.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    print("wrote", p)
    print("wrote", os.path.join(root, "interface_probability.csv"))


if __name__ == "__main__":
    main()
