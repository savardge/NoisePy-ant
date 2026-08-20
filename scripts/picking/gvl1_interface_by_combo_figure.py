#!/usr/bin/env python3
"""GVL-1 cells by input combo, rendered as ensemble INTERFACE PROBABILITY.

The companion of gvl1_sandwich_figure.py: identical layout and cells, but each column is
P(z) -- the histogram of every layer boundary of every posterior model in that cell,
normalised to sum 1 down the column -- instead of Vs. Where the Vs version answers "what
velocity does this combo want", this one answers "where does this combo want BOUNDARIES",
which is the manuscript's Fig-6b statistic and the thing its stratigraphic-tie claims rest on.

Colour scale is clipped at the p99 of values BELOW 0.5 km depth. The near-surface boundary
peak is ~10x anything deeper; left unclipped it saturates the panel and hides everything at
basement level. The shallow band is therefore deliberately over-saturated here.

Right panel: P(z) profiles at the nearest cell for all combos, with the four key stratigraphic
tops marked, so trim/combo differences at the Muschelkalk, Buntsandstein, Permo-Carboniferous
and basement contacts can be read directly.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
STRAT = "/Users/genevievesavard/Data/hautesorne/stratigraphy/GVL-1_Stratigraphy_v2.xlsx"
STRAT_CSV = (f"{EHM}/hautesorne/tomo/2_vs_depth_inversion/gvl1_stratigraphy_groups.csv")
COMBOS = ["R0g", "R0gL0g", "R0pL0p", "R0gL0gR0pL0p"]
LAB = {"R0g": "R0g\n(manuscript config)", "R0gL0g": "R0g+L0g",
       "R0pL0p": "R0p+L0p (phase)", "R0gL0gR0pL0p": "all four"}
CELLS = ["41_21", "42_21", "41_22", "42_22"]
DIST = {"41_21": 0.27, "42_21": 0.32, "41_22": 0.40, "42_22": 0.44}
KEY = {"Muschelkalk": 1.074, "Buntsandstein": 1.420,
       "Permo-Carboniferous": 1.474, "Crystalline Basement": 2.240}
DZ = 0.05
ZMAX = 8.0


def strat():
    if os.path.exists(STRAT_CSV):
        g = pd.read_csv(STRAT_CSV)
    else:
        g = pd.read_excel(STRAT, sheet_name="Groups")
    g = g.rename(columns={"MD Top [m]": "top_m", "MD Base [m]": "base_m"})
    g["top_km"] = g.top_m / 1000.0
    g["base_km"] = g.base_m / 1000.0
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="test_2026-08-07_gvl1_iso_combos_ens")
    a = ap.parse_args()
    B = f"{EHM}/hautesorne/tomo/2_vs_depth_inversion/tests/{a.tag}/"
    S = strat()
    edges = np.arange(0, ZMAX + DZ, DZ)
    ctr = 0.5 * (edges[:-1] + edges[1:])

    P = {}
    for combo in COMBOS:
        for cell in CELLS:
            f = f"{B}GVL1_cell_{cell}/{combo}/bayhunter_result.npz"
            if not os.path.exists(f):
                continue
            z = np.load(f, allow_pickle=True)
            h, _ = np.histogram(np.asarray(z["iface_depths"], float), bins=edges)
            P[(combo, cell)] = h / max(h.sum(), 1)
    if not P:
        raise SystemExit("no ensemble runs found (need --save-ensemble)")
    deep = ctr >= 0.5
    vmax = np.percentile(np.concatenate([v[deep] for v in P.values()]), 99)

    fig = plt.figure(figsize=(19, 9))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.30, 2.5, 1.9], wspace=0.22)
    axS = fig.add_subplot(gs[0])
    for _, g in S.iterrows():
        axS.axhspan(g.top_km, min(g.base_km, ZMAX), color=g["Hex Color"], alpha=0.9)
        if g.base_km - g.top_km > 0.3:
            axS.text(0.5, (g.top_km + min(g.base_km, ZMAX)) / 2, g.Group, ha="center",
                     va="center", fontsize=7, rotation=90)
    axS.set_ylim(ZMAX, 0); axS.set_xticks([]); axS.set_ylabel("depth [km]")
    axS.set_title("GVL-1", fontsize=9)
    axS.axhline(KEY["Crystalline Basement"], color="k", lw=1.6)

    gsm = gs[1].subgridspec(1, len(COMBOS), wspace=0.10)
    ce = np.concatenate([[edges[0]], 0.5 * (ctr[:-1] + ctr[1:]), [edges[-1]]])
    for ci, combo in enumerate(COMBOS):
        ax = fig.add_subplot(gsm[ci])
        M = np.array([P.get((combo, c), np.full(len(ctr), np.nan)) for c in CELLS]).T
        im = ax.pcolormesh(np.arange(len(CELLS) + 1), ce, M, cmap="inferno",
                           vmin=0, vmax=vmax, shading="flat")
        for zt in KEY.values():
            ax.axhline(zt, color="w", ls=":", lw=1.0, alpha=0.8)
        ax.axhline(KEY["Crystalline Basement"], color="w", lw=1.6)
        ax.set_ylim(ZMAX, 0)
        ax.set_xticks(np.arange(len(CELLS)) + 0.5)
        ax.set_xticklabels([f"{DIST[c]:.2f}" for c in CELLS], fontsize=7)
        ax.set_title(LAB[combo], fontsize=9)
        ax.set_xlabel("km from well", fontsize=8)
        if ci == 0:
            ax.set_ylabel("depth [km]")
        else:
            ax.set_yticklabels([])
        if ci == len(COMBOS) - 1:
            plt.colorbar(im, ax=ax, label=f"P(interface) / {DZ*1000:.0f} m bin "
                                          f"(clipped at p99 below 0.5 km)")

    axP = fig.add_subplot(gs[2])
    for _, g in S.iterrows():
        axP.axhspan(g.top_km, min(g.base_km, ZMAX), color=g["Hex Color"], alpha=0.10, zorder=0)
    col = dict(zip(COMBOS, ["tab:purple", "tab:blue", "tab:orange", "tab:green"]))
    for combo in COMBOS:
        v = P.get((combo, "41_21"))
        if v is None:
            continue
        axP.plot(v, ctr, "-", color=col[combo], lw=2.0,
                 label=LAB[combo].replace("\n", " "))
    for nm, zt in KEY.items():
        axP.axhline(zt, color="0.35", ls="--", lw=1.0)
        axP.text(axP.get_xlim()[1], zt, " " + nm, fontsize=7, va="center", color="0.35")
    axP.set_ylim(ZMAX, 0); axP.grid(alpha=0.3)
    axP.set_xlabel(f"P(interface) / {DZ*1000:.0f} m bin")
    axP.set_title("cell (41,21), 0.27 km from GVL-1", fontsize=9)
    axP.legend(fontsize=8, loc="lower right")
    fig.suptitle("GVL-1 — ensemble INTERFACE PROBABILITY by input combination "
                 "(the Fig-6b statistic)\nwhite line = drilled basement top (2.24 km); "
                 "dotted = Muschelkalk / Buntsandstein / Permo-Carb tops",
                 fontsize=13, fontweight="bold")
    out = os.path.join(B, "ms_figure_clones", "interface_prob_by_combo.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
