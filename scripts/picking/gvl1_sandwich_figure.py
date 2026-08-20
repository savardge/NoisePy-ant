#!/usr/bin/env python3
"""Visualise the basement 'low-high sandwich' at the GVL-1 cells, per input combo.

Left block: Vs(z) colour strips, one column per cell, drawn with the SAME convention as the
manuscript's Fig 8 cross-sections (red = slow, blue = fast, 1.5-4.0 km/s) so a column here is
directly comparable to a pixel column there. Alternating red/blue banding below the basement
top IS the sandwich.

Right block: the same models as line profiles, with velocity reversals below the basement top
marked, and the 68% posterior band. A reversal is a sign change of dVs/dz, i.e. a local
maximum or minimum -- the thing that makes a sandwich.

Colour choice: red-slow / blue-fast is the manuscript's own convention for these sections
(elsewhere in this project absolute-velocity maps use inferno).
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
COMBOS = ["R0g", "R0gL0g", "R0pL0p", "R0gL0gR0pL0p"]
LAB = {"R0g": "R0g\n(manuscript config)", "R0gL0g": "R0g+L0g",
       "R0pL0p": "R0p+L0p (phase)", "R0gL0gR0pL0p": "all four"}
CELLS = ["41_21", "42_21", "41_22", "42_22"]
DIST = {"41_21": 0.27, "42_21": 0.32, "41_22": 0.40, "42_22": 0.44}
ZTOP = 2.24          # basement top at GVL-1


def reversals(d, v, zmin):
    """Depths where dVs/dz changes sign below zmin.

    Flat runs (dVs/dz exactly 0) are COLLAPSED before counting: a trans-D layer spanning
    several depth nodes produces a zero-gradient run, and treating that as a break would
    split one reversal into none. Counting without collapsing under-reports ~2x."""
    m = d >= zmin
    dd, vv = d[m], v[m]
    g = np.sign(np.diff(vv))
    keep = np.flatnonzero(g != 0)
    if keep.size < 2:
        return np.array([]), np.array([])
    gs = g[keep]
    turn = keep[1:][np.diff(gs) != 0]
    return dd[turn], vv[turn]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="test_2026-08-07_gvl1_iso_combos_ens")
    a = ap.parse_args()
    B = f"{EHM}/hautesorne/tomo/2_vs_depth_inversion/tests/{a.tag}/"
    S = pd.read_excel(STRAT, sheet_name="Groups").rename(
        columns={"MD Top [m]": "top_m", "MD Base [m]": "base_m"})
    S["top_km"] = S.top_m / 1000.0; S["base_km"] = S.base_m / 1000.0

    fig = plt.figure(figsize=(19, 9))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.30, 2.5, 1.9], wspace=0.22)
    axS = fig.add_subplot(gs[0])
    for _, g in S.iterrows():
        axS.axhspan(g.top_km, min(g.base_km, 8.0), color=g["Hex Color"], alpha=0.9)
        if g.base_km - g.top_km > 0.3:
            axS.text(0.5, (g.top_km + min(g.base_km, 8.0)) / 2, g.Group, ha="center",
                     va="center", fontsize=7, rotation=90)
    axS.set_ylim(8, 0); axS.set_xticks([]); axS.set_ylabel("depth [km]")
    axS.set_title("GVL-1", fontsize=9)
    axS.axhline(ZTOP, color="k", lw=1.6)

    gsm = gs[1].subgridspec(1, len(COMBOS), wspace=0.10)
    for ci, combo in enumerate(COMBOS):
        ax = fig.add_subplot(gsm[ci])
        M = []
        for cell in CELLS:
            f = f"{B}GVL1_cell_{cell}/{combo}/bayhunter_result.npz"
            z = np.load(f, allow_pickle=True)
            M.append(z["vs_median"]); d = z["depth"]
        M = np.array(M).T
        # shading='flat' wants C one smaller than BOTH edge arrays; d is the depth NODE
        # vector, so build depth edges too rather than silently dropping a row.
        de = np.concatenate([[d[0] - (d[1] - d[0]) / 2],
                             0.5 * (d[:-1] + d[1:]),
                             [d[-1] + (d[-1] - d[-2]) / 2]])
        im = ax.pcolormesh(np.arange(len(CELLS) + 1), de, M, cmap="RdYlBu",
                           vmin=1.5, vmax=4.0, shading="flat")
        ax.axhline(ZTOP, color="k", lw=1.6)
        ax.set_ylim(8, 0)
        ax.set_xticks(np.arange(len(CELLS)) + 0.5)
        ax.set_xticklabels([f"{DIST[c]:.2f}" for c in CELLS], fontsize=7)
        ax.set_title(LAB[combo], fontsize=9)
        if ci == 0:
            ax.set_ylabel("depth [km]")
        else:
            ax.set_yticklabels([])
        if ci == len(COMBOS) - 1:
            plt.colorbar(im, ax=ax, label="Vs [km/s]  (Fig-8 convention: red slow, blue fast)")
        ax.set_xlabel("km from well", fontsize=8)

    axP = fig.add_subplot(gs[2])
    for _, g in S.iterrows():
        axP.axhspan(g.top_km, min(g.base_km, 8.0), color=g["Hex Color"], alpha=0.10, zorder=0)
    col = dict(zip(COMBOS, ["tab:purple", "tab:blue", "tab:orange", "tab:green"]))
    txt = []
    for combo in COMBOS:
        z = np.load(f"{B}GVL1_cell_41_21/{combo}/bayhunter_result.npz", allow_pickle=True)
        d, v = z["depth"], z["vs_median"]
        axP.plot(v, d, "-", color=col[combo], lw=2.2, label=LAB[combo].replace("\n", " "))
        axP.fill_betweenx(d, z["vs_p16"], z["vs_p84"], color=col[combo], alpha=0.12)
        rz, rv = reversals(d, v, ZTOP)
        axP.plot(rv, rz, "o", color=col[combo], ms=6, mec="k", mew=0.7, zorder=5)
        txt.append(f"{LAB[combo].splitlines()[0]}: {len(rz)} reversals")
    axP.axhline(ZTOP, color="k", lw=1.6)
    axP.text(0.98, 0.02, "\n".join(txt) + "\n(markers = dVs/dz sign changes below 2.24 km)",
             transform=axP.transAxes, ha="right", va="bottom", fontsize=8,
             bbox=dict(fc="white", alpha=0.85, ec="0.7"))
    axP.set_ylim(8, 0); axP.set_xlim(1.2, 4.0); axP.grid(alpha=0.3)
    axP.set_xlabel("Vs [km/s]"); axP.set_title("cell (41,21), 0.27 km from GVL-1", fontsize=9)
    axP.legend(fontsize=8, loc="upper left")
    fig.suptitle("GVL-1 basement structure by input combination — the 'low-high sandwich' is "
                 "GROUP-specific and dissolves with phase\n"
                 "black line = drilled basement top (2.24 km)",
                 fontsize=13, fontweight="bold")
    out = os.path.join(B, "ms_figure_clones", "sandwich_by_combo.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
