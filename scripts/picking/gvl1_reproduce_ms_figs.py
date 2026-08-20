#!/usr/bin/env python3
"""Reproduce manuscript Figure 5 and Figure S10 from the CURRENT workflow, per input combo.

Purpose: decide whether the present pipeline converges better than the manuscript's depth
inversion, on figures that are directly superimposable on the originals.

  FIG 5 clone  -- for the 4 grid cells nearest GVL-1, per combo: posterior Vs density vs
                  depth, median / mean / mean+-std / minimum-misfit profiles, the interface
                  PROBABILITY histogram, and the stratigraphic column.
  FIG S10 clone -- per cell: (1) misfit distribution with median, (2) running cumulative mean
                  +- std of misfit vs sample index, (3) noise sigma distribution with the
                  retained-chain count, (4) posterior predicted dispersion curves over the
                  observed data.

Both need the model ENSEMBLE (`--save-ensemble`): posterior percentiles on a depth grid have
already lost the per-model layer boundaries and the per-model misfits.

HONEST DIFFERENCES from the originals, stated on the figures themselves:
  * panel (3) of the manuscript's S10 is captioned "log-likelihood distribution" but shows the
    NOISE parameter; this clone plots the noise parameter and says so.
  * the manuscript keeps 3-4 of 30 chains via a +-5%-of-best-likelihood rule. The current
    workflow reports `frac_chains_ok` from chain disagreement in Vs, which is the gate the
    convergence calibration selected; both numbers are printed so the comparison is explicit.
  * predicted curves here are forward-modelled with disba from a random subsample of the
    posterior ensemble, not stored per-model predictions.

Usage:
  python gvl1_reproduce_ms_figs.py --tag test_2026-08-07_gvl1_iso_combos_ens
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/genevievesavard/Codes/NoisePy-ant")
from noisepy import vs_inversion as vi                       # noqa: E402

EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
STRAT = "/Users/genevievesavard/Data/hautesorne/stratigraphy/GVL-1_Stratigraphy_v2.xlsx"
COMBOS = ["R0g", "R0gL0g", "R0pL0p", "R0gL0gR0pL0p"]
# the 4 cells nearest the well, matching the manuscript's Fig 5 (its 4 cells are 0.22-0.49 km)
CELLS4 = ["41_21", "42_21", "41_22", "42_22"]
DIST = {"41_21": 0.27, "42_21": 0.32, "41_22": 0.40, "42_22": 0.44, "41_20": 0.70}


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


def fig5(tag, combo, S, out):
    files = [f"{EHM}/hautesorne/tomo/2_vs_depth_inversion/tests/{tag}/GVL1_cell_{c}/{combo}/"
             "bayhunter_result.npz" for c in CELLS4]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        return None
    n = len(files)
    fig, axs = plt.subplots(1, 3 * n, figsize=(4.6 * n, 8.4), sharey=True,
                            gridspec_kw={"width_ratios": [1.5, 0.5, 0.32] * n})
    rows = []
    for i, f in enumerate(files):
        z = np.load(f, allow_pickle=True)
        d = z["depth"]; E = np.asarray(z["ens_vs"], float)
        cell = os.path.basename(os.path.dirname(os.path.dirname(f))).replace("GVL1_cell_", "")
        # --- Vs posterior density ---
        ax = axs[3 * i]
        vb = np.linspace(0.5, 4.2, 160)
        H = np.stack([np.histogram(E[:, k], bins=vb)[0] for k in range(len(d))], axis=1)
        ax.pcolormesh(0.5 * (vb[:-1] + vb[1:]), d, H.T, cmap="Reds", shading="auto")
        ax.plot(np.median(E, axis=0), d, "-", color="green", lw=1.8, label="Median")
        ax.plot(E.mean(axis=0), d, "-", color="blue", lw=1.5, label="Mean")
        sd = E.std(axis=0)
        ax.plot(E.mean(axis=0) - sd, d, ":", color="red", lw=1.0)
        ax.plot(E.mean(axis=0) + sd, d, ":", color="red", lw=1.0, label="Mean $\\pm$ std")
        if "ens_misfit" in z:
            _m = np.asarray(z["ens_misfit"], float).ravel()
            if _m.size and _m.size % len(E) == 0:
                tot = _m.reshape(len(E), -1)[:, -1]
                ax.plot(E[int(np.argmin(tot)), :], d, "-", color="red", lw=1.3,
                        label="Min misfit")
        ax.set_ylim(d.max(), 0); ax.set_xlim(0.5, 4.2)
        ax.set_title(f"{DIST.get(cell, float('nan')):.2f} km from well\n"
                     f"{int(z['n_models']):,} models", fontsize=9)
        ax.set_xlabel("Vs (km/s)")
        if i == 0:
            ax.set_ylabel("Depth (km)"); ax.legend(fontsize=6.5, loc="lower left")
        # --- interface probability ---
        ax = axs[3 * i + 1]
        ifd = np.asarray(z["iface_depths"], float)
        eb = np.arange(0, d.max() + 0.1, 0.1)
        h, _ = np.histogram(ifd, bins=eb)
        ax.barh(0.5 * (eb[:-1] + eb[1:]), h / max(h.sum(), 1), height=0.09,
                color="0.55")
        ax.set_xlabel("Interfaces", fontsize=8); ax.set_xticks([])
        # --- stratigraphy ---
        ax = axs[3 * i + 2]
        for _, g in S.iterrows():
            ax.axhspan(g.top_km, min(g.base_km, d.max()), color=g["Hex Color"], alpha=0.9)
        ax.set_xticks([]); ax.set_xlim(0, 1)
        rows.append(dict(combo=combo, cell=cell, n_models=int(z["n_models"]),
                         frac_ok=float(z["frac_chains_ok"]),
                         n_chains_kept=int(z["n_chains_kept"]),
                         n_chains_used=int(z["n_chains_used"]),
                         chain_disagree=round(float(z["chain_disagree"]), 3),
                         confidence=str(z["confidence"]),
                         med_sd_0_3km=round(float(sd[d <= 3].mean()), 3),
                         med_sd_3_8km=round(float(sd[d > 3].mean()), 3)))
    fig.suptitle(f"Fig-5 clone — {combo} — current workflow (BayHunter, 24 chains, "
                 f"150k+120k, 8 km box, prod3 k2 blanket, v1 periods)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out, f"fig5_clone_{combo}.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("  wrote", os.path.basename(p))
    return rows


def figS10(tag, combo, out, nsub=250):
    files = [f"{EHM}/hautesorne/tomo/2_vs_depth_inversion/tests/{tag}/GVL1_cell_{c}/{combo}/"
             "bayhunter_result.npz" for c in CELLS4]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        return
    fig, axs = plt.subplots(len(files), 4, figsize=(17, 3.3 * len(files)), squeeze=False)
    rng = np.random.default_rng(0)
    for i, f in enumerate(files):
        z = np.load(f, allow_pickle=True)
        cell = os.path.basename(os.path.dirname(os.path.dirname(f))).replace("GVL1_cell_", "")
        # ens_misfit is (n_models, n_targets+1) FLATTENED: per-target misfits then the total.
        # Reading it as a flat per-model sequence interleaves the targets, which made the
        # running-mean panel average unrelated series.
        _m = np.asarray(z.get("ens_misfit", []), float).ravel()
        _nm = int(z["n_models"])
        mis = _m.reshape(_nm, -1)[:, -1] if (_m.size and _m.size % _nm == 0) else _m
        ax = axs[i][0]
        if mis.size:
            ax.hist(mis, bins=60, color="darkblue")
            ax.axvline(np.median(mis), color="red", ls="--", lw=1.2,
                       label=f"median={np.median(mis):.4f}")
            ax.legend(fontsize=7)
        ax.set_title(f"Cell ({cell})  misfit distribution", fontsize=9)
        ax = axs[i][1]
        if mis.size:
            k = np.arange(1, mis.size + 1)
            rm = np.cumsum(mis) / k
            rs = np.sqrt(np.maximum(np.cumsum(mis ** 2) / k - rm ** 2, 0))
            ax.plot(k, rm, "-", color="navy", lw=1.2)
            ax.fill_between(k, rm - rs, rm + rs, color="slateblue", alpha=0.3)
        ax.set_title("running cumulative mean $\\pm$ std", fontsize=9)
        ax.set_xlabel("sample index")
        ax = axs[i][2]
        npo = np.asarray(z.get("noise_post", []), float).ravel()
        if npo.size:
            ax.hist(npo, bins=60, color="green")
            ax.axvline(np.median(npo), color="red", ls="--", lw=1.2)
        ax.set_title(f"noise $\\sigma$   (kept {int(z['n_chains_kept'])}/"
                     f"{int(z['n_chains_used'])} chains, disagree "
                     f"{float(z['chain_disagree']):.2f})", fontsize=9)
        ax = axs[i][3]
        E = np.asarray(z["ens_vs"], float); d = z["depth"]
        waves = [str(w) for w in np.atleast_1d(z["waves"])]
        idx = rng.choice(len(E), size=min(nsub, len(E)), replace=False)
        th = np.full(len(d), float(d[1] - d[0])); th[-1] = 100.0
        for w in waves:
            T = z[f"obsT_{w}"]; obs = z[f"obs_{w}"]
            dw, mode, meas = vi.curve_def(w)
            for j in idx[:120]:
                pr = vi.dispersion_velocity(th, E[j], T, mode, measure=meas, disba_wave=dw)
                ax.plot(T, pr, "-", color="tab:blue", lw=0.4, alpha=0.12)
            ax.plot(T, obs, "o", color="red", ms=2.6, label=f"observed {w}")
        ax.legend(fontsize=6.5); ax.set_xlabel("Period (s)"); ax.set_ylabel("U (km/s)")
        ax.set_title("posterior predicted vs observed", fontsize=9)
    fig.suptitle(f"Fig-S10 clone — {combo} — panel 3 is the NOISE parameter "
                 f"(the manuscript caption calls it log-likelihood)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out, f"figS10_clone_{combo}.png")
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print("  wrote", os.path.basename(p))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="test_2026-08-07_gvl1_iso_combos_ens")
    a = ap.parse_args()
    out = f"{EHM}/hautesorne/tomo/2_vs_depth_inversion/tests/{a.tag}/ms_figure_clones"
    os.makedirs(out, exist_ok=True)
    S = strat()
    allrows = []
    for combo in COMBOS:
        print(combo)
        r = fig5(a.tag, combo, S, out)
        if r:
            allrows += r
            figS10(a.tag, combo, out)
    if allrows:
        D = pd.DataFrame(allrows)
        D.to_csv(os.path.join(out, "convergence_summary.csv"), index=False)
        print("\n=== convergence, current workflow (4 cells nearest GVL-1) ===")
        print(D.groupby("combo")[["frac_ok", "chain_disagree", "n_chains_kept",
                                  "med_sd_0_3km", "med_sd_3_8km"]].mean().round(3).to_string())
        print("\nmanuscript Fig S10 for reference: 3-4 of 30 chains kept (frac 0.10-0.13)")


if __name__ == "__main__":
    main()
