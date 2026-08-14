#!/usr/bin/env python3
"""Cross-combination summary of the waveset Vs test runs (test_2026-08-06_waveset_combos).

For each well cell, the sweep inverted 8 input combinations (R0g ... R0gL0gR0pL0p) with both
engines. This script answers, per well:

  1. WHAT does each combination do to the posterior Vs(z)?      -> profile overlay figure
  2. HOW MUCH do combinations disagree, depth-resolved?          -> pairwise |dVs| matrix
  3. Which combinations CONVERGE (BayHunter chain_disagree)?     -> summary CSV
  4. Does adding a target improve or degrade the fit of the
     targets already present?                                    -> misfit table CSV

Conventions: the pairwise disagreement uses the SAME metric as the runner's chain_disagree
(mean |dVs| between posterior medians over the depth grid, in km/s) so combo-vs-combo numbers
are directly comparable with the chain-vs-chain convergence gate.

Usage:
  python waveset_combo_summary.py            # all wells found under both networks
  python waveset_combo_summary.py --tag test_2026-08-06_waveset_combos
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
COMBOS = ["R0g", "R0gR1g", "R0gR0p", "L0g", "L0gL0p", "R0gL0g", "R0pL0p", "R0gL0gR0pL0p"]
COL = dict(zip(COMBOS, plt.cm.tab10(np.linspace(0, 0.9, len(COMBOS)))))
TARGETS = ["fund", "overtone", "love", "fund_phase", "overtone_phase", "love_phase"]


def parse_log(log):
    """{engine: {target: misfit}} from a run.log."""
    txt = open(log).read()
    out = {}
    for eng, pat in (("bayesbay", r"=== bayesbay ===.*?misfit \{(.*?)\}"),
                     ("bayhunter", r"=== BayHunter \(subprocess\) ===.*?misfit \{(.*?)\}")):
        m = re.search(pat, txt, re.S)
        if m:
            out[eng] = {k: float(v) for k, v in re.findall(r"'([a-z_]+)'\)?: ([\d.]+)", m.group(1))}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="test_2026-08-06_waveset_combos")
    a = ap.parse_args()

    wells = sorted(glob.glob(f"{EHM}/*/tomo/2_vs_depth_inversion/tests/{a.tag}/*_cell_*"))
    if not wells:
        raise SystemExit("no well directories under tag " + a.tag)
    outdir = os.path.dirname(wells[0])          # summary lives beside the well dirs (riehen tree)

    rows = []
    for wdir in wells:
        net = wdir.split("/Projects/")[1].split("/")[0]
        well = os.path.basename(wdir).rsplit("_cell_", 1)[0]
        prof = {}                               # (combo, engine) -> npz
        for combo in COMBOS:
            log = os.path.join(wdir, combo, "run.log")
            if not os.path.exists(log):
                continue
            mis = parse_log(log)
            for eng in ("bayesbay", "bayhunter"):
                npz = os.path.join(wdir, combo, f"{eng}_result.npz")
                r = dict(net=net, well=well, combo=combo, engine=eng)
                if eng in mis:
                    r.update({t: round(mis[eng].get(t, np.nan), 2) for t in TARGETS})
                if os.path.exists(npz):
                    z = np.load(npz, allow_pickle=True)
                    prof[(combo, eng)] = z
                    # keys individually guarded: bayesbay results share the schema only
                    # partially (e.g. chain_disagree without frac_chains_ok)
                    if "chain_disagree" in z:
                        r["chain_disagree"] = round(float(z["chain_disagree"]), 3)
                    if "frac_chains_ok" in z:
                        r["frac_ok"] = float(z["frac_chains_ok"])
                    if "confidence" in z:
                        r["confidence"] = str(z["confidence"])
                rows.append(r)

        if not prof:
            continue
        # ---------- figure: profile overlay (one panel per engine) + disagreement matrix ----
        fig, axs = plt.subplots(1, 3, figsize=(15.5, 6.4),
                                gridspec_kw={"width_ratios": [1, 1, 1.15]})
        for k, eng in enumerate(("bayesbay", "bayhunter")):
            ax = axs[k]
            for combo in COMBOS:
                z = prof.get((combo, eng))
                if z is None:
                    continue
                d, v = z["depth"], z["vs_median"]
                ax.plot(v, d, "-", color=COL[combo], lw=1.9, label=combo)
                if combo == "R0gL0gR0pL0p" and "vs_p16" in z:
                    ax.fill_betweenx(d, z["vs_p16"], z["vs_p84"], color=COL[combo], alpha=0.15)
            ax.invert_yaxis() if k == 0 else ax.set_ylim(ax.get_ylim())
            ax.set_ylim(6, 0)
            ax.set_xlabel("Vs [km/s]"); ax.grid(alpha=0.3)
            ax.set_title(eng + ("  (band = full combo p16-84)" if k == 0 else ""), fontsize=10)
            if k == 0:
                ax.set_ylabel("depth [km]")
                ax.legend(fontsize=7.5, loc="lower left")
        # pairwise combo disagreement (BayHunter posteriors; same metric as chain_disagree)
        ax = axs[2]
        have = [c for c in COMBOS if (c, "bayhunter") in prof]
        M = np.full((len(have), len(have)), np.nan)
        for i, ci in enumerate(have):
            for j, cj in enumerate(have):
                vi = prof[(ci, "bayhunter")]["vs_median"]
                vj = prof[(cj, "bayhunter")]["vs_median"]
                M[i, j] = np.mean(np.abs(vi - vj))
        im = ax.imshow(M, cmap="inferno", vmin=0)
        ax.set_xticks(range(len(have))); ax.set_xticklabels(have, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(have))); ax.set_yticklabels(have, fontsize=8)
        for i in range(len(have)):
            for j in range(len(have)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7,
                            color="w" if M[i, j] < np.nanmax(M) * 0.6 else "k")
        plt.colorbar(im, ax=ax, shrink=0.8, label="mean |dVs| between combo medians [km/s]")
        ax.set_title("combo disagreement (BayHunter)\nsame metric as chain_disagree", fontsize=10)
        fig.suptitle(f"{net} — {well}: posterior Vs by input combination "
                     f"(16 chains, v1 ranges, group=scaled / phase=blanket Cd)",
                     fontsize=12.5, fontweight="bold")
        fig.tight_layout()
        p = os.path.join(wdir, "combo_overlay.png")
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("wrote", p)

    D = pd.DataFrame(rows)
    csv = os.path.join(outdir, "combo_summary.csv")
    D.to_csv(csv, index=False)
    print("wrote %s  (%d rows)" % (csv, len(D)))


if __name__ == "__main__":
    main()
