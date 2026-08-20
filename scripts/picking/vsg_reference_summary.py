#!/usr/bin/env python3
"""Assess the VSG-reference inversions: are the four reference curves mutually consistent, and
is the whole-network average Vs physically sensible?

Reads test_2026-08-16_vsg_reference/<cfg>/bayhunter_result.npz and reports, per config:

  * per-target misfit -- THE branch diagnostic. A mis-branched reference curve is smooth, so it
    gets a small pick-scatter sigma and still cannot be fitted by any model that also fits the
    others. Look for a target whose misfit jumps when it is added to a set that was fitting well.
  * convergence (chain_disagree on Vs, retained chains).
  * the average Vs(z), and the IMPLIED Vp/Vs against the Michel (2016) sonic Vp -- the same
    plausibility test used at the wells, because the log's own Vs assumes 1.90/1.75 and is not
    ground truth for Vs.
  * agreement between configs: if dropping a curve moves Vs by more than the chain disagreement,
    that curve is driving the model.

Usage:
  python vsg_reference_summary.py --net riehen
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
MODEL = "/Volumes/T7blue/riehen-data/well-data/Michel2016_gpdc.model"
# production combo naming; every VSG reference curve is PHASE
ORDER = ["R0p", "R0pR1p", "R0pL0p", "R0pR1pL0p", "R0pR1pL0pL1p"]
LABEL = {"R0p": "R0p", "R0pR1p": "R0p+R1p", "R0pL0p": "R0p+L0p",
         "R0pR1pL0p": "R0p+R1p+L0p", "R0pR1pL0pL1p": "R0p+R1p+L0p+L1p"}
COL = {"R0p": "tab:blue", "R0pR1p": "tab:orange", "R0pL0p": "tab:green",
       "R0pR1pL0p": "tab:red", "R0pR1pL0pL1p": "tab:purple"}


def michel():
    rows = [l.split() for l in open(MODEL).read().split("\n")[1:] if l.strip()]
    th = np.array([float(r[0]) for r in rows]) / 1000.0
    vp = np.array([float(r[1]) for r in rows]) / 1000.0
    vs = np.array([float(r[2]) for r in rows]) / 1000.0
    return np.concatenate([[0.0], np.cumsum(th)[:-1]]), vp, vs


def step_at(top, val, z):
    return val[np.clip(np.searchsorted(top, z, side="right") - 1, 0, len(val) - 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", default="riehen")
    a = ap.parse_args()
    root = f"{EHM}/{a.net}/tomo/2_vs_depth_inversion/tests/test_2026-08-16_vsg_reference"
    top, vp, vs_assumed = michel()

    rows, prof = [], {}
    for cfg in ORDER:
        f = f"{root}/{cfg}/bayhunter_result.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        d, v = z["depth"], z["vs_median"]
        prof[cfg] = (d, v, z)
        waves = [str(w) for w in np.atleast_1d(z["waves"])]
        # Per-target misfit computed HERE from the stored obs/pred/sigma. It cannot be parsed
        # from run.log: the "misfit {...}" line is emitted by run_vs_inversion.py, and these
        # runs call run_bayhunter_cell.py directly, so no such line exists.
        mis = {}
        for w in waves:
            ko, kp, ks = f"obs_{w}", f"pred_{w}", f"obssig_{w}"
            if ko in z and kp in z:
                obs = np.asarray(z[ko], float).ravel()
                pred = np.asarray(z[kp], float)
                pred = pred.mean(axis=0) if pred.ndim > 1 else pred.ravel()
                sig = (np.asarray(z[ks], float).ravel() if ks in z
                       else np.full(obs.size, 1.0))
                n = min(obs.size, pred.size, sig.size)
                ok = np.isfinite(obs[:n]) & np.isfinite(pred[:n]) & (sig[:n] > 0)
                if ok.any():
                    mis[w] = float(np.sqrt(np.mean(
                        ((obs[:n][ok] - pred[:n][ok]) / sig[:n][ok]) ** 2)))
        r = dict(cfg=cfg, waves="+".join(waves),
                 chain_disagree=round(float(z["chain_disagree"]), 3),
                 chains=f"{int(z['n_chains_kept'])}/{int(z['n_chains_used'])}",
                 z_reliable=round(float(z["z_reliable_max"]), 2))
        for w in ("fund", "overtone", "love", "love_ot"):
            for key in (w, f"{w}_phase"):
                if key in mis:
                    r[f"chi_{w}"] = round(mis[key], 2)
        rows.append(r)

    if not rows:
        raise SystemExit("no completed runs yet")
    D = pd.DataFrame(rows)
    print("=== per-target misfit and convergence ===")
    print(D.to_string(index=False))

    # config-to-config Vs agreement vs chain disagreement
    print("\n=== does dropping a curve move Vs more than the chains disagree? ===")
    for i, c1 in enumerate(ORDER):
        for c2 in ORDER[i + 1:]:
            if c1 not in prof or c2 not in prof:
                continue
            d1, v1, z1 = prof[c1]; _, v2, z2 = prof[c2]
            zc = min(float(z1["z_reliable_max"]), float(z2["z_reliable_max"]))
            m = (d1 >= 0.2) & (d1 <= zc)
            dv = float(np.mean(np.abs(v1[m] - v2[m])))
            gate = max(float(z1["chain_disagree"]), float(z2["chain_disagree"]))
            print(f"   {c1:<6} vs {c2:<6} |dVs| {dv:.3f}  (chain gate {gate:.3f})  "
                  f"{'DRIVEN by the differing curve' if dv > gate else 'consistent'}")

    # figure
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 8.0), sharey=True,
                            gridspec_kw={"width_ratios": [1, 1, 1]})
    zz = np.linspace(0, 6, 900)
    axs[0].step(step_at(top, vp, zz), zz, where="post", color="k", lw=2.0,
                label="Vp (sonic, measured)")
    axs[0].step(step_at(top, vs_assumed, zz), zz, where="post", color="0.5", lw=1.6,
                ls="--", label="log Vs (ASSUMED 1.90/1.75)")
    for cfg in ORDER:
        if cfg not in prof:
            continue
        d, v, z = prof[cfg]
        zc = float(z["z_reliable_max"])
        m = d <= zc
        axs[0].plot(v[m], d[m], "-", color=COL[cfg], lw=2.0, label=LABEL[cfg])
        axs[0].plot(v[~m], d[~m], ":", color=COL[cfg], lw=1.0, alpha=0.5)
        axs[1].fill_betweenx(d[m], z["vs_p16"][m], z["vs_p84"][m], color=COL[cfg], alpha=0.18)
        axs[1].plot(v[m], d[m], "-", color=COL[cfg], lw=2.0, label=LABEL[cfg])
        axs[2].plot(step_at(top, vp, d[m]) / v[m], d[m], "-", color=COL[cfg], lw=2.0,
                    label=LABEL[cfg])
    for lo, hi, col in ((1.65, 1.80, "#4575b4"), (1.80, 2.00, "#91bfdb"),
                        (2.00, 2.30, "#fee090"), (2.30, 2.60, "#fc8d59")):
        axs[2].axvspan(lo, hi, color=col, alpha=0.35)
    axs[2].step(step_at(top, vp / vs_assumed, zz), zz, where="post", color="0.5", lw=1.6,
                ls="--", label="log's assumed ratio")
    for ax, xl, ti in ((axs[0], "velocity [km/s]", "network-average Vs vs the sonic log"),
                       (axs[1], "Vs [km/s]", "posterior (band = 16-84%)"),
                       (axs[2], "implied Vp/Vs", "implied Vp$_{log}$/Vs$_{VSG}$")):
        ax.set_ylim(6, 0); ax.grid(alpha=0.3); ax.set_xlabel(xl); ax.set_title(ti, fontsize=10)
        ax.legend(fontsize=7.5, loc="lower left")
    axs[0].set_ylabel("depth [km]"); axs[0].set_xlim(0.5, 6.2); axs[2].set_xlim(1.4, 3.2)
    fig.suptitle(f"{a.net}: VSG reference curves inverted for a whole-network average Vs\n"
                 "dotted = below each run's z_reliable_max (prior-filled, not a measurement)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(root, "vsg_reference_summary.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    D.to_csv(os.path.join(root, "vsg_reference_summary.csv"), index=False)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
