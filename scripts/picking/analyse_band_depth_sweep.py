#!/usr/bin/env python3
"""Does the basement velocity depend on the period cut, the depth box, or both?

The `res_diag >= 0.05` threshold that produced the trimmed arm was chosen by hand, and the
period-to-threshold mapping is smooth (0.02 -> 6.09 s, 0.05 -> 5.12 s, 0.10 -> 4.06 s), so no
value is picked out by the data. The production module uses a RELATIVE criterion instead
(period_resolution.DEFAULTS["R_frac"] = 0.5, i.e. half of each cell's own peak res_diag) because
absolute res_diag is low everywhere under Tarantola-Valette regularisation; at the GVL-1 cells
that lands at 3.84 s, a harder cut than the 5.12 s actually used.

The depth box is not independent of that choice. Trimming shortens the longest wavelength, which
changes how deep the data see, which changes what box is defensible. Measured at the GVL-1
cells (phase velocity from the HS phase maps; group kernels by finite difference on the GVL-1
posterior over a standard crust):

    T_max    c     lambda   lambda/3   z90 (90% of |K| above)
    3.84   2.82    10.8 km    3.6 km      6.9 km
    5.12   2.78    14.2 km    4.7 km      9.0 km
    8.61   3.26    28.1 km    9.4 km     14.0 km

CORRECTION (verified with disba's own PhaseSensitivity/GroupSensitivity on one model): the
earlier claim here that group kernels "run roughly twice as deep" as phase kernels was WRONG.
The group kernel's positive PEAK is if anything SHALLOWER than the phase peak, and lambda/3
tracks that group peak quite well (3.9 km vs a 3.3 km peak at 5.12 s). What is true is that the
group kernel is oscillatory -- a positive shallow lobe over a large NEGATIVE lobe at depth -- so
in |K| terms it extends only 6-35% deeper than phase, not 2x. The box argument survives but for
a different reason: a box must span the whole sensitive range INCLUDING the negative lobe, which
is much deeper than any peak-depth rule. That is a peak-vs-extent distinction, not a
group-vs-phase one. The sweep tests it directly: depth 5 km is approximately lambda/3 for the
5.12 s band,
6 km is the manuscript's box, 8 km is what we have been using, and 12 km clears 1.5*z90 for
every band.

Grid: 3 period bands x 4 depth boxes x the 4 GVL-1 cells = 48 BayHunter runs, 24 chains,
150k+120k, --save-ensemble. Reports basement velocity against the drilled 2.24 km top, the
deep gradient, chain agreement and fit, so band and box effects can be separated.
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
ROOT = f"{EHM}/hautesorne/tomo/2_vs_depth_inversion/manuscript_comparison_2026-08/2_sweep/test_2026-08-16_gvl1_band_depth_sweep"
BANDS = {"T384": 3.84, "T512": 5.12, "T861": 8.61}
BLAB = {"T384": "3.84 s\n(production R_frac=0.5)", "T512": "5.12 s\n(res_diag>=0.05)",
        "T861": "8.61 s\n(untrimmed)"}
DEPTHS = [5.0, 6.0, 8.0, 12.0]
Z90 = {"T384": 6.9, "T512": 9.0, "T861": 14.0}
LAM3 = {"T384": 3.6, "T512": 4.7, "T861": 9.4}
BASEMENT = 2.24         # drilled crystalline basement top at GVL-1, km below ground
BAND = (2.24, 3.5)      # where "basement velocity" is measured


def collect():
    out = []
    for band in BANDS:
        for dz in DEPTHS:
            vals = []
            for f in sorted(glob.glob(f"{ROOT}/{band}_z{dz:.1f}/cell_*/bayhunter_result.npz")):
                z = np.load(f, allow_pickle=True)
                d = np.asarray(z["depth"]); vs = np.asarray(z["vs_median"])
                m = (d >= BAND[0]) & (d <= BAND[1])
                if not m.any():
                    continue
                deep = d >= min(BAND[1], 0.9 * d.max())
                vals.append(dict(
                    vs_base=float(np.mean(vs[m])),
                    grad=float(np.polyfit(d[deep], vs[deep], 1)[0]) if deep.sum() > 3 else np.nan,
                    vs_bot=float(vs[-1]),
                    dis=float(z["chain_disagree"]),
                    kept=int(z["n_chains_kept"]),
                    zrel=float(z["z_reliable_max"]) if "z_reliable_max" in z.files else np.nan,
                    mis=float(np.median(np.asarray(z["ens_misfit"], float)))
                    if "ens_misfit" in z.files else np.nan))
            if vals:
                out.append(dict(band=band, dz=dz, n=len(vals),
                                **{k: float(np.mean([v[k] for v in vals])) for k in vals[0]},
                                vs_base_sd=float(np.std([v["vs_base"] for v in vals]))))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=ROOT)
    a = ap.parse_args()
    R = collect()
    if not R:
        raise SystemExit(f"no runs yet under {ROOT}")
    print(f"{'band':>6} {'T_max':>6} {'box':>6} {'n':>3} | {'Vs 2.24-3.5km':>13} {'+-':>5} "
          f"{'deep grad':>10} {'Vs at box base':>15} {'chain_dis':>10} {'kept':>5} {'misfit':>8}")
    for r in sorted(R, key=lambda r: (BANDS[r["band"]], r["dz"])):
        print(f"{r['band']:>6} {BANDS[r['band']]:6.2f} {r['dz']:6.1f} {r['n']:3d} | "
              f"{r['vs_base']:13.3f} {r['vs_base_sd']:5.3f} {r['grad']:10.3f} "
              f"{r['vs_bot']:15.3f} {r['dis']:10.3f} {r['kept']:5.1f} {r['mis']:8.4f}")

    # separate the two effects
    print("\nEFFECT OF THE DEPTH BOX (mean over bands, relative to the 12 km box):")
    ref = {r["band"]: r["vs_base"] for r in R if r["dz"] == 12.0}
    for dz in DEPTHS:
        d = [r["vs_base"] - ref[r["band"]] for r in R if r["dz"] == dz and r["band"] in ref]
        if d:
            print(f"  box {dz:5.1f} km: basement Vs {np.mean(d):+.3f} km/s "
                  f"(spread {np.ptp(d):.3f})")
    print("\nEFFECT OF THE PERIOD BAND (mean over boxes, relative to the untrimmed band):")
    ref2 = {r["dz"]: r["vs_base"] for r in R if r["band"] == "T861"}
    for band in BANDS:
        d = [r["vs_base"] - ref2[r["dz"]] for r in R if r["band"] == band and r["dz"] in ref2]
        if d:
            print(f"  {band} (T<={BANDS[band]:.2f} s): basement Vs {np.mean(d):+.3f} km/s "
                  f"(spread {np.ptp(d):.3f})")

    fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.0))
    col = dict(zip(BANDS, ["tab:blue", "tab:green", "tab:red"]))
    ax = axs[0]
    for band in BANDS:
        q = sorted([r for r in R if r["band"] == band], key=lambda r: r["dz"])
        if not q:
            continue
        ax.errorbar([r["dz"] for r in q], [r["vs_base"] for r in q],
                    yerr=[r["vs_base_sd"] for r in q], fmt="o-", color=col[band], lw=2,
                    capsize=3, label=BLAB[band].replace("\n", " "))
        ax.axvline(Z90[band], color=col[band], ls=":", lw=1.2, alpha=0.7)
        ax.axvline(LAM3[band], color=col[band], ls="--", lw=1.0, alpha=0.5)
    ax.set_xlabel("model box depth [km]")
    ax.set_ylabel(f"Vs over {BAND[0]}-{BAND[1]} km [km/s]")
    ax.set_title("basement velocity vs box depth\n"
                 "dotted = z90, dashed = lambda/3 for that band", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axs[1]
    G = np.full((len(BANDS), len(DEPTHS)), np.nan)
    for i, band in enumerate(BANDS):
        for j, dz in enumerate(DEPTHS):
            v = [r["vs_base"] for r in R if r["band"] == band and r["dz"] == dz]
            if v:
                G[i, j] = v[0]
    im = ax.imshow(G, cmap="inferno", aspect="auto", origin="lower")
    ax.set_xticks(range(len(DEPTHS))); ax.set_xticklabels([f"{d:.0f}" for d in DEPTHS])
    ax.set_yticks(range(len(BANDS)))
    ax.set_yticklabels([BLAB[b].replace("\n", " ") for b in BANDS], fontsize=8)
    for i in range(len(BANDS)):
        for j in range(len(DEPTHS)):
            if np.isfinite(G[i, j]):
                ax.text(j, i, f"{G[i,j]:.2f}", ha="center", va="center", color="w",
                        fontsize=9, fontweight="bold")
    ax.set_xlabel("model box depth [km]")
    ax.set_title("basement Vs across the full grid", fontsize=10)
    plt.colorbar(im, ax=ax, label="Vs [km/s]")

    ax = axs[2]
    for band in BANDS:
        q = sorted([r for r in R if r["band"] == band], key=lambda r: r["dz"])
        if q:
            ax.plot([r["dz"] for r in q], [r["dis"] for r in q], "o-", color=col[band], lw=2,
                    label=BLAB[band].replace("\n", " "))
    ax.axhline(0.3, color="k", ls="--", lw=1.2)
    ax.set_xlabel("model box depth [km]"); ax.set_ylabel("chain_disagree(Vs) [km/s]")
    ax.set_title("does the configuration also affect convergence?", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("GVL-1 — is the basement velocity set by the period cut or by the depth box?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(a.out, "band_depth_sweep.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); print("\nwrote", p)


if __name__ == "__main__":
    main()
