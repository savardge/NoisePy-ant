#!/usr/bin/env python3
"""How much does the Cd parameterisation move the velocity maps, and at which periods?

Compares the production runs that differ ONLY in their data-covariance model:

    blanket   Cd = (0.10 tau)^2                     -- every pick weighted equally (legacy)
    measured  Cd = tau_std^2                        -- substack-jackknife pick repeatability
    scaled    Cd = (k tau_std)^2, k per period      -- same ranking, chi2_red driven to 1

Two questions, two products:

  1. VELOCITY DISTRIBUTION per map -- for each mode: the mean, the spatial spread (std over
     shown cells) and the p5-p95 span. The spread is the amplitude of lateral structure the
     inversion is willing to put in; a tighter Cd buys more structure, a looser one lets the
     prior smooth it away.

  2. WHICH PERIODS MOVE MOST -- RMS difference between modes over cells shown in BOTH, in
     km/s and as a percentage of the map mean, ranked. Only common cells are used: the
     resolution mask itself shifts with Cd, and counting cells that exist in one map and not
     the other would report a masking change as a velocity change.

Writes <out>/cd_comparison_<net>_<measure>.csv and a per-wave figure.

Usage:
  python compare_cd_modes.py --production-root <.../1_velocity_maps/production> \
      --net riehen --measure group [--dx 0.2] [--out <dir>]
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODES = ("blanket", "measured", "scaled")
WAVES = ("fund", "overtone", "love")


def load_run(root, net, measure, mode, dx, wave):
    """{period: dict} for one run, or {} when that variant was not produced."""
    d = os.path.join(root, "tspws_%s_%s_dx%s" % (measure, mode, dx), "production", wave)
    out = {}
    for f in glob.glob(os.path.join(d, "map_T*.npz")):
        try:
            z = np.load(f)
            out[round(float(z["period"]), 3)] = {
                "vel": z["vel"], "var_red": float(z["var_red"]),
                "chi2": float(z["chi2_red"]),
                "scale": float(z["cd_scale"]) if "cd_scale" in z.files else 1.0}
        except Exception:
            continue
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--production-root", required=True)
    ap.add_argument("--net", required=True)
    ap.add_argument("--measure", default="group", choices=("group", "phase"))
    ap.add_argument("--dx", default=None, help="grid label, e.g. 0.2 (auto-detected if omitted)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dx = a.dx
    if dx is None:
        cand = glob.glob(os.path.join(a.production_root, "tspws_%s_*_dx*" % a.measure))
        dx = sorted({c.rsplit("_dx", 1)[1] for c in cand})[0] if cand else "0.2"
    out = a.out or a.production_root
    os.makedirs(out, exist_ok=True)

    rows = []
    fig, axs = plt.subplots(2, len(WAVES), figsize=(5.2 * len(WAVES), 8), squeeze=False)
    for j, wave in enumerate(WAVES):
        runs = {m: load_run(a.production_root, a.net, a.measure, m, dx, wave) for m in MODES}
        have = [m for m in MODES if runs[m]]
        if len(have) < 2:
            for i in (0, 1):
                axs[i][j].set_title("%s: need >=2 modes" % wave, fontsize=9)
                axs[i][j].axis("off")
            continue
        periods = sorted(set.intersection(*[set(runs[m]) for m in have]))

        # ---- panel 1: structure amplitude (spatial std) per mode
        for m in have:
            sd = [np.nanstd(runs[m][T]["vel"]) for T in periods]
            axs[0][j].plot(periods, sd, "o-", ms=3, lw=1.2, label=m)
        axs[0][j].set_title("%s %s: lateral structure amplitude" % (a.net, wave), fontsize=10)
        axs[0][j].set_xlabel("period [s]"); axs[0][j].set_ylabel("std of V over map [km/s]")
        # log: a Cd model can destabilise individual periods (Riehen fund T=0.51 s reaches
        # std 25 km/s under measured Cd), and on a linear axis that one point flattens the
        # rest into a line.
        axs[0][j].set_yscale("log")
        axs[0][j].legend(fontsize=8); axs[0][j].grid(alpha=0.3, which="both")

        # ---- panel 2: pairwise RMS difference, common cells only
        pairs = [(x, y) for i, x in enumerate(have) for y in have[i + 1:]]
        for x, y in pairs:
            rms = []
            for T in periods:
                A, B = runs[x][T]["vel"], runs[y][T]["vel"]
                both = np.isfinite(A) & np.isfinite(B)
                mu = np.nanmean(np.where(both, A, np.nan))
                r = (np.sqrt(np.nanmean((A[both] - B[both]) ** 2)) if both.any() else np.nan)
                rms.append(100.0 * r / mu if mu else np.nan)
                rows.append(dict(wave=wave, period=T, pair="%s-vs-%s" % (x, y),
                                 rms_kms=round(float(r), 4),
                                 rms_pct=round(float(100.0 * r / mu) if mu else np.nan, 3),
                                 n_common=int(both.sum()),
                                 chi2_x=round(runs[x][T]["chi2"], 3),
                                 chi2_y=round(runs[y][T]["chi2"], 3),
                                 varred_x=round(runs[x][T]["var_red"], 4),
                                 varred_y=round(runs[y][T]["var_red"], 4),
                                 scale_y=round(runs[y][T]["scale"], 3)))
            axs[1][j].plot(periods, rms, "o-", ms=3, lw=1.2, label="%s vs %s" % (x, y))
        axs[1][j].set_title("%s %s: map change from Cd choice" % (a.net, wave), fontsize=10)
        axs[1][j].set_xlabel("period [s]"); axs[1][j].set_ylabel("RMS difference [% of mean V]")
        axs[1][j].set_yscale("log")
        axs[1][j].axhline(10, color="0.6", lw=0.8, ls="--")
        axs[1][j].legend(fontsize=8); axs[1][j].grid(alpha=0.3, which="both")

    fig.suptitle("%s %s: sensitivity of the velocity maps to the Cd model "
                 "(common cells only)" % (a.net, a.measure), y=1.0)
    fig.tight_layout()
    fp = os.path.join(out, "cd_comparison_%s_%s.png" % (a.net, a.measure))
    fig.savefig(fp, dpi=130); plt.close(fig)

    if rows:
        import pandas as pd
        df = pd.DataFrame(rows).sort_values(["wave", "pair", "period"])
        cp = os.path.join(out, "cd_comparison_%s_%s.csv" % (a.net, a.measure))
        df.to_csv(cp, index=False)
        print("wrote %s and %s" % (os.path.basename(fp), os.path.basename(cp)))
        for pair, g in df.groupby("pair"):
            top = g.nlargest(5, "rms_pct")
            print("  %-22s median %5.2f%%  |  largest changes: %s"
                  % (pair, g["rms_pct"].median(),
                     ", ".join("T=%.2f (%.1f%%)" % (r.period, r.rms_pct)
                               for r in top.itertuples())))
    else:
        print("wrote", os.path.basename(fp))


if __name__ == "__main__":
    main()
