#!/usr/bin/env python3
"""Per-period comparison of runs that differ ONLY in the prior correlation length LC.

Aggregate medians hide what matters here: with `--lc-mode fresnel` the correlation length
is close to the `--lc` floor at short period and only diverges from a fixed alternative at
long period, so a single median mixes periods where the two runs are nearly identical with
periods where they differ by a factor of eight.

Four panels per wave, all against period:
  LC(T)              what each variant actually used -- the x-ray of the experiment
  structure std      spatial std of V over the shown cells: how much lateral structure
  var_red            variance reduction: whether that structure improves the fit
  RMS difference     between variants, over cells shown in BOTH (a shifting resolution
                     mask would otherwise register as a velocity change)

Reading it: LC enters the prior as CM = (sigma*L0/LC)^2 * exp(-r/LC), so shrinking LC
shrinks the prior VARIANCE as well as its range. A variant with smaller LC and the same
--se is therefore more strongly damped, and can show LESS structure despite the shorter
correlation length. Compare LC(T) against struct-std(T) before concluding anything about
smoothing.

Usage:
  python compare_lc_variants.py --production-root <.../production> --net riehen \
      --measure group --cd scaled [--tags "" _lcfix1 _lc0.5]
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WAVES = ("fund", "overtone", "love")


def load(root, measure, cd, dx, tag, wave):
    d = os.path.join(root, "tspws_%s_%s_dx%s%s" % (measure, cd, dx, tag), "production", wave)
    out = {}
    for f in glob.glob(os.path.join(d, "map_T*.npz")):
        try:
            z = np.load(f)
            out[round(float(z["period"]), 3)] = dict(
                vel=z["vel"], lc=float(z["LC"]), var_red=float(z["var_red"]),
                chi2=float(z["chi2_red"]), n=int(z["N"]))
        except Exception:
            continue
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--production-root", required=True)
    ap.add_argument("--net", required=True)
    ap.add_argument("--measure", default="group", choices=("group", "phase"))
    ap.add_argument("--cd", default="scaled", choices=("blanket", "measured", "scaled"))
    ap.add_argument("--dx", default=None)
    ap.add_argument("--tags", nargs="+", default=["", "_lcfix1", "_lc0.5"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dx = a.dx
    if dx is None:
        cand = glob.glob(os.path.join(a.production_root,
                                      "tspws_%s_%s_dx*" % (a.measure, a.cd)))
        dx = sorted({c.rsplit("_dx", 1)[1].split("_")[0] for c in cand})[0] if cand else "0.2"

    name = {"": "fresnel (prod)"}
    rows = []
    fig, axs = plt.subplots(4, len(WAVES), figsize=(5.4 * len(WAVES), 15), squeeze=False)
    for j, wave in enumerate(WAVES):
        runs = {t: load(a.production_root, a.measure, a.cd, dx, t, wave) for t in a.tags}
        have = [t for t in a.tags if runs[t]]
        if not have:
            for i in range(4):
                axs[i][j].axis("off")
            continue
        periods = sorted(set.intersection(*[set(runs[t]) for t in have]))
        for t in have:
            lab = name.get(t, t.lstrip("_") or t)
            P = periods
            axs[0][j].plot(P, [runs[t][x]["lc"] for x in P], "o-", ms=3, lw=1.2, label=lab)
            axs[1][j].plot(P, [np.nanstd(runs[t][x]["vel"]) for x in P], "o-", ms=3, lw=1.2,
                           label=lab)
            axs[2][j].plot(P, [runs[t][x]["var_red"] for x in P], "o-", ms=3, lw=1.2, label=lab)
        base = have[0]
        for t in have[1:]:
            d = []
            for x in periods:
                A, B = runs[base][x]["vel"], runs[t][x]["vel"]
                both = np.isfinite(A) & np.isfinite(B)
                mu = np.nanmean(np.where(both, A, np.nan))
                r = np.sqrt(np.nanmean((A[both] - B[both]) ** 2)) if both.any() else np.nan
                d.append(100.0 * r / mu if mu else np.nan)
                rows.append(dict(wave=wave, period=x,
                                 pair="%s-vs-%s" % (name.get(base, base), name.get(t, t)),
                                 lc_a=runs[base][x]["lc"], lc_b=runs[t][x]["lc"],
                                 std_a=round(float(np.nanstd(runs[base][x]["vel"])), 4),
                                 std_b=round(float(np.nanstd(runs[t][x]["vel"])), 4),
                                 varred_a=round(runs[base][x]["var_red"], 4),
                                 varred_b=round(runs[t][x]["var_red"], 4),
                                 rms_pct=round(float(d[-1]), 3), n_rays=runs[base][x]["n"]))
            axs[3][j].plot(periods, d, "o-", ms=3, lw=1.2,
                           label="%s vs %s" % (name.get(base, base), name.get(t, t)))
        axs[2][j].axhline(0, color="0.6", lw=0.8, ls="--")
        for i, (ylab, ttl) in enumerate((
                ("LC [km]", "prior correlation length"),
                ("std of V over map [km/s]", "lateral structure amplitude"),
                ("var_red", "variance reduction"),
                ("RMS difference [% of mean V]", "map change"))):
            axs[i][j].set_xlabel("period [s]"); axs[i][j].set_ylabel(ylab)
            axs[i][j].set_title("%s %s: %s" % (a.net, wave, ttl), fontsize=10)
            axs[i][j].legend(fontsize=8); axs[i][j].grid(alpha=0.3)

    fig.suptitle("%s %s / Cd=%s: effect of the correlation length, PER PERIOD"
                 % (a.net, a.measure, a.cd), y=1.0)
    fig.tight_layout()
    out = a.out or os.path.join(a.production_root,
                                "lc_comparison_%s_%s_%s.png" % (a.net, a.measure, a.cd))
    fig.savefig(out, dpi=130); plt.close(fig)
    print("wrote", os.path.basename(out))
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        cp = out.replace(".png", ".csv")
        df.to_csv(cp, index=False)
        print("wrote", os.path.basename(cp))
        for (pair, wave), g in df.groupby(["pair", "wave"]):
            for lo, hi, lab in ((0, 1, "T<1s"), (1, 2, "1-2s"), (2, 4, "2-4s"), (4, 99, "T>4s")):
                s = g[(g.period >= lo) & (g.period < hi)]
                if not len(s):
                    continue
                print("  %-9s %-8s LC %4.1f->%4.1f | struct %.3f->%.3f | var_red %+.3f->%+.3f"
                      " | RMS %5.1f%%"
                      % (wave, lab, s.lc_a.median(), s.lc_b.median(),
                         s.std_a.median(), s.std_b.median(),
                         s.varred_a.median(), s.varred_b.median(), s.rms_pct.median()))


if __name__ == "__main__":
    main()
