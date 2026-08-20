"""Per-(net,cfg) progress, recent rate and ETA for the vs_prod3 campaign.

Aggregate cells/min is a misleading ETA here: each array is capped by its own
ArrayTaskThrottle, so a fast array's freed slots return to the cluster rather than to a slow
one. The campaign ends when its SLOWEST array ends, so report per array and rank by ETA.

Rate is measured over a recent window (default 2 h) from file mtimes, not since-start, so it
reflects the current worker count rather than an average diluted by earlier conditions.

  python progress.py [runs_dir] [--window-h 2]
"""
import argparse
import glob
import os
import time

# expected cell counts, from each array's "N cells x 1 wavesets" startup line
EXPECTED = {
    ("riehen", "R0g"): 4388, ("riehen", "R0p"): 4430, ("riehen", "L0g"): 4824,
    ("riehen", "L0p"): 4806, ("riehen", "RLg_radial"): 4824, ("riehen", "RLg_iso"): 4824,
    ("aargau", "R0g"): 1783, ("aargau", "R0p"): 1778, ("aargau", "L0g"): 1839,
    ("aargau", "L0p"): 1836, ("aargau", "RLg_radial"): 1839, ("aargau", "RLg_iso"): 1839,
    ("hautesorne", "R0g"): 2287, ("hautesorne", "R0p"): 2333, ("hautesorne", "L0g"): 2402,
    ("hautesorne", "L0p"): 2399, ("hautesorne", "RLg_radial"): 2402,
    ("hautesorne", "RLg_iso"): 2402,
}
BOOSTED = {("riehen", "RLg_radial"), ("riehen", "RLg_iso"), ("riehen", "R0g"),
           ("hautesorne", "RLg_radial"), ("riehen", "R0p")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="?",
                    default="/srv/beegfs/scratch/users/s/savardg/vs_prod3/runs")
    ap.add_argument("--window-h", type=float, default=2.0)
    args = ap.parse_args()

    cutoff = time.time() - args.window_h * 3600
    rows, tot_done, tot_exp = [], 0, 0
    for (net, cfg), exp in EXPECTED.items():
        d = os.path.join(args.runs, net, cfg, "cells")
        fs = [f for f in glob.glob(os.path.join(d, "cell_*.npz")) if ".tmp" not in f]
        recent = sum(1 for f in fs if os.path.getmtime(f) >= cutoff)
        rate = recent / args.window_h
        rem = exp - len(fs)
        eta = rem / rate if rate > 0 else float("inf")
        rows.append((eta, net, cfg, len(fs), exp, rate, rem))
        tot_done += len(fs)
        tot_exp += exp

    rows.sort(reverse=True)
    print(f"{'net/cfg':<26}{'done':>13}{'%':>7}{'cells/h':>9}{'ETA':>10}")
    for eta, net, cfg, done, exp, rate, rem in rows:
        # a finished config also reports 0 cells/h; call that DONE, not stalled
        if rem <= 0:
            etas = "DONE"
        elif eta == float("inf"):
            etas = "stalled"
        else:
            etas = f"{eta:.0f}h" if eta >= 1 else "<1h"
        star = " *" if (net, cfg) in BOOSTED else ""
        print(f"{net + '/' + cfg + star:<26}{f'{done}/{exp}':>13}"
              f"{100.0 * done / exp:>6.1f}%{rate:>9.1f}{etas:>10}")
    print(f"\nTOTAL {tot_done}/{tot_exp} ({100.0 * tot_done / tot_exp:.1f}%)   "
          f"* = has a booster array")
    finite = [r[0] for r in rows if r[0] != float("inf")]
    if finite:
        print(f"critical path (slowest array): {max(finite):.0f}h")


if __name__ == "__main__":
    main()
