#!/usr/bin/env python
"""Exact per-cell gate drop counts, by calling the production gate itself.

Recomputed rather than parsed from run logs: the logs only carry the shard-level total, and
the question here is which INDIVIDUAL cells lost samples. Cells the gate left untouched are
the control group -- they re-ran on identical data, so any Vs difference between the gated
and ungated arm there is pure MCMC scatter, not gate effect.
"""
import argparse, sys
import numpy as np
sys.path.insert(0, "/home/users/s/savardg/NoisePy-ant")
from noisepy.vs_inversion import (load_cell_curves, mode_id_gate,
                                  read_period_ranges, restrict_periods)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--waves", default="love")
    ap.add_argument("--group-root", required=True)
    ap.add_argument("--phase-root", required=True)
    ap.add_argument("--margin", type=float, default=1.0)
    # the driver ALWAYS trims to the decided valid band before gating, so gating the
    # untrimmed curve counts drops the inversion never saw (19.3% vs the true 3.4%)
    ap.add_argument("--period-ranges", default=None)
    ap.add_argument("--net", default=None)
    ap.add_argument("--cells", required=True, help="volume npz whose cells define the list")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    waves = tuple(a.waves.split(","))
    cells = np.load(a.cells, allow_pickle=True)["cells"]
    pr = read_period_ranges(a.period_ranges, a.net) if a.period_ranges else None

    rows, first_err = [], None
    for ix, iy in cells:
        rep = {}
        try:
            cur = load_cell_curves(a.group_root, int(ix), int(iy), waves=waves)
            if pr:
                cur = restrict_periods(cur, pr)
            mode_id_gate(cur, int(ix), int(iy), phase_root=a.phase_root, waves=waves,
                         margin=a.margin, report=rep)
        except Exception as e:
            if first_err is None:
                first_err = repr(e)
            rows.append((int(ix), int(iy), -1, -1))
            continue
        dropped = sum(v[0] for v in rep.values())
        judged = sum(v[1] for v in rep.values())
        rows.append((int(ix), int(iy), dropped, judged))

    with open(a.out, "w") as fh:
        fh.write("ix,iy,dropped,judged\n")
        for r in rows:
            fh.write("%d,%d,%d,%d\n" % r)

    d = np.array([r[2] for r in rows])
    ok = d >= 0
    print(f"cells {len(rows)}  errors {int(np.sum(~ok))}")
    if first_err:
        print("first error:", first_err)
    if ok.any():
        print(f"  dropped 0 periods: {int(np.sum(d[ok]==0))} ({100*np.mean(d[ok]==0):.1f}%)")
        print(f"  dropped >0:        {int(np.sum(d[ok]>0))}  (max {int(d[ok].max())})")
        print(f"  total dropped {int(d[ok].sum())} of {int(sum(r[3] for r in rows if r[3]>=0))} judged")
    print("wrote", a.out)


main()
