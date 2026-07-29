"""ffscan step 2: cut the ffscan BASE pick tables at multiple minimum r/lambda thresholds.

Input = the ffscan base exports (export_unified_tomo_picks.py --src <ffscan qc>/picks_unified_QCd.csv
--out-suffix _nf / _nf_phase), i.e. the pool whose only far-field trim is the picker's own
1-lambda floor (the QC farfield gate was lowered to 1.0 for the ffscan re-run). Group and phase
therefore both extend down to ~1 lambda and every threshold arm is a strict subset of the SAME
QC'd pool — the scan varies ONLY the r/lambda cut.

lambda convention = REFERENCE CURVE (decided 2026-07-27 after the per-pick trial failed):

    keep row iff   distance >= X * lam_ref(T),      lam_ref(T) = v_ref(T) * T

with v_ref(T) the median velocity at that period over the BASE pool, smoothed in log-T. The
cut therefore depends only on (distance, period) -- never on the row's own velocity.

WHY NOT PER-PICK (d >= X*v_row*T), which we tried first: it is velocity-SELECTIVE, and since
velocity varies spatially it is therefore SPATIALLY selective. At fixed (T, d) it keeps slow
picks and rejects fast ones, so the surviving population is chosen partly BECAUSE it is slow,
and the tomography maps that selection. Measured on the base pools: the kept median never
rose (max +1.1%) and fell by up to 67% at long period -- e.g. aargau fund at 5 s can only
keep v <= d_max/(X*T) = 1.65 km/s while the true branch sits at ~2.1, so the branch is
deleted entirely and only the slow outlier tail survives. A reference curve cannot do this:
every pick at a given period faces the same distance threshold. Its own weakness -- lam_ref
is mis-sized where the medium is laterally heterogeneous -- is second order and, crucially,
is not correlated with the mapped quantity per pick.

No r/lambda upper cap and no period cap are applied (unlike the hautesorne ffv2 production rule):
the scan isolates the minimum-threshold variable; interpret long-T group maps accordingly.

Outputs, next to the inputs:  picks_{wave}_uni[_phase]_ff{X:.1f}.csv (+ .meta.json)

Usage: python ffscan_filter_picks.py --net riehen|aargau|hautesorne
                                     [--inputs-dir .../inputs/ffscan]
                                     [--thresholds 1.0,1.5,2.0,2.5,3.0]
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--net", required=True, choices=("riehen", "aargau", "hautesorne"))
ap.add_argument("--inputs-dir", default=None,
                help="dir with the base exports (default {project}/tomo/1_velocity_maps/inputs/ffscan)")
ap.add_argument("--thresholds", default="1.0,1.5,2.0,2.5,3.0")
ap.add_argument("--min-n", type=int, default=30,
                help="min picks at a period for its median to seed v_ref (sparser rungs "
                     "are interpolated from the smoothed curve)")
ap.add_argument("--smooth", type=int, default=5,
                help="rolling-median width (in scale rungs) applied to v_ref(T)")
args = ap.parse_args()

indir = args.inputs_dir or os.path.join(EHM, args.net, "tomo", "1_velocity_maps",
                                        "inputs", "ffscan")
thresholds = [float(x) for x in args.thresholds.split(",")]

bases = sorted(glob.glob(os.path.join(indir, "picks_*_uni_nf.csv"))
               + glob.glob(os.path.join(indir, "picks_*_uni_nf_phase.csv")))
if not bases:
    raise SystemExit(f"no base tables picks_*_uni_nf[_phase].csv in {indir}")

for fn in bases:
    base = os.path.basename(fn)
    wave = base.split("picks_")[1].split("_uni")[0]
    phase = base.endswith("_nf_phase.csv")
    d = pd.read_csv(fn)
    if not len(d):
        print(f"{base}: empty, skipped")
        continue
    # reference lambda(T) = v_ref(T)*T from the BASE pool, smoothed over the period ladder
    # (a raw per-period median jitters where a rung is sparse, which would put a staircase
    # into the distance threshold). Identical for every threshold arm -> arms stay nested.
    g = d.groupby("inst_period")["group_velocity"]
    med, n = g.median(), g.size()
    good = med[n >= args.min_n]
    Ts = np.sort(d["inst_period"].unique())
    if len(good) >= 3:
        sm = good.rolling(args.smooth, center=True, min_periods=1).median()
        vref = pd.Series(np.interp(Ts, sm.index.values, sm.values), index=Ts)
    else:
        vref = med.reindex(Ts).ffill().bfill()
    lam_ref = vref * vref.index
    rml = d["distance"].values / lam_ref.reindex(d["inst_period"]).values
    tag = "_phase" if phase else ""
    for X in thresholds:
        keep = d[rml >= X]
        out = os.path.join(indir, f"picks_{wave}_uni{tag}_ff{X:.1f}.csv")
        keep.to_csv(out, index=False, float_format="%.4f")
        with open(out + ".meta.json", "w") as fh:
            json.dump({"generator": "ffscan_filter_picks.py", "source": fn,
                       "measure": "phase" if phase else "group", "wave": wave,
                       "min_r_over_lambda": X, "lambda_convention":
                       "reference curve: distance >= X * v_ref(T) * T, v_ref = smoothed "
                       "median velocity per period of the base pool (NOT per-pick -- a "
                       "per-pick lambda is velocity- and hence spatially-selective)",
                       "smooth_rungs": args.smooth, "min_n": args.min_n,
                       "no_upper_cap": True, "no_period_cap": True,
                       "rows": int(len(keep)), "rows_base": int(len(d)),
                       "pairs": int(keep.station_pair.nunique())}, fh, indent=2)
        print(f"{args.net} {wave:8s} {'phase' if phase else 'group':5s} ff{X:.1f}: "
              f"{len(keep):>9,}/{len(d):,} rows "
              f"({100 * len(keep) / len(d):5.1f}%) -> {os.path.basename(out)}")
