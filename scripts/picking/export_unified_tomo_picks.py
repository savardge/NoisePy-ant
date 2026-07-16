"""Export QC'd UNIFIED picks to swtomotv tomography inputs (fund / overtone / love per network).

Reads {project}/dispersion_unified/picks_unified_QCd.csv (qc_unified_picks.py output; the
pick-level QC — snr, 2-lambda far-field, suppression, T-floored leak vetoes, scale dedupe — is
already applied there) and writes, into {project}/tomo/:

    picks_fund_uni.csv       rayleigh fundamental, group_ok==1
    picks_overtone_uni.csv   rayleigh overtone,    group_ok==1
    picks_love_uni.csv       love fundamental,     group_ok==1

in the swtomotv canonical PickColumns schema (station_pair, stasrc, starcv, inst_period,
group_velocity, std, count, std_percent, distance, azimuth).

Aggregation: one row per (pair, nominal_period) = median over the surviving picks (argmax +
topology streams, post-dedupe). std/count are the real cross-method spread. (pair, T) cells with
count >= 2 and std > --max-std are DROPPED as branch-ambiguous (median of two distinct ridges is
meaningless); singles pass with std=0. Production policy: pairs touching a station flagged in
{project}/station_qc.csv are excluded (validated 2026-07-08: interior stable, var_red improves).

GROUP-velocity bounds are applied HERE (fund/love <= 3.6, overtone 1.5-4.5 km/s): the QC script's
vbounds default was relaxed to 5.0 so valid long-period PHASE picks are not clipped, but this
export consumes group picks only, where the original calibrated bounds still hold (a ~5 km/s
"fundamental group velocity" on these paths is a contaminant, not a surface wave).

Usage:  python export_unified_tomo_picks.py --net riehen|aargau [--max-std 0.2] [--keep-flagged]
                                            [--out-suffix _keepflag]

--out-suffix renames the outputs to picks_{wave}_uni{suffix}.csv; the default '' reproduces the
production filenames exactly. Control runs (--keep-flagged) MUST pass a suffix, otherwise they
overwrite the production pick tables.
"""
import argparse
import os

import numpy as np
import pandas as pd

EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
WAVES = {"fund": ("rayleigh", "fundamental"),
         "overtone": ("rayleigh", "overtone"),
         "love": ("love", "fundamental")}
GROUP_VB = {"fund": (0.5, 3.6), "overtone": (1.5, 4.5), "love": (0.5, 3.6)}

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--net", required=True, choices=("riehen", "aargau"))
ap.add_argument("--max-std", type=float, default=0.2,
                help="drop (pair,T) with count>=2 and cross-method std above this [km/s]")
ap.add_argument("--keep-flagged", action="store_true",
                help="keep pairs touching station_qc.csv-flagged stations")
ap.add_argument("--out-suffix", default="",
                help="suffix appended to the output basenames: picks_{wave}_uni{suffix}.csv "
                     "(default '' = the production filenames). Use it for control/experiment "
                     "runs (e.g. --keep-flagged --out-suffix _keepflag) so production picks "
                     "are not overwritten.")
args = ap.parse_args()

proj = os.path.join(EHM, args.net)
src = os.path.join(proj, "dispersion_unified", "picks_unified_QCd.csv")
outdir = os.path.join(proj, "tomo")
print(f"reading {src} ...")
df = pd.read_csv(src, usecols=["pair", "nominal_period", "group_velocity", "wave_type", "mode",
                               "group_ok", "distance", "azimuth"])

flagged = set()
if not args.keep_flagged:
    qcf = os.path.join(proj, "station_qc.csv")
    if os.path.exists(qcf):
        q = pd.read_csv(qcf, index_col=0)
        flagged = set(q.index[q["flag"].fillna("").astype(str).str.strip() != ""].astype(str))
        print(f"excluding {len(flagged)} flagged stations: {', '.join(sorted(flagged))}")

if flagged:
    parts = df["pair"].str.split("_")
    df = df[~(parts.str[0].isin(flagged) | parts.str[-1].isin(flagged))]

cols = ["station_pair", "stasrc", "starcv", "inst_period", "group_velocity",
        "std", "count", "std_percent", "distance", "azimuth"]
for wkey, (wt, md) in WAVES.items():
    lo, hi = GROUP_VB[wkey]
    d = df[(df.wave_type == wt) & (df["mode"] == md) & (df.group_ok == 1)
           & df.group_velocity.between(lo, hi)]
    g = d.groupby(["pair", "nominal_period"])
    A = g.agg(group_velocity=("group_velocity", "median"), std=("group_velocity", "std"),
              count=("group_velocity", "size"), distance=("distance", "first"),
              azimuth=("azimuth", "first")).reset_index()
    A["std"] = A["std"].fillna(0.0)
    n0 = len(A)
    A = A[~((A["count"] >= 2) & (A["std"] > args.max_std))]        # branch-ambiguous cells
    A["stasrc"] = A["pair"].str.split("_").str[0]
    A["starcv"] = A["pair"].str.split("_").str[-1]
    A = A.rename(columns={"pair": "station_pair", "nominal_period": "inst_period"})
    A["std_percent"] = (100 * A["std"] / A["group_velocity"]).round(2)
    A["count"] = A["count"].astype(int)
    fn = os.path.join(outdir, f"picks_{wkey}_uni{args.out_suffix}.csv")
    A[cols].to_csv(fn, index=False, float_format="%.4f")
    print(f"{wkey:9s}: {len(A):,} (pair,T) rows ({n0 - len(A):,} branch-ambiguous dropped) | "
          f"{A.station_pair.nunique():,} pairs | T {A.inst_period.min():.1f}-"
          f"{A.inst_period.max():.1f} s | U {A.group_velocity.min():.2f}-"
          f"{A.group_velocity.max():.2f} -> {fn}")
