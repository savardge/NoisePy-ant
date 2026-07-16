"""Export QC'd UNIFIED picks to swtomotv tomography inputs (fund / overtone / love per network).

Reads {project}/dispersion_unified/picks_unified_QCd.csv (qc_unified_picks.py output; the
pick-level QC — snr, 2-lambda far-field, suppression, T-floored leak vetoes, scale dedupe — is
already applied there) and writes, into {project}/tomo/:

    --measure group (default)          --measure phase
    picks_fund_uni.csv                 picks_fund_uni_phase.csv       (rayleigh fundamental)
    picks_overtone_uni.csv             picks_overtone_uni_phase.csv   (rayleigh overtone)
    picks_love_uni.csv                 picks_love_uni_phase.csv       (love fundamental)

in the swtomotv canonical PickColumns schema (station_pair, stasrc, starcv, inst_period,
group_velocity, std, count, std_percent, distance, azimuth).

⚠ SCHEMA TRAP: swtomotv's column is named `group_velocity` whatever it holds, so a phase export
writes PHASE velocity into a column called `group_velocity`, and the two files' headers are
byte-identical. The ONLY in-band discriminators are the filename suffix and the sidecar
`<output>.meta.json` this script writes. Never infer measure from the file contents.

Aggregation: one row per (pair, T) = median over the surviving picks (argmax + topology streams,
post-dedupe). std/count are the real cross-method spread. (pair, T) cells with count >= 2 and
std > --max-std are DROPPED as branch-ambiguous (median of two distinct ridges is meaningless);
singles pass with std=0. Production policy: pairs touching a station flagged in
{project}/station_qc.csv are excluded (--keep-flagged to override).

THE PERIOD AXIS DIFFERS BY MEASURE, and this is not cosmetic:
  group -> `nominal_period`.
  phase -> `round(T_scale, 1)`, the discrete CWT scale the phase was actually measured on.
Phase is read off discrete wavelet scales, so T_scale is its true period axis; nominal_period is
a label. Grouping phase by nominal_period silently merges picks from different scales and moves
~0.5% of rows to the wrong period. (Recovered 2026-07-16 by reproducing the lost Jul-14 export;
see VELOCITY bounds note below.)

VELOCITY bounds are applied HERE and differ by measure — the QC script's vbounds were relaxed to
5.0 so valid long-period phase picks are not clipped:
  group: fund/love <= 3.6, overtone 1.5-4.5   (a ~5 km/s "fundamental group velocity" on these
                                               paths is a contaminant, not a surface wave)
  phase: fund/love 0.6-4.5, overtone 1.6-5.0  (c > U, so the phase bounds sit higher)

Usage:  python export_unified_tomo_picks.py --net riehen|aargau [--measure group|phase]
                                            [--max-std 0.2] [--keep-flagged] [--out-suffix ...]

--out-suffix renames the outputs to picks_{wave}_uni{suffix}.csv; the defaults ('' for group,
'_phase' for phase) reproduce the production filenames exactly. Control runs (--keep-flagged)
MUST pass a suffix, otherwise they overwrite the production pick tables.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
WAVES = {"fund": ("rayleigh", "fundamental"),
         "overtone": ("rayleigh", "overtone"),
         "love": ("love", "fundamental")}
# velocity bounds per (measure, wave) -- see the module docstring for why they differ
VBOUNDS = {"group": {"fund": (0.5, 3.6), "overtone": (1.5, 4.5), "love": (0.5, 3.6)},
           "phase": {"fund": (0.6, 4.5), "overtone": (1.6, 5.0), "love": (0.6, 4.5)}}
# (velocity column, ok flag, period column) per measure. The period axis is NOT shared: phase is
# measured on discrete CWT scales, so T_scale (rounded to the 0.1 s map grid) is its real period.
MEASURE = {"group": ("group_velocity", "group_ok", "nominal_period"),
           "phase": ("phase_velocity", "phase_ok", "T_scale")}

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--net", required=True, choices=("riehen", "aargau"))
ap.add_argument("--measure", default="group", choices=("group", "phase"),
                help="which velocity to export (default group). phase reads phase_velocity/"
                     "phase_ok on the T_scale period axis and defaults --out-suffix to '_phase'.")
ap.add_argument("--max-std", type=float, default=0.2,
                help="drop (pair,T) with count>=2 and cross-method std above this [km/s]")
ap.add_argument("--keep-flagged", action="store_true",
                help="keep pairs touching station_qc.csv-flagged stations")
ap.add_argument("--out-suffix", default=None,
                help="suffix appended to the output basenames: picks_{wave}_uni{suffix}.csv "
                     "(defaults: '' for group, '_phase' for phase = the production filenames). "
                     "Use it for control/experiment runs (e.g. --keep-flagged --out-suffix "
                     "_keepflag) so production picks are not overwritten.")
ap.add_argument("--outdir", default=None,
                help="where to write the pick CSVs. Default {project}/tomo -- but pass the new "
                     "inputs dir explicitly after the tomo/ reorg, or the picks land in the old "
                     "tomo/ root (which still exists) while the YAMLs read the new location.")
args = ap.parse_args()
if args.out_suffix is None:
    args.out_suffix = "_phase" if args.measure == "phase" else ""
VCOL, OKCOL, TCOL = MEASURE[args.measure]

proj = os.path.join(EHM, args.net)
src = os.path.join(proj, "dispersion_unified", "picks_unified_QCd.csv")
outdir = args.outdir or os.path.join(proj, "tomo")
os.makedirs(outdir, exist_ok=True)
print(f"reading {src} (measure={args.measure}, velocity={VCOL}, period={TCOL}) ...")
df = pd.read_csv(src, usecols=["pair", TCOL, VCOL, "wave_type", "mode", OKCOL,
                               "distance", "azimuth"])
# map grid is 0.1 s; T_scale is continuous, nominal_period already discrete (round is a no-op)
df["_T"] = df[TCOL].round(1)

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
    lo, hi = VBOUNDS[args.measure][wkey]
    d = df[(df.wave_type == wt) & (df["mode"] == md) & (df[OKCOL] == 1)
           & df[VCOL].between(lo, hi)]
    g = d.groupby(["pair", "_T"])
    A = g.agg(group_velocity=(VCOL, "median"), std=(VCOL, "std"),
              count=(VCOL, "size"), distance=("distance", "first"),
              azimuth=("azimuth", "first")).reset_index()
    A["std"] = A["std"].fillna(0.0)
    n0 = len(A)
    A = A[~((A["count"] >= 2) & (A["std"] > args.max_std))]        # branch-ambiguous cells
    A["stasrc"] = A["pair"].str.split("_").str[0]
    A["starcv"] = A["pair"].str.split("_").str[-1]
    A = A.rename(columns={"pair": "station_pair", "_T": "inst_period"})
    A["std_percent"] = (100 * A["std"] / A["group_velocity"]).round(2)
    A["count"] = A["count"].astype(int)
    fn = os.path.join(outdir, f"picks_{wkey}_uni{args.out_suffix}.csv")
    A[cols].to_csv(fn, index=False, float_format="%.4f")
    # sidecar: the CSV column is called group_velocity whatever it holds, so record the truth
    with open(fn + ".meta.json", "w") as fh:
        json.dump({"measure": args.measure, "velocity_type": args.measure,
                   "velocity_column_is_misnamed": "group_velocity holds %s velocity" % args.measure,
                   "source": src, "source_column": VCOL, "ok_column": OKCOL,
                   "period_axis": TCOL, "period_rounding": 1, "wave": wkey,
                   "wave_type": wt, "mode": md, "vbounds_km_s": [lo, hi],
                   "max_std": args.max_std, "flagged_excluded": sorted(flagged),
                   "rows": int(len(A)), "pairs": int(A.station_pair.nunique()),
                   "generator": "export_unified_tomo_picks.py"}, fh, indent=2)
    print(f"{wkey:9s}: {len(A):,} (pair,T) rows ({n0 - len(A):,} branch-ambiguous dropped) | "
          f"{A.station_pair.nunique():,} pairs | T {A.inst_period.min():.1f}-"
          f"{A.inst_period.max():.1f} s | {args.measure} {A.group_velocity.min():.2f}-"
          f"{A.group_velocity.max():.2f} -> {fn}")
