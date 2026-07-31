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

THE PERIOD AXIS IS THE PICKER'S CWT SCALE LADDER (`T_scale`) for BOTH measures -- uneven,
log-spaced at ~5.95%, e.g. ... 2.031, 2.153, 2.281, 2.416, 2.560 ... s. Rationale (2026-07-27):
the picks are measured ON those scales; the 0.1 s `nominal_period` grid is only a label the
FTAN image is interpolated onto. Forcing that uniform grid corrupts BOTH ends of the band:
  * T > ~1.7 s (scales coarser than 0.1 s): one scale is split across two nominal periods, so
    the tomography builds two STARVED maps from one measurement -- 43 of 89 Haute-Sorne group
    maps, with ray counts split e.g. 69/385/391/2381.
  * T < ~1.7 s (scales finer than 0.1 s): 10-11 scales per network collapse onto a shared
    nominal period and never get their own map.
Downstream, swtomotv must render periods with enough decimals to keep the rungs distinct
(`period_decimals: 3` in the dataset YAML); at 1 decimal two scales silently share a cache
file. `--period-axis nominal` restores the legacy behaviour.

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
         "love": ("love", "fundamental"),
         "love_ot": ("love", "overtone")}
# velocity bounds per (measure, wave) -- see the module docstring for why they differ
VBOUNDS = {"group": {"fund": (0.5, 3.6), "overtone": (1.5, 4.5), "love": (0.5, 3.6),
                     "love_ot": (1.5, 4.5)},
           "phase": {"fund": (0.6, 4.5), "overtone": (1.6, 5.0), "love": (0.6, 4.5),
                     "love_ot": (1.6, 5.0)}}
# (velocity column, ok flag, period column) per measure. BOTH measures are exported on the
# picker's native CWT scale ladder (`T_scale`), not on the 0.1 s nominal grid -- see the
# --period-axis note in the docstring.
MEASURE = {"group": ("group_velocity", "group_ok", "T_scale"),
           "phase": ("phase_velocity", "phase_ok", "T_scale")}

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--net", required=True, choices=("riehen", "aargau", "hautesorne"))
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
ap.add_argument("--period-axis", default="scale", choices=("scale", "nominal"),
                help="scale (default) = the picker's native CWT scale ladder (uneven, "
                     "log-spaced ~5.95%%); nominal = the legacy uniform 0.1 s FTAN grid "
                     "for group. See the docstring for why scale is correct.")
ap.add_argument("--vbounds", default=None,
                help="override the built-in VBOUNDS for the chosen measure, as "
                     "'fund=0.2:3.6,overtone=1.5:4.5,love=0.2:3.6[,love_ot=1.5:4.5]'. "
                     "Waves not listed keep the built-in bounds. The built-in group floor "
                     "is 0.5 -- on the vmin=0.2 pick trees pass the 0.2 floor explicitly "
                     "or slow picks are silently re-clipped here (third-floor trap).")
ap.add_argument("--bounds-file", default=None,
                help="period-DEPENDENT bounds JSON from pick_vbounds.py; applied to the "
                     "chosen measure's velocity before aggregation. Complements --vbounds "
                     "(both apply). Ignored with a warning if the file does not exist.")
ap.add_argument("--src", default=None,
                help="alternate picks_unified_QCd.csv to read (default "
                     "{project}/dispersion_unified/picks_unified_QCd.csv). Use for control QC "
                     "runs, e.g. the ffscan --farfield 1.0 re-run; pair it with --out-suffix.")
args = ap.parse_args()
if args.out_suffix is None:
    args.out_suffix = "_phase" if args.measure == "phase" else ""
VCOL, OKCOL, TCOL = MEASURE[args.measure]
if args.period_axis == "nominal":            # legacy behaviour, group only
    TCOL = "nominal_period" if args.measure == "group" else "T_scale"

if args.vbounds:
    for part in args.vbounds.split(","):
        wkey, rng = part.split("=")
        lo, hi = rng.split(":")
        if wkey not in VBOUNDS[args.measure]:
            raise SystemExit("unknown wave key %r in --vbounds" % wkey)
        VBOUNDS[args.measure][wkey] = (float(lo), float(hi))
    print("vbounds override:", VBOUNDS[args.measure])

proj = os.path.join(EHM, args.net)
src = args.src or os.path.join(proj, "dispersion_unified", "picks_unified_QCd.csv")
outdir = args.outdir or os.path.join(proj, "tomo")
os.makedirs(outdir, exist_ok=True)
print(f"reading {src} (measure={args.measure}, velocity={VCOL}, period={TCOL}) ...")
df = pd.read_csv(src, usecols=["pair", TCOL, VCOL, "wave_type", "mode", OKCOL,
                               "distance", "azimuth"])
# On the scale axis the period IS the CWT scale: keep its value, rounding only to strip
# float noise so the ladder collapses to its ~47 discrete rungs. On the legacy nominal axis
# the 0.1 s grid is the label.
df["_T"] = df[TCOL].round(4 if args.period_axis == "scale" else 1)

if args.bounds_file:
    if not os.path.exists(args.bounds_file):
        print("WARNING: --bounds-file %s does not exist -- skipping" % args.bounds_file)
    else:
        from pick_vbounds import load_bounds, apply_bounds
        keep = apply_bounds(load_bounds(args.bounds_file), df["wave_type"], df["mode"],
                            df[TCOL], df[VCOL])
        print("bounds_file: dropping %s of %s rows" % (format(int((~keep).sum()), ","),
                                                       format(len(df), ",")))
        df = df[keep]

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
