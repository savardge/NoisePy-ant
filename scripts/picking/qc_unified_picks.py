"""Merge-time QC gating for unified dispersion picks -- with a per-gate rejection budget.

Reads the <pair>_unified.csv tree written by dispersion_unified.py (schema v2 with the QC columns
xmode_amp / ot_flag / dU_rayfund / dU_rayot / mode_overlap / env_ratio / scale_j) and applies the
ordered gate set below. Nothing is dropped silently: every gate's kill count is reported per
(wave, mode) and per measure (group / phase) in qc_rejection_budget.txt.

A row can survive as a group pick but lose its phase (or vice versa): the output carries
`group_ok` / `phase_ok` (int 1/0) columns and keeps rows where either survives. Downstream
consumers filter on those columns.

Gates (defaults; disable any with --disable g1,g2,... / tune via CLI):
    snr           all    : snr_nbG >= 5
    vbounds       all    : fundamental 0.5-5.0 km/s, overtone 1.5-5.0 (group U and phase c each;
                           kept loose so histograms are not truncated -- tighten via CLI if needed)
    farfield      group  : ratio_d_lambda >= 2.0    (deep-LVZ finding; phase stays at picker's 1-lambda)
    suppression   rayleigh OVERTONE only: xmode_amp <= 0.6 (mutual suppression, validate_modes
                           CONTRAST_MAX). NOT applied to the fundamental -- at long T / short paths
                           the stacks cannot separate the modes, so the dominant fundamental fails
                           suppression spuriously (28-42%% of G_LR0 picks at T>1.5 s on Aargau);
                           the production validator confirmed the fundamental by raw consensus, not
                           suppression.
    ot_res        rayleigh overtone: drop ot_flag in {slow, unresolved}
    love_env      love   : env_ratio >= --env-min, DISABLED by default (env-min 0). Aargau evidence:
                           Love phase picks sitting ON the reference have median env_ratio 0.88 --
                           an absolute >=1.5 gate kills 74%% of provably good picks. Cross-term SNR
                           comparable to TT is normal on short paths; env_ratio stays as a column
                           for joint merge-time tuning, and rf_leak/ot_leak are the actual
                           contamination discriminators.
    love_overlap  love overtone: mode_overlap == 0
    rf_leak       love   : drop |dU_rayfund| <= 0.15 (R-fundamental-on-TT leakage fingerprint),
                           but ONLY where the veto is diagnostic. AUDIT FINDING (2026-07-14, full
                           Aargau): unconditional, the kill fraction rises monotonically with
                           period (0.10 at T<1 s -> 0.40 at 3-4 s) because Love-fund and R-fund
                           group velocities genuinely CONVERGE at long T -- an unconditional veto
                           censors the surviving Love set AGAINST Love~Rayleigh and would
                           manufacture spurious gamma(z) anisotropy at depth. Fix: the veto fires
                           at T <= --leak-tmax (ALWAYS -- leakage physics is short-period; the
                           T-cap is a floor) OR where |ref_love_group(T) - ref_rayfund_group(T)|
                           > --leak-sep-factor x --leak-tol (refs from --ref-dir, phase->group via
                           the dispersion relation) -- separation extends the veto to longer T
                           only where the network curves genuinely diverge. (Re-verify 2026-07-14:
                           the sep-only form under-vetoed Riehen's graben band at 1.4-1.6 s because
                           its network-average Love ref is east-fast-dominated; the T-floor fixes
                           that while still killing the long-T censoring bias.)
    ot_leak       love   : drop |dU_rayot| <= 0.15 (R-overtone-on-TT leakage), same
                           separation-conditioned logic vs ref_overtone_phase.
    phase_phys    phase  : phase_velocity > group_velocity (2*pi*N branch physicality)
    scale_dedupe  phase  : one phase pick per (pair, component, lag, mode, scale_j) -- same CWT
                           scale = same measurement (phase-step2); keeps the pick nearest T_scale
    group_scale_dedupe group: same principle for GROUP picks -- the FTAN image interpolates the
                           log-spaced CWT scales onto the 0.1 s nominal grid, so above ~2 s adjacent
                           nominal-period picks are the same measurement (Aargau: x1.35 duplication
                           at 2-3 s, x1.85 at 3-4 s, x2.28 at 4-6 s; within-scale U spread 0.02 km/s).
                           Key includes pick_method (both streams preserved) and a 0.2 km/s velocity
                           bin so distinct topology branches at one scale (different wave packets)
                           are never collapsed. The retained pick is the one nearest T_scale.
    station       all    : optional --station-qc csv -> flagged_sta column; --drop-flagged to kill

Usage:
    python qc_unified_picks.py --dir <dispersion_unified_dir> [--out-dir D] [--ref-dir R]
                               [--disable rf_leak,ot_leak] [--snr-min 5] [--farfield 2.0]
                               [--xmode-max 0.6] [--env-min 1.5] [--leak-tol 0.15]
                               [--station-qc station_qc.csv [--drop-flagged]]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--dir", required=True)
ap.add_argument("--out-dir", default=None)
ap.add_argument("--ref-dir", default=None, help="vsg_modesep dir (phase refs on the figure)")
ap.add_argument("--disable", default="", help="comma-separated gate names to skip")
ap.add_argument("--snr-min", type=float, default=5.0)
ap.add_argument("--vbounds-fund", default="0.5,5.0")
ap.add_argument("--vbounds-ot", default="1.5,5.0")
ap.add_argument("--farfield", type=float, default=2.0)
ap.add_argument("--xmode-max", type=float, default=0.6)
ap.add_argument("--env-min", type=float, default=0.0, help="0 = env gate off (see docstring)")
ap.add_argument("--leak-tol", type=float, default=0.15)
ap.add_argument("--leak-sep-factor", type=float, default=2.0,
                help="leak vetoes fire only where the Love/Rayleigh group REFERENCE curves are "
                     "separated by > factor*leak_tol (censoring-bias fix)")
ap.add_argument("--leak-tmax", type=float, default=2.0,
                help="fallback where refs are unavailable/converged-indeterminate: veto only T <= this")
ap.add_argument("--station-qc", default=None)
ap.add_argument("--drop-flagged", action="store_true")
args = ap.parse_args()
OUT = args.out_dir or args.dir
DISABLED = set(x.strip() for x in args.disable.split(",") if x.strip())
VB = {"fundamental": tuple(float(x) for x in args.vbounds_fund.split(",")),
      "overtone": tuple(float(x) for x in args.vbounds_ot.split(","))}

# ----------------------------------------------------------------------------- load
files = sorted(glob.glob(os.path.join(args.dir, "*", "*_unified.csv")))
if not files:
    raise SystemExit(f"no *_unified.csv under {args.dir}")
print(f"loading {len(files)} pair CSVs ...")
frames = []
for fn in files:
    try:
        d = pd.read_csv(fn)
    except Exception as e:
        print(f"  skip {os.path.basename(fn)}: {e}")
        continue
    if len(d) == 0:
        continue
    d["pair"] = os.path.basename(fn).replace("_unified.csv", "")
    frames.append(d)
df = pd.concat(frames, ignore_index=True)
if "xmode_amp" not in df.columns:
    raise SystemExit("old-schema CSVs (no xmode_amp column) -- rerun dispersion_unified.py first")
print(f"{len(df):,} rows, {df['pair'].nunique()} pairs")

# optional station flags
df["flagged_sta"] = 0
if args.station_qc:
    # station_qc.py writes the station code as the (unnamed) index, not a "station" column;
    # unflagged rows carry NaN in "flag", which str-casts to "nan" -- strip/compare to "" instead
    # of testing length. Same idiom as export_unified_tomo_picks.py.
    qc = pd.read_csv(args.station_qc, index_col=0)
    codes = qc["station"].astype(str) if "station" in qc.columns else qc.index.to_series().astype(str)
    if "flag" in qc.columns:
        flagged = set(codes[qc["flag"].fillna("").astype(str).str.strip() != ""])
    else:
        flagged = set(codes)
    print(f"station QC: {len(flagged)} flagged of {len(qc)}: {', '.join(sorted(flagged))}")
    def _nflag(pair):
        return sum(1 for s in pair.split("_") if s in flagged)
    df["flagged_sta"] = df["pair"].map(_nflag)

# ----------------------------------------------------------------------------- gate machinery
df["group_ok"] = np.isfinite(df["group_velocity"]) & (df["group_velocity"] > 0)
df["phase_ok"] = np.isfinite(df["phase_velocity"]) & (df["phase_velocity"] > 0)

budget = []          # (gate, wave, mode, measure, killed)


def apply_gate(name, fail_mask, measures=("group", "phase")):
    """Kill picks failing `fail_mask` on the given measures; record per-(wave,mode) kills."""
    if name in DISABLED:
        return
    for meas in measures:
        col = f"{meas}_ok"
        kill = df[col] & fail_mask
        if kill.any():
            for (w, m), n in kill.groupby([df["wave_type"], df["mode"]]).sum().items():
                if n:
                    budget.append((name, w, m, meas, int(n)))
        df.loc[kill, col] = False


is_love = df["wave_type"] == "love"
is_ray = df["wave_type"] == "rayleigh"
is_ot = df["mode"] == "overtone"

# 1 snr
apply_gate("snr", ~(df["snr_nbG"] >= args.snr_min))
# 2 velocity bounds (per measure, per mode)
for mode, (lo, hi) in VB.items():
    sel = df["mode"] == mode
    apply_gate("vbounds", sel & ~df["group_velocity"].between(lo, hi), measures=("group",))
    apply_gate("vbounds", sel & ~df["phase_velocity"].between(lo, hi), measures=("phase",))
# 3 far-field (group only; picker already gates phase at 1 lambda)
apply_gate("farfield", ~(df["ratio_d_lambda"] >= args.farfield), measures=("group",))
# 4 mutual suppression (Rayleigh OVERTONE only -- see docstring; fundamental fails it spuriously
#   wherever the path is too short to separate the modes)
apply_gate("suppression",
           is_ray & is_ot & np.isfinite(df["xmode_amp"]) & (df["xmode_amp"] > args.xmode_max))
# 5 overtone resolution flags (Rayleigh overtone)
apply_gate("ot_res", is_ray & is_ot & df["ot_flag"].isin(["slow", "unresolved"]))
# 6 Love cross-term energy (off by default; env_ratio is not an absolute discriminator here)
if args.env_min > 0:
    apply_gate("love_env", is_love & ~(df["env_ratio"] >= args.env_min))
# 7 Love overtone mode overlap
apply_gate("love_overlap", is_love & is_ot & (df["mode_overlap"] != 0))


# 8/9 Rayleigh-on-TT leakage vetoes, CONDITIONED on reference-curve separation (censoring-bias
# fix, see docstring): coincidence with the Rayleigh curve is only diagnostic of leakage where the
# network reference curves say Love and Rayleigh should differ; where they genuinely converge
# (long T), Love ~ Rayleigh is expected physics and must not be culled.
def _leak_diagnostic(ray_ref_name):
    """Boolean per-row mask: is the coincidence veto diagnostic at this row's period?"""
    Tarr = df["nominal_period"].to_numpy(float)
    sep = np.full(len(df), np.nan)
    if args.ref_dir:
        try:
            from noisepy import dispersion, unified_picking as up
            pg = np.arange(0.2, 8.0, 0.1)
            Ulove = up.phase_ref_to_group_ref(
                dispersion.load_reference_curve(os.path.join(args.ref_dir, "ref_love_phase.txt")), pg)
            Uray = up.phase_ref_to_group_ref(
                dispersion.load_reference_curve(os.path.join(args.ref_dir, ray_ref_name)), pg)
            if Ulove is not None and Uray is not None:
                sep = np.abs(np.asarray(Ulove(Tarr), float) - np.asarray(Uray(Tarr), float))
        except Exception as e:
            print(f"WARN: leak-veto reference separation unavailable ({e}); using T-cap only")
    # T-cap is a FLOOR, not just a fallback: leakage physics is short-period, so the veto is ALWAYS
    # diagnostic at T <= leak_tmax. Reference separation only EXTENDS it to longer T where the
    # network curves genuinely diverge. This matters in bimodal networks (Riehen W-slow/E-fast):
    # the network-average Love ref is east-dominated, so sep drops below threshold across the
    # diagnosed graben contamination band (1.4-1.6 s) even though the graben's LOCAL curves are
    # separated -- the T-floor keeps the veto active there.
    return (Tarr <= args.leak_tmax) | (np.isfinite(sep) & (sep > args.leak_sep_factor * args.leak_tol))


apply_gate("rf_leak", is_love & (df["dU_rayfund"].abs() <= args.leak_tol)
           & _leak_diagnostic("ref_fundamental_phase.txt"))
apply_gate("ot_leak", is_love & (df["dU_rayot"].abs() <= args.leak_tol)
           & _leak_diagnostic("ref_overtone_phase.txt"))
# 10 phase branch physicality
apply_gate("phase_phys", ~(df["phase_velocity"] > df["group_velocity"]), measures=("phase",))
# 11 station flags (optional kill; always carried as a column)
if args.drop_flagged:
    apply_gate("station", df["flagged_sta"] > 0)

# 12 scale dedupe (phase): same (pair, component, lag, scale_j) = one measurement.
if "scale_dedupe" not in DISABLED:
    ph = df[df["phase_ok"] & (df["scale_j"] >= 0)].copy()
    ph["pm_rank"] = (ph["pick_method"] != "argmax").astype(int)   # argmax preferred
    # among duplicates, keep the pick whose nominal period is closest to the scale's true Fourier
    # period T_scale -- the physically correct label for a single-scale measurement
    ph["dT_scale"] = (ph["nominal_period"] - ph["T_scale"]).abs()
    keep_idx = (ph.sort_values(["pm_rank", "dT_scale", "score"], ascending=[True, True, False])
                  .drop_duplicates(subset=["pair", "component", "lag", "mode", "scale_j"]).index)
    kill = df.index.isin(ph.index.difference(keep_idx))
    for (w, m), n in pd.Series(kill, index=df.index).groupby(
            [df["wave_type"], df["mode"]]).sum().items():
        if n:
            budget.append(("scale_dedupe", w, m, "phase", int(n)))
    df.loc[kill, "phase_ok"] = False

# 13 group scale dedupe: same single-scale logic for group picks (see docstring). The velocity bin
# keeps genuinely distinct branches at one scale (topology multi-ridge) as separate measurements.
if "group_scale_dedupe" not in DISABLED:
    gp = df[df["group_ok"] & (df["scale_j"] >= 0)].copy()
    gp["dT_scale"] = (gp["nominal_period"] - gp["T_scale"]).abs()
    gp["u_bin"] = (gp["group_velocity"] / 0.2).round().astype(int)
    keep_idx = (gp.sort_values(["dT_scale", "score"], ascending=[True, False])
                  .drop_duplicates(subset=["pair", "component", "lag", "mode", "pick_method",
                                           "scale_j", "u_bin"]).index)
    kill = df.index.isin(gp.index.difference(keep_idx))
    for (w, m), n in pd.Series(kill, index=df.index).groupby(
            [df["wave_type"], df["mode"]]).sum().items():
        if n:
            budget.append(("group_scale_dedupe", w, m, "group", int(n)))
    df.loc[kill, "group_ok"] = False

# ----------------------------------------------------------------------------- outputs
survivors = df[df["group_ok"] | df["phase_ok"]].copy()
survivors["group_ok"] = survivors["group_ok"].astype(int)
survivors["phase_ok"] = survivors["phase_ok"].astype(int)
out_csv = os.path.join(OUT, "picks_unified_QCd.csv")
survivors.to_csv(out_csv, index=False)

# rejection budget
bud = pd.DataFrame(budget, columns=["gate", "wave", "mode", "measure", "killed"])
lines = [f"QC rejection budget -- {len(files)} pairs, {len(df):,} input rows",
         f"gates disabled: {sorted(DISABLED) if DISABLED else 'none'}", ""]
order = [g for g in ["snr", "vbounds", "farfield", "suppression", "ot_res", "love_env",
                     "love_overlap", "rf_leak", "ot_leak", "phase_phys", "station",
                     "scale_dedupe", "group_scale_dedupe"] if g not in DISABLED]
for g in order:
    sub = bud[bud.gate == g]
    tot = int(sub["killed"].sum())
    lines.append(f"[{g}] total killed: {tot:,}")
    for _, r in sub.sort_values("killed", ascending=False).iterrows():
        lines.append(f"    {r['wave']:8s} {r['mode']:11s} {r['measure']:5s}: {r['killed']:,}")
lines.append("")
lines.append("survivors per (wave, mode):")
for (w, m), sub in survivors.groupby(["wave_type", "mode"]):
    lines.append(f"    {w:8s} {m:11s}: group {int(sub['group_ok'].sum()):,} | "
                 f"phase {int(sub['phase_ok'].sum()):,}")
report = "\n".join(lines)
with open(os.path.join(OUT, "qc_rejection_budget.txt"), "w") as f:
    f.write(report + "\n")
print("\n" + report)

# ----------------------------------------------------------------------------- before/after figure
refs = {}
if args.ref_dir:
    for key, fn in {("rayleigh", "fundamental"): "ref_fundamental_phase.txt",
                    ("rayleigh", "overtone"): "ref_overtone_phase.txt",
                    ("love", "fundamental"): "ref_love_phase.txt",
                    ("love", "overtone"): "ref_love_overtone_phase.txt"}.items():
        try:
            refs[key] = np.loadtxt(os.path.join(args.ref_dir, fn))
        except Exception:
            refs[key] = None

Tb = np.arange(0.2, 6.05, 0.1)
Vb = np.arange(0.5, 5.05, 0.05)
ROWS = [(w, m) for w in ("rayleigh", "love") for m in ("fundamental", "overtone")]
COLS = [("group", "before"), ("group", "after"), ("phase", "before"), ("phase", "after")]
fig, axs = plt.subplots(4, 4, figsize=(22, 17))
for ir, (w, m) in enumerate(ROWS):
    base = df[(df["wave_type"] == w) & (df["mode"] == m)]
    for ic, (meas, stage) in enumerate(COLS):
        ax = axs[ir, ic]
        vcol = "group_velocity" if meas == "group" else "phase_velocity"
        sub = base if stage == "before" else base[base[f"{meas}_ok"]]
        T, V = sub["nominal_period"].to_numpy(), sub[vcol].to_numpy()
        good = np.isfinite(T) & np.isfinite(V) & (V > 0)
        T, V = T[good], V[good]
        if len(T):
            H, xe, ye = np.histogram2d(T, V, bins=[Tb, Vb])
            pm = ax.pcolormesh(xe, ye, np.where(H.T > 0, H.T, np.nan), cmap="viridis",
                               norm=LogNorm())
            plt.colorbar(pm, ax=ax, label="picks / cell")
        r = refs.get((w, m))
        if meas == "phase" and r is not None and np.ndim(r) == 2 and len(r):
            ax.plot(r[:, 0], r[:, 1], "r--", lw=1.5)
        ax.set(title=f"{w} {m} -- {meas} {stage} (n={len(T):,})", xlim=(0.2, 6), ylim=(0.5, 5.0))
        if ic == 0:
            ax.set_ylabel("velocity [km/s]")
        if ir == 3:
            ax.set_xlabel("Period [s]")
fig.suptitle(f"Unified picks QC: before vs after ({len(files)} pairs)", y=0.995, fontsize=15)
fig.tight_layout(rect=(0, 0, 1, 0.99))
figpath = os.path.join(OUT, "qc_before_after.png")
fig.savefig(figpath, dpi=110)
plt.close(fig)
print(f"\nwrote {out_csv}\n      {os.path.join(OUT, 'qc_rejection_budget.txt')}\n      {figpath}")
