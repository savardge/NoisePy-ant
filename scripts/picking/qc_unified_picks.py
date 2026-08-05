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
                           Key is (pair, component, lag, mode, pick_method, scale_j) -- pick_method
                           keeps both streams. The retained pick is the one nearest T_scale.
                           CHANGED 2026-07-28: the 0.2 km/s velocity bin that used to be part of the
                           key is OFF by default (--u-bin 0.2 restores it). Both pickers emit exactly
                           one pick per (pair, component, lag, mode, nominal_period) -- measured on
                           Haute-Sorne, mean 1.000 / max 1 for argmax AND topology -- so several
                           picks at one scale_j are always duplicate nominal periods, not distinct
                           wave packets (99.6%% of argmax same-scale groups span < 0.2 km/s, median
                           0.030). The bin split those duplicates whenever they straddled an EDGE,
                           keeping two picks and piling a 1.3-1.5x count excess on 1.9/2.1/2.3/2.5
                           km/s. Removing it drops 4.0%% of picks and moves no median.
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

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--dir", required=True)
ap.add_argument("--out-dir", default=None)
ap.add_argument("--ref-dir", default=None, help="vsg_modesep dir (phase refs on the figure)")
ap.add_argument("--disable", default="", help="comma-separated gate names to skip")
ap.add_argument("--snr-min", type=float, default=5.0)
ap.add_argument("--vbounds-fund", default="0.5,5.0")
ap.add_argument("--vbounds-ot", default="1.5,5.0")
ap.add_argument("--farfield", type=float, default=1.5,
                help="LOWER d/lambda bound (group). CHANGED 2026-07-30: default 2.0 -> 1.5 "
                     "on the substack-jackknife evidence (repeatability optimum d/lambda "
                     "2-6, degrading below ~1.5). CAUTION: the 2.0 default came from the "
                     "deep-LVZ ACCURACY finding (near-field group picks bias slow at long "
                     "T); repeatability does not rule that bias out -- the ffscan campaign "
                     "is the arbiter. Restore --farfield 2.0 to keep the old behaviour.")
ap.add_argument("--farfield-max", type=float, default=10.0,
                help="UPPER d/lambda bound (group), NEW 2026-07-30: beyond ~10 wavelengths "
                     "pick repeatability degrades steeply (cycle skipping; jackknife sigma "
                     "climbs to ~0.6 km/s on Aargau). 0 disables.")
ap.add_argument("--xmode-max", type=float, default=0.6)
ap.add_argument("--env-min", type=float, default=0.0, help="0 = env gate off (see docstring)")
ap.add_argument("--leak-tol", type=float, default=0.15)
ap.add_argument("--leak-sep-factor", type=float, default=2.0,
                help="leak vetoes fire only where the Love/Rayleigh group REFERENCE curves are "
                     "separated by > factor*leak_tol (censoring-bias fix)")
ap.add_argument("--leak-tmax", type=float, default=2.0,
                help="fallback where refs are unavailable/converged-indeterminate: veto only T <= this")
ap.add_argument("--band-edge-rungs", type=int, default=1,
                help="drop picks sitting on the last N CWT rungs below the pair's band edge "
                     "Tmax = distance/vave. The ladder is clipped there, so the final rung is "
                     "an accident of path length and the pick sits on the limit. 0 = off. "
                     "Riehen cost at N=1: 2.28%% of group picks, reach 1.92 -> 1.81 s.")
ap.add_argument("--vave", type=float, default=3.0,
                help="must match the picker's Config.vave (3.0); sets Tmax = distance/vave.")
ap.add_argument("--u-bin", type=float, default=0.0,
                help="velocity-bin width [km/s] in the group_scale_dedupe key. 0 (default) "
                     "= no velocity term, one group pick per (pair, component, lag, mode, "
                     "method, scale_j). 0.2 restores the pre-2026-07-28 behaviour, which "
                     "imprinted a 1.3-1.5x count excess at 1.9/2.1/2.3/2.5 km/s.")
ap.add_argument("--group-scale-dedupe", action="store_true",
                help="re-enable the group scale dedupe (one group pick per pair per CWT "
                     "scale). OFF BY DEFAULT since 2026-07-30: the tomography export "
                     "aggregates by (pair, T_scale rung) with a median, which collapses "
                     "the duplicates anyway, and per-period maps never double-count "
                     "within a map -- so the dedupe only mattered for count statistics "
                     "and it stripes period histograms. Enable it if exporting on the "
                     "legacy NOMINAL axis, where duplicates would smear one measurement "
                     "across adjacent period maps.")
ap.add_argument("--fold-love-overtone", action="store_true",
                help="relabel love/overtone rows as love/fundamental before any gate runs, "
                     "reproducing the picker's PICK_LOVE_OVERTONE=False default (all TT "
                     "ridges stay fundamental, nothing dropped). Exact for group picks; "
                     "phase needs a re-pick because c_ref is chosen by mode.")
ap.add_argument("--short-vmax", default="",
                help="per-wave GROUP velocity ceiling below --short-vmax-tmax, as "
                     "'fund=2.0,love=2.0' (empty = off; omitted waves unaffected). At "
                     "T<1 s the Rayleigh/Love FUNDAMENTAL branches sit at 1.0-1.8 km/s, so "
                     "picks above ~2 km/s there are the fast spray that made the "
                     "T=0.2-0.3 s maps come out at a uniform ~3 km/s. Deliberately NOT "
                     "applied to the Rayleigh overtone, whose branch legitimately runs "
                     "1.8-3.5 km/s at short period -- a blanket cut there removes 26-29%% "
                     "of the overtone data, i.e. the branch itself. Group only: phase "
                     "velocity exceeds group, so the same ceiling would over-cut it.")
ap.add_argument("--short-vmax-tmax", type=float, default=1.0,
                help="period below which --short-vmax applies [s]")
ap.add_argument("--station-qc", default=None)
ap.add_argument("--drop-flagged", action="store_true")
ap.add_argument("--sigma-trim", type=float, default=0.0,
                help="also write qc_before_after_<K>sigma.png: raw vs QC gates vs a recursive "
                     "K-sigma velocity trim of the RAW picks per period bin (an ALTERNATIVE to "
                     "the gates, for comparison). 0 = off")
ap.add_argument("--core-audit", type=float, default=0.0,
                help="write qc_core_audit.txt: per-stream kill rate for CORE picks (within "
                     "this many km/s of the per-period median) vs tail picks, attributed to "
                     "the gate responsible. 0 = off")
ap.add_argument("--figs-only", action="store_true",
                help="skip writing picks_unified_QCd.csv and the budget (figure regeneration "
                     "only -- the CSV is deterministic, so rewriting multi-GB output is waste)")
args = ap.parse_args()
OUT = args.out_dir or args.dir
# Create it up front: everything below is expensive and the writes happen last.
os.makedirs(OUT, exist_ok=True)
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

# Fold the Love overtone stream back into the fundamental, reproducing what the picker does
# with Config.PICK_LOVE_OVERTONE = False (its default): every TT ridge stays fundamental,
# nothing is dropped. Use where the overtone label is not credible -- on Haute-Sorne the
# whole stream sits in T = 1.1-1.4 s at U >= 2.08 km/s, the only band where the overtone
# REFERENCE curve is defined, and it carves a matching hole out of the Love fundamental
# distribution (exactly the failure the picker's own comment warns about).
# Folding here rather than post-merge means the fundamental gates apply to these rows.
# NOTE: group picks come out identical to a re-pick with the flag off (label affects neither
# T nor U); PHASE picks do not, because the picker chooses c_ref by mode -- for exact phase,
# re-run dispersion_unified.py instead.
if args.fold_love_overtone:
    m = (df["wave_type"] == "love") & (df["mode"] == "overtone")
    if m.any():
        df.loc[m, "mode"] = "fundamental"
        df.loc[m, "mode_overlap"] = 0        # flag-off path emits overlap=0 for every ridge
    print(f"[fold_love_overtone] {int(m.sum()):,} love-overtone rows relabeled fundamental")

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

df["group_killer"] = ""
df["phase_killer"] = ""
budget = []          # (gate, wave, mode, measure, killed)


def apply_gate(name, fail_mask, measures=("group", "phase")):
    """Kill picks failing `fail_mask` on the given measures; record per-(wave,mode) kills.
    Also stamps the FIRST gate that killed each pick (`{meas}_killer`) so the rejections
    can be audited against the mode of the velocity distribution -- a gate that removes
    core (near-median) picks is doing something different from one that removes tails."""
    if name in DISABLED:
        return
    for meas in measures:
        col = f"{meas}_ok"
        kill = df[col] & fail_mask
        if kill.any():
            for (w, m), n in kill.groupby([df["wave_type"], df["mode"]]).sum().items():
                if n:
                    budget.append((name, w, m, meas, int(n)))
        df.loc[kill, f"{meas}_killer"] = name
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
_ff_bad = ~(df["ratio_d_lambda"] >= args.farfield)
if args.farfield_max > 0:
    _ff_bad |= df["ratio_d_lambda"] > args.farfield_max
apply_gate("farfield", _ff_bad, measures=("group",))
# 3b band edge. compute_cwt clips each pair's scale ladder at Tmax = dist/vave, so the LAST
#    surviving rung is set by the exact path length and the pick on it sits right on the limit
#    (Riehen: median nominal_period 0.90 s against median Tmax 0.90 s). Those picks are marginal
#    by construction and are what populate the freak rungs used by 2-41 pairs of 19,017. They
#    pass farfield (T = d/vave with U ~ 1 gives d/lambda ~ 3), so nothing else removes them.
#    Cost of dropping one rung on Riehen: 2.28% of surviving group picks, 19 of 19,017 pairs
#    emptied, median picks per pair 33 -> 32, median period reach 1.92 -> 1.81 s.
if args.band_edge_rungs > 0 and "band_edge" not in DISABLED:
    RUNG = 2.0 ** (1.0 / 12.0)                      # dj = 1/12 -> one rung of the ladder
    Tmax_pair = df["distance"] / args.vave
    apply_gate("band_edge",
               df["T_scale"].notna()
               & (df["T_scale"] * RUNG ** args.band_edge_rungs > Tmax_pair),
               measures=("group", "phase"))
# 3b short-period fast spray, per wave (user-specified 2026-07-31). Slow short-period
#    picks are NOT cut: unconsolidated sediments genuinely produce them, and their small
#    jackknife sigma reflects a clean repeatable arrival, not an artifact -- the weighting
#    problem those caused is handled by winsorizing sigma, not by deleting picks.
if args.short_vmax:
    _WKEY = {"fund": ("rayleigh", "fundamental"), "love": ("love", "fundamental"),
             "overtone": ("rayleigh", "overtone")}
    _Tv = df["nominal_period"].to_numpy(float)
    for _item in args.short_vmax.split(","):
        _item = _item.strip()
        if not _item:
            continue
        _k, _, _v = _item.partition("=")
        _k = _k.strip()
        if _k not in _WKEY:
            raise SystemExit("--short-vmax: unknown wave key %r (use %s)"
                             % (_k, "/".join(_WKEY)))
        _w, _m = _WKEY[_k]
        apply_gate("short_vmax",
                   (df["wave_type"] == _w) & (df["mode"] == _m)
                   & (_Tv < args.short_vmax_tmax)
                   & (df["group_velocity"].to_numpy(float) > float(_v)),
                   measures=("group",))

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
    df.loc[kill, "phase_killer"] = "scale_dedupe"
    df.loc[kill, "phase_ok"] = False

# 13 group scale dedupe: same single-scale logic for group picks (see docstring). The velocity bin
# keeps genuinely distinct branches at one scale (topology multi-ridge) as separate measurements.
if args.group_scale_dedupe and "group_scale_dedupe" not in DISABLED:
    gp = df[df["group_ok"] & (df["scale_j"] >= 0)].copy()
    gp["dT_scale"] = (gp["nominal_period"] - gp["T_scale"]).abs()
    # Velocity term in the key: DISABLED by default since 2026-07-28 (--u-bin 0.2 restores it).
    # It was meant to stop distinct wave packets at one scale being collapsed, but BOTH pickers
    # already emit exactly one pick per (pair, component, lag, mode, nominal_period) -- measured
    # on Haute-Sorne: mean 1.000, max 1 for argmax AND topology. So several picks sharing a
    # scale_j are always several nominal periods landing on the same CWT scale, i.e. duplicates:
    # 99.6% of argmax same-scale groups span < 0.2 km/s (median 0.030). The absolute bin therefore
    # almost never separated real packets; what it did do was split duplicates that straddled a
    # bin EDGE, so those groups kept two picks instead of one and piled a 1.3-1.5x excess onto
    # 1.9 / 2.1 / 2.3 / 2.5 km/s. Dropping it removes 4.0% of picks and leaves the median group
    # velocity unchanged to 4 decimals at every well-sampled period.
    key = ["pair", "component", "lag", "mode", "pick_method", "scale_j"]
    if args.u_bin > 0:
        gp["u_bin"] = (gp["group_velocity"] / args.u_bin).round().astype(int)
        key = key + ["u_bin"]
    # NOTE: `score` is constant 1.0 for every topology pick (57% of Haute-Sorne picks), so for
    # that stream this sort degenerates to dT_scale and the survivor is decided by row order.
    keep_idx = (gp.sort_values(["dT_scale", "score"], ascending=[True, False])
                  .drop_duplicates(subset=key).index)
    kill = df.index.isin(gp.index.difference(keep_idx))
    for (w, m), n in pd.Series(kill, index=df.index).groupby(
            [df["wave_type"], df["mode"]]).sum().items():
        if n:
            budget.append(("group_scale_dedupe", w, m, "group", int(n)))
    df.loc[kill, "group_killer"] = "group_scale_dedupe"
    df.loc[kill, "group_ok"] = False

# ----------------------------------------------------------------------------- outputs
survivors = df[df["group_ok"] | df["phase_ok"]].copy()
survivors["group_ok"] = survivors["group_ok"].astype(int)
survivors["phase_ok"] = survivors["phase_ok"].astype(int)
out_csv = os.path.join(OUT, "picks_unified_QCd.csv")
if args.figs_only:
    print(f"--figs-only: NOT rewriting {out_csv}")
else:
    survivors.to_csv(out_csv, index=False)

# rejection budget
bud = pd.DataFrame(budget, columns=["gate", "wave", "mode", "measure", "killed"])
lines = [f"QC rejection budget -- {len(files)} pairs, {len(df):,} input rows",
         f"gates disabled: {sorted(DISABLED) if DISABLED else 'none'}", ""]
order = [g for g in ["snr", "vbounds", "farfield", "band_edge", "suppression", "ot_res", "love_env",
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
if not args.figs_only:
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

def _scale_bins(scales, tmin, tmax, min_width=0.1):
    """Period-bin edges matched to the picker's CWT scale grid: geometric midpoints
    between the scales actually present, thinned to >= min_width (the linear FTAN
    step) so every bin holds at least one nominal period. Uniform 0.1 s bins sawtooth
    above ~2 s, where the log-spaced scales are sparser than the linear grid and
    scale-less bins collect only FTAN-stream picks. (ffscan_common has a net-keyed
    twin of this for the ffscan diagnostics.)"""
    s = np.unique(scales[np.isfinite(scales) & (scales > 0)])
    s = s[(s >= tmin / 1.2) & (s <= tmax * 1.2)]
    if len(s) < 3:
        return np.arange(tmin, tmax + 0.05, 0.1)
    e = np.concatenate([[s[0] * np.sqrt(s[0] / s[1])], np.sqrt(s[:-1] * s[1:]),
                        [s[-1] * np.sqrt(s[-1] / s[-2])]])
    while e[0] > tmin:
        e = np.concatenate([[e[0] * s[0] / s[1]], e])
    while e[-1] < tmax:
        e = np.concatenate([e, [e[-1] * s[-1] / s[-2]]])
    keep = [0]
    for i in range(1, len(e)):
        if e[i] - e[keep[-1]] >= min_width:
            keep.append(i)
    return e[keep]


Tb = _scale_bins(df["T_scale"].to_numpy(), 0.2, 6.0)
Vb = np.arange(0.5, 5.05, 0.05)
# Love overtone is NOT picked on these networks (unified_picking.PICK_LOVE_OVERTONE is
# False; the TT overtone label is not credible where the two Love modes overlap), so it
# gets no row -- an all-empty row only invites the reader to look for data that the
# workflow deliberately does not produce.
ROWS = [("rayleigh", "fundamental"), ("rayleigh", "overtone"), ("love", "fundamental")]
COLS = [("group", "before"), ("group", "after"), ("phase", "before"), ("phase", "after")]
fig, axs = plt.subplots(len(ROWS), 4, figsize=(22, 4.3 * len(ROWS)))
for ir, (w, m) in enumerate(ROWS):
    base = df[(df["wave_type"] == w) & (df["mode"] == m)]
    # histogram first, then draw: before/after of one measure share a LINEAR color
    # scale (saturated at the 99th percentile of occupied cells) so the panels are
    # directly comparable and dense short-period cells do not flatten the rest
    Hs = {}
    for meas, stage in COLS:
        vcol = "group_velocity" if meas == "group" else "phase_velocity"
        # phase is measured on the discrete CWT scales -- T_scale is its real period
        # axis; nominal_period is only a label (see export_unified_tomo_picks.py)
        tcol = "nominal_period" if meas == "group" else "T_scale"
        sub = base if stage == "before" else base[base[f"{meas}_ok"]]
        T, V = sub[tcol].to_numpy(), sub[vcol].to_numpy()
        good = np.isfinite(T) & np.isfinite(V) & (V > 0)
        Hs[(meas, stage)] = np.histogram2d(T[good], V[good], bins=[Tb, Vb])[0]
    vmax = {meas: max(float(np.percentile(H[H > 0], 99)) if (H > 0).any() else 1.0
                      for st in ("before", "after") for H in [Hs[(meas, st)]])
            for meas in ("group", "phase")}
    for ic, (meas, stage) in enumerate(COLS):
        ax = axs[ir, ic]
        vcol = "group_velocity" if meas == "group" else "phase_velocity"
        tcol = "nominal_period" if meas == "group" else "T_scale"
        sub = base if stage == "before" else base[base[f"{meas}_ok"]]
        T, V = sub[tcol].to_numpy(), sub[vcol].to_numpy()
        good = np.isfinite(T) & np.isfinite(V) & (V > 0)
        T, V = T[good], V[good]
        if len(T):
            H = Hs[(meas, stage)]
            pm = ax.pcolormesh(Tb, Vb, np.where(H.T > 0, H.T, np.nan), cmap="viridis",
                               vmin=0, vmax=vmax[meas])
            plt.colorbar(pm, ax=ax, extend="max", label="picks / cell")
        r = refs.get((w, m))
        if meas == "phase" and r is not None and np.ndim(r) == 2 and len(r):
            ax.plot(r[:, 0], r[:, 1], "r--", lw=1.5)
        ax.set(title=f"{w} {m} -- {meas} {stage} (n={len(T):,})", xlim=(0.2, 6), ylim=(0.5, 5.0))
        if ic == 0:
            ax.set_ylabel("velocity [km/s]")
        if ir == len(ROWS) - 1:
            ax.set_xlabel("Period [s]  (group: nominal; phase: T_scale)")
fig.suptitle(f"Unified picks QC: before vs after ({len(files)} pairs)", y=0.995, fontsize=15)
fig.tight_layout(rect=(0, 0, 1, 0.99))
figpath = os.path.join(OUT, "qc_before_after.png")
fig.savefig(figpath, dpi=110)
plt.close(fig)
print(f"\nwrote {figpath}" if args.figs_only else
      f"\nwrote {out_csv}\n      {os.path.join(OUT, 'qc_rejection_budget.txt')}\n      {figpath}")


# ------------------------------------------------- optional: gates vs k-sigma trim figure
def _recursive_trim(v, k, max_iter=100):
    """Recursive mean/std k-sigma rejection until stable; returns the keep mask."""
    keep = np.ones(v.size, bool)
    for _ in range(max_iter):
        m, s = v[keep].mean(), v[keep].std()
        if s == 0:
            break
        new = keep & (np.abs(v - m) <= k * s)
        if new.sum() == keep.sum():
            break
        keep = new
    return keep


if args.core_audit:
    # Does the gate battery remove CORE picks (near the per-period mode) or only tails?
    # Core = within +/- args.core_audit km/s of the per-period MEDIAN of the raw picks
    # (median, not mean: robust to the very tails under test). Reported per stream as
    # the core kill rate, then attributed to the gate that killed each core pick.
    lines = [f"core-vs-tail rejection audit (core = |v - median(v|T)| <= "
             f"{args.core_audit:g} km/s, raw per-period median)", ""]
    for w, m in ROWS:
        for meas in ("group", "phase"):
            vcol = "group_velocity" if meas == "group" else "phase_velocity"
            tcol = "nominal_period" if meas == "group" else "T_scale"
            sub = df[(df["wave_type"] == w) & (df["mode"] == m)
                     & np.isfinite(df[tcol]) & np.isfinite(df[vcol]) & (df[vcol] > 0)]
            if not len(sub):
                continue
            bi = np.clip(np.digitize(sub[tcol].to_numpy(), Tb) - 1, 0, len(Tb) - 2)
            med = pd.Series(sub[vcol].to_numpy()).groupby(bi).transform("median").to_numpy()
            core = np.abs(sub[vcol].to_numpy() - med) <= args.core_audit
            killed = ~sub[f"{meas}_ok"].to_numpy()
            nc, nt = int(core.sum()), int((~core).sum())
            if not nc:
                continue
            # what the tomography actually consumes is (pair, T) CELLS -- the export takes
            # one median per cell, so duplicate picks at the mode add no information
            cell_raw = sub.groupby(["pair", sub[tcol].round(1)]).ngroups
            surv = sub[sub[f"{meas}_ok"]]
            cell_qc = surv.groupby(["pair", surv[tcol].round(1)]).ngroups if len(surv) else 0
            lines.append(f"{w} {m} -- {meas}: core {nc:,} ({100 * killed[core].mean():.1f}% "
                         f"killed) | tail {nt:,} ({100 * killed[~core].mean():.1f}% killed)"
                         f"  ||  (pair,T) cells {cell_raw:,} -> {cell_qc:,} "
                         f"({100 * cell_qc / max(cell_raw, 1):.1f}% kept)")
            kil = sub[f"{meas}_killer"].to_numpy()[core & killed]
            for gate, n in pd.Series(kil).value_counts().items():
                lines.append(f"      {gate:20s} {n:>9,}  ({100 * n / nc:5.1f}% of core)")
    rep = "\n".join(lines)
    with open(os.path.join(OUT, "qc_core_audit.txt"), "w") as f:
        f.write(rep + "\n")
    print("\n" + rep)

if args.sigma_trim:
    K = args.sigma_trim
    SCOLS = [(meas, stage) for meas in ("group", "phase")
             for stage in ("before", "QC gates", f"{K:g}sigma")]
    fig, axs = plt.subplots(len(ROWS), 6, figsize=(31, 4.3 * len(ROWS)))
    for ir, (w, m) in enumerate(ROWS):
        base = df[(df["wave_type"] == w) & (df["mode"] == m)]
        Hs = {}
        for meas, stage in SCOLS:
            vcol = "group_velocity" if meas == "group" else "phase_velocity"
            tcol = "nominal_period" if meas == "group" else "T_scale"
            raw = base[np.isfinite(base[tcol]) & np.isfinite(base[vcol]) & (base[vcol] > 0)]
            if stage == "before":
                sub = raw
            elif stage == "QC gates":
                sub = raw[raw[f"{meas}_ok"]]
            else:                       # k-sigma trim of the RAW set, per period bin
                T, V = raw[tcol].to_numpy(), raw[vcol].to_numpy()
                keep = np.zeros(V.size, bool)
                bi = np.clip(np.digitize(T, Tb) - 1, 0, len(Tb) - 2)
                for ib in np.unique(bi):
                    sel = np.where(bi == ib)[0]
                    keep[sel] = (_recursive_trim(V[sel], K) if sel.size >= 5
                                 else np.ones(sel.size, bool))
                sub = raw[keep]
            Hs[(meas, stage)] = (np.histogram2d(sub[tcol].to_numpy(), sub[vcol].to_numpy(),
                                                bins=[Tb, Vb])[0], len(sub))
        vmax = {meas: max(float(np.percentile(H[H > 0], 99)) if (H > 0).any() else 1.0
                          for st in ("before", "QC gates", f"{K:g}sigma")
                          for H in [Hs[(meas, st)][0]])
                for meas in ("group", "phase")}
        for ic, (meas, stage) in enumerate(SCOLS):
            ax = axs[ir, ic]
            H, n = Hs[(meas, stage)]
            pm = ax.pcolormesh(Tb, Vb, np.where(H.T > 0, H.T, np.nan), cmap="viridis",
                               vmin=0, vmax=vmax[meas])
            plt.colorbar(pm, ax=ax, extend="max", label="picks / cell")
            r = refs.get((w, m))
            if meas == "phase" and r is not None and np.ndim(r) == 2 and len(r):
                ax.plot(r[:, 0], r[:, 1], "r--", lw=1.5)
            ax.set(title=f"{w} {m} -- {meas} {stage} (n={n:,})",
                   xlim=(0.2, 6), ylim=(0.5, 5.0))
            if ic == 0:
                ax.set_ylabel("velocity [km/s]")
            if ir == len(ROWS) - 1:
                ax.set_xlabel("Period [s]  (group: nominal; phase: T_scale)")
    fig.suptitle(f"Unified picks: raw vs QC gates vs recursive {K:g}-sigma trim "
                 f"({len(files)} pairs). The sigma trim is an ALTERNATIVE to the gates "
                 f"(applied to the raw picks per period bin), not applied on top of them.",
                 y=0.995, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    sigpath = os.path.join(OUT, f"qc_before_after_{K:g}sigma.png")
    fig.savefig(sigpath, dpi=110)
    plt.close(fig)
    print(f"      {sigpath}")
