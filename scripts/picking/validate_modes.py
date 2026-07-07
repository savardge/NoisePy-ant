"""
Mode validation / QA-QC for the V6 dispersion picks.

Post-processes a per-pair dispersion CSV from dispersion_curves_V6_modesep.py (no re-measurement).
The Nayak & Thurber synthesis (G_LR0 = fundamental, G_LR1 = 1st higher mode) is only trusted where
it is independently corroborated by the picks already made on the single components and the
amplitude-product combos:

  Fundamental (G_LR0) is CONFIRMED at a period when its group velocity agrees (within TOL_V) with
  at least MIN_SUPPORT_FUND of {ZZ argmax, RR argmax, all4 argmax}. Those single-component argmax
  ridges already trace the fundamental where it dominates, so they are the independent witnesses.

  Overtone (G_LR1) is CONFIRMED at a period only if (a) it is faster than the fundamental and
  separated from it by more than SEP_MIN (else the modes are not resolved -- osculation / single
  mode), and (b) a *secondary* topology branch of ZZ and/or RR sits within TOL_V of it (the
  topology picker returns multiple branches per period -- the overtone is one of them).

Robustness uses the votes already in the CSV: velocities are aggregated (median) across the
stack methods (pws/linear/nroot) for the chosen lag, and the cross-stack spread is reported.

Outputs next to the input CSV:
  <spair>_modes_validated.csv : period, U_fund, conf_fund, fund_flag,
                                U_overtone, conf_overtone, ot_flag, snr_fund
  <spair>_modeQA.png          : all component picks overlaid with the confirmed modes circled.

Usage:  python validate_modes.py <dispersion_all.csv | directory_of_such_csvs>
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------- tunables
LAG = "sym"                 # lag to validate on (paper uses the symmetric component)
TOL_V = 0.20                # km/s: velocity match tolerance between images
SEP_MIN = 0.30              # km/s: floor of the fundamental<->overtone separation requirement
# The actual separation requirement is resolution-adaptive: two group arrivals are only resolved
# when their arrival-time difference exceeds the FTAN envelope width. For the Morlet (f0=6) CWT the
# envelope time-spread at period T is ~T, giving dU_min ~ U^2 * T / dist. Requirement used:
#   sep_req(T) = max(SEP_MIN, RES_FACTOR * U_fund^2 * T / dist)
# This is the osculation guard (Boaga 2013 / Boue 2016 discussion in Nayak & Thurber 2020): where
# the modes approach each other, or the path is too short to time-separate them, we abstain.
RES_FACTOR = 1.0            # envelope-widths of separation required (1 = one sigma_t)
# Mutual-suppression test (the paper's core physics): if U1 is a genuine prograde overtone, the
# retrograde stack G_LR0 must be WEAK at (T, U1) and G_LR1 weak at (T, U0). Checked on the exported
# per-period-normalized pws/sym _group.dat images when present (authoritative anchor test); when
# images are absent we fall back to the dual-branch topology criterion.
CONTRAST_MAX = 0.6          # max normalized amplitude of the OTHER mode's image at a pick
SNR_MIN = 3.0               # min narrowband SNR (snr_nbG) for a confirmed pick
MIN_SUPPORT_FUND = 2        # how many of {ZZ,RR,all4} must agree with G_LR0
MIN_SUPPORT_OT = 1          # how many of {ZZ,RR} topology branches must support G_LR1
FUND_SUPPORTERS = ["ZZ", "RR", "all4"]
OT_SUPPORTERS = ["ZZ", "RR"]


def _median_by_period(df, component, pick_method):
    """Median velocity (+spread, +median snr) per period for one component/pick_method, across
    stack methods, for the chosen lag. Returns dict period -> (U_median, U_std, snr_median)."""
    s = df[(df.component == component) & (df.pick_method == pick_method) & (df.lag == LAG)]
    out = {}
    for T, g in s.groupby("period_r"):
        out[T] = (float(np.median(g.group_velocity)),
                  float(np.std(g.group_velocity)),
                  float(np.median(g.snr_nbG)))
    return out


def _topology_branches(df, component):
    """All topology-branch velocities per period for one component (across stacks, chosen lag)."""
    s = df[(df.component == component) & (df.pick_method == "topology") & (df.lag == LAG)]
    out = {}
    for T, g in s.groupby("period_r"):
        out[T] = np.asarray(g.group_velocity, dtype=float)
    return out


def _load_glr_images(csv_path):
    """Load the exported G_LR0/G_LR1 group images (pws/sym) if present, for the suppression test.
    Accepts the batch-driver npz bundle or the V6 .dat/.npz exports.
    Returns (per, vel, {mode: image[n_per, n_vel]}) or None."""
    spair = os.path.basename(csv_path).replace("_dispersion_all.csv", "")
    imdir = os.path.join(os.path.dirname(csv_path), "images")
    npz = os.path.join(imdir, f"{spair}_glr_images.npz")
    if os.path.exists(npz):
        try:
            z = np.load(npz)
            return (np.asarray(z["period"]), np.asarray(z["velocity"]),
                    {"G_LR0": np.asarray(z["G_LR0"]), "G_LR1": np.asarray(z["G_LR1"])})
        except Exception:
            return None
    try:
        ax = np.load(os.path.join(imdir, f"{spair}_pws_G_LR0_sym_axes.npz"))
        per, vel = np.asarray(ax["period"]), np.asarray(ax["velocity"])
        A = {m: np.loadtxt(os.path.join(imdir, f"{spair}_pws_{m}_sym_group.dat")).T
             for m in ("G_LR0", "G_LR1")}          # .dat is (vel, per) -> transpose
        return per, vel, A
    except Exception:
        return None


def _amp_at(per, vel, img, T, U):
    """Nearest-node normalized image amplitude at (period T, velocity U)."""
    return float(img[int(np.argmin(np.abs(per - T))), int(np.argmin(np.abs(vel - U)))])


def validate_csv(csv_path, plot=True, verbose=True):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["period_r"] = df["nominal_period"].round(2)
    dist = float(df["distance"].iloc[0])
    glr_imgs = _load_glr_images(csv_path)

    g0 = _median_by_period(df, "G_LR0", "argmax")
    g1 = _median_by_period(df, "G_LR1", "argmax")
    fund_argmax = {c: _median_by_period(df, c, "argmax") for c in FUND_SUPPORTERS}
    ot_topo = {c: _topology_branches(df, c) for c in OT_SUPPORTERS}

    periods = sorted(set(g0) | set(g1))
    out_rows = []
    for T in periods:
        U0 = U0std = snr0 = np.nan
        conf_f = 0
        fund_flag = "no_glr0"
        if T in g0:
            U0, U0std, snr0 = g0[T]
            conf_f = sum(1 for c in FUND_SUPPORTERS
                         if T in fund_argmax[c] and abs(fund_argmax[c][T][0] - U0) <= TOL_V)
            if not (snr0 >= SNR_MIN):          # NaN-safe: NaN SNR must NOT pass the gate
                fund_flag = "low_snr"
            elif conf_f >= MIN_SUPPORT_FUND:
                fund_flag = "ok"
            else:
                fund_flag = "unconfirmed"

        U1 = U1std = np.nan
        conf_o = 0
        ot_flag = "no_glr1"
        if T in g1:
            U1, U1std, _ = g1[T]
            if np.isnan(U0):
                ot_flag = "no_fundamental_ref"
            else:
                # resolution-adaptive separation requirement (see RES_FACTOR note above)
                sep_req = max(SEP_MIN, RES_FACTOR * U0 ** 2 * T / dist)
                if (U1 - U0) <= sep_req:       # not resolved as a distinct faster mode
                    ot_flag = "unseparated"
                    U1 = np.nan                # do not report a velocity we don't trust as a mode
                else:
                    # (1) The overtone pick must be OBSERVED in the raw data: a single-component
                    # topology branch near U1 and away from U0.
                    conf_o = 0
                    fund_support = False
                    for c in OT_SUPPORTERS:
                        br = ot_topo.get(c, {}).get(T, np.empty(0))
                        if br.size:
                            if np.any(np.abs(br - U0) <= TOL_V):
                                fund_support = True
                            if np.any((np.abs(br - U1) <= TOL_V) & (np.abs(br - U0) > sep_req)):
                                conf_o += 1
                    if conf_o < MIN_SUPPORT_OT:
                        ot_flag = "glr1_unconfirmed"
                    # (2) Anchor validity. Preferred: MUTUAL SUPPRESSION on the G_LR images --
                    # a genuine prograde overtone is suppressed in the retrograde stack (G_LR0
                    # weak at U1) and vice versa (G_LR1 weak at U0). This is the discriminating
                    # physics of the +-pi/2 stack, and catches the mislabel failure where G_LR0
                    # picks an artifact and "U1" is really the fundamental. Fallback when no
                    # images were exported: require the U0 anchor branch to be observed too
                    # (dual-branch criterion).
                    elif glr_imgs is not None:
                        per_i, vel_i, A = glr_imgs
                        a_fund_at_U1 = _amp_at(per_i, vel_i, A["G_LR0"], T, U1)
                        a_ot_at_U0 = _amp_at(per_i, vel_i, A["G_LR1"], T, U0)
                        if a_fund_at_U1 <= CONTRAST_MAX and a_ot_at_U0 <= CONTRAST_MAX:
                            ot_flag = "ok"
                        else:
                            ot_flag = "mode_mixing"    # stacks did not separate cleanly here
                    elif fund_support:
                        ot_flag = "ok"
                    else:
                        ot_flag = "glr0_unsupported"   # U0 anchor unseen in any raw component

        out_rows.append((T, U0, U0std, conf_f, fund_flag, U1, U1std, conf_o, ot_flag, snr0))

    cols = ["period", "U_fund", "U_fund_std", "conf_fund", "fund_flag",
            "U_overtone", "U_ot_std", "conf_overtone", "ot_flag", "snr_fund"]
    out = pd.DataFrame(out_rows, columns=cols)

    base = csv_path.replace("_dispersion_all.csv", "")
    out_csv = base + "_modes_validated.csv"
    out.to_csv(out_csv, index=False, float_format="%.3f")

    if plot:
        _qa_plot(df, out, base, g0, g1)
    n_ok_f = int((out.fund_flag == "ok").sum())
    n_ok_o = int((out.ot_flag == "ok").sum())
    if verbose:
        print(f"{os.path.basename(csv_path)}: {len(out)} periods | confirmed fundamental "
              f"{n_ok_f}, confirmed overtone {n_ok_o} -> {os.path.basename(out_csv)}")
    return out


def _qa_plot(df, out, base, g0, g1):
    sub = df[df.lag == LAG]
    fig, ax = plt.subplots(figsize=(11, 6))
    # background: every component's picks, faint
    def scat(comp, pm, **kw):
        s = sub[(sub.component == comp) & (sub.pick_method == pm)]
        if len(s):
            ax.scatter(s.nominal_period, s.group_velocity, **kw)
    scat("ZZ", "topology", s=10, c="#9ecae1", label="ZZ/RR topology branches", zorder=1)
    scat("RR", "topology", s=10, c="#9ecae1", zorder=1)
    scat("ZZ", "argmax", s=14, c="0.55", marker=".", label="ZZ argmax", zorder=2)
    scat("RR", "argmax", s=14, c="0.35", marker="x", label="RR argmax", zorder=2)
    scat("all4", "argmax", s=18, c="green", marker="^", label="all4 product", zorder=2)
    # synthesized modes
    if g0:
        T0 = sorted(g0); ax.plot(T0, [g0[t][0] for t in T0], "-o", c="red", ms=4,
                                 label="G_LR0 (fundamental)", zorder=4)
    if g1:
        T1 = sorted(g1); ax.plot(T1, [g1[t][0] for t in T1], "-o", c="darkorange", ms=4,
                                 label="G_LR1 (1st higher)", zorder=4)
    # circle confirmed picks
    okf = out[out.fund_flag == "ok"]
    ax.scatter(okf.period, okf.U_fund, s=130, facecolors="none", edgecolors="red", lw=1.6,
               label="confirmed fundamental", zorder=5)
    oko = out[out.ot_flag == "ok"]
    ax.scatter(oko.period, oko.U_overtone, s=130, facecolors="none", edgecolors="darkorange",
               lw=1.6, label="confirmed overtone", zorder=5)
    ax.set_xlabel("Period [s]"); ax.set_ylabel("Group velocity [km/s]")
    ax.set_title(f"Mode validation QA  {os.path.basename(base)}  (lag={LAG})")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout(); fig.savefig(base + "_modeQA.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_plot = "--no-plot" not in sys.argv
    target = args[0]
    if os.path.isdir(target):
        files = sorted(glob.glob(os.path.join(target, "**", "*_dispersion_all.csv"), recursive=True))
    else:
        files = [target]
    if not files:
        print("no *_dispersion_all.csv found"); sys.exit(1)
    for f in files:
        try:
            validate_csv(f, plot=do_plot)
        except Exception as e:
            print(f"{f}: FAILED ({e})")
