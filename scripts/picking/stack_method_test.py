"""Does tf-PWS actually beat plain PWS (or linear) for VELOCITY PICKING?

Tests the stacking method at the TWO independent levels where it enters the pipeline, scoring only
what matters: agreement of the measured dispersion with the EXTERNAL network VSG reference.

  EXPERIMENT A -- time-window stacking (the production choice, `Allstack_pws`):
      combine a pair's ~200 per-window 9C substacks with linear / pws / tf-pws, rotate ENZ->RTZ
      (stack-then-rotate, matching production; PWS is non-linear so the order matters), then
      synthesize G_LR0 with a FIXED method and measure. Isolates the time-window stack.

  EXPERIMENT B -- G_LR mode-synthesis stacking (`unified_picking.Config.GLR_STACK`):
      fix the input to the production-style pws window stack, then combine the FOUR phase-corrected
      components into G_LR0 with linear / pws / tf-pws and measure. Isolates the synthesis stack.
      (Nayak & Thurber use tf-PWS here on real data; 'linear' = their eqs 3/4 exactly.)

Metrics per (pair, method), all vs the external VSG reference so nothing is circular:
    phase_absdev median |c_phase - c_ref|  [PRIMARY -- closeness to the reference, includes bias]
    phase_mad  MAD of (c_phase - c_ref) about its own median      [scatter only, bias-blind]
    ridge_dev  median |U_argmax(T) - U_ref(T)|                     [group velocity accuracy]
    n_pick     ridge points passing the 2-lambda far-field rule    [usable band]
    snr_med    median narrowband SNR                               [waveform quality]
Paired per pair (same pairs for every method), so the comparison is a paired test, and the winner
is reported by median paired delta + win rate.

Usage: /opt/anaconda3/bin/python stack_method_test.py --net {riehen,aargau} [--n-pairs 20]
"""
import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from noisepy import dispersion  # noqa: E402
from noisepy import unified_picking as up  # noqa: E402
from noisepy.stacking import pws, rotation  # noqa: E402

ENZ_ORDER = ["EE", "EN", "EZ", "NE", "NN", "NZ", "ZE", "ZN", "ZZ"]
RTZ_ORDER = ["ZR", "ZT", "ZZ", "RR", "RT", "RZ", "TR", "TT", "TZ"]
NETS = {
    "aargau": {"stack": "/Volumes/T7blue/aargau-data/STACK_CHAA_normZ", "code": "AA",
               "proj": "/Users/genevievesavard/Codes/extract_higher_modes/Projects/aargau"},
    "riehen": {"stack": "/Volumes/T7blue/riehen-data/STACK_CHRI_normZ", "code": "RI",
               "proj": "/Users/genevievesavard/Codes/extract_higher_modes/Projects/riehen"},
}
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--net", required=True, choices=list(NETS))
ap.add_argument("--n-pairs", type=int, default=20)
ap.add_argument("--min-dist", type=float, default=3.0)
args = ap.parse_args()
PROJ = NETS[args.net]["proj"]
OUT = os.path.join(PROJ, "regime_pilot")
os.makedirs(OUT, exist_ok=True)
CFG = up.Config

# external references
cref_fund = dispersion.load_reference_curve(os.path.join(PROJ, "vsg_modesep",
                                                         "ref_fundamental_phase.txt"))
uref_fund = up.phase_ref_to_group_ref(cref_fund, np.arange(0.2, 8.0, 0.1))


def stack_traces(arr, dt, method):
    """arr: (nwin, npts) -> stacked trace by 'linear' | 'pws' | 'tfpws'."""
    if method == "linear":
        return arr.mean(axis=0)
    if method == "pws":
        return pws(arr, 1.0 / dt)
    if method == "tfpws":
        return dispersion.tf_pws(list(arr), dt)
    raise ValueError(method)


def measure(sig, dist, dt):
    """Velocity accuracy of one stacked trace vs the external VSG reference."""
    out = {"phase_absdev": np.nan, "phase_mad": np.nan, "ridge_dev": np.nan, "n_pick": 0,
           "snr_med": np.nan}
    Tmax = dist / CFG.vave
    per_grid = np.arange(CFG.Tmin, Tmax, CFG.dT)
    if len(per_grid) < 3:
        return out
    try:
        snr, _, _, _ = dispersion.nb_filt_gauss(sig, dt, 1.0 / per_grid, dist,
                                                alpha=CFG.gauss_alpha, vmin=CFG.vmin, vmax=CFG.vmax)
        out["snr_med"] = float(np.median(snr))
        cwt = dispersion.compute_cwt(sig, dist, dt, Tmin=CFG.Tmin, vmin=CFG.vmin, vmax=CFG.vmax,
                                     vave=CFG.vave)
        amp, per, vel, coi = dispersion.disp_image_from_cwt(
            cwt, dist, Tmin=CFG.Tmin, dT=CFG.dT, vmin=CFG.vmin, vmax=CFG.vmax, dvel=CFG.dvel,
            vave=CFG.vave)
        nper, gv, sc = dispersion.extract_dispersion(amp, per, vel, dist, vmax=CFG.vmax,
                                                     maxgap=CFG.maxgap, minlambda=2.0,
                                                     segments=True, min_seg=CFG.MIN_SEG)
        nper, gv, sc = dispersion.remove_picks_coi(np.asarray(nper), np.asarray(gv),
                                                   np.asarray(sc), vel, coi)
        out["n_pick"] = int(len(nper))
        if len(nper):
            d = np.abs(np.asarray(gv, float) - np.asarray(uref_fund(np.asarray(nper, float)), float))
            d = d[np.isfinite(d)]
            if len(d):
                out["ridge_dev"] = float(np.median(d))
            corr = dispersion.measure_corrections_and_phase(
                cwt, nper, gv, dist, c_ref=cref_fund, phase_shift=CFG.PHASE_SHIFT["rayleigh"],
                phase_offset=CFG.PHASE_OFFSET, use_period=CFG.PHASE_USE_PERIOD,
                joint=CFG.PHASE_JOINT, smooth_weight=CFG.PHASE_SMOOTH_WEIGHT)
            cph = np.asarray(corr["phase_velocity"], float)
            cr = np.asarray([float(cref_fund(t)) for t in nper])
            r = (cph - cr)[np.isfinite(cph) & np.isfinite(cr)]
            if len(r) >= 3:
                out["phase_mad"] = float(np.median(np.abs(r - np.median(r))))
                out["phase_absdev"] = float(np.median(np.abs(r)))
    except Exception:
        pass
    return out


def synth_g0(sym, dt, method):
    c0, _c1 = dispersion.phase_corrected_components(sym["ZZ"], sym["RR"], sym["RZ"], sym["ZR"])
    return stack_traces(np.asarray(c0), dt, method)


def to_sym(rt):
    d = {c: rt[i] for i, c in enumerate(RTZ_ORDER)}
    return {c: 0.5 * (v[len(v) // 2:] + v[: len(v) // 2 + 1][::-1]) for c, v in d.items()}


# ---------------------------------------------------------------- pair list
import glob  # noqa: E402
cands = sorted(glob.glob(os.path.join(NETS[args.net]["stack"], f"{NETS[args.net]['code']}.*",
                                      f"{NETS[args.net]['code']}.*_{NETS[args.net]['code']}.*.h5")))
cands = cands[:: max(1, len(cands) // (6 * args.n_pairs))]
rows_a, rows_b = [], []
done = 0
for f in cands:
    if done >= args.n_pairs:
        break
    pair = os.path.basename(f)[:-3]
    try:
        with h5py.File(f, "r") as h:
            g = h["AuxiliaryData"]
            tkeys = sorted(k for k in g if k.startswith("T"))
            if len(tkeys) < 30:
                continue
            a = g[tkeys[0]]["ZZ"].attrs
            dist, dt = float(a["dist"]), float(a["dt"])
            azi, baz = float(a["azi"]), float(a["baz"])
            if dist < args.min_dist:
                continue
            win = {c: [] for c in ENZ_ORDER}
            for tk in tkeys:
                grp = g[tk]
                if any(c not in grp for c in ENZ_ORDER):
                    continue
                for c in ENZ_ORDER:
                    win[c].append(np.asarray(grp[c][:], np.float64))
            nwin = len(win["ZZ"])
            if nwin < 30:
                continue
            win = {c: np.asarray(v) for c, v in win.items()}
    except Exception as e:
        print(f"  {pair}: read failed ({e})", flush=True)
        continue

    # ---- EXPERIMENT A: time-window stacking method (G_LR synthesis fixed at tfpws) ----
    sym_by_method = {}
    for meth in ("linear", "pws", "tfpws"):
        try:
            big = np.stack([stack_traces(win[c], dt, meth) for c in ENZ_ORDER])
            rt = rotation(big.astype(np.float32), {"azi": azi, "baz": baz}, {})
            sym = to_sym(rt)
            sym_by_method[meth] = sym
            g0 = synth_g0(sym, dt, "tfpws")
            rows_a.append({"pair": pair, "dist": dist, "nwin": nwin, "method": meth,
                           **measure(g0, dist, dt)})
        except Exception as e:
            print(f"  {pair}/A/{meth}: {e}", flush=True)

    # ---- EXPERIMENT B: G_LR synthesis method (input fixed at the pws window stack) ----
    if "pws" in sym_by_method:
        for meth in ("linear", "pws", "tfpws"):
            try:
                g0 = synth_g0(sym_by_method["pws"], dt, meth)
                rows_b.append({"pair": pair, "dist": dist, "method": meth,
                               **measure(g0, dist, dt)})
            except Exception as e:
                print(f"  {pair}/B/{meth}: {e}", flush=True)
    done += 1
    print(f"  {done}/{args.n_pairs} pairs done ({pair}, {nwin} windows)", flush=True)

A = pd.DataFrame(rows_a)
B = pd.DataFrame(rows_b)
A.to_csv(os.path.join(OUT, "stack_method_A_window.csv"), index=False)
B.to_csv(os.path.join(OUT, "stack_method_B_synth.csv"), index=False)

METRICS = [("phase_absdev", "lower"), ("phase_mad", "lower"), ("ridge_dev", "lower"),
           ("n_pick", "higher"), ("snr_med", "higher")]


def report(df, title, baseline):
    L = [f"--- {title} (n={df.pair.nunique()} pairs, baseline = {baseline}) ---",
         df.groupby("method")[[m for m, _ in METRICS]].median().to_string(), ""]
    piv = {m: df.pivot_table(index="pair", columns="method", values=m) for m, _ in METRICS}
    for m, better in METRICS:
        p = piv[m].dropna()
        if baseline not in p.columns or p.empty:
            continue
        for meth in p.columns:
            if meth == baseline:
                continue
            d = p[meth] - p[baseline]
            wins = int((d < 0).sum() if better == "lower" else (d > 0).sum())
            rel = 100 * np.median(d) / abs(np.median(p[baseline])) if np.median(p[baseline]) else np.nan
            pv = binomtest(wins, len(p), 0.5).pvalue if len(p) else np.nan
            verdict = ("SIGNIFICANT " + ("better" if wins > len(p) / 2 else "WORSE")
                       if pv < 0.05 else "n.s.")
            L.append(f"  {m:12s}: {meth:6s} vs {baseline:6s} -> wins {wins:2d}/{len(p):2d} "
                     f"({100*wins/len(p):3.0f}%), median delta {rel:+6.1f}%, p={pv:.3f}  {verdict}")
    return "\n".join(L)


lines = [f"=== stacking-method test: {args.net} ===",
         "Scored ONLY on velocity accuracy vs the external VSG reference (phase_mad = primary).", ""]
if not A.empty:
    lines.append(report(A, "EXPERIMENT A: time-window stack (linear/pws/tfpws), synthesis fixed=tfpws",
                        "pws"))
    lines.append("")
if not B.empty:
    lines.append(report(B, "EXPERIMENT B: G_LR synthesis stack (linear/pws/tfpws), window stack fixed=pws",
                        "tfpws"))
rep = "\n".join(lines)
print("\n" + rep)
with open(os.path.join(OUT, "stack_method_report.txt"), "w") as fo:
    fo.write(rep + "\n")
print(f"\nwrote {OUT}/stack_method_report.txt")
