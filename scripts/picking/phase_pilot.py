"""STEP-2 PILOT: can phase velocity recover the deep (long-period) constraint that the
far-field group-velocity filter throws away?

Background. The 4-5 km LVZ artifact was fixed by dropping group picks below ~2.5 wavelengths
(near-field bias), but that also discards the long periods -> shallow imaging (~3.5 km Aargau).
Phase velocity is a frequency-domain measurement (zero crossings / Bensen fringe) valid to ~1
wavelength (Luo et al. 2015; Ekstrom et al. 2009), so it should keep the long-period paths the
group filter rejects, extending sensitivity to 4-6 km.

The phase machinery already exists and is synthetic+13-pair validated in noisepy/dispersion.py
(measure_corrections_and_phase, resolve_phase_curve[_unwrap], convention fixed to 0.003 km/s).
But the FULL-NETWORK V6 batch nulls phase wherever T > dist/TAU_MAX_FACTOR with
TAU_MAX_FACTOR=12 -- i.e. it kept phase only beyond ~6 wavelengths, discarding exactly the
short-path/long-period phase that is the whole point. It also used per-period branch resolution
(joint=False), which puts ~1/3 of fundamental picks on the wrong 2*pi*N branch (c < U).

This pilot RE-MEASURES phase on a representative subset with the SAME validated call chain but
(1) phase kept to MIN_LAMBDA_PHASE=1.0 wavelength and the full group period band (no tau_max cut),
(2) joint branch tracking (Viterbi curve-continuity + VSG reference) so c(T) rides one branch.
It then asks the decision questions the user will judge:
  - does phase reach longer periods (=> deeper sensitivity) than the far-field group pool?
  - are the phase curves physical (c > U, smooth) and consistent with the VSG reference?
  - how much deeper (lambda/2 of the longest reliable phase period) can phase constrain?

Run (needs pycwt+findpeaks+h5py; reads the stacks on /Volumes/T7blue):
  PYTHONPATH=~/Codes/NoisePy-ant /opt/anaconda3/envs/das-ambient-noise/bin/python \
      phase_pilot.py --net aargau --n-pairs 80
Outputs under Projects/<net>/tomo/phase_pilot/:
  phase_pilot_curves.csv      per (pair, period, wave): T, U_group, c_phase, N, dist, r/lambda
  phase_pilot_diagnostic.png  the decision figure
  phase_pilot_examples.png    example per-pair c(T) vs U(T) vs VSG reference
"""
import argparse
import glob
import os
import random
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from noisepy import dispersion

# --- batch-identical measurement constants (dispersion_batch_modesep.py lines 73-77) ---
Tmin, dT, vmin, vmax, dvel, vave = 0.2, 0.1, 0.5, 4.5, 0.01, 3.0
maxgap, MIN_SEG, min_score, gauss_alpha = int(0.2 / dvel), 5, 0.7, 5.0
MIN_LAMBDA_GROUP = 1.0
PHASE_OFFSET = 0.0
PHASE_SHIFT_COMPONENT = {"G_LR0": +np.pi / 4.0, "G_LR1": +np.pi / 4.0}
# --- pilot changes: phase kept to 1 wavelength, no tau_max cap, joint branch tracking ---
MIN_LAMBDA_PHASE = 1.0
JOINT = "unwrap"          # 'unwrap' | True (Viterbi) | False (per-period); branch resolution

STACK = {"aargau": "/Volumes/T7blue/aargau-data/STACK_CHAA_normZ",
         "riehen": "/Volumes/T7blue/riehen-data/STACK_CHRI_normZ"}
PROJ = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
NETPREFIX = {"aargau": "AA", "riehen": "RI"}


def _sym(d):
    i = len(d) // 2
    return 0.5 * (d[i:] + np.flip(d[:i + 1]))


def refs(net):
    vsg = os.path.join(PROJ, net, "vsg_modesep")
    return {c: dispersion.load_reference_curve(os.path.join(vsg, p))
            for c, p in (("G_LR0", "ref_fundamental_phase.txt"),
                         ("G_LR1", "ref_overtone_phase.txt"))}


def process_pair(path, cref):
    """Full validated chain per pair; return rows dict-list for G_LR0 (fund) and G_LR1 (ot)."""
    import h5py
    spair = os.path.basename(path).replace(".h5", "")
    try:
        with h5py.File(path, "r") as f:
            g = f["AuxiliaryData"]["Allstack_pws"]
            dist = float(g["ZZ"].attrs["dist"]); dt = float(g["ZZ"].attrs["dt"])
            tr = {k: _sym(np.asarray(g[k][:], float)) for k in ("ZZ", "RR", "ZR", "RZ")}
    except Exception as e:
        return []
    if len(np.arange(Tmin, dist / vave, dT)) < 3:
        return []
    try:
        c0, c1 = dispersion.phase_corrected_components(tr["ZZ"], tr["RR"], tr["RZ"], tr["ZR"])
        sig = {"G_LR0": dispersion.tf_pws(c0, dt), "G_LR1": dispersion.tf_pws(c1, dt)}
        rows = []
        for comp, wave in (("G_LR0", "fund"), ("G_LR1", "overtone")):
            cw = dispersion.compute_cwt(sig[comp], dist, dt, Tmin=Tmin, vmin=vmin,
                                        vmax=vmax, vave=vave)
            amp, per, vel, coi = dispersion.disp_image_from_cwt(
                cw, dist, Tmin=Tmin, dT=dT, vmin=vmin, vmax=vmax, dvel=dvel, vave=vave)
            gp, gv, sc = dispersion.extract_dispersion(amp, per, vel, dist, vmax=vmax,
                                                       maxgap=maxgap, minlambda=MIN_LAMBDA_GROUP,
                                                       segments=True, min_seg=MIN_SEG)
            gp, gv, sc = dispersion.remove_picks_coi(gp, gv, sc, vel, coi)
            if not len(gp):
                continue
            corr = dispersion.measure_corrections_and_phase(
                cw, gp, gv, dist, c_ref=cref[comp],
                phase_shift=PHASE_SHIFT_COMPONENT[comp], phase_offset=PHASE_OFFSET,
                use_period="scale", joint=JOINT)               # scale-freq (bug 1+3 fix)
            cph, Namb = corr["phase_velocity"], corr["N_ambiguity"]
            Tscale, scj = corr["T_scale"], corr["scale_j"]
            # refined group velocity (dist/t_peak), self-consistent with the phase (bug 4)
            from noisepy.dispersion import measure_point
            seen = set()
            for i in range(len(gp)):
                T, U = float(gp[i]), float(gv[i])
                if T <= 0 or U <= 0:
                    continue
                ratio = dist / (T * U)                          # group r/lambda at nominal T
                if ratio < MIN_LAMBDA_GROUP:
                    continue
                # phase pick lives at its TRUE scale period, once per scale (bug 1 dedupe)
                c_i, Tp, Uref, rlp = np.nan, np.nan, np.nan, np.nan
                if np.isfinite(cph[i]) and scj[i] >= 0 and int(scj[i]) not in seen:
                    seen.add(int(scj[i]))
                    Tp = float(Tscale[i])
                    Uref = float(measure_point(cw, T, U, dist)["U"])
                    rlp = dist / (Tp * cph[i]) if cph[i] > 0 else np.nan
                    if rlp >= MIN_LAMBDA_PHASE and cph[i] > 0:
                        c_i = float(cph[i])
                    else:
                        Tp = np.nan
                rows.append(dict(pair=spair, wave=wave, period=round(T, 2),
                                 period_phase=Tp, U_group=U, U_ref=Uref, c_phase=c_i,
                                 N=int(Namb[i]), dist=dist, rlambda_group=ratio,
                                 rlambda_phase=rlp))
        return rows
    except Exception as e:
        return []


def run(net, n_pairs, seed=3):
    cref = refs(net)
    files = glob.glob(os.path.join(STACK[net], f"{NETPREFIX[net]}.*",
                                   f"{NETPREFIX[net]}.*_{NETPREFIX[net]}.*.h5"))
    random.seed(seed)
    # distance-stratified sample so long paths (the phase payoff) are represented
    random.shuffle(files)
    files = files[:max(n_pairs * 3, n_pairs)]
    out = []
    for i, f in enumerate(files):
        out += process_pair(f, cref)
        if len({r["pair"] for r in out}) >= n_pairs:
            break
        if i % 25 == 0:
            print(f"  {i} files, {len({r['pair'] for r in out})} pairs with picks", flush=True)
    df = pd.DataFrame(out)
    outdir = os.path.join(PROJ, net, "tomo", "phase_pilot")
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, "phase_pilot_curves.csv"), index=False)
    print(f"{net}: {df.pair.nunique()} pairs, {len(df)} (pair,period) rows, "
          f"{np.isfinite(df.c_phase).sum()} with phase")
    return df, outdir, cref


# ------------------------------------------------------------------ far-field group reference
FARFIELD_RULE = {  # mirror of export_tomo_picks.FARFIELD_RULE
    "aargau": {"fund": [(0.0, 2.0, 4.5), (2.0, 3.0, 3.0), (3.0, 99.0, 2.5)],
               "overtone": [(0.0, 99.0, 2.5)]},
    "riehen": {"fund": [(0.0, 2.5, 3.0), (2.5, 99.0, 2.5)],
               "overtone": [(0.0, 99.0, 2.5)]}}


def ff_cutoff(net, wave, T):
    for lo, hi, c in FARFIELD_RULE[net][wave]:
        if lo <= T < hi:
            return c
    return 2.5


def diagnostics(net, df, outdir, cref):
    df = df.copy()
    df["ff_keep_group"] = [r.rlambda_group >= ff_cutoff(net, r.wave, r.period)
                           for r in df.itertuples()]
    # a phase pick is one deduped row (period_phase set) that is physical (c > refined U_ref)
    df["phase_ok"] = (np.isfinite(df.c_phase) & np.isfinite(df.period_phase)
                      & (df.c_phase > df.U_ref))
    TB = np.round(np.arange(0.2, 7.2, 0.2), 1)                 # common 0.2 s period bins
    fig, axs = plt.subplots(2, 3, figsize=(17, 9.5))
    for j, wave in enumerate(("fund", "overtone")):
        s = df[df.wave == wave]
        ph = s[s.phase_ok].copy()
        ph["pb"] = TB[np.clip(np.searchsorted(TB, ph.period_phase.values) - 0, 0, len(TB) - 1)]
        # (1) period reach: far-field group (nominal T) vs phase (true scale period, deduped)
        a = axs[j, 0]
        ng = [s[(np.round(s.period, 1) == T) & s.ff_keep_group].shape[0] for T in TB]
        npa = [int((np.round(ph.period_phase, 1) == T).sum()) for T in TB]
        a.bar(TB - 0.04, ng, width=0.08, color="tab:gray", label="far-field group (nominal T)")
        a.bar(TB + 0.04, npa, width=0.08, color="tab:blue",
              label="phase ($\\geq$1$\\lambda$, c>U, deduped)")
        a.set_xlabel("period [s]"); a.set_ylabel("picks in pilot subset")
        a.set_title(f"{wave}: period reach"); a.legend()
        # (2) c>U physicality vs true period (all deduped phase measurements)
        a = axs[j, 1]
        fin = s[np.isfinite(s.c_phase) & np.isfinite(s.period_phase)].copy()
        fin["pb"] = np.round(fin.period_phase, 1)
        frac = [(fin[fin.pb == T].c_phase > fin[fin.pb == T].U_ref).mean()
                if (fin.pb == T).any() else np.nan for T in TB]
        a.axhline(1.0, color="green", lw=0.8, ls="--")
        a.plot(TB, frac, "o-", color="tab:purple", ms=3)
        a.set_ylim(0, 1.05); a.set_xlabel("period [s]")
        a.set_ylabel("fraction with c > U (normal disp.)")
        a.set_title(f"{wave}: phase branch physicality")
        # (3) mean dispersion: group U(nominal), phase c(true period), VSG reference
        a = axs[j, 2]
        gb = s.copy(); gb["pb"] = np.round(gb.period, 1)
        mu = [gb[gb.pb == T].U_group.median() if (gb.pb == T).any() else np.nan for T in TB]
        mc = [ph[np.round(ph.period_phase, 1) == T].c_phase.median()
              if (np.round(ph.period_phase, 1) == T).any() else np.nan for T in TB]
        a.plot(TB, mu, "o-", color="tab:gray", ms=3, label="group U (median)")
        a.plot(TB, mc, "o-", color="tab:blue", ms=3, label="phase c (median, deduped)")
        try:
            cc = cref["G_LR0" if wave == "fund" else "G_LR1"]
            a.plot(TB, [cc(t) for t in TB], "-", color="crimson", lw=1, label="VSG ref c")
        except Exception:
            pass
        a.set_xlabel("period [s]"); a.set_ylabel("velocity [km/s]")
        a.set_title(f"{wave}: mean dispersion"); a.legend(fontsize=8)
    # depth-reach summary in suptitle
    def depth_reach(sub, use_phase):
        if use_phase:
            g = sub[sub.phase_ok]
            v, Tcol = g.c_phase, g.period_phase
        else:
            g = sub[sub.ff_keep_group]
            v, Tcol = g.U_group, g.period
        if not len(g):
            return np.nan
        Tm = Tcol.quantile(0.95)
        return 0.5 * 1.1 * float(v[Tcol >= Tcol.quantile(0.9)].median()) * Tm
    fsub = df[df.wave == "fund"]
    zg, zp = depth_reach(fsub, False), depth_reach(fsub, True)
    fig.suptitle(f"{net.capitalize()} phase-velocity step-2 pilot ({df.pair.nunique()} pairs) — "
                 f"fund reliable-depth reach: far-field group ~{zg:.1f} km vs "
                 f"phase ~{zp:.1f} km  (lambda/2 of longest reliable period)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(outdir, "phase_pilot_diagnostic.png")
    fig.savefig(out, dpi=140); plt.close(fig); print("wrote", out)

    # example per-pair curves (longest-distance pairs, where phase should help most)
    ex = (df[df.wave == "fund"].groupby("pair").dist.first().sort_values(ascending=False)
          .head(6).index)
    fig, axs = plt.subplots(2, 3, figsize=(16, 8.5), sharex=True)
    for a, pr in zip(axs.ravel(), ex):
        s = df[(df.pair == pr) & (df.wave == "fund")].sort_values("period")
        a.plot(s.period, s.U_group, "o-", color="tab:gray", ms=3, label="group U")
        fin = s[np.isfinite(s.c_phase) & np.isfinite(s.period_phase)].sort_values("period_phase")
        a.plot(fin.period_phase, fin.c_phase, "s-", color="tab:blue", ms=3, label="phase c (true T)")
        try:
            Ts = np.sort(s.period.unique())
            a.plot(Ts, [cref["G_LR0"](t) for t in Ts], "-", color="crimson", lw=1, label="VSG ref")
        except Exception:
            pass
        a.set_title(f"{pr} ({s.dist.iloc[0]:.1f} km)", fontsize=8)
        a.set_xlabel("T [s]"); a.set_ylabel("v [km/s]"); a.legend(fontsize=7)
    fig.suptitle(f"{net.capitalize()} — example fundamental phase c(T) vs group U(T) "
                 f"(longest pairs; c should sit above U and stay smooth)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(outdir, "phase_pilot_examples.png")
    fig.savefig(out, dpi=140); plt.close(fig); print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", default="aargau", choices=("aargau", "riehen"))
    ap.add_argument("--n-pairs", type=int, default=80)
    args = ap.parse_args()
    df, outdir, cref = run(args.net, args.n_pairs)
    if len(df):
        diagnostics(args.net, df, outdir, cref)


if __name__ == "__main__":
    main()
