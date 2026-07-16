"""Lean full-network Love-wave (TT) batch -- sibling of dispersion_batch_modesep.py.

The Rayleigh batch (dispersion_batch_modesep.py) never reads TT, so its runs contain zero
Love picks by construction; the V4 Love script's QC columns are unusable (see the bug audit
in dispersion_curves_V4_love.py). This script picks TT with V6-grade QC and carries every
Rayleigh-leakage discriminator as a COLUMN, so contamination thresholds are tuned at merge
time without re-running the batch.

Per pair (pws stack, neg/pos/sym lags):
  * read TT + cross-terms RT/TR/TZ/ZT, fold each lag;
  * CWT-FTAN image on TT; segment-aware argmax AND topology picks (topology keeps multiple
    ridges -- TT genuinely carries two branches: the Love fundamental and a fast branch that
    is either the Love overtone or leaked Rayleigh overtone);
  * per-period cross-term QC done right: narrowband signal-window ENVELOPE-max ratio
    env(TT)/max(env(RT,TR,TZ,ZT)) (`env_ratio`; a real Love wave puts no energy on the
    cross-terms, leaked Rayleigh does), plus V6-style per-period `snr_nbG_other` and
    broadband `snr_bb_other`;
  * Rayleigh-branch discriminators from the pair's validated V6 output
    (<v6_dir>/<src>/<pair>_modes_validated.csv):
      dU_fund, dU_ot  = pick velocity minus the validated fundamental/overtone group
                        velocity at that period (NaN when the branch is not measured there)
      ot_flag         = 'veto' (|dU_ot| <= OT_VETO_DV: velocity coincides with the Rayleigh
                        overtone -> leakage suspect unless env_ratio is high), 'clear', 'no_ref'
  * flagged_sta = number of pair endpoints (0/1/2) in the station-QC orientation/polarity
    list (station_qc.csv, any non-empty flag).

Outputs <out>/<src>/<pair>_love.csv + images/<pair>_tt_images.npz (sym-lag TT image).
Skips pairs whose _love.csv already exists (resume).

Config (preferred):
  python dispersion_batch_love.py --config param_files/modesep_params.yaml
                                  [--out DIR] [--nproc N] [--limit K]
Legacy (mirrors dispersion_batch_modesep.py; used when --config absent):
  python dispersion_batch_love.py <stack_root> <out_root> [nproc]
  env: DISP_NET (glob prefix), DISP_LIMIT (pilot subset),
       DISP_V6_DIR (modes_validated dir; default: sibling 'dispersion_V6' of out_root),
       DISP_STATION_QC (station_qc.csv; default: 'station_qc.csv' next to DISP_V6_DIR)

Riehen production run (T7blue is READ-ONLY full -- outputs stay local):
  DISP_NET=RI \
  DISP_V6_DIR=~/Codes/extract_higher_modes/Projects/riehen/dispersion_V6 \
  PYTHONPATH=~/Codes/NoisePy-ant /opt/anaconda3/envs/das-ambient-noise/bin/python \
  dispersion_batch_love.py /Volumes/T7blue/riehen-data/STACK_CHRI_normZ \
  ~/Codes/extract_higher_modes/Projects/riehen/dispersion_V6_love 10

Resolution happens at import time (not under __main__) so spawned workers reconstruct the
same globals; sys.argv is inherited by the workers.
"""
import argparse
import glob
import logging
import os
import sys
import numpy as np


def _resolve():
    """Return (STACK_ROOT, OUT_ROOT, NPROC, NET, LIMIT, V6_DIR, STATION_QC)."""
    ap = argparse.ArgumentParser(add_help=("--config" in sys.argv or "-h" in sys.argv
                                           or "--help" in sys.argv))
    ap.add_argument("--config")
    ap.add_argument("--out")
    ap.add_argument("--nproc", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("pos", nargs="*")           # legacy: stack_root out_root [nproc]
    a, _ = ap.parse_known_args()
    if a.config:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import modesep_config
        cfg = modesep_config.load_config(a.config)
        stack = cfg["paths"]["stack_root"]
        v6 = cfg["paths"]["dispersion_dir"]
        out = a.out or (v6.rstrip("/") + "_love")
        nproc = a.nproc or int(cfg["batch"].get("nproc", 10))
        net = cfg["network"]["code"]
        limit = a.limit if a.limit is not None else int(cfg["batch"].get("limit", 0))
        proj = cfg["paths"].get("project_dir") or os.path.dirname(v6.rstrip("/"))
        sqc = os.path.join(proj, "station_qc.csv")
        return stack, out, nproc, net, limit, v6, sqc
    # ---- legacy: positional args + DISP_* env ----
    stack = a.pos[0]
    out = a.pos[1]
    nproc = int(a.pos[2]) if len(a.pos) > 2 else 10
    net = os.environ.get("DISP_NET", "AA")
    limit = int(os.environ.get("DISP_LIMIT", "0"))
    v6 = os.environ.get("DISP_V6_DIR",
                        os.path.join(os.path.dirname(out.rstrip("/")), "dispersion_V6"))
    sqc = os.environ.get("DISP_STATION_QC",
                         os.path.join(os.path.dirname(v6.rstrip("/")), "station_qc.csv"))
    return stack, out, nproc, net, limit, v6, sqc


STACK_ROOT, OUT_ROOT, NPROC, NET, LIMIT, V6_DIR, STATION_QC = _resolve()

# ---- configuration (grid mirrors dispersion_batch_modesep.py so images are comparable) ----
Tmin, dT, vmin, vmax, dvel, vave = 0.2, 0.1, 0.5, 4.5, 0.01, 3.0
maxgap, MIN_SEG, min_score, gauss_alpha = int(0.2 / dvel), 5, 0.5, 5.0
# min_score 0.5 (vs 0.7 in the Rayleigh batch): on TT the DOMINANT ridge is often the fast
# branch (Love overtone / leaked Rayleigh overtone), so a high persistence threshold favors
# the contaminant and drops the weaker Love fundamental; the env_ratio/dU_ot columns carry
# the discrimination instead.
MIN_LAMBDA_GROUP = 1.0
LAGS = ("neg", "pos", "sym")     # per-lag agreement is a merge-time Love QC (directional noise)
XTERMS = ("RT", "TR", "TZ", "ZT")
OT_VETO_DV = 0.15                # |U_pick - U_overtone| below this -> Rayleigh-leakage suspect
REF_T_TOL = 0.15                 # max period distance [s] to a validated ref sample for dU_*
IMAGE_LAGS = ("sym",)            # TT image bundle export

HEADER = ("nominal_period,group_velocity,score,snr_nbG,snr_bb,snr_nbG_other,snr_bb_other,"
          "env_ratio,dU_fund,dU_ot,ot_flag,flagged_sta,ratio_d_lambda,azimuth,backazimuth,"
          "distance,lag,component,wave_type,stack_method,pick_method\n")


def _load_flagged(path):
    """Stations with any non-empty flag in station_qc.csv (empty set if unreadable)."""
    try:
        import pandas as pd
        s = pd.read_csv(path, index_col=0)
        return set(s.index[s["flag"].fillna("") != ""].astype(str))
    except Exception:
        return set()


FLAGGED = _load_flagged(STATION_QC)

_G = {}   # per-worker globals


def _init():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TQDM_DISABLE"] = "1"
    for name in ("findpeaks", "findpeaks.stats", "matplotlib", "noisepy"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.CRITICAL)
        lg.propagate = False
        lg.handlers = [logging.NullHandler()]
    from noisepy import dispersion
    _G["disp"] = dispersion


def _fold(d):
    """Two-sided CCF -> {'neg','pos','sym'} one-sided signals."""
    i = len(d) // 2
    return {"neg": d[:i + 1][::-1], "pos": d[i:],
            "sym": 0.5 * (d[i:] + np.flip(d[:i + 1]))}


def _load_ref_branches(vfile):
    """(T_fund, U_fund, T_ot, U_ot) from a modes_validated.csv, keeping *_use==1 rows only."""
    import pandas as pd
    try:
        v = pd.read_csv(vfile)
    except Exception:
        return None
    def branch(ucol, use):
        m = (pd.to_numeric(v[use], errors="coerce") == 1) & np.isfinite(
            pd.to_numeric(v[ucol], errors="coerce"))
        t = v.loc[m, "period"].values.astype(float)
        u = v.loc[m, ucol].values.astype(float)
        o = np.argsort(t)
        return t[o], u[o]
    tf, uf = branch("U_fund", "fund_use")
    to, uo = branch("U_overtone", "ot_use")
    return tf, uf, to, uo


def _ref_at(T, rt, ru):
    """Reference velocity at period T, NaN when no validated sample within REF_T_TOL."""
    if rt is None or len(rt) == 0:
        return np.nan
    i = int(np.argmin(np.abs(rt - T)))
    if abs(rt[i] - T) > REF_T_TOL:
        return np.nan
    return float(np.interp(T, rt, ru)) if len(rt) > 1 else float(ru[i])


def process(path):
    import h5py
    dispersion = _G["disp"]
    spair = os.path.basename(path).replace(".h5", "")
    sta1 = os.path.basename(os.path.dirname(path))
    outdir = os.path.join(OUT_ROOT, sta1)
    lcfile = os.path.join(outdir, spair + "_love.csv")
    if os.path.exists(lcfile):
        return "skip"
    try:
        with h5py.File(path, "r") as f:
            g = f["AuxiliaryData"]["Allstack_pws"]
            if "TT" not in g:
                return "no-TT"
            dist = float(g["TT"].attrs["dist"]); dt = float(g["TT"].attrs["dt"])
            azi = float(g["TT"].attrs["azi"]); baz = float(g["TT"].attrs["baz"])
            tr = {k: _fold(np.asarray(g[k][:], float))
                  for k in ("TT",) + XTERMS if k in g}
    except Exception as e:
        return f"read-fail {e}"
    per_grid = np.arange(Tmin, dist / vave, dT)
    if len(per_grid) < 3:
        return "too-short"

    # leakage references (may legitimately be absent: too-short pairs, no validated rows)
    refs = _load_ref_branches(os.path.join(V6_DIR, sta1, spair + "_modes_validated.csv"))
    tf, uf, to, uo = refs if refs is not None else (None, None, None, None)
    s1, s2 = spair.split("_")[0], spair.split("_")[-1]
    n_flagged = int(s1 in FLAGGED) + int(s2 in FLAGGED)

    try:
        fn = 1.0 / per_grid
        sig_i0, sig_i1 = int(dist / vmax / dt), int(dist / vmin / dt)

        # narrowband SNR + signal-window envelope maxima, per lag per component
        snr_bank, envmax_bank, snrbb = {}, {}, {}
        for lag in LAGS:
            for c, s in tr.items():
                try:
                    nb, bb, _, env = dispersion.nb_filt_gauss(
                        s[lag], dt, fn, dist, alpha=gauss_alpha, vmin=vmin, vmax=vmax)
                    snr_bank[(c, lag)] = nb
                    snrbb[(c, lag)] = float(bb)
                    envmax_bank[(c, lag)] = env[:, sig_i0:min(sig_i1, env.shape[1])].max(axis=1)
                except Exception:
                    pass

        rows = []

        def emit(nper, gv, score, lag, pm, xsnr, xbb, xenv, tsnr, tbb, tenv):
            for i in range(len(nper)):
                T, U = float(nper[i]), float(gv[i])
                if T <= 0 or U <= 0:
                    continue
                ratio = dist / (T * U)
                if ratio < MIN_LAMBDA_GROUP:
                    continue
                k = int(np.argmin(np.abs(per_grid - T)))
                sn = float(tsnr[k]) if tsnr is not None else np.nan
                sno = float(xsnr[k]) if xsnr is not None else np.nan
                er = (float(tenv[k] / xenv[k])
                      if tenv is not None and xenv is not None and xenv[k] > 0 else np.nan)
                du_f = U - _ref_at(T, tf, uf)
                du_o = U - _ref_at(T, to, uo)
                otf = ("no_ref" if not np.isfinite(du_o)
                       else "veto" if abs(du_o) <= OT_VETO_DV else "clear")
                rows.append(f"{T:.2f},{U:.2f},{float(score[i]):.2f},{sn:.2f},{tbb:.2f},"
                            f"{sno:.2f},{xbb:.2f},{er:.2f},{du_f:.2f},{du_o:.2f},{otf},"
                            f"{n_flagged:d},{ratio:.2f},{azi:.2f},{baz:.2f},{dist:.3f},"
                            f"{lag},TT,love,pws,{pm}\n")

        img_export = {}
        for lag in LAGS:
            cw = dispersion.compute_cwt(tr["TT"][lag], dist, dt, Tmin=Tmin,
                                        vmin=vmin, vmax=vmax, vave=vave)
            amp, per, vel, coi = dispersion.disp_image_from_cwt(
                cw, dist, Tmin=Tmin, dT=dT, vmin=vmin, vmax=vmax, dvel=dvel, vave=vave)
            if lag in IMAGE_LAGS:
                img_export[lag] = (amp, per, vel)

            # per-period cross-term context for this lag
            xs = [snr_bank[(c, lag)] for c in XTERMS if (c, lag) in snr_bank]
            xe = [envmax_bank[(c, lag)] for c in XTERMS if (c, lag) in envmax_bank]
            xb = [snrbb[(c, lag)] for c in XTERMS if (c, lag) in snrbb]
            xsnr = np.max(xs, axis=0) if xs else None
            xenv = np.max(xe, axis=0) if xe else None
            xbb = float(np.max(xb)) if xb else np.nan
            tsnr = snr_bank.get(("TT", lag))
            tenv = envmax_bank.get(("TT", lag))
            tbb = snrbb.get(("TT", lag), np.nan)

            gp, gv, sc = dispersion.extract_dispersion(amp, per, vel, dist, vmax=vmax,
                                                       maxgap=maxgap,
                                                       minlambda=MIN_LAMBDA_GROUP,
                                                       segments=True, min_seg=MIN_SEG)
            gp, gv, sc = dispersion.remove_picks_coi(gp, gv, sc, vel, coi)
            emit(gp, gv, sc, lag, "argmax", xsnr, xbb, xenv, tsnr, tbb, tenv)

            tp, tv, ts = dispersion.extract_curves_topology(amp, per, vel, limit=min_score)
            tp, tv, ts = dispersion.remove_picks_coi(np.asarray(tp), np.asarray(tv),
                                                     np.asarray(ts), vel, coi)
            emit(tp, tv, ts, lag, "topology", xsnr, xbb, xenv, tsnr, tbb, tenv)

        os.makedirs(os.path.join(outdir, "images"), exist_ok=True)
        with open(lcfile, "w") as f:
            f.write(HEADER)
            f.writelines(rows)
        if img_export:
            lag0 = IMAGE_LAGS[0]
            amp, per, vel = img_export[lag0]
            np.savez_compressed(os.path.join(outdir, "images", f"{spair}_tt_images.npz"),
                                period=per, velocity=vel, TT=amp.astype(np.float32))
        return "ok"
    except Exception as e:
        return f"fail {type(e).__name__}: {e}"


if __name__ == "__main__":
    import multiprocessing as mp
    files = sorted(glob.glob(os.path.join(STACK_ROOT, f"{NET}.*", f"{NET}.*_{NET}.*.h5")))
    if LIMIT and len(files) > LIMIT:
        files = files[:: max(1, len(files) // LIMIT)][:LIMIT]   # evenly-spread pilot subset
    print(f"{len(files)} pair files (NET={NET}, limit={LIMIT or 'none'}); {NPROC} workers\n"
          f"V6 refs: {V6_DIR}\nstation QC: {STATION_QC} ({len(FLAGGED)} flagged)", flush=True)
    n = {"ok": 0, "skip": 0, "other": 0}
    with mp.Pool(NPROC, initializer=_init, maxtasksperchild=200) as pool:
        for i, st in enumerate(pool.imap_unordered(process, files, chunksize=4)):
            n["ok" if st == "ok" else "skip" if st == "skip" else "other"] += 1
            if st not in ("ok", "skip") and n["other"] <= 30:
                print(f"  note: {st}", flush=True)
            if (i + 1) % 250 == 0:
                print(f"[{i+1}/{len(files)}] ok={n['ok']} skip={n['skip']} "
                      f"other={n['other']}", flush=True)
    print(f"DONE: ok={n['ok']} skip={n['skip']} other={n['other']}", flush=True)
