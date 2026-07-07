"""Lean full-network batch of the validated V6 mode-separation workflow.

Per pair (pws stack, sym lag only -- the validated configuration):
  * read ZZ/RR/ZR/RZ, symmetric fold, tf-PWS synthesis of G_LR0/G_LR1;
  * CWT-FTAN images for ZZ, RR, RZ, ZR (+ all4 product) and G_LR0/G_LR1;
  * picks: segment-aware argmax on ZZ/RR/all4/G_LR0/G_LR1, topology on ZZ/RR (validator
    witnesses); narrowband SNR bank for G_LR0;
  * phase velocity on G_LR0/G_LR1 argmax picks: per-period 2piN vs the VSG-picked references
    (G_LR0 -> fundamental, G_LR1 -> overtone), far-field gates as in V6;
  * writes the V6-schema CSV, the G_LR0/G_LR1 image bundle (suppression test), and runs the
    consensus validator inline (no plots).

Skips pairs whose *_modes_validated.csv already exists (resume). Skips corrections
(T_centroid/T_inst) and neg/pos lags and linear/nroot stacks -- not used downstream.

Config (preferred):
  python dispersion_batch_modesep.py --config param_files/modesep_params.yaml
                                     [--out DIR] [--nproc N] [--limit K]
Legacy (kept for backward compatibility / dispersion.slurm; used when --config absent):
  python dispersion_batch_modesep.py <stack_root> <out_root> [nproc]
  with env DISP_NET / DISP_REF_DIR / DISP_LIMIT.

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
    """Return (STACK_ROOT, OUT_ROOT, NPROC, NET, LIMIT, REFS). --config wins over legacy."""
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
        out = a.out or cfg["paths"]["dispersion_dir"]
        nproc = a.nproc or int(cfg["batch"].get("nproc", 10))
        net = cfg["network"]["code"]
        limit = a.limit if a.limit is not None else int(cfg["batch"].get("limit", 0))
        refs_by_branch = modesep_config.ref_curve_paths(cfg)
        refs = {"G_LR0": refs_by_branch["fundamental"], "G_LR1": refs_by_branch["overtone"]}
        return stack, out, nproc, net, limit, refs
    # ---- legacy: positional args + DISP_* env (defaults = Aargau) ----
    stack = a.pos[0]
    out = a.pos[1]
    nproc = int(a.pos[2]) if len(a.pos) > 2 else 10
    net = os.environ.get("DISP_NET", "AA")
    limit = int(os.environ.get("DISP_LIMIT", "0"))
    vsg = os.environ.get(
        "DISP_REF_DIR",
        "/Users/genevievesavard/Codes/extract_higher_modes/Projects/aargau/vsg_modesep")
    refs = {"G_LR0": os.path.join(vsg, "ref_fundamental_phase.txt"),
            "G_LR1": os.path.join(vsg, "ref_overtone_phase.txt")}
    return stack, out, nproc, net, limit, refs


STACK_ROOT, OUT_ROOT, NPROC, NET, LIMIT, REFS = _resolve()

# ---- validated configuration (mirrors dispersion_curves_V6_modesep.py) ----
Tmin, dT, vmin, vmax, dvel, vave = 0.2, 0.1, 0.5, 4.5, 0.01, 3.0
maxgap, MIN_SEG, min_score, gauss_alpha = int(0.2 / dvel), 5, 0.7, 5.0
MIN_LAMBDA_GROUP, MIN_LAMBDA_PHASE, TAU_MAX_FACTOR = 1.0, 3.0, 12.0
PHASE_OFFSET = 0.0
PHASE_SHIFT_COMPONENT = {"G_LR0": +np.pi / 4.0, "G_LR1": +np.pi / 4.0}
# REFS (reference-curve paths per component) is resolved in _resolve() above.

HEADER = ("nominal_period,T_centroid,T_inst,group_velocity,phase_velocity,N_ambiguity,"
          "U_from_phase,score,snr_nbG,snr_bb,ratio_d_lambda,azimuth,backazimuth,distance,"
          "lag,component,wave_type,stack_method,pick_method,snr_bb_other,snr_nbG_other\n")

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
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import validate_modes
    _G["disp"] = dispersion
    _G["validate"] = validate_modes
    _G["cref"] = {c: dispersion.load_reference_curve(p) for c, p in REFS.items()}


def _sym(d):
    i = len(d) // 2
    return 0.5 * (d[i:] + np.flip(d[:i + 1]))


def process(path):
    import h5py
    dispersion = _G["disp"]
    spair = os.path.basename(path).replace(".h5", "")
    sta1 = os.path.basename(os.path.dirname(path))   # source-station subdir name (network-agnostic)
    outdir = os.path.join(OUT_ROOT, sta1)
    dcfile = os.path.join(outdir, spair + "_dispersion_all.csv")
    vfile = os.path.join(outdir, spair + "_modes_validated.csv")
    if os.path.exists(vfile):
        return "skip"
    try:
        with h5py.File(path, "r") as f:
            g = f["AuxiliaryData"]["Allstack_pws"]
            dist = float(g["ZZ"].attrs["dist"]); dt = float(g["ZZ"].attrs["dt"])
            azi = float(g["ZZ"].attrs["azi"]); baz = float(g["ZZ"].attrs["baz"])
            tr = {k: _sym(np.asarray(g[k][:], float)) for k in ("ZZ", "RR", "ZR", "RZ")}
    except Exception as e:
        return f"read-fail {e}"
    per_grid = np.arange(Tmin, dist / vave, dT)
    if len(per_grid) < 3:
        return "too-short"
    try:
        c0, c1 = dispersion.phase_corrected_components(tr["ZZ"], tr["RR"], tr["RZ"], tr["ZR"])
        sig = dict(tr)
        sig["G_LR0"] = dispersion.tf_pws(c0, dt)
        sig["G_LR1"] = dispersion.tf_pws(c1, dt)

        imgs, cwts = {}, {}
        per = vel = None
        for k, s in sig.items():
            cw = dispersion.compute_cwt(s, dist, dt, Tmin=Tmin, vmin=vmin, vmax=vmax, vave=vave)
            amp, per, vel, coi = dispersion.disp_image_from_cwt(
                cw, dist, Tmin=Tmin, dT=dT, vmin=vmin, vmax=vmax, dvel=dvel, vave=vave)
            imgs[k] = (amp, coi)
            cwts[k] = cw
        all4 = (imgs["ZZ"][0] * imgs["RR"][0] * imgs["RZ"][0] * imgs["ZR"][0]) ** 0.25
        imgs["all4"] = (all4, imgs["ZZ"][1])

        snr_g0 = None
        try:
            snr_nbG, snr_bb, _, _ = dispersion.nb_filt_gauss(
                sig["G_LR0"], dt, 1.0 / per_grid, dist, alpha=gauss_alpha, vmin=vmin, vmax=vmax)
            snr_g0 = (per_grid, snr_nbG, snr_bb)
        except Exception:
            pass

        tau_max = dist / TAU_MAX_FACTOR
        rows = []

        def emit(nper, gv, score, comp, pm, cph=None, Namb=None, snr_bank=None):
            for i in range(len(nper)):
                T, U = float(nper[i]), float(gv[i])
                if T <= 0 or U <= 0:
                    continue
                ratio = dist / (T * U)
                if ratio < MIN_LAMBDA_GROUP:
                    continue
                c_i = np.nan if cph is None else float(cph[i])
                N_i = 0 if Namb is None else int(Namb[i])
                if np.isfinite(c_i) and (ratio < MIN_LAMBDA_PHASE or T > tau_max):
                    c_i, N_i = np.nan, 0
                sn = np.nan
                if snr_bank is not None:
                    sn = float(snr_bank[1][int(np.argmin(np.abs(snr_bank[0] - T)))])
                sb = snr_bank[2] if snr_bank is not None else np.nan
                rows.append(f"{T:.2f},nan,nan,{U:.2f},{c_i:.3f},{N_i:d},nan,"
                            f"{float(score[i]):.2f},{sn:.2f},{sb:.2f},{ratio:.2f},"
                            f"{azi:.2f},{baz:.2f},{dist:.3f},sym,{comp},rayleigh,pws,{pm},"
                            f"nan,nan\n")

        for comp in ("ZZ", "RR", "all4", "G_LR0", "G_LR1"):
            amp, coi = imgs[comp]
            gp, gv, sc = dispersion.extract_dispersion(amp, per, vel, dist, vmax=vmax,
                                                       maxgap=maxgap, minlambda=MIN_LAMBDA_GROUP,
                                                       segments=True, min_seg=MIN_SEG)
            gp, gv, sc = dispersion.remove_picks_coi(gp, gv, sc, vel, coi)
            cph = Namb = None
            if comp in ("G_LR0", "G_LR1") and len(gp):
                corr = dispersion.measure_corrections_and_phase(
                    cwts[comp], gp, gv, dist, c_ref=_G["cref"][comp],
                    phase_shift=PHASE_SHIFT_COMPONENT[comp], phase_offset=PHASE_OFFSET,
                    use_period="nominal", joint=False)
                cph, Namb = corr["phase_velocity"], corr["N_ambiguity"]
            emit(gp, gv, sc, comp, "argmax", cph, Namb,
                 snr_g0 if comp == "G_LR0" else None)
            if comp in ("ZZ", "RR"):
                tp, tv, ts = dispersion.extract_curves_topology(amp, per, vel, limit=min_score)
                tp, tv, ts = dispersion.remove_picks_coi(np.asarray(tp), np.asarray(tv),
                                                         np.asarray(ts), vel, imgs[comp][1])
                emit(tp, tv, ts, comp, "topology")

        os.makedirs(os.path.join(outdir, "images"), exist_ok=True)
        with open(dcfile, "w") as f:
            f.write(HEADER)
            f.writelines(rows)
        np.savez_compressed(os.path.join(outdir, "images", f"{spair}_glr_images.npz"),
                            period=per, velocity=vel,
                            G_LR0=imgs["G_LR0"][0].astype(np.float32),
                            G_LR1=imgs["G_LR1"][0].astype(np.float32))
        _G["validate"].validate_csv(dcfile, plot=False, verbose=False)
        return "ok"
    except Exception as e:
        return f"fail {type(e).__name__}: {e}"


if __name__ == "__main__":
    import multiprocessing as mp
    files = sorted(glob.glob(os.path.join(STACK_ROOT, f"{NET}.*", f"{NET}.*_{NET}.*.h5")))
    if LIMIT and len(files) > LIMIT:
        files = files[:: max(1, len(files) // LIMIT)][:LIMIT]   # evenly-spread pilot subset
    print(f"{len(files)} pair files (NET={NET}, limit={LIMIT or 'none'}); {NPROC} workers",
          flush=True)
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
