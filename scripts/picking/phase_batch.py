"""STEP-2: network-wide phase-velocity measurement on the mode-separated stacks, with the
bug-fixed pipeline (scale-frequency labelling + per-scale de-duplication + relaxed 1-lambda /
no-tau_max gates + joint 'unwrap' branch tracking). Companion to phase_pilot.py (same per-pair
chain) but over ALL pairs, parallel and resumable, writing tomography-ready phase picks.

Per pair (Allstack_pws, sym fold): synth G_LR0/G_LR1 (tf-PWS) -> CWT -> group argmax picks ->
measure_corrections_and_phase(use_period='scale', joint='unwrap'). Each phase pick is emitted at
its TRUE CWT-scale period (deduped by scale_j, so no long-period multi-count) if it is physical
(c > refined U_ref) and >= MIN_LAMBDA_PHASE wavelengths.

Output: Projects/<net>/tomo/phase_picks/<s1>__<s2>.csv  (resume-safe), columns:
  wave, period, c_phase, U_ref, N, dist, rlambda_phase, azimuth
Aggregate + QC + swtomotv export is done by export_phase_picks.py.

Run (needs pycwt+findpeaks+h5py; reads /Volumes/T7blue):
  PYTHONPATH=~/Codes/NoisePy-ant /opt/anaconda3/envs/das-ambient-noise/bin/python \
      phase_batch.py --net aargau --nproc 8
"""
import argparse
import glob
import os
import sys

import numpy as np

from noisepy import dispersion

# batch-identical measurement constants (dispersion_batch_modesep.py 73-77) + pilot relaxations
Tmin, dT, vmin, vmax, dvel, vave = 0.2, 0.1, 0.5, 4.5, 0.01, 3.0
maxgap, MIN_SEG, MIN_LAMBDA_GROUP = int(0.2 / dvel), 5, 1.0
MIN_LAMBDA_PHASE = 1.0
PHASE_OFFSET = 0.0
PHASE_SHIFT = +np.pi / 4.0                 # G_LR0/G_LR1, validated -phi convention
JOINT = "unwrap"

STACK = {"aargau": "/Volumes/T7blue/aargau-data/STACK_CHAA_normZ",
         "riehen": "/Volumes/T7blue/riehen-data/STACK_CHRI_normZ"}
PROJ = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
NETPREFIX = {"aargau": "AA", "riehen": "RI"}
_CREF = {}


def _refs(net):
    if net not in _CREF:
        vsg = os.path.join(PROJ, net, "vsg_modesep")
        _CREF[net] = {c: dispersion.load_reference_curve(os.path.join(vsg, p))
                      for c, p in (("G_LR0", "ref_fundamental_phase.txt"),
                                   ("G_LR1", "ref_overtone_phase.txt"))}
    return _CREF[net]


def _sym(d):
    i = len(d) // 2
    return 0.5 * (d[i:] + np.flip(d[:i + 1]))


def _azimuth(la1, lo1, la2, lo2):
    f1, f2 = np.radians(la1), np.radians(la2)
    dl = np.radians(lo2 - lo1)
    y = np.sin(dl) * np.cos(f2)
    x = np.cos(f1) * np.sin(f2) - np.sin(f1) * np.cos(f2) * np.cos(dl)
    return float(np.degrees(np.arctan2(y, x)) % 360.0)


def process_pair(args):
    """(path, net, outdir) -> status string; writes per-pair phase-pick CSV."""
    import h5py
    path, net, outdir = args
    spair = os.path.basename(path).replace(".h5", "")
    outf = os.path.join(outdir, spair + ".csv")
    if os.path.exists(outf):
        return "skip"
    cref = _refs(net)
    try:
        with h5py.File(path, "r") as f:
            g = f["AuxiliaryData"]["Allstack_pws"]
            dist = float(g["ZZ"].attrs["dist"]); dt = float(g["ZZ"].attrs["dt"])
            laS = float(g["ZZ"].attrs.get("latS", np.nan)); loS = float(g["ZZ"].attrs.get("lonS", np.nan))
            laR = float(g["ZZ"].attrs.get("latR", np.nan)); loR = float(g["ZZ"].attrs.get("lonR", np.nan))
            tr = {k: _sym(np.asarray(g[k][:], float)) for k in ("ZZ", "RR", "ZR", "RZ")}
    except Exception as e:
        return f"read-fail"
    if len(np.arange(Tmin, dist / vave, dT)) < 3:
        return "too-short"
    azi = _azimuth(laS, loS, laR, loR) if np.isfinite(laS) else np.nan
    try:
        c0, c1 = dispersion.phase_corrected_components(tr["ZZ"], tr["RR"], tr["RZ"], tr["ZR"])
        sig = {"G_LR0": dispersion.tf_pws(c0, dt), "G_LR1": dispersion.tf_pws(c1, dt)}
        lines = []
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
                cw, gp, gv, dist, c_ref=cref[comp], phase_shift=PHASE_SHIFT,
                phase_offset=PHASE_OFFSET, use_period="scale", joint=JOINT)
            cph, Namb, Tsc, scj = (corr["phase_velocity"], corr["N_ambiguity"],
                                   corr["T_scale"], corr["scale_j"])
            seen = set()
            for i in range(len(gp)):
                if not (np.isfinite(cph[i]) and scj[i] >= 0 and int(scj[i]) not in seen):
                    continue
                Tp, c_i = float(Tsc[i]), float(cph[i])
                if c_i <= 0 or Tp <= 0:
                    continue
                Uref = float(dispersion.measure_point(cw, float(gp[i]), float(gv[i]), dist)["U"])
                rlp = dist / (Tp * c_i)
                if rlp < MIN_LAMBDA_PHASE or not (c_i > Uref):      # physical + far enough
                    continue
                seen.add(int(scj[i]))
                lines.append(f"{wave},{Tp:.3f},{c_i:.4f},{Uref:.4f},{int(Namb[i])},"
                             f"{dist:.3f},{rlp:.2f},{azi:.2f}\n")
        with open(outf, "w") as fo:
            fo.write("wave,period,c_phase,U_ref,N,dist,rlambda_phase,azimuth\n")
            fo.writelines(lines)
        return "ok" if lines else "no-phase"
    except Exception as e:
        return "proc-fail"


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", required=True, choices=("aargau", "riehen"))
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    pre = NETPREFIX[args.net]
    files = sorted(glob.glob(os.path.join(STACK[args.net], f"{pre}.*", f"{pre}.*_{pre}.*.h5")))
    if args.limit:
        files = files[:args.limit]
    outdir = os.path.join(PROJ, args.net, "tomo", "phase_picks")
    os.makedirs(outdir, exist_ok=True)
    tasks = [(f, args.net, outdir) for f in files]
    print(f"{args.net}: {len(tasks)} pairs, {args.nproc} procs -> {outdir}", flush=True)
    tally = {}
    with mp.Pool(args.nproc) as pool:
        for i, st in enumerate(pool.imap_unordered(process_pair, tasks, chunksize=8), 1):
            tally[st] = tally.get(st, 0) + 1
            if i % 500 == 0:
                print(f"  {i}/{len(tasks)}  {dict(tally)}", flush=True)
    print(f"done {args.net}: {dict(tally)}", flush=True)


if __name__ == "__main__":
    main()
