#!/usr/bin/env python3
"""VALIDATE the VSG reference curves by inverting them for a whole-network average 1D Vs model.

WHY. The four curves in Projects/<net>/vsg_modesep/ are not inverted anywhere in the pipeline --
they are the `c_ref` used to resolve the 2*pi*N branch ambiguity when picking per-pair PHASE.
That makes them a single point of failure: a wrong branch in the reference propagates as a
SYSTEMATIC error across every phase pick, not as random scatter. Inverting them is the direct
test: if one physically sensible Vs(z) explains all four simultaneously, the branches are
mutually consistent; if a curve can only be fitted by an implausible model, or forces the others
to degrade, that curve's branch is suspect.

The result doubles as a whole-network average Vs model, useful as an independent reference.

TWO METHODOLOGICAL CHOICES, both of which change the answer if got wrong:

1. DECIMATION. The curves carry 85-225 samples each, read off a smooth picked ridge. Those are
   NOT independent measurements -- the ridge is continuous, so neighbouring samples repeat the
   same information. Inverting all of them with independent sigma shrinks the effective
   uncertainty by ~sqrt(N) and forces the sampler to over-fit picking jitter. We therefore
   decimate onto a log-period grid (~N_DEC points per curve), which is closer to the number of
   genuinely independent points the ridge carries.

2. SIGMA. The files have no uncertainty column. Rather than invent a flat value, sigma is
   estimated per curve from the pick scatter itself: the residual of the curve about a smooth
   low-order fit, floored at SIG_FLOOR. This is a repeatability estimate, not an accuracy one --
   a systematically mis-branched curve is smooth and would get a SMALL sigma. Read the per-target
   misfits, not sigma, for branch errors.

Usage:
  python vsg_reference_inversion.py --net riehen
  python vsg_reference_inversion.py --net riehen --configs Rf,RfRo,RfLf,all4
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
NP = "/Users/genevievesavard/Codes/NoisePy-ant"
PY_BH = "/opt/anaconda3/envs/bayhunter/bin/python"
RUNNER = f"{NP}/scripts/picking/run_bayhunter_cell.py"
REF = {"fund": "ref_fundamental_phase.txt", "overtone": "ref_overtone_phase.txt",
       "love": "ref_love_phase.txt", "love_ot": "ref_love_overtone_phase.txt"}
# Named with the SAME convention as the production well-cell combos so the two studies are
# directly comparable. Every VSG reference curve is a PHASE curve, hence "p" throughout:
#   R0p = Rayleigh fundamental, R1p = Rayleigh overtone, L0p = Love fundamental,
#   L1p = Love overtone (the curve the picker documents as unreliable by construction).
CONFIGS = {"R0p": ["fund"],
           "R0pR1p": ["fund", "overtone"],
           "R0pL0p": ["fund", "love"],
           "R0pR1pL0p": ["fund", "overtone", "love"],
           "R0pR1pL0pL1p": ["fund", "overtone", "love", "love_ot"]}
# first-pass directory names, kept so completed runs are reused rather than recomputed
LEGACY = {"Rf": "R0p", "RfRo": "R0pR1p", "RfLf": "R0pL0p", "all4": "R0pR1pL0pL1p"}
N_DEC = 22            # decimated samples per curve (see docstring note 1)
SIG_FLOOR = 0.02      # km/s; below this the quantisation of the files (0.01) dominates


def load_ref(net, wave, n_dec=N_DEC, suffix=""):
    """(T, c, sigma) decimated onto a log-period grid, sigma from the pick scatter.

    `suffix` selects a variant of the reference file, e.g. "_FJ" -> ref_fundamental_phase_FJ.txt
    (the F-J topology re-pick). The slant-stack originals are never overwritten."""
    fn = REF[wave].replace(".txt", f"{suffix}.txt")
    a = np.loadtxt(f"{EHM}/{net}/vsg_modesep/{fn}")
    T, c = a[:, 0], a[:, 1]
    o = np.argsort(T); T, c = T[o], c[o]
    # scatter about a smooth fit = repeatability of the ridge pick
    deg = 4 if len(T) > 30 else 2
    resid = c - np.polyval(np.polyfit(np.log(T), c, deg), np.log(T))
    sig = max(float(np.std(resid)), SIG_FLOOR)
    grid = np.exp(np.linspace(np.log(T.min()), np.log(T.max()), n_dec))
    cd = np.interp(grid, T, c)
    return grid, cd, np.full(n_dec, sig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", default="riehen")
    ap.add_argument("--configs", default="R0p,R0pR1p,R0pL0p,R0pR1pL0p,R0pR1pL0pL1p")
    ap.add_argument("--depth-max", type=float, default=6.0)
    ap.add_argument("--vs-max", type=float, default=5.0,
                    help="MUST exceed the largest phase velocity in the curves being inverted. "
                         "Guided modes require c < Vs(halfspace), so a cap below max(c) makes "
                         "those points unfittable and the misfit reports the CAP, not the data. "
                         "The Rayleigh overtone reference reaches c=4.51 km/s (51%% of it above "
                         "4.0), so 3.6 and even 4.0 are both too low for any overtone config.")
    ap.add_argument("--out-tag", default=None,
                    help="output dir name; defaults to test_2026-08-16_vsg_reference plus a "
                         "_vmax<X> suffix whenever vs-max differs from 4.0, so cap variants are "
                         "never silently mixed in one tree")
    ap.add_argument("--ref-suffix", default="",
                    help="reference-file variant, e.g. _FJ for the F-J re-picked curves. Runs go "
                         "to a tag with the same suffix so slant-stack and F-J inversions never "
                         "share a directory.")
    ap.add_argument("--nchains", type=int, default=24)
    ap.add_argument("--burnin", type=int, default=150_000)
    ap.add_argument("--main", type=int, default=120_000)
    a = ap.parse_args()

    tag = a.out_tag or ("test_2026-08-16_vsg_reference" if abs(a.vs_max - 4.0) < 1e-9
                        else f"test_2026-08-16_vsg_reference_vmax{a.vs_max:g}")
    if a.ref_suffix:
        tag += a.ref_suffix
    out_root = f"{EHM}/{a.net}/tomo/2_vs_depth_inversion/tests/{tag}"
    os.makedirs(out_root, exist_ok=True)
    for old, new in LEGACY.items():                 # adopt first-pass results under the new names
        o, n = os.path.join(out_root, old), os.path.join(out_root, new)
        if os.path.isdir(o) and not os.path.isdir(n):
            os.rename(o, n)
            print(f"  renamed {old} -> {new} (reusing completed run)")
    print(f"VSG reference curves -> whole-network average Vs  ({a.net})\n")
    print(f"{'wave':<10}{'n_raw':>7}{'T range':>15}{'sigma':>9}")
    curves = {}
    for w in REF:
        fn = REF[w].replace(".txt", f"{a.ref_suffix}.txt")
        if not os.path.exists(f"{EHM}/{a.net}/vsg_modesep/{fn}"):
            continue                              # e.g. no F-J overtone file: skip that wave
        raw = np.loadtxt(f"{EHM}/{a.net}/vsg_modesep/{fn}")
        T, c, s = load_ref(a.net, w, suffix=a.ref_suffix)
        curves[w] = (T, c, s)
        print(f"{w:<10}{len(raw):>7}{f'{T.min():.2f}-{T.max():.2f}':>15}{s[0]:>9.3f}")
    print(f"\ndecimated to {N_DEC} pts/curve; sigma = scatter about a smooth fit "
          f"(floor {SIG_FLOOR})\n")

    for name in [c.strip() for c in a.configs.split(",") if c.strip()]:
        waves = CONFIGS[name]
        out = os.path.join(out_root, name)
        os.makedirs(out, exist_ok=True)
        npz = os.path.join(out, "bayhunter_result.npz")
        if os.path.exists(npz):
            print(f"=== {name}: already done, skip ==="); continue
        work = os.path.join(out, "work"); os.makedirs(work, exist_ok=True)
        phasefiles = {}
        for w in waves:
            T, c, s = curves[w]
            fp = os.path.join(work, f"disp_{w}.txt")
            np.savetxt(fp, np.column_stack([T, c, s]), fmt="%.6f")
            phasefiles[w] = fp
        cfg = dict(curves=phasefiles, curves_phase={}, measure="phase",
                   out_npz=npz, depth_max=a.depth_max, vs_bounds=[0.3, a.vs_max],
                   n_layers=[1, 20], maxfrac=0.5, nchains=a.nchains,
                   iter_burnin=a.burnin, iter_main=a.main, save_ensemble=True,
                   use_mp=True, mp_nthreads=8, noise_regime="bounded",
                   cell=[0, 0, float("nan"), float("nan")], net=a.net, tag=f"vsg_{name}")
        cfgp = os.path.join(work, "config.json")
        json.dump(cfg, open(cfgp, "w"))
        print(f"=== {name}: {', '.join(waves)} ===", flush=True)
        env = dict(os.environ, OBJC_DISABLE_INITIALIZE_FORK_SAFETY="YES",
                   VECLIB_MAXIMUM_THREADS="1", OMP_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        r = subprocess.run([PY_BH, RUNNER, cfgp], env=env,
                           capture_output=True, text=True)
        log = os.path.join(out, "run.log")
        open(log, "w").write(r.stdout + "\n=== STDERR ===\n" + r.stderr)
        tail = [l for l in r.stdout.strip().split("\n") if l.strip()][-3:]
        print("   " + "\n   ".join(tail) if tail else "   (no output)")
        if r.returncode != 0:
            print(f"   FAILED rc={r.returncode}; see {log}")
    print(f"\noutputs under {out_root}")


if __name__ == "__main__":
    main()
