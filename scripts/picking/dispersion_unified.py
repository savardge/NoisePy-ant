"""Unified dispersion-picking driver: all EIGHT pick types for a whole network, one CSV per pair.

    Rayleigh fundamental/overtone (group + phase, via G_LR0/G_LR1 mode synthesis)
    Love     fundamental/overtone (group + phase, via TT ridge labeling against Love references)

Thin batch wrapper around noisepy.unified_picking.pick_all_modes (which holds the algorithm and the
theory notes). Mirrors dispersion_batch_modesep.py's config/discovery/Pool/skip-if-exists structure.

Requires the four data-derived reference curves under paths.ref_dir:
    ref_fundamental_phase.txt   ref_overtone_phase.txt        (Rayleigh; pick_reference_ridges.py)
    ref_love_phase.txt          ref_love_overtone_phase.txt   (Love; vsg_love_reference.py)

Usage:
    python dispersion_unified.py --config ../../param_files/modesep_aargau.yaml [--nproc N] [--limit K]
    # legacy: DISP_NET / DISP_LIMIT / DISP_REF_DIR env + positional  stack_root out_root [nproc]
    python dispersion_unified.py <stack_root> <out_root> [nproc]

Output: <out_root>/<src>/<pair>_unified.csv (unified_picking.HEADER schema).
"""
import argparse
import glob
import logging
import multiprocessing as mp
import os
import sys

import numpy as np


def _resolve():
    """Return (STACK_ROOT, OUT_ROOT, NPROC, NET, LIMIT, REF_DIR). --config wins over legacy env."""
    ap = argparse.ArgumentParser(add_help=("--config" in sys.argv or "-h" in sys.argv
                                           or "--help" in sys.argv))
    ap.add_argument("--config")
    ap.add_argument("--out")
    ap.add_argument("--nproc", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("pos", nargs="*")                 # legacy: stack_root out_root [nproc]
    a, _ = ap.parse_known_args()
    if a.config:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import modesep_config
        cfg = modesep_config.load_config(a.config)
        stack = cfg["paths"]["stack_root"]
        out = a.out or (cfg["paths"]["dispersion_dir"] + "_unified")
        nproc = a.nproc or int(cfg["batch"].get("nproc", 10))
        net = cfg["network"]["code"]
        limit = a.limit if a.limit is not None else int(cfg["batch"].get("limit", 0))
        ref_dir = cfg["paths"]["ref_dir"]
        return stack, out, nproc, net, limit, ref_dir
    # ---- legacy positional + DISP_* env (Aargau defaults) ----
    stack = a.pos[0]
    out = a.pos[1] if len(a.pos) > 1 else stack.rstrip("/") + "_unified"
    nproc = int(a.pos[2]) if len(a.pos) > 2 else 10
    net = os.environ.get("DISP_NET", "AA")
    limit = int(os.environ.get("DISP_LIMIT", "0"))
    ref_dir = os.environ.get(
        "DISP_REF_DIR",
        "/Users/genevievesavard/Codes/extract_higher_modes/Projects/aargau/vsg_modesep")
    return stack, out, nproc, net, limit, ref_dir


STACK_ROOT, OUT_ROOT, NPROC, NET, LIMIT, REF_DIR = _resolve()
STACK_METHOD = os.environ.get("DISP_STACK", "pws")    # single stack method (validated production = pws)
OVERWRITE = "--overwrite" in sys.argv or os.environ.get("DISP_OVERWRITE") == "1"
LOVE_OT = os.environ.get("DISP_LOVE_OT") == "1"       # Love overtone extraction (default off; judged
#                                                       not credible -- see unified_picking.Config)

# The four data-derived reference curves (all under REF_DIR), keyed by (wave, mode).
REF_FILES = {
    ("rayleigh", "fundamental"): "ref_fundamental_phase.txt",
    ("rayleigh", "overtone"): "ref_overtone_phase.txt",
    ("love", "fundamental"): "ref_love_phase.txt",
    ("love", "overtone"): "ref_love_overtone_phase.txt",
}

_G = {}   # per-worker globals


def _init():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TQDM_DISABLE"] = "1"
    for name in ("findpeaks", "findpeaks.stats", "matplotlib", "noisepy"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.CRITICAL)
        lg.propagate = False
        lg.handlers = [logging.NullHandler()]
    from noisepy import unified_picking as up
    from noisepy import dispersion
    up.Config.PICK_LOVE_OVERTONE = LOVE_OT
    _G["up"] = up
    refs = {}
    for key, fn in REF_FILES.items():
        path = os.path.join(REF_DIR, fn)
        try:
            refs[key] = dispersion.load_reference_curve(path)
        except Exception as e:
            print(f"WARN: could not load {key} reference ({fn}): {e}; phase disabled for it.")
            refs[key] = None
    _G["refs"] = refs
    _G["comps"] = list(up.RAYLEIGH_COMPS) + ["TT"] + list(up.Config.LOVE_CONTEXT)


def process(path):
    up = _G["up"]
    src = os.path.basename(os.path.dirname(path))
    pair = os.path.basename(path).replace(".h5", "")
    out_dir = os.path.join(OUT_ROOT, src)
    out_csv = os.path.join(out_dir, pair + "_unified.csv")
    if os.path.exists(out_csv) and not OVERWRITE:
        return "skip"
    try:
        params, ccf = up.load_pair(path, STACK_METHOD, _G["comps"])
        if "ZZ" not in ccf:
            return "no-ZZ"
        rows = up.pick_all_modes(params, ccf, _G["refs"], STACK_METHOD, cfg=up.Config)
    except Exception as e:
        return f"err:{type(e).__name__}"
    os.makedirs(out_dir, exist_ok=True)
    with open(out_csv, "w") as f:
        f.write(up.rows_to_csv(rows))
    return f"ok:{len(rows)}"


def main():
    files = sorted(glob.glob(os.path.join(STACK_ROOT, f"{NET}.*", f"{NET}.*_{NET}.*.h5")))
    if LIMIT and len(files) > LIMIT:                  # decimate evenly for a subset run
        files = files[:: max(1, len(files) // LIMIT)][:LIMIT]
    os.makedirs(OUT_ROOT, exist_ok=True)
    print(f"[unified] {len(files)} pairs | net={NET} stack={STACK_METHOD} nproc={NPROC}\n"
          f"          refs={REF_DIR}\n          out={OUT_ROOT}", flush=True)
    from collections import Counter
    tally = Counter()
    with mp.Pool(NPROC, initializer=_init, maxtasksperchild=200) as pool:
        for i, res in enumerate(pool.imap_unordered(process, files, chunksize=4), 1):
            tally[res.split(":")[0]] += 1
            if i % 200 == 0 or i == len(files):
                print(f"  {i}/{len(files)}  {dict(tally)}", flush=True)
    print(f"[unified] done: {dict(tally)}")


if __name__ == "__main__":
    main()
