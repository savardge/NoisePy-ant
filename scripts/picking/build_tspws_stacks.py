#!/usr/bin/env python3
"""Build an `Allstack_tspws` stack tree from the substack windows, for production picking.

Wavelet-domain phase-weighted stack (ts-PWS, Ventosa et al. 2017 -- `dispersion.tf_pws`)
across a pair's T* substack windows, per ENZ component, then rotated to RTZ. Replaces the
stored time-domain `Allstack_pws`, which distorts dispersed weak bands (see the 2026-07-30
substack-jackknife evidence: 13-33% of picks >2 sigma from their own substack consensus).

The output H5 is minimal but is read by the UNCHANGED production picker via
`DISP_STACK=tspws` (unified_picking.read_stack_components' h5py fallback needs only
AuxiliaryData/Allstack_tspws/<comp> plus dist/dt/azi/baz attrs on ZZ).

Two cost controls, both validated:
  * windows are lag-trimmed to +-(dist/vmin + pad) before any CWT (~6x)
  * --pre-block K averages K consecutive windows into one PWS element first; the pilot
    showed pick scatter is FLAT from ~4 h to ~4 days, so K=2 (~4-7 h elements) halves the
    transform count without changing what the coherence measures.

Usage:
  PYTHONPATH=... python build_tspws_stacks.py \
      --stack-root /Volumes/T7blue/riehen-data/STACK_CHRI_normZ \
      --out /Volumes/T7blue/riehen-data/STACK_CHRI_tspws --nproc 8 [--pre-block 2] [--limit N]
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import h5py

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "..", "..")))
from noisepy import dispersion
from noisepy.stacking import rotation

ENZ = ["EE", "EN", "EZ", "NE", "NN", "NZ", "ZE", "ZN", "ZZ"]
RTZ = ["ZR", "ZT", "ZZ", "RR", "RT", "RZ", "TR", "TT", "TZ"]
KEEP = {"ZR": 0, "ZT": 1, "ZZ": 2, "RR": 3, "RT": 4, "RZ": 5, "TR": 6, "TT": 7, "TZ": 8}
MIN_WINDOWS = 6

A = argparse.ArgumentParser(description=__doc__.splitlines()[0])
A.add_argument("--stack-root", required=True)
A.add_argument("--out", required=True)
A.add_argument("--nproc", type=int, default=8)
A.add_argument("--pre-block", type=int, default=2, help="windows averaged per PWS element")
A.add_argument("--vmin-trim", type=float, default=0.2, help="lag trim uses dist/this")
A.add_argument("--pad", type=int, default=64, help="extra samples kept beyond the trim")
A.add_argument("--limit", type=int, default=0)
A.add_argument("--shard", default=None, metavar="I/N",
               help="process only shard I of N (0-based), for a Slurm job array. Strided "
                    "(files[I::N]), not contiguous, so long-substack pairs spread evenly "
                    "across tasks instead of piling into one. Output is per-pair and "
                    "skip-if-exists, so shards never collide and a re-run resumes.")
args = A.parse_args()

SHARD_I, SHARD_N = 0, 1
if args.shard:
    SHARD_I, SHARD_N = (int(x) for x in args.shard.split("/"))
    if not 0 <= SHARD_I < SHARD_N:
        A.error("--shard I/N needs 0 <= I < N (got %s)" % args.shard)


def one_pair(path):
    pair = os.path.basename(path)
    src = os.path.basename(os.path.dirname(path))
    ofile = os.path.join(args.out, src, pair)
    if os.path.exists(ofile):
        return "skip"
    try:
        with h5py.File(path, "r") as f:
            aux = f["AuxiliaryData"]
            tg = sorted(k for k in aux if k.startswith("T"))
            if not tg:
                return "no-substacks"
            at = None
            for k in tg:                      # the first window does not always carry ZZ
                if "ZZ" in aux[k]:
                    at = aux[k]["ZZ"].attrs
                    break
            if at is None:
                return "no-ZZ"
            params = {k: float(at[k]) for k in ("dist", "dt", "azi", "baz")}
            npts = aux[tg[0]][sorted(aux[tg[0]].keys())[0]].shape[0]
            mid = npts // 2
            L = min(int(params["dist"] / args.vmin_trim / params["dt"]) + args.pad, mid)
            per_comp = {c: [] for c in ENZ}
            K = max(1, args.pre_block)
            for i in range(0, len(tg), K):
                acc, cnt = {}, {}
                for k in tg[i:i + K]:
                    g = aux[k]
                    for c in ENZ:
                        if c in g:
                            acc[c] = acc.get(c, 0) + g[c][mid - L:mid + L + 1].astype(np.float64)
                            cnt[c] = cnt.get(c, 0) + 1
                for c, v in acc.items():
                    per_comp[c].append(v / cnt[c])
        if any(len(v) < MIN_WINDOWS for v in per_comp.values()):
            return "few-windows"
        stacked = np.stack([dispersion.tf_pws(np.asarray(per_comp[c]), params["dt"])
                            for c in ENZ])
        rt = rotation(stacked, params, {})
    except Exception as e:
        return "err:%s" % type(e).__name__
    # The WRITE is inside try/except too: an uncaught error here propagates out of the
    # worker and kills the whole Pool (an HDF5 lock clash between two concurrent builds
    # took down a 19,503-pair run at ~5,100 pairs).
    try:
        os.makedirs(os.path.dirname(ofile), exist_ok=True)
        tmp = "%s.%d.tmp" % (ofile, os.getpid())   # per-PID: two builds cannot collide
        with h5py.File(tmp, "w") as f:
            g = f.create_group("AuxiliaryData/Allstack_tspws")
            for c, i in KEEP.items():
                d = g.create_dataset(c, data=rt[i].astype(np.float32))
                for k, v in params.items():
                    d.attrs[k] = v
        os.replace(tmp, ofile)                # atomic: a killed run leaves no partial file
    except Exception as e:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        return "werr:%s" % type(e).__name__
    return "ok"


def main():
    files = sorted(glob.glob(os.path.join(args.stack_root, "*", "*.h5")))
    if args.limit:
        files = files[:args.limit]
    total = len(files)
    if SHARD_N > 1:
        files = files[SHARD_I::SHARD_N]
    print("[tspws] %d pairs (shard %d/%d of %d) | pre-block %d | nproc %d -> %s"
          % (len(files), SHARD_I, SHARD_N, total, args.pre_block, args.nproc, args.out),
          flush=True)
    os.makedirs(args.out, exist_ok=True)
    stats = {}
    with Pool(args.nproc) as pool:
        for i, r in enumerate(pool.imap_unordered(one_pair, files, chunksize=4)):
            stats[r] = stats.get(r, 0) + 1
            if (i + 1) % 250 == 0:
                print("  %d/%d %s" % (i + 1, len(files), stats), flush=True)
    print("[tspws] done:", stats)


if __name__ == "__main__":
    main()
