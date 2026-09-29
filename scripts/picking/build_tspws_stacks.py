#!/usr/bin/env python3
"""Build an `Allstack_tspws` stack tree from the substack windows, for production picking.

Wavelet-domain phase-weighted stack (ts-PWS, Ventosa et al. 2017 -- `dispersion.ts_pws`)
across a pair's T* substack windows, per ENZ component, then rotated to RTZ. Replaces the
stored time-domain `Allstack_pws`, which distorts dispersed weak bands (see the 2026-07-30
substack-jackknife evidence: 13-33% of picks >2 sigma from their own substack consensus).

The output H5 is minimal but is read by the UNCHANGED production picker via
`DISP_STACK=tspws` (unified_picking.read_stack_components' h5py fallback needs only
AuxiliaryData/Allstack_tspws/<comp> plus dist/dt/azi/baz attrs on ZZ).

Since 2026-09-26 each file also holds AuxiliaryData/Allstack_linear: the LINEAR mean of the same
pre-block elements, kept to +-(lin_mult*dist/vmin + pad) samples (capped at the record). The
picker measures snr_nbG on it (unified_picking.Config.SNR_STACK) because the lag-trimmed ts-PWS
trace has no noise window -- the old SNR wrapped into the signal -- and a ts-PWS trace is too
noise-free for an SNR threshold to mean anything. lin_mult 2.2 leaves room for the full
'after' noise window (signal to dist/vmin, 2 T gap, then one signal length) at every period.
The ts-PWS input is the central +-(dist/vmin + pad) slice of the same elements, bit-identical to
the pre-2026-09-26 build. The inverse CWT is inlined in dispersion.ts_pws (Torrence & Compo
eq. 11), so the output no longer depends on the installed pycwt; datasets carry icwt='tc98'.
--reuse-tspws copies Allstack_tspws from an existing tree instead of recomputing it (use only
for a tree already built with the tc98 inverse, e.g. the laptop-built Riehen stacks).
See extract_higher_modes/Projects/method_tests/2_pick_qc/test_2026-09-26_tspws_snr_noise_window/.

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
A.add_argument("--lin-mult", type=float, default=2.2,
               help="Allstack_linear keeps +-(lin_mult*dist/vmin_trim + pad) samples (0 = no linear)")
A.add_argument("--reuse-tspws", default=None, metavar="TREE",
               help="copy Allstack_tspws from TREE/<sta>/<pair>.h5 when present (must be a tc98 "
                    "build), compute only Allstack_linear")
A.add_argument("--limit", type=int, default=0)
A.add_argument("--code", default=None, metavar="XX",
               help="station-code prefix (RI/AA/SS). Restricts the pair glob to "
                    "<code>.*/<code>.*_<code>.*.h5, matching what dispersion_unified.py "
                    "picks. Without it the glob takes every *.h5, which on the Riehen and "
                    "Aargau trees includes CH.* broadband cross-correlations the picker "
                    "then ignores -- 24%% and 18%% of the files respectively, built for "
                    "nothing.")
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
            # elements are read to the LONGER linear half-width; ts-PWS takes the central +-L
            Lr = max(L, min(int(args.lin_mult * params["dist"] / args.vmin_trim / params["dt"])
                            + args.pad, mid)) if args.lin_mult > 0 else L
            per_comp = {c: [] for c in ENZ}
            K = max(1, args.pre_block)
            for i in range(0, len(tg), K):
                acc, cnt = {}, {}
                for k in tg[i:i + K]:
                    g = aux[k]
                    for c in ENZ:
                        if c in g:
                            acc[c] = acc.get(c, 0) + g[c][mid - Lr:mid + Lr + 1].astype(np.float64)
                            cnt[c] = cnt.get(c, 0) + 1
                for c, v in acc.items():
                    per_comp[c].append(v / cnt[c])
        if any(len(v) < MIN_WINDOWS for v in per_comp.values()):
            return "few-windows"
        rt, tspws_src = None, "computed"
        reuse = os.path.join(args.reuse_tspws, src, pair) if args.reuse_tspws else None
        if reuse and os.path.exists(reuse):
            with h5py.File(reuse, "r") as f:
                g = f["AuxiliaryData/Allstack_tspws"]
                old = {c: g[c][()] for c in KEEP}
            if all(len(v) == 2 * L + 1 for v in old.values()):
                rt = [old[c] for c in RTZ]
                tspws_src = "reused:%s" % args.reuse_tspws
        if rt is None:
            sl = slice(Lr - L, Lr + L + 1)
            stacked = np.stack([dispersion.ts_pws(np.asarray(per_comp[c])[:, sl], params["dt"])
                                for c in ENZ])
            rt = rotation(stacked, params, {})
        lin = None
        if args.lin_mult > 0:
            lin = rotation(np.stack([np.mean(np.asarray(per_comp[c]), axis=0) for c in ENZ]),
                           params, {})
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
                d = g.create_dataset(c, data=np.asarray(rt[i]).astype(np.float32))
                for k, v in params.items():
                    d.attrs[k] = v
                d.attrs["icwt"] = "tc98"
                d.attrs["tspws_source"] = tspws_src
            if lin is not None:
                g = f.create_group("AuxiliaryData/Allstack_linear")
                for c, i in KEEP.items():
                    d = g.create_dataset(c, data=lin[i].astype(np.float32))
                    for k, v in params.items():
                        d.attrs[k] = v
                    d.attrs["pre_block"] = max(1, args.pre_block)
        os.replace(tmp, ofile)                # atomic: a killed run leaves no partial file
    except Exception as e:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        return "werr:%s" % type(e).__name__
    return "ok"


def main():
    pat = (os.path.join(args.stack_root, "%s.*" % args.code,
                        "%s.*_%s.*.h5" % (args.code, args.code)) if args.code
           else os.path.join(args.stack_root, "*", "*.h5"))
    files = sorted(glob.glob(pat))
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
