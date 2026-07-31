#!/usr/bin/env python3
"""Substack jackknife over a whole stack tree: per-(pair, period) dispersion pick
repeatability, plus a flag where the stored Allstack_pws pick deviates from the consensus.

Method (validated in the 2026-07-29 pilot): group the T* substack windows into blocks of
--k consecutive windows (~4 h at k=2 with the Riehen cadence), accumulate PER COMPONENT
per block with per-component MEANS (components drop in and out per window independently),
rotate ENZ->RTZ, synthesize G_LR0 (Rayleigh fundamental) and take TT (Love), per-period
argmax on each block. Pilot findings this rests on:
  * scatter is FLAT vs block length (systematic-dominated), so short blocks are valid and
    sigma(blocks) ~ sigma of the allstack pick (0.03-0.06 km/s on the pilot pairs);
  * median over blocks agrees with the Allstack_linear pick to <= 0.04 km/s;
  * the stored Allstack_pws (time-domain PWS, stacking.py) deviates by up to 0.7 km/s in
    weak bands, hence the flag column.

Output: one <pair>_jk.csv per pair under --out (skip-if-exists resume, like the picker),
columns: stream, period, U_med_blocks, sigma_mad, n_blocks, U_pws, U_lin, flag_pws.
flag_pws = |U_pws - U_med_blocks| > max(2*sigma_mad, 0.05).

Usage:
  PYTHONPATH=... python substack_jackknife.py --stack-root /Volumes/T7blue/riehen-data/STACK_CHRI_normZ \
      --out .../Projects/riehen/substack_jackknife_k2 --k 2 --nproc 8 [--limit N]
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from noisepy import dispersion
from noisepy.stacking import rotation
from noisepy.unified_picking import split_lags

ENZ = ["EE", "EN", "EZ", "NE", "NN", "NZ", "ZE", "ZN", "ZZ"]
RTZ = ["ZR", "ZT", "ZZ", "RR", "RT", "RZ", "TR", "TT", "TZ"]
VMIN, VMAX, TMIN, VAVE, DT_GRID = 0.2, 4.5, 0.4, 3.0, 0.1
MIN_DIST = 1.5          # [km] band too narrow below this (Tmax = dist/vave)
HEADER = "stream,period,U_med_blocks,sigma_mad,n_blocks,U_pws,U_lin,flag_pws\n"

A = argparse.ArgumentParser(description=__doc__.splitlines()[0])
A.add_argument("--stack-root", required=True)
A.add_argument("--out", required=True)
A.add_argument("--k", type=int, default=2, help="substack windows per block (2 ~ 4 h)")
A.add_argument("--nproc", type=int, default=8)
A.add_argument("--limit", type=int, default=0)
A.add_argument("--shard", default=None, metavar="I/N",
               help="process only shard I of N (0-based), for a Slurm job array. Strided "
                    "(files[I::N]) so cost spreads evenly. Output is one CSV per pair and "
                    "skip-if-exists, so shards never collide and a re-run resumes.")
args = A.parse_args()

SHARD_I, SHARD_N = 0, 1
if args.shard:
    SHARD_I, SHARD_N = (int(x) for x in args.shard.split("/"))
    if not 0 <= SHARD_I < SHARD_N:
        A.error("--shard I/N needs 0 <= I < N (got %s)" % args.shard)


def argmax_curve(trace, dist, dt):
    cw = dispersion.compute_cwt(trace.astype(np.float64), dist, dt,
                                Tmin=TMIN, vmin=VMIN, vmax=VMAX, vave=VAVE)
    amp, per, vel, _ = dispersion.disp_image_from_cwt(cw, dist, Tmin=TMIN, dT=DT_GRID,
                                                      vmin=VMIN, vmax=VMAX, vave=VAVE)
    # NaN-safe: a degenerate block can leave whole periods all-NaN after the per-period
    # normalization; nanargmax would raise. Those periods stay NaN in the curve.
    amp = np.where(np.isfinite(amp), amp, -np.inf)
    U = np.full(amp.shape[0], np.nan)
    ok = np.isfinite(amp).any(axis=1)
    U[ok] = vel[np.argmax(amp[ok], axis=1)]
    return per, U


def curves(ten9_or_comp, params, rotated=False):
    if rotated:
        sym = {c: split_lags(ten9_or_comp[c])["sym"] for c in ten9_or_comp}
    else:
        rt = rotation(ten9_or_comp, params, {})
        sym = {c: split_lags(rt[RTZ.index(c)])["sym"] for c in ("ZZ", "RR", "RZ", "ZR", "TT")}
    g0, _ = dispersion.synthesize_rayleigh_modes(sym["ZZ"], sym["RR"], sym["RZ"], sym["ZR"])
    return {"rayleigh_G_LR0": argmax_curve(g0, params["dist"], params["dt"]),
            "love_TT": argmax_curve(sym["TT"], params["dist"], params["dt"])}


def one_pair(path):
    pair = os.path.basename(path).replace(".h5", "")
    src = os.path.basename(os.path.dirname(path))
    odir = os.path.join(args.out, src)
    ocsv = os.path.join(odir, pair + "_jk.csv")
    if os.path.exists(ocsv):
        return "skip"
    try:
        with h5py.File(path, "r") as f:
            aux = f["AuxiliaryData"]
            tg = sorted(k for k in aux if k.startswith("T"))
            if not tg:
                return "no-substacks"
            # attrs live on each component dataset, and the FIRST window does not always
            # carry ZZ (RI.BAS04's early windows lack Z entirely -> 779 KeyErrors in the
            # first campaign). Take attrs from the first window that has ZZ.
            at = None
            for k in tg:
                if "ZZ" in aux[k]:
                    at = aux[k]["ZZ"].attrs
                    break
            if at is None:
                return "no-ZZ"
            params = {k: float(at[k]) for k in ("dist", "dt", "azi", "baz")}
            if params["dist"] < MIN_DIST:
                return "short"
            minw = 1 if args.k <= 3 else max(2, args.k // 3)
            blocks = []
            for i in range(0, len(tg) - args.k + 1, args.k):
                acc, cnt = {}, {}
                for k in tg[i:i + args.k]:
                    g = aux[k]
                    for c in ENZ:
                        if c in g:
                            acc[c] = acc.get(c, 0) + g[c][:].astype(np.float64)
                            cnt[c] = cnt.get(c, 0) + 1
                if all(cnt.get(c, 0) >= minw for c in ENZ):
                    blocks.append(np.stack([acc[c] / cnt[c] for c in ENZ]))
            if len(blocks) < 6:
                return "few-blocks"
            alls = {m: {c: aux["Allstack_" + m][c][:] for c in ("ZZ", "RR", "RZ", "ZR", "TT")}
                    for m in ("linear", "pws")}
    except Exception as e:
        return "err:%s" % type(e).__name__

    try:
        bcur = [curves(b, params) for b in blocks]
        ref = {m: curves(alls[m], params, rotated=True) for m in alls}
    except Exception as e:
        return "err:%s" % type(e).__name__

    os.makedirs(odir, exist_ok=True)
    lines = [HEADER]
    for st in ("rayleigh_G_LR0", "love_TT"):
        pers = ref["linear"][st][0]
        M = np.full((len(bcur), len(pers)), np.nan)
        for i, b in enumerate(bcur):
            M[i, :len(b[st][1])] = b[st][1][:len(pers)]
        med = np.nanmedian(M, axis=0)
        mad = 1.4826 * np.nanmedian(np.abs(M - med[None, :]), axis=0)
        n = np.sum(np.isfinite(M), axis=0)
        upws = ref["pws"][st][1][:len(pers)]
        ulin = ref["linear"][st][1][:len(pers)]
        flag = np.abs(upws - med) > np.maximum(2 * mad, 0.05)
        for i, T in enumerate(pers):
            lines.append("%s,%.2f,%.3f,%.4f,%d,%.3f,%.3f,%d\n"
                         % (st, T, med[i], mad[i], n[i], upws[i], ulin[i], int(flag[i])))
    with open(ocsv, "w") as fh:
        fh.writelines(lines)
    return "ok"


def main():
    files = sorted(glob.glob(os.path.join(args.stack_root, "*", "*.h5")))
    if args.limit:
        files = files[:args.limit]
    total = len(files)
    if SHARD_N > 1:
        files = files[SHARD_I::SHARD_N]
    print("[jk] %d pairs (shard %d/%d of %d) | k=%d | nproc=%d | out=%s"
          % (len(files), SHARD_I, SHARD_N, total, args.k, args.nproc, args.out))
    os.makedirs(args.out, exist_ok=True)
    stats = {}
    with Pool(args.nproc) as pool:
        for i, r in enumerate(pool.imap_unordered(one_pair, files, chunksize=4)):
            stats[r] = stats.get(r, 0) + 1
            if (i + 1) % 200 == 0:
                print("  %d/%d %s" % (i + 1, len(files), stats), flush=True)
    print("[jk] done:", stats)


if __name__ == "__main__":
    main()
