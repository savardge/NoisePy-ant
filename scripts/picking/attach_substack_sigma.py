#!/usr/bin/env python3
"""Attach data-driven per-pick uncertainties to the exported tomography pick tables.

Replaces the exported `std` column -- which is meaningless, because the two pick methods
(argmax / topology) return the IDENTICAL velocity in 99.9% of cells, so their spread is a
column of zeros -- with the substack-jackknife sigma: the scatter of that station pair's
own dispersion pick across independent ~4 h substack blocks (Bensen-style repeatability).

Mapping: the jackknife runs on a 0.1 s nominal period grid; the export uses the CWT scale
ladder. For each (pair, stream) sigma(T) is interpolated in LOG period onto the export's
rungs, then clipped to the pair's measured range (no extrapolation -- outside the jackknife
band we fall back, see below).

Columns written (originals preserved with a _cross suffix):
    std          <- sigma_mad from the jackknife (km/s)
    std_percent  <- 100 * std / velocity
    n_blocks     <- how many substack blocks the sigma came from
    sigma_src    <- 'jackknife' | 'fallback_stream_median' | 'fallback_global'
    std_cross    <- the previous cross-method std (kept for reference)

Cells with no jackknife coverage take the stream's median sigma at that period (or the
global median), flagged in `sigma_src` so the tomography can down-weight or exclude them.

Usage:
  python attach_substack_sigma.py --picks-dir <.../tomo/1_velocity_maps/inputs> \
      --jackknife <.../substack_jackknife_k2> [--suffix ""] [--dry-run]
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

# exported wave key -> jackknife stream name
STREAM = {"fund": "rayleigh_G_LR0", "love": "love_TT"}
# the jackknife measures the FUNDAMENTAL streams only (G_LR0 / TT). The Rayleigh overtone
# has no direct counterpart, so it inherits the Rayleigh-fundamental sigma of the same pair
# and period, flagged accordingly -- overtone picks are not more repeatable than the
# fundamental, so this is a lower bound rather than an invention.
INHERIT = {"overtone": "rayleigh_G_LR0"}


def load_jackknife(jk_dir, min_blocks=6):
    """{(pair, stream): (periods, sigmas)} from the per-pair jk CSVs."""
    out = {}
    files = glob.glob(os.path.join(jk_dir, "*", "*_jk.csv"))
    print("reading %s jackknife files ..." % format(len(files), ","))
    for i, f in enumerate(files):
        pair = os.path.basename(f).replace("_jk.csv", "")
        try:
            d = pd.read_csv(f, usecols=["stream", "period", "sigma_mad", "n_blocks"])
        except Exception:
            continue
        d = d[(d["n_blocks"] >= min_blocks) & np.isfinite(d["sigma_mad"])]
        for st, g in d.groupby("stream"):
            g = g.sort_values("period")
            out[(pair, st)] = (g["period"].values, g["sigma_mad"].values,
                               g["n_blocks"].values)
        if (i + 1) % 5000 == 0:
            print("  %d/%d" % (i + 1, len(files)), flush=True)
    print("  %s (pair, stream) sigma curves" % format(len(out), ","))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--picks-dir", required=True)
    ap.add_argument("--jackknife", required=True)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--min-blocks", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    jk = load_jackknife(a.jackknife, a.min_blocks)

    for wkey in ("fund", "overtone", "love"):
        fn = os.path.join(a.picks_dir, "picks_%s_uni%s.csv" % (wkey, a.suffix))
        if not os.path.exists(fn):
            continue
        d = pd.read_csv(fn)
        stream = STREAM.get(wkey) or INHERIT[wkey]
        sig = np.full(len(d), np.nan)
        nbl = np.zeros(len(d), dtype=int)
        for pair, idx in d.groupby("station_pair").groups.items():
            key = (pair, stream)
            if key not in jk:
                continue
            P, Sg, Nb = jk[key]
            T = d.loc[idx, "inst_period"].values
            # interpolate in log-period, no extrapolation beyond the measured band
            ok = (T >= P.min()) & (T <= P.max())
            if ok.any():
                v = np.interp(np.log(T[ok]), np.log(P), Sg)
                nb = np.interp(np.log(T[ok]), np.log(P), Nb)
                ii = np.asarray(idx)[ok]
                sig[d.index.get_indexer(ii)] = v
                nbl[d.index.get_indexer(ii)] = np.round(nb).astype(int)
        src = np.where(np.isfinite(sig), "jackknife", "")
        # fallback 1: stream median sigma at that period; fallback 2: global median
        if (~np.isfinite(sig)).any():
            have = np.isfinite(sig)
            if have.any():
                med_by_T = pd.Series(sig[have]).groupby(
                    d.loc[have, "inst_period"].round(4).values).median()
                fill = d.loc[~have, "inst_period"].round(4).map(med_by_T).values
                gmed = float(np.nanmedian(sig[have]))
                src[~have] = np.where(np.isfinite(fill), "fallback_stream_median",
                                      "fallback_global")
                sig[~have] = np.where(np.isfinite(fill), fill, gmed)
            else:
                src[:] = "fallback_global"
                sig[:] = np.nan
        d["std_cross"] = d["std"]
        d["std"] = np.round(sig, 4)
        d["std_percent"] = np.round(100 * d["std"] / d["group_velocity"], 2)
        d["n_blocks"] = nbl
        d["sigma_src"] = src
        cov = 100 * (src == "jackknife").mean()
        print("%-9s %s rows | jackknife sigma on %.1f%% | median sigma %.3f km/s "
              "(was cross-method %.3f)"
              % (wkey, format(len(d), ","), cov, np.nanmedian(sig),
                 float(d["std_cross"].median())))
        if a.dry_run:
            continue
        d.to_csv(fn, index=False, float_format="%.4f")
        mf = fn + ".meta.json"
        meta = json.load(open(mf)) if os.path.exists(mf) else {}
        meta.update({"std_column": "substack jackknife sigma_mad (km/s)",
                     "std_source_dir": os.path.abspath(a.jackknife),
                     "std_min_blocks": a.min_blocks,
                     "std_jackknife_coverage_pct": round(cov, 2),
                     "std_cross_column": "previous argmax-vs-topology std (degenerate)",
                     "sigma_inherited_from": INHERIT.get(wkey)})
        json.dump(meta, open(mf, "w"), indent=2)
    if a.dry_run:
        print("(dry run -- nothing written)")


if __name__ == "__main__":
    main()
