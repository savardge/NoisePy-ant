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
    ap.add_argument("--winsor-pct", type=float, default=25.0,
                    help="winsorize sigma from below at this PER-PERIOD percentile "
                         "(0 disables). The jackknife measures repeatability, not "
                         "accuracy: a pick locked onto the same wrong arrival in every "
                         "substack has MAD -> 0, pins at the floor, and a measured-Cd "
                         "inversion then hands it ~500x the weight of an honest pick. On "
                         "Riehen fund that population is 9-15%% of short-period picks and "
                         "14-24%% slower than the rest; it drove map structure to 26 km/s "
                         "and a NEGATIVE mean velocity at T=0.51 s. Winsorizing at p25 "
                         "(not p10 -- at the bad periods p10 IS the floor) removed every "
                         "negative-var_red period and beat deleting the picks outright. "
                         "The raw value is preserved in std_jk.")
    ap.add_argument("--sigma-floor", type=float, default=0.0297,
                    help="clamp sigma from below [km/s]. Default 1.4826*0.02 = the "
                         "jackknife's OWN resolution: its picks live on a 0.02 km/s "
                         "velocity grid, so a MAD finer than one quantum is not a "
                         "measurement. Without it 0.02-0.3%% of rows carry std=0 exactly "
                         "and others go to 5e-4, which a Tarantola-Valette inversion "
                         "reads as (near-)zero data variance = near-infinite weight.")
    ap.add_argument("--proxy", action="store_true",
                    help="the sigma being attached does not measure this table's own "
                         "quantity. Use for PHASE tables: the jackknife re-picks GROUP "
                         "velocity per substack block, so there is no phase sigma "
                         "anywhere. Group sigma is a defensible RELATIVE weight (same "
                         "waveform quality drives both) but its absolute scale is wrong "
                         "for phase, which is generally better determined. Every "
                         "sigma_src value gets a _group_proxy suffix so no downstream "
                         "step can mistake it for a measured phase uncertainty.")
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
        tag = "_group_proxy" if a.proxy else ""
        # dtype=object, NOT the default: np.where(..., "jackknife", "") yields dtype "<U9"
        # and every longer label assigned later is silently clipped to 9 chars, which made
        # "fallback_stream_median" and "fallback_global" both read as "fallback_".
        src = np.where(np.isfinite(sig), "jackknife" + tag, "").astype(object)
        # fallback 1: stream median sigma at that period; fallback 2: global median
        if (~np.isfinite(sig)).any():
            have = np.isfinite(sig)
            if have.any():
                med_by_T = pd.Series(sig[have]).groupby(
                    d.loc[have, "inst_period"].round(4).values).median()
                fill = d.loc[~have, "inst_period"].round(4).map(med_by_T).values
                gmed = float(np.nanmedian(sig[have]))
                src[~have] = np.where(np.isfinite(fill),
                                      "fallback_stream_median" + tag,
                                      "fallback_global" + tag)
                sig[~have] = np.where(np.isfinite(fill), fill, gmed)
            else:
                src[:] = "fallback_global" + tag
                sig[:] = np.nan
        d["std_cross"] = d["std"]
        n_floored = int(np.sum(np.isfinite(sig) & (sig < a.sigma_floor)))
        sig = np.where(np.isfinite(sig), np.maximum(sig, a.sigma_floor), sig)
        d["std_jk"] = np.round(sig, 4)          # pre-winsorizing jackknife value
        n_wins = 0
        if a.winsor_pct and a.winsor_pct > 0:
            ser = pd.Series(sig, index=d.index)
            cut = ser.groupby(d["inst_period"]).transform(
                lambda x: np.nanpercentile(x, a.winsor_pct) if np.isfinite(x).any() else np.nan)
            lift = np.isfinite(sig) & np.isfinite(cut.values) & (sig < cut.values)
            n_wins = int(lift.sum())
            sig = np.where(lift, cut.values, sig)
        d["std"] = np.round(sig, 4)
        d["std_percent"] = np.round(100 * d["std"] / d["group_velocity"], 2)
        d["n_blocks"] = nbl
        d["sigma_src"] = src
        cov = 100 * (src == "jackknife" + tag).mean()
        print("%-9s %s rows | jackknife sigma on %.1f%% | median sigma %.3f km/s "
              "(was cross-method %.3f) | floored %s at %.4f | winsorized %s at p%g"
              % (wkey, format(len(d), ","), cov, np.nanmedian(sig),
                 float(d["std_cross"].median()), format(n_floored, ","), a.sigma_floor,
                 format(n_wins, ","), a.winsor_pct))
        if a.dry_run:
            continue
        d.to_csv(fn, index=False, float_format="%.4f")
        mf = fn + ".meta.json"
        meta = json.load(open(mf)) if os.path.exists(mf) else {}
        meta.update({"std_column": (
                         "substack jackknife sigma_mad (km/s) -- GROUP-velocity "
                         "repeatability applied as a PROXY on this phase table; "
                         "relative weight only, absolute scale NOT calibrated for phase"
                         if a.proxy else "substack jackknife sigma_mad (km/s)"),
                     "std_is_proxy": bool(a.proxy),
                     "std_floor_km_s": a.sigma_floor,
                     "std_winsor_pct": a.winsor_pct,
                     "std_n_winsorized": n_wins,
                     "std_jk_column": "jackknife sigma BEFORE winsorizing",
                     "std_n_floored": n_floored,
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
