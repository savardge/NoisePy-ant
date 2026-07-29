"""ffscan QC figure: the 2-D pick distribution EFFECTIVELY used by each arm's inversion.

One figure per wave inside the arm's output_root ({output_root}/pick_distributions/{wave}.png):
  top    : period x velocity density (LINEAR counts, CWT-scale period bins)
  middle : period x distance density with the arm's PER-PICK r/lambda cut as a band --
           the cut d >= X*v*T depends on each pick's own velocity, so the admissible
           minimum distance spans X*[p10..p90 of v(T)]*T (dashed = X*median(v|T)*T)
  bottom : picks per period, kept vs cut, and which periods produced a map

Both panels draw the arm's KEPT picks in color OVER the base (pre-cut) pool in grey, so
what the cut removed is visible — and a non-binding cut (e.g. ff1.0: the picker's own
emission floor is ~1.4-2.6 lambda depending on T) shows as a band below the grey edge
with no grey fringe, rather than floating in empty space.
Markers in the bottom panel show the periods that actually produced a map
(production_{wave}.csv), so "picks exist but no map" periods are visible. Non-continuous velocity/distance ranges
across neighbouring periods (vbounds clipping, sparse discrete CWT scales for phase,
lam_ref jumps) show up directly as vertical seams.

Usage:
  python ffscan_pick_hists.py --manifest .../ffscan_manifest.json [--only <run substr>]
  python ffscan_pick_hists.py --yaml <arm.yaml> --ff X       (single arm)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yaml as _yaml

from ffscan_common import scale_bin_edges, populated_bin_edges


def arm_figs(ypath, ff, measure=None):
    net = next(n for n in ("riehen", "aargau", "hautesorne") if n in ypath)
    cfg = _yaml.safe_load(open(ypath))
    root = cfg["output_root"]
    measure = measure or ("phase" if "phase" in os.path.basename(ypath) else "group")
    outdir = os.path.join(root, "pick_distributions")
    os.makedirs(outdir, exist_ok=True)
    made = []
    for wave, csv in cfg["pick_files"].items():
        if not os.path.exists(csv):
            continue
        d = pd.read_csv(csv, usecols=["inst_period", "group_velocity", "distance",
                                      "station_pair"])
        if not len(d):
            continue
        # base (pre-cut) pool for the grey underlay: picks_{w}_uni_ffX.csv -> _uni_nf.csv,
        # picks_{w}_uni_phase_ffX.csv -> _uni_nf_phase.csv
        bname = os.path.basename(csv)
        base_csv = os.path.join(
            os.path.dirname(csv),
            (bname.split("_uni_phase_ff")[0] + "_uni_nf_phase.csv") if "_uni_phase_ff" in bname
            else (bname.split("_uni_ff")[0] + "_uni_nf.csv"))
        b = pd.read_csv(base_csv, usecols=["inst_period", "group_velocity", "distance"]) \
            if os.path.exists(base_csv) else d
        T, V, R = d.inst_period.values, d.group_velocity.values, d.distance.values
        # bins = the ladder rungs the data actually populates (see ffscan_common)
        tb = populated_bin_edges(b.inst_period.values)
        if tb is None:
            tb = scale_bin_edges(net, b.inst_period.min(), b.inst_period.max())
        tc = np.sqrt(tb[:-1] * tb[1:])
        # mapped periods, if the iso stage already ran
        pcsv = os.path.join(root, "production", wave, f"production_{wave}.csv")
        mapped = pd.read_csv(pcsv)["T"].values if os.path.exists(pcsv) else []

        # three stacked panels sharing the period axis, colorbars in their own column so
        # every panel spans exactly the same x-range (a colorbar stolen from the axes
        # would shrink it and break the alignment with the count strip)
        fig = plt.figure(figsize=(12.5, 10.2))
        gs = fig.add_gridspec(3, 2, width_ratios=[1, 0.022],
                              height_ratios=[1, 1, 0.62], hspace=0.13, wspace=0.03)
        ax_v = fig.add_subplot(gs[0, 0])
        ax_d = fig.add_subplot(gs[1, 0], sharex=ax_v)
        ax_n = fig.add_subplot(gs[2, 0], sharex=ax_v)
        for ax, cax, Y, Yb, ylab in (
                (ax_v, fig.add_subplot(gs[0, 1]), V, b.group_velocity.values,
                 f"{measure} velocity (km/s)"),
                (ax_d, fig.add_subplot(gs[1, 1]), R, b.distance.values, "distance (km)")):
            yb = np.histogram_bin_edges(Yb, bins=60)
            hb, _, _ = np.histogram2d(b.inst_period.values, Yb, bins=[tb, yb])
            h, _, _ = np.histogram2d(T, Y, bins=[tb, yb])
            # LINEAR counts, saturated at the 99th pct of occupied cells (a log scale
            # made a handful of dense short-period cells set the whole colour range)
            vmax = max(float(np.percentile(h[h > 0], 99)) if (h > 0).any() else 1.0, 1.0)
            hb[hb == 0] = np.nan
            h[h == 0] = np.nan
            ax.pcolormesh(tb, yb, hb.T, cmap="Greys", vmin=0, vmax=vmax, alpha=0.55)
            im = ax.pcolormesh(tb, yb, h.T, cmap="viridis", vmin=0, vmax=vmax)
            fig.colorbar(im, cax=cax, extend="max", label="picks / cell")
            ax.set_ylabel(ylab)
            ax.tick_params(labelbottom=False)
        # median of the KEPT picks = the curve the inversion actually sees
        s = pd.Series(V, index=np.digitize(T, tb)).groupby(level=0)
        med = s.median()
        idx = med.index[(med.index > 0) & (med.index <= len(tc))]
        ax_v.plot(tc[idx - 1], med.loc[idx].values, "-", color="w", lw=3.0)
        ax_v.plot(tc[idx - 1], med.loc[idx].values, "-", color="crimson", lw=1.6,
                  label="median of kept picks")
        ax_v.legend(loc="upper right", fontsize=8)
        # the cut is REFERENCE-lambda: d >= ff*v_ref(T)*T, one curve for every pick at a
        # given period (v_ref = smoothed median of the base pool, as in ffscan_filter_picks)
        sb = pd.Series(b.group_velocity.values,
                       index=np.digitize(b.inst_period.values, tb)).groupby(level=0)
        bmed = sb.median()
        bidx = bmed.index[(bmed.index > 0) & (bmed.index <= len(tc))]
        t = tc[bidx - 1]
        vref = pd.Series(bmed.loc[bidx].values).rolling(5, center=True,
                                                        min_periods=1).median().values
        dmax = float(np.nanmax(b.distance.values))
        ax_d.plot(t, np.clip(ff * vref * t, 0, dmax * 1.02), "-", color="crimson", lw=1.8,
                  label=f"cut: d ≥ {ff:g}·v_ref(T)·T   (v_ref = smoothed median)")
        ax_d.set_ylim(0, dmax * 1.02)
        ax_d.legend(loc="upper left", fontsize=8)
        # counts per period: kept vs cut, plus which periods produced a map
        n_kept = np.histogram(T, bins=tb)[0]
        n_base = np.histogram(b.inst_period.values, bins=tb)[0]
        w = 0.9 * np.diff(tb)
        ax_n.bar(tc, n_base, width=w, color="0.75", label="cut by r/λ")
        ax_n.bar(tc, n_kept, width=w, color="tab:green", label="kept (inverted)")
        if len(mapped):
            ax_n.plot(mapped, np.full(len(mapped), 0.6), "v", color="k", ms=4,
                      clip_on=False, label=f"period mapped ({len(mapped)})")
        ax_n.set_yscale("log")
        ax_n.set_ylim(bottom=0.6)
        ax_n.set_ylabel("picks / period")
        ax_n.set_xlabel("period (s)")
        ax_n.legend(loc="upper right", fontsize=8, ncol=3)
        ax_n.grid(alpha=0.3, axis="y")
        kept_pct = 100 * len(d) / max(len(b), 1)
        fig.suptitle(
            f"{os.path.basename(root)}   {wave}  ({measure})\n"
            f"{len(d):,} of {len(b):,} picks kept by r/λ ≥ {ff:g} ({kept_pct:.0f}%)  |  "
            f"{d.station_pair.nunique() if 'station_pair' in d else 0:,} pairs  |  "
            f"{len(mapped)} periods mapped  |  grey = removed, colour = inverted",
            fontsize=10.5, y=0.985)
        fn = os.path.join(outdir, f"{wave}.png")
        fig.savefig(fn, dpi=130)
        plt.close(fig)
        made.append(fn)
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--yaml", default=None)
    ap.add_argument("--ff", type=float, default=None)
    ap.add_argument("--only", default=None, help="substring filter on the run name")
    args = ap.parse_args()
    if args.yaml:
        for fn in arm_figs(args.yaml, args.ff):
            print(fn)
        return
    with open(args.manifest) as fh:
        jobs = json.load(fh)
    for j in jobs:
        if args.only and args.only not in f"{j['net']}_{j['run']}":
            continue
        made = arm_figs(j["yaml"], j["ff"], j["measure"])
        print(f"{j['net']}_{j['run']}: {len(made)} figs")


if __name__ == "__main__":
    main()
