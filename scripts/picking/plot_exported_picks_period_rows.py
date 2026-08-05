#!/usr/bin/env python3
"""Per-inversion-period velocity histograms of the EXPORTED tomography pick tables.

Same input as plot_exported_picks_hist2d.py (picks_<wave>_uni*.csv, one row per
station pair and period after median aggregation) but laid out as a ridgeline:
one ROW per period actually inverted -- i.e. per unique `inst_period`, which is
the picker's CWT scale rung and therefore one tomographic map -- period
increasing downwards, velocity on a shared x axis. Each row carries the p10,
p25, p50, p75 and p90 of its own velocity distribution as vertical lines, so the
spread feeding each period map is directly readable.

Two layouts, both written by default (--layout):
  columns  one column per wave table; rows are the union of their periods, so the
           same row is the same period across columns.
  overlay  all wave tables on the SAME axes per period, semi-transparent and
           colour-coded by wave, percentile lines in the wave colour -- for
           reading the mode separation at a glance instead of comparing columns.

    python plot_exported_picks_period_rows.py --dir <.../tomo/1_velocity_maps/inputs> \
        --title "Riehen" [--suffix _phase] [--layout overlay]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WAVES = [("fund", "Rayleigh fundamental"), ("overtone", "Rayleigh overtone"),
         ("love", "Love fundamental"), ("love_ot", "Love overtone")]
PCTS = [10, 25, 50, 75, 90]
# Single-wave panels: colour carries the percentile, p50 standing out and the
# pairs bracketing it sharing a style so the nesting is legible.
PCT_STYLE = {10: dict(color="#4d9be6", ls=":", lw=1.1),
             25: dict(color="#0b4fbf", ls="--", lw=1.2),
             50: dict(color="crimson", ls="-", lw=1.7),
             75: dict(color="#0b4fbf", ls="--", lw=1.2),
             90: dict(color="#4d9be6", ls=":", lw=1.1)}
# Overlay panels: colour is spent on the wave instead, so the percentile ladder
# has only linestyle/width/alpha left to distinguish it.
WAVE_COLOR = {"fund": "#1f4e9c", "overtone": "#c9401f", "love": "#1f8a3c",
              "love_ot": "#7b3fa0"}
PCT_LS = {10: dict(ls=":", lw=1.0, alpha=0.55), 25: dict(ls="--", lw=1.2, alpha=0.8),
          50: dict(ls="-", lw=1.9, alpha=1.0), 75: dict(ls="--", lw=1.2, alpha=0.8),
          90: dict(ls=":", lw=1.0, alpha=0.55)}


def load(dirname, suffix):
    found = []
    for key, label in WAVES:
        fn = os.path.join(dirname, "picks_%s_uni%s.csv" % (key, suffix))
        if not os.path.exists(fn):
            continue
        d = pd.read_csv(fn)
        if not len(d):
            print("  %-9s empty -- skipped" % key)
            continue
        meta = {}
        if os.path.exists(fn + ".meta.json"):
            meta = json.load(open(fn + ".meta.json"))
        d = d.assign(_T=d["inst_period"].round(4))
        found.append((key, label, d, meta))
        print("  %-9s %s rows | %s pairs | %d periods | U %.2f-%.2f"
              % (key, format(len(d), ","), format(d.station_pair.nunique(), ","),
                 d["_T"].nunique(), d.group_velocity.min(), d.group_velocity.max()))
    return found


def bare_row_axis(ax, lo, hi):
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.set_yticks([])
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1.18)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.grid(axis="x", alpha=0.25, lw=0.5)


def curve_at(meta, T):
    """Period-dependent upper bound (derive_vbounds track curve) recorded in the meta."""
    c = meta.get("vbounds_curve_km_s")
    if not c:
        return None
    c = np.asarray(c, float)
    return float(np.interp(T, c[:, 0], c[:, 1], left=c[0, 1], right=c[-1, 1]))


def shape(v, edges, denom=None):
    """Histogram of one period's picks, normalized to its own peak (or to `denom`).

    Passing the ghost row's peak as `denom` keeps a culled row on the same scale as the
    uncut row drawn behind it, so the removed picks show up as a gap instead of being
    renormalized away.
    """
    h, _ = np.histogram(v, bins=edges)
    return 0.5 * (edges[:-1] + edges[1:]), h / max(denom if denom else h.max(), 1)


def figure_columns(found, periods, limits, measure, a, out, ghost=None):
    nrow, ncol = len(periods), len(found)
    figh = 0.42 * nrow + 2.0
    fig, axes = plt.subplots(nrow, ncol, squeeze=False, sharex="col",
                             figsize=(4.8 * ncol, figh),
                             gridspec_kw={"hspace": 0.0, "wspace": 0.10})
    for j, (key, label, d, meta) in enumerate(found):
        lo, hi = limits[j]
        edges = np.arange(lo, hi + 0.5 * a.dv, a.dv)
        by_T = {t: g["group_velocity"].values for t, g in d.groupby("_T")}
        gby_T = {}
        if ghost is not None:
            gd = dict((k, v) for k, _, v, _ in ghost).get(key)
            if gd is not None:
                gby_T = {t: g["group_velocity"].values for t, g in gd.groupby("_T")}
        for i, T in enumerate(periods):
            ax = axes[i][j]
            bare_row_axis(ax, lo, hi)
            v = by_T.get(T)
            n = 0 if v is None else len(v)
            gv = gby_T.get(T)
            denom = None
            if gv is not None and len(gv):
                hg, _ = np.histogram(gv, bins=edges)
                denom = hg.max()
                ax.fill_between(0.5 * (edges[:-1] + edges[1:]), hg / max(denom, 1),
                                step="mid", lw=0, color="0.86")
            if n:
                weak = n < a.min_count
                x, y = shape(v, edges, denom)
                ax.fill_between(x, y, step="mid", lw=0,
                                color="0.75" if weak else "#3b4a6b")
                if not weak:
                    for p, xp in zip(PCTS, np.percentile(v, PCTS)):
                        ax.axvline(xp, **PCT_STYLE[p])
            xb = curve_at(meta, T)
            if xb is not None and lo < xb < hi:
                ax.axvline(xb, color="darkorange", lw=1.2, ls="-", alpha=0.9)
            lost = "" if not (gv is not None and len(gv) and len(gv) - n > 0) \
                else "  -%d" % (len(gv) - n)
            ax.text(1.002, 0.12, "" if not n else format(n, ",") + lost,
                    transform=ax.transAxes,
                    ha="left", va="center", fontsize=6,
                    color="0.5" if n < a.min_count else "0.25")
            if j == 0:
                # Anchored on the row's own baseline, not its middle, so a label can
                # never be read against the histogram of the row above.
                ax.set_ylabel("%.2f" % T, rotation=0, ha="right", va="bottom", y=0.0,
                              fontsize=7.5, labelpad=6)
            if i == 0:
                ax.set_title("%s\n%s rows | %s pairs" % (
                    label, format(len(d), ","), format(d.station_pair.nunique(), ",")),
                    fontsize=10, pad=8)
            if i == nrow - 1:
                ax.set_xlabel("%s velocity [km/s]" % measure.capitalize())
            else:
                ax.tick_params(labelbottom=False)
        for y in meta.get("vbounds_km_s") or []:
            if lo < y < hi:
                for i in range(nrow):
                    axes[i][j].axvline(y, color="darkgreen", lw=0.8, alpha=0.45)

    handles = [plt.Line2D([], [], label="p%d" % p, **PCT_STYLE[p]) for p in PCTS]
    if any(m.get("vbounds_km_s") for _, _, _, m in found):
        handles.append(plt.Line2D([], [], color="darkgreen", lw=0.8, alpha=0.45,
                                  label="export vbounds"))
    if any(m.get("vbounds_curve_km_s") for _, _, _, m in found):
        handles.append(plt.Line2D([], [], color="darkorange", lw=1.2,
                                  label="applied bound"))
    if ghost is not None:
        handles.append(plt.Rectangle((0, 0), 1, 1, color="0.86", label="before cull"))
    finish(fig, figh, handles, len(handles),
           "%s -- %s velocity distribution per inverted period\n"
           "one row per period (CWT scale rung = one tomographic map), increasing "
           "downward; %s, pick count at right"
           % (a.title or a.dir, measure,
              "each row scaled to the peak of the before-cull row, kept and removed "
              "counts at right" if ghost is not None else "each row scaled to its own "
              "peak"), out)


def figure_overlay(found, periods, lo, hi, measure, a, out):
    nrow = len(periods)
    figh = 0.58 * nrow + 2.2
    edges = np.arange(lo, hi + 0.5 * a.dv, a.dv)
    fig, axes = plt.subplots(nrow, 1, squeeze=False, sharex=True, figsize=(11.0, figh),
                             gridspec_kw={"hspace": 0.0})
    by_wave = [(key, WAVE_COLOR.get(key, "0.4"),
                {t: g["group_velocity"].values for t, g in d.groupby("_T")})
               for key, _, d, _ in found]
    for i, T in enumerate(periods):
        ax = axes[i][0]
        bare_row_axis(ax, lo, hi)
        for k, (key, color, by_T) in enumerate(by_wave):
            v = by_T.get(T)
            n = 0 if v is None else len(v)
            # Counts stack down the right margin in the wave colour, so a row that
            # is thin for one mode only cannot be mistaken for a narrow spread.
            ax.text(1.004, 0.62 - 0.25 * k, "" if not n else format(n, ","),
                    transform=ax.transAxes, ha="left", va="center", fontsize=6.5,
                    color=color, alpha=0.45 if n < a.min_count else 1.0)
            if not n:
                continue
            weak = n < a.min_count
            x, y = shape(v, edges)
            ax.fill_between(x, y, step="mid", color=color, lw=0,
                            alpha=0.16 if weak else 0.30)
            ax.step(x, y, where="mid", color=color, lw=0.9, alpha=0.4 if weak else 0.9)
            if not weak:
                for p, xp in zip(PCTS, np.percentile(v, PCTS)):
                    ax.axvline(xp, color=color, **PCT_LS[p])
        ax.set_ylabel("%.2f" % T, rotation=0, ha="right", va="bottom", y=0.0,
                      fontsize=8, labelpad=6)
        if i == nrow - 1:
            ax.set_xlabel("%s velocity [km/s]" % measure.capitalize())
        else:
            ax.tick_params(labelbottom=False)

    handles = [plt.Line2D([], [], color=WAVE_COLOR.get(key, "0.4"), lw=6, alpha=0.45,
                          label=label) for key, label, _, _ in found]
    handles += [plt.Line2D([], [], color="0.35", label="p%d" % p, **PCT_LS[p])
                for p in PCTS]
    finish(fig, figh, handles, len(found),
           "%s -- %s velocity distribution per inverted period, modes overlaid\n"
           "one row per period (CWT scale rung = one tomographic map), increasing "
           "downward; each mode scaled to its own peak within the row, counts at right"
           % (a.title or a.dir, measure), out)


def finish(fig, figh, handles, ncol, suptitle, out):
    fig.legend(handles=handles, loc="lower center", ncol=ncol, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.004))
    fig.suptitle(suptitle, fontsize=12)
    fig.supylabel("Period [s]", fontsize=10)
    fig.subplots_adjust(top=1 - 1.5 / figh, bottom=1.25 / figh)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote %s" % out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--suffix", default="", help="pick table suffix (e.g. _phase)")
    ap.add_argument("--title", default="")
    ap.add_argument("--layout", default="both", choices=("columns", "overlay", "both"))
    ap.add_argument("--out", default=None,
                    help="output path; honoured only for a single --layout, since "
                         "'both' writes two figures")
    ap.add_argument("--measure", default=None, choices=("group", "phase"),
                    help="label only; inferred from --suffix when omitted. The phase "
                         "tables reuse swtomotv's group_velocity COLUMN (see the export "
                         "meta), so the column name cannot be trusted for the axis label.")
    ap.add_argument("--dv", type=float, default=0.05,
                    help="velocity bin width [km/s]. Wider than the 0.02 of the 2D "
                         "histogram because a single period holds far fewer picks.")
    ap.add_argument("--vlim", default=None, metavar="MIN,MAX",
                    help="force the shared velocity limits, e.g. 0.3,4.6")
    ap.add_argument("--per-wave-vlim", action="store_true",
                    help="columns layout only: give each column its own limits instead "
                         "of one shared range (limits stay common to all periods within "
                         "a column either way). The overlay layout is one axes, so it "
                         "always uses the shared range.")
    ap.add_argument("--clip-pct", type=float, default=99.5,
                    help="percentile of the pooled velocities setting the upper limit")
    ap.add_argument("--min-count", type=int, default=30,
                    help="below this many picks a period row is drawn muted and its "
                         "percentiles are omitted as not meaningful")
    ap.add_argument("--csv", action="store_true",
                    help="also write the percentile table as CSV")
    ap.add_argument("--ghost-dir", default=None,
                    help="columns layout only: draw the pick tables of ANOTHER tree in "
                         "grey behind each row, on that tree's normalization, so what a "
                         "cull removed reads as a gap rather than being renormalized "
                         "away. Point it at the uncut tree.")
    a = ap.parse_args()
    measure = a.measure or ("phase" if "phase" in a.suffix.lower() else "group")

    found = load(a.dir, a.suffix)
    if not found:
        raise SystemExit("no pick tables in %s" % a.dir)
    ghost = load(a.ghost_dir, a.suffix) if a.ghost_dir else None
    # Union with the ghost's periods: a period a cull emptied completely still deserves
    # its row, showing the before-histogram over nothing.
    periods = np.sort(np.unique(np.concatenate(
        [d["_T"].unique() for _, _, d, _ in found + (ghost or [])])))

    stats = pd.DataFrame(
        [dict(wave=key, period_s=T, n_picks=len(g),
              **{"p%d" % p: x for p, x in zip(PCTS, np.percentile(g, PCTS))})
         for key, _, d, _ in found
         for T, g in ((t, gg["group_velocity"].values) for t, gg in d.groupby("_T"))])
    drawn = stats[stats.n_picks >= a.min_count]

    def rng(v, pct):
        """Tail-clipped range, then widened until it encloses every percentile drawn
        in it: a p10 or p90 line falling outside the axis would silently make the
        row look narrower than the picks actually are."""
        lo = np.percentile(v, 100 - a.clip_pct)
        hi = np.percentile(v, a.clip_pct)
        if len(pct):
            lo = min(lo, pct.p10.min() - a.dv)
            hi = max(hi, pct.p90.max() + a.dv)
        return np.floor(lo / a.dv) * a.dv, np.ceil(hi / a.dv) * a.dv

    if a.vlim:
        shared = tuple(float(x) for x in a.vlim.split(","))
    else:
        shared = rng(np.concatenate([d["group_velocity"].values for _, _, d, _ in found]),
                     drawn)
    if a.per_wave_vlim and not a.vlim:
        limits = [rng(d["group_velocity"].values, drawn[drawn.wave == key])
                  for key, _, d, _ in found]
    else:
        limits = [shared] * len(found)
    print("  velocity axis %.2f-%.2f km/s | %d periods | %d rows below --min-count %d"
          % (shared[0], shared[1], len(periods), len(stats) - len(drawn), a.min_count))

    single = a.layout != "both"
    if a.layout in ("columns", "both"):
        out = (a.out if single and a.out else
               os.path.join(a.dir, "picks_period_rows_%s.png" % measure))
        figure_columns(found, periods, limits, measure, a, out, ghost)
    if a.layout in ("overlay", "both"):
        out = (a.out if single and a.out else
               os.path.join(a.dir, "picks_period_rows_overlay_%s.png" % measure))
        figure_overlay(found, periods, shared[0], shared[1], measure, a, out)

    if a.csv:
        fn = os.path.join(a.dir, "picks_period_rows_%s.csv" % measure)
        stats.to_csv(fn, index=False, float_format="%.4f")
        print("wrote %s" % fn)


if __name__ == "__main__":
    main()
