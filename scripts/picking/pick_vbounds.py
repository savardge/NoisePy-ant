#!/usr/bin/env python3
"""Interactively define PERIOD-DEPENDENT velocity bounds on the QC'd dispersion picks.

A single flat ceiling does not fit these data: the fast envelope falls with period (Rayleigh
fundamental reaches ~4 km/s at 0.3 s but ~2.5 by 3 s), so a constant bound cuts hard at short
period and does nothing at long period. Here you click the bound where it belongs.

Usage
-----
  # define bounds (opens a window; needs a GUI backend, run it from a normal terminal)
  python pick_vbounds.py --qc-csv .../picks_unified_QCd.csv --out vbounds_riehen.json

  # look at what you saved, or at someone else's file, without a GUI
  python pick_vbounds.py --qc-csv .../picks_unified_QCd.csv --load vbounds_riehen.json \
                         --preview vbounds_riehen.png

Controls
--------
  left click    add a vertex to the ACTIVE bound of the panel you clicked in
  u / l         switch the active bound to Upper / Lower  (starts on lower)
  backspace     remove the last vertex from the active bound of the last panel used
  c             clear the active bound in the last panel used
  enter         save and close
  escape        close WITHOUT saving

Two vertices give a straight line in (period, velocity); add more for a polyline. Between
vertices the bound is linearly interpolated, and OUTSIDE the clicked period range it is held
flat at the end values (never extrapolated -- an extrapolated ceiling silently deletes the
short-period end, which is where the picks are densest).

Output
------
JSON, one entry per (wave, mode), each with `lower` / `upper` vertex lists in (period [s],
velocity [km/s]). Consume it with `load_bounds()` / `apply_bounds()` from this module:

    from pick_vbounds import load_bounds, apply_bounds
    b = load_bounds("vbounds_riehen.json")
    keep = apply_bounds(b, df["wave_type"], df["mode"], df["T_scale"], df["group_velocity"])
"""
import argparse
import json
import os

import matplotlib
import numpy as np
import pandas as pd

CLIP_PCT = 99.0
V_LIMITS = (0.1, 5.0)
ORDER = [("rayleigh", "fundamental"), ("rayleigh", "overtone"),
         ("love", "fundamental"), ("love", "overtone")]
LABELS = {k: "%s %s" % k for k in ORDER}


# --------------------------------------------------------------------------- consume the file
def load_bounds(path):
    with open(path) as fh:
        return json.load(fh)


def _edge(vertices, periods):
    """Interpolate one bound at `periods`; held FLAT outside the clicked range."""
    v = np.asarray(sorted(vertices), dtype=float)
    if len(v) == 0:
        return None
    if len(v) == 1:
        return np.full(len(periods), v[0, 1])
    return np.interp(periods, v[:, 0], v[:, 1], left=v[0, 1], right=v[-1, 1])


def apply_bounds(bounds, wave_type, mode, period, velocity):
    """Boolean keep-mask. Streams absent from the file are kept untouched."""
    wave_type = np.asarray(wave_type); mode = np.asarray(mode)
    period = np.asarray(period, float); velocity = np.asarray(velocity, float)
    keep = np.ones(len(period), dtype=bool)
    for key, bd in bounds.get("bounds", {}).items():
        wt, md = key.split("|")
        m = (wave_type == wt) & (mode == md)
        if not m.any():
            continue
        lo = _edge(bd.get("lower", []), period[m])
        hi = _edge(bd.get("upper", []), period[m])
        ok = np.ones(int(m.sum()), dtype=bool)
        if lo is not None:
            ok &= velocity[m] >= lo
        if hi is not None:
            ok &= velocity[m] <= hi
        keep[m] = ok
    return keep


# --------------------------------------------------------------------------- data
def scale_period_edges(qc_csv, min_share=0.05, chunk=500_000):
    """Period edges at geometric midpoints between the WELL-POPULATED CWT rungs.

    Uniform or geomspace period bins do NOT align with the rung ladder, leaving half the
    columns empty and the image striped -- unusable for clicking a bound on. Rungs carrying
    < min_share of the median rung's picks are folded into their neighbour (Riehen: 17 of 46
    rungs are reached by only 2-41 pairs out of 19,017, being each pair's band-edge rung).
    """
    cnt = {}
    for ch in pd.read_csv(qc_csv, usecols=["T_scale", "group_ok"], chunksize=chunk):
        v = ch.loc[ch["group_ok"] == 1, "T_scale"].dropna().round(4)
        for k, n in v.value_counts().items():
            cnt[k] = cnt.get(k, 0) + int(n)
    ser = pd.Series(cnt).sort_index()
    keep = ser[ser >= min_share * ser.median()]
    s = np.sort((keep if len(keep) > 2 else ser).index.values.astype(float))
    mid = np.sqrt(s[:-1] * s[1:])
    r = s[1] / s[0]
    print("period axis: %d CWT rungs (%d folded as under-populated)" % (len(s), len(ser) - len(s)))
    return np.concatenate([[s[0] / np.sqrt(r)], mid, [s[-1] * np.sqrt(r)]])



def load_hist(qc_csv, tedges, vedges, chunk=500_000):
    """Accumulate per-stream 2D histograms of the surviving GROUP picks."""
    hists = {}
    n = 0
    for ch in pd.read_csv(qc_csv, usecols=["T_scale", "group_velocity", "wave_type",
                                           "mode", "group_ok"], chunksize=chunk):
        ch = ch[ch["group_ok"] == 1].dropna(subset=["T_scale", "group_velocity"])
        n += len(ch)
        for key, s in ch.groupby(["wave_type", "mode"], observed=True):
            h, _, _ = np.histogram2d(s["T_scale"], s["group_velocity"], bins=[tedges, vedges])
            hists[key] = hists.get(key, 0) + h
    print("%s group picks" % format(n, ","))
    return hists


def draw_panel(ax, h, tedges, vedges, title):
    occ = h[h > 0]
    vmax = np.percentile(occ, CLIP_PCT) if occ.size else 1.0
    ax.pcolormesh(tedges, vedges, np.ma.masked_where(h.T == 0, h.T),
                  cmap="magma", vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(tedges[0], tedges[-1])
    ax.set_ylim(vedges[0], vedges[-1])
    ax.set_xscale("log")
    ax.set_xticks([0.3, 0.5, 1, 2, 3, 5])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Period T [s]  (CWT scale)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--qc-csv", required=True)
    ap.add_argument("--out", default=None, help="JSON to write (required unless --preview)")
    ap.add_argument("--load", default=None, help="start from an existing bounds file")
    ap.add_argument("--preview", default=None,
                    help="render the loaded bounds to this PNG and exit (no GUI needed)")
    ap.add_argument("--net", default="", help="recorded in the JSON for provenance")
    ap.add_argument("--tmin", type=float, default=0.18)
    ap.add_argument("--tmax", type=float, default=6.3)
    a = ap.parse_args()
    if not a.out and not a.preview:
        ap.error("give --out (to save) or --preview (to render)")

    if a.preview:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # period edges from the rung ladder itself (see scale_period_edges); velocity on the
    # 0.01 km/s pick grid with edges offset half a node so no pick lands on one.
    tedges = scale_period_edges(a.qc_csv)
    vedges = np.arange(V_LIMITS[0] - 0.005, V_LIMITS[1] + 0.025, 0.02)

    hists = load_hist(a.qc_csv, tedges, vedges)
    present = [k for k in ORDER if k in hists]
    if not present:
        raise SystemExit("no picks found")

    state = {"%s|%s" % k: {"lower": [], "upper": []} for k in present}
    if a.load:
        prev = load_bounds(a.load).get("bounds", {})
        for k in state:
            if k in prev:
                state[k] = {"lower": [list(map(float, p)) for p in prev[k].get("lower", [])],
                            "upper": [list(map(float, p)) for p in prev[k].get("upper", [])]}
        print("loaded bounds from %s" % a.load)

    fig, axes = plt.subplots(1, len(present), figsize=(6.0 * len(present), 5.6),
                             sharey=True, squeeze=False)
    axes = list(axes[0])
    lines = {}
    for ax, key in zip(axes, present):
        draw_panel(ax, hists[key], tedges, vedges, LABELS[key])
        k = "%s|%s" % key
        lines[k] = {
            "lower": ax.plot([], [], "-o", color="deepskyblue", lw=2, ms=5, label="lower")[0],
            "upper": ax.plot([], [], "-o", color="lime", lw=2, ms=5, label="upper")[0],
        }
    axes[0].set_ylabel("U [km/s]")
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.75)

    active = {"bound": "lower", "last": None}

    def redraw():
        for k, ln in lines.items():
            for which in ("lower", "upper"):
                v = np.array(sorted(state[k][which]), dtype=float).reshape(-1, 2)
                ln[which].set_data(v[:, 0], v[:, 1]) if len(v) else ln[which].set_data([], [])
        fig.suptitle("active bound: %s   |   click=add  u/l=switch  backspace=undo  "
                     "c=clear  enter=save  esc=cancel" % active["bound"].upper(), fontsize=11)
        fig.canvas.draw_idle()

    def on_click(ev):
        if ev.inaxes is None or ev.xdata is None:
            return
        for ax, key in zip(axes, present):
            if ev.inaxes is ax:
                k = "%s|%s" % key
                state[k][active["bound"]].append([float(ev.xdata), float(ev.ydata)])
                active["last"] = k
                redraw()
                return

    def on_key(ev):
        if ev.key in ("u", "l"):
            active["bound"] = "upper" if ev.key == "u" else "lower"
        elif ev.key == "backspace" and active["last"]:
            s = state[active["last"]][active["bound"]]
            if s:
                s.pop()
        elif ev.key == "c" and active["last"]:
            state[active["last"]][active["bound"]] = []
        elif ev.key == "enter":
            save()
            plt.close(fig)
            return
        elif ev.key == "escape":
            print("cancelled -- nothing written")
            plt.close(fig)
            return
        redraw()

    def save():
        out = {"network": a.net, "source": os.path.abspath(a.qc_csv),
               "period_axis": "T_scale",
               "note": ("vertices are (period [s], velocity [km/s]); linear between them, "
                        "held FLAT outside the clicked period range"),
               "bounds": {k: {w: sorted(v) for w, v in b.items() if v}
                          for k, b in state.items() if b["lower"] or b["upper"]}}
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print("wrote %s" % a.out)
        for k, b in out["bounds"].items():
            print("   %-24s lower %d vertices | upper %d vertices"
                  % (k, len(b.get("lower", [])), len(b.get("upper", []))))

    redraw()
    if a.preview:
        fig.savefig(a.preview, dpi=140, bbox_inches="tight")
        print("wrote %s" % a.preview)
        # report what the loaded bounds would keep
        if a.load:
            tot = keep = 0
            b = load_bounds(a.load)
            for ch in pd.read_csv(a.qc_csv, usecols=["T_scale", "group_velocity", "wave_type",
                                                     "mode", "group_ok"], chunksize=500_000):
                ch = ch[ch["group_ok"] == 1].dropna(subset=["T_scale", "group_velocity"])
                m = apply_bounds(b, ch["wave_type"], ch["mode"], ch["T_scale"],
                                 ch["group_velocity"])
                tot += len(ch); keep += int(m.sum())
            print("bounds would keep %s of %s group picks (%.2f%% cut)"
                  % (format(keep, ","), format(tot, ","), 100 * (1 - keep / max(1, tot))))
        return

    if matplotlib.get_backend().lower() == "agg":
        raise SystemExit("no interactive backend -- run from a terminal, or use --preview")
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


if __name__ == "__main__":
    main()
