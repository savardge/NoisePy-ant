"""Compare the grid Vs volumes against deep-well logs, one figure per well.

For every well in well_vs_qc.WELLS[net] this pulls the Vs(z) posterior of the volume cell
nearest the well head, for each config given, and overlays the well's own log converted to Vs.
Configs share a panel so a Rayleigh-vs-Love or group-vs-phase difference at the well is
immediately visible.

What the "log" actually is, per network -- these are NOT all velocity measurements:
  aargau      NAGRA blocky geological-interval Vp, shown as Vs = Vp/1.73, /1.90, /2.50. The
              ratio is an assumption, so treat the spread between those curves as the real
              uncertainty band, not the individual lines.
  riehen      Michel (2016) in-situ Vs + its Vp, when the external drive holding it is mounted.
  hautesorne  GVL-1 has stratigraphy but NO sonic log -- nothing to overlay; use
              gvl1_stratigraphy_compare.py to compare against formation tops instead.

  python well_profile_compare.py --net aargau \
      --configs R0g:volume_fund.npz L0g:volume_love.npz L0p:volume_love.npz \
      --root <.../2_vs_depth_inversion/vs_prod3> --out <.../well_compare>

An arm that ran only a handful of cells (a `--well-cells` run, or a Dinver arm) has no assembled
volume; give its cells DIRECTORY instead of a volume file and the cells are stacked into one on
the fly, so those arms sit in the same panel as the full-grid ones:

      --configs R0g:volume_fund.npz R0g_wellcells:cells dinver_R0gR1g_wellcells:cells
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from well_vs_qc import WELLS, overlay_curves                      # noqa: E402

# diagnostic/re-run arms get their own colors so multi-arm panels stay readable
VARIANT_SUFFIX = "_vmax4.5"      # same arm, higher Vs prior ceiling -> control's colour, dotted

CFG_COLORS = {"R0g_T1.08": "navy",
              "L0g_modegate": "darkred", "RLg_iso_modegate": "darkolivegreen",
              "RLg_radial_modegate": "darkgreen", "RLg_radial_freevpvs": "limegreen",
              "R0g_R0R1g": "purple", "R0p_R0R1p": "saddlebrown",
              "R0g_R0R1gp": "darkorange", "RLg_iso_RLgp": "dimgray",
              "R0g_g5": "black", "R0g_wellcells": "steelblue",
              "dinver_R0gR1g_wellcells": "sienna", "dinver_R0gR1gLVZ_wellcells": "goldenrod",
              "R0g": "tab:blue", "R0p": "tab:cyan", "L0g": "tab:red", "L0p": "tab:pink",
              "R0gp_g5": "midnightblue",
              "RLg_radial": "tab:green", "RLg_iso": "tab:olive",
              "L0gp_wellpilot": "tab:brown"}


# arms that inverted only a few cells write cells/cell_<ix>_<iy>_<waveset>.npz and never an
# assembled volume; stacking those files gives the same fields this figure needs.
_VOL_KEYS = ("vs_median", "vs_p16", "vs_p84", "z_reliable_max", "z_reliable_min")



STRAT_CSV = "/Users/genevievesavard/Data/aargau/nagra-wells-vp/well_stratigraphy_aargau.csv"


def read_stratigraphy(path=STRAT_CSV):
    """{well: [(depth_km, kind, label), ...]} from the Aargau well stratigraphy table.

    Sources and the depth convention are documented in the CSV header. Four wells have real
    columns (Boettstein NTB 85-02, Leuggern NTB 86-05, Riniken NTB 86-02, Boezberg-1 NAB 21-21
    Dossier X Tab. 1-2); the rest have none and simply get no markers.
    """
    out = {}
    if not os.path.exists(path):
        return out
    for ln in open(path):
        if ln.startswith("#") or ln.startswith("well,"):
            continue
        f = ln.rstrip("\n").split(",")
        if len(f) < 3 or not f[1]:
            continue
        out.setdefault(f[0], []).append(
            (float(f[1]) / 1000.0, f[2], f[3] if len(f) > 3 else ""))
    return out


def draw_stratigraphy(ax, tops, zmax, fs=6.5):
    """Group tops as labelled rules across the panel; formations as fainter unlabelled ticks.

    Labels are nudged apart so a tight sequence stays readable, and a leader is drawn when a
    label had to move. Text is right-aligned inside the axes so it never collides with the
    Vs curves, which occupy the left.
    """
    vis = sorted([t for t in tops if t[0] <= zmax], key=lambda t: t[0])
    if not vis:
        return
    for z_, kind, lab in vis:
        if kind != "group":
            ax.axhline(z_, color="#d4756b", lw=0.6, ls=":", alpha=0.65, zorder=1)
    grp = [t for t in vis if t[1] == "group"]
    if not grp:
        return
    ax.figure.canvas.draw()
    h_pt = ax.get_window_extent().height * 72.0 / ax.figure.dpi
    minsep = zmax * (fs * 1.25) / h_pt
    y = [t[0] for t in grp]
    for i in range(1, len(y)):
        y[i] = max(y[i], y[i - 1] + minsep)
    if y and y[-1] > zmax - 0.02:
        y[-1] = zmax - 0.02
        for i in range(len(y) - 2, -1, -1):
            y[i] = min(y[i], y[i + 1] - minsep)
    for (z_, kind, lab), yl in zip(grp, y):
        ax.axhline(z_, color="tab:red", lw=0.9, ls="--", alpha=0.75, zorder=1)
        if abs(yl - z_) > 0.35 * minsep:
            ax.plot([0.955, 0.978], [z_, yl], transform=ax.get_yaxis_transform(),
                    color="tab:red", lw=0.5, alpha=0.7, zorder=1, clip_on=False)
        ax.text(0.982, yl, lab, transform=ax.get_yaxis_transform(), fontsize=fs,
                color="tab:red", ha="right", va="center", zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.75))


STRAT = {}


def load_volume(path):
    """Volume-shaped dict from either a volume_*.npz or a directory of per-cell npz files."""
    if os.path.isfile(path):
        v = np.load(path, allow_pickle=True)
        return {k: v[k] for k in v.files}
    files = sorted(glob.glob(os.path.join(path, "cell_*.npz")))
    if not files:
        raise SystemExit(f"{path}: no cell_*.npz to stack")
    cells = [np.load(f, allow_pickle=True) for f in files]
    z = np.asarray(cells[0]["depth"], float)
    for f, c in zip(files, cells):
        if not np.array_equal(np.asarray(c["depth"], float), z):
            raise SystemExit(f"{f}: depth axis differs from {files[0]} -- cannot stack")
    out = {"depth": z,
           "lonlat": np.array([np.asarray(c["cell_lonlat"], float) for c in cells]),
           "cells": np.array([np.asarray(c["cell_ixiy"], int) for c in cells])}
    for k in _VOL_KEYS:
        if all(k in c.files for c in cells):
            out[k] = np.array([np.asarray(c[k], float) for c in cells])
    return out


def nearest_cell(vol, lon, lat):
    """Index of the volume cell nearest (lon, lat), plus the separation in km."""
    ll = vol["lonlat"]
    # local-flat approximation is plenty at these separations (a few km)
    coslat = np.cos(np.deg2rad(lat))
    dx = (ll[:, 0] - lon) * 111.32 * coslat
    dy = (ll[:, 1] - lat) * 110.57
    d = np.hypot(dx, dy)
    i = int(np.argmin(d))
    return i, float(d[i])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", required=True, choices=("riehen", "aargau", "hautesorne"))
    ap.add_argument("--root", required=True, help="dir holding <config>/volume_*.npz")
    ap.add_argument("--configs", nargs="+", required=True,
                    help="label:volume_file, e.g. R0g:volume_fund.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-dist-km", type=float, default=1.5,
                    help="skip a well whose nearest cell is further than this")
    ap.add_argument("--depth-max", type=float, default=6.0)
    ap.add_argument("--no-strat", action="store_true",
                    help="do not draw the Aargau well stratigraphy (group tops as red dashed "
                         "rules, formation tops as fainter dotted ones). Four wells have a "
                         "column on file; the rest are unaffected either way.")
    ap.add_argument("--no-band", action="store_true",
                    help="median curves only; use when so many arms overlap that the stacked "
                         "p16-p84 bands hide the curves they belong to")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    global STRAT
    STRAT = read_stratigraphy() if a.net == "aargau" else {}
    if a.net == "aargau":
        print(f"stratigraphy: {len(STRAT)} well(s) with a column -- "
              + ", ".join(sorted(STRAT)) if STRAT else "stratigraphy: none found")

    vols, cellonly = {}, set()      # cellonly arms are drawn dashed: a few cells, not a grid
    for spec in a.configs:
        label, fn = spec.split(":", 1)
        p = os.path.join(a.root, label, fn)
        if not os.path.exists(p):
            print(f"skip {label}: {p} not found")
            continue
        vols[label] = load_volume(p)
        if os.path.isdir(p):
            cellonly.add(label)
    if not vols:
        raise SystemExit("no volumes loaded")
    # ad-hoc arms (test variants) get stable distinct colors instead of all falling to black
    # unnamed arms get colours off tab20; sized to the panel so a 28-arm run cannot exhaust it
    def base(lb):
        return lb[:-len(VARIANT_SUFFIX)] if lb.endswith(VARIANT_SUFFIX) else lb

    # colours picked to miss everything in CFG_COLORS, so a fallback arm never twins a named one
    spare = ("dimgray", "indigo", "deeppink", "darkturquoise", "chocolate", "slateblue",
             "crimson", "seagreen", "peru", "cadetblue", "orangered", "darkslategray")
    unnamed = [lb for lb in vols if base(lb) not in CFG_COLORS]
    fallback = {lb: spare[i % len(spare)] for i, lb in enumerate(unnamed)}
    colors = {lb: CFG_COLORS.get(base(lb), fallback.get(lb)) for lb in vols}

    n_made = 0
    for nm, wla, wlo, wdep in WELLS.get(a.net, []):
        ov = overlay_curves(a.net, nm)
        fig, ax = plt.subplots(figsize=(5.4, 7.4))
        drew = False
        for label, v in vols.items():
            i, dist = nearest_cell(v, wlo, wla)
            if dist > a.max_dist_km:
                print(f"{nm}/{label}: nearest cell {dist:.2f} km away -- skipped")
                continue
            z = np.asarray(v["depth"], float)
            med = np.asarray(v["vs_median"], float)[i]
            p16 = np.asarray(v["vs_p16"], float)[i]
            p84 = np.asarray(v["vs_p84"], float)[i]
            # mask OUTSIDE this cell's resolvable range -- prior fill is not a measurement, and
            # neither is structure thinner than lam_min/3 (Vantassel & Cox 2021), which is what
            # z_reliable_min encodes. A group-only arm can have that at 0.5-1 km.
            bad = np.zeros(len(z), bool)
            if "z_reliable_max" in v:
                zr = float(np.asarray(v["z_reliable_max"], float)[i])
                if np.isfinite(zr):
                    bad |= z > zr
            if "z_reliable_min" in v:
                zn = float(np.asarray(v["z_reliable_min"], float)[i])
                if np.isfinite(zn) and zn > 0:
                    bad |= z < zn
            if bad.any():
                med = np.where(bad, np.nan, med)
                p16 = np.where(bad, np.nan, p16)
                p84 = np.where(bad, np.nan, p84)
            c = colors[label]
            lstyle = ("--" if label in cellonly
                      else ":" if label.endswith(VARIANT_SUFFIX) else "-")
            ax.plot(med, z, color=c, lw=1.8, ls=lstyle, label=f"{label} ({dist:.2f} km)", zorder=3)
            if not a.no_band:
                ax.fill_betweenx(z, p16, p84, color=c, alpha=0.18, lw=0, zorder=2)
            drew = True
        if not drew:
            plt.close(fig)
            continue
        for vs_, z_, lab, col, ls in ov:
            ax.plot(vs_, z_, color=col, ls=ls, lw=1.2, label=lab, zorder=4)
        if a.net == "aargau" and not a.no_strat:
            draw_stratigraphy(ax, STRAT.get(nm, []), a.depth_max)
        ax.axhline(wdep / 1000.0, color="0.4", lw=0.8, ls=":", zorder=1)
        ax.annotate(f"well TD {wdep} m", (0.02, wdep / 1000.0), xycoords=("axes fraction", "data"),
                    fontsize=7, color="0.35", va="bottom")
        ax.set_ylim(a.depth_max, 0)
        ax.set_xlabel("Vs [km/s]")
        ax.set_ylabel("depth below surface [km]")
        ax.set_title(f"{a.net} — {nm}", fontsize=11)
        ax.grid(alpha=0.25)
        if len(vols) > 8:
            ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                      borderaxespad=0.0)
        else:
            ax.legend(fontsize=6.5, loc="lower left")
        out = os.path.join(a.out, f"well_{nm.replace(' ', '')}.png")
        fig.savefig(out, dpi=145, bbox_inches="tight")
        plt.close(fig)
        n_made += 1
        print(f"wrote {out}")
    if not n_made:
        print("no well figures written (no wells in range, or none defined for this net)")
    if a.net == "hautesorne":
        print("note: GVL-1 has no sonic log -- the panel shows the inversions only; use "
              "gvl1_stratigraphy_compare.py for the formation-top comparison")


if __name__ == "__main__":
    main()
