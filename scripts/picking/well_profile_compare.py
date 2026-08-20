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
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from well_vs_qc import WELLS, overlay_curves                      # noqa: E402

CFG_COLORS = {"R0g": "tab:blue", "R0p": "tab:cyan", "L0g": "tab:red", "L0p": "tab:pink",
              "RLg_radial": "tab:green", "RLg_iso": "tab:olive"}


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
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    vols = {}
    for spec in a.configs:
        label, fn = spec.split(":", 1)
        p = os.path.join(a.root, label, fn)
        if not os.path.exists(p):
            print(f"skip {label}: {p} not found")
            continue
        vols[label] = np.load(p, allow_pickle=True)
    if not vols:
        raise SystemExit("no volumes loaded")

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
            if "z_reliable_max" in v.files:
                zr = float(np.asarray(v["z_reliable_max"], float)[i])
                if np.isfinite(zr):
                    bad |= z > zr
            if "z_reliable_min" in v.files:
                zn = float(np.asarray(v["z_reliable_min"], float)[i])
                if np.isfinite(zn) and zn > 0:
                    bad |= z < zn
            if bad.any():
                med = np.where(bad, np.nan, med)
                p16 = np.where(bad, np.nan, p16)
                p84 = np.where(bad, np.nan, p84)
            c = CFG_COLORS.get(label, "k")
            ax.plot(med, z, color=c, lw=1.8, label=f"{label} ({dist:.2f} km)", zorder=3)
            ax.fill_betweenx(z, p16, p84, color=c, alpha=0.18, lw=0, zorder=2)
            drew = True
        if not drew:
            plt.close(fig)
            continue
        for vs_, z_, lab, col, ls in ov:
            ax.plot(vs_, z_, color=col, ls=ls, lw=1.2, label=lab, zorder=4)
        ax.axhline(wdep / 1000.0, color="0.4", lw=0.8, ls=":", zorder=1)
        ax.annotate(f"well TD {wdep} m", (0.02, wdep / 1000.0), xycoords=("axes fraction", "data"),
                    fontsize=7, color="0.35", va="bottom")
        ax.set_ylim(a.depth_max, 0)
        ax.set_xlabel("Vs [km/s]")
        ax.set_ylabel("depth below surface [km]")
        ax.set_title(f"{a.net} — {nm}", fontsize=11)
        ax.grid(alpha=0.25)
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
