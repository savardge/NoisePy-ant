#!/usr/bin/env python3
"""Where does the S9 group-velocity tail come from -- is U ~ 3.5-4.0 km/s at 3-4 s a valid
FUNDAMENTAL Rayleigh measurement?

si_s9_s10_check.py established that the tail is real, spatially compact, and sits on the
BEST-resolved map cells (res_diag 0.118 vs 0.074 for the rest), so it cannot be waved away as a
prior-dominated corner of the map. That makes the measurement itself the open question, because
a fast deep body in the Vs section is only as good as the U that demands it.

Three candidate origins, each with a signature this script measures:

  1. OVERTONE LEAKAGE. R1 is faster than R0 at these periods, so fundamental picks contaminated
     by the first overtone would read fast. Signature: at the tail cells the fundamental map
     velocity approaches (or crosses) the overtone map velocity, while elsewhere they are well
     separated. This is the one that would invalidate the measurement.
  2. GENUINE FAST ROCK. Signature: fundamental stays well below the overtone, the tail is
     smooth in space, and the cells' uncertainty is normal.
  3. THIN / EDGE DATA. Signature: elevated unc_s, or the cells sit at the coverage margin.
     Partly excluded already by the res_diag result, checked here on unc_s.

Prints a per-cell table for the tail population and writes a three-panel figure.
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer

EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
BASE = (f"{EHM}/hautesorne/tomo/1_velocity_maps/1_production/"
        f"tspws_group_blanket_dx0.5_prod3_k2/production")
YAML = ("/Users/genevievesavard/Codes/NoisePy-ant/param_files/cluster/tomo/"
        "hautesorne_tspws_group_blanket.yaml")
TBAND = (2.8, 4.2)
UHI = 3.5


def load(wave):
    out = {}
    for f in sorted(glob.glob(f"{BASE}/{wave}/map_T*.npz")):
        z = np.load(f, allow_pickle=True)
        out[float(z["period"])] = dict(vel=np.where(z["mask"].astype(bool), z["vel"], np.nan),
                                       res=z["res_diag"], unc=z["unc_s"])
    return dict(sorted(out.items()))


def band_stack(M, lo, hi, key="vel"):
    T = np.array(list(M))
    sel = (T >= lo) & (T <= hi)
    return T[sel], np.stack([M[t][key] for t in T[sel]])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=f"{EHM}/hautesorne/tomo/2_vs_depth_inversion/manuscript_comparison_2026-08/"
                                     f"3_diagnostics/si_s9_s10_check")
    a = ap.parse_args()
    F, O = load("fund"), load("overtone")
    Tf, Vf = band_stack(F, *TBAND)
    _, Rf = band_stack(F, *TBAND, key="res")
    _, Uf = band_stack(F, *TBAND, key="unc")
    print(f"fundamental periods in band: {np.round(Tf,2).tolist()}")

    with np.errstate(invalid="ignore"):
        hot = np.nanmax(Vf, axis=0)
    tail = np.isfinite(hot) & (hot > UHI)
    rest = np.isfinite(hot) & ~tail
    print(f"tail cells: {tail.sum()}   rest: {rest.sum()}")

    # 1. overtone leakage --------------------------------------------------------------------
    To = np.array(list(O))
    ov = (To >= TBAND[0]) & (To <= TBAND[1])
    if ov.sum() == 0:
        print("no overtone maps in band -- leakage test unavailable")
        Vo = None
    else:
        Vo = np.stack([O[t]["vel"] for t in To[ov]])
        print(f"overtone periods in band: {np.round(To[ov],2).tolist()}")
        with np.errstate(invalid="ignore"):
            omed = np.nanmedian(Vo, axis=0)
        # gap = how far the fundamental sits BELOW the overtone at the same place
        gap = omed - hot
        for nm, m in (("tail", tail), ("rest", rest)):
            g = gap[m & np.isfinite(gap)]
            if g.size:
                print(f"  {nm}: overtone - fundamental = {np.median(g):+.2f} km/s "
                      f"(median), {np.mean(g < 0)*100:.0f}% of cells have fundamental FASTER "
                      f"than the overtone")

    # 3. thin / edge data --------------------------------------------------------------------
    with np.errstate(invalid="ignore"):
        um = np.nanmedian(Uf, axis=0)
        rm = np.nanmax(Rf, axis=0)
    for nm, m in (("tail", tail), ("rest", rest)):
        print(f"  {nm}: unc_s median {np.nanmedian(um[m]):.4f}   "
              f"res_diag max {np.nanmedian(rm[m]):.3f}")

    # is the tail smooth in space, or salt-and-pepper? ---------------------------------------
    from swtomotv.config import DatasetConfig
    from swtomotv.geometry import make_grid, xy2ll
    grid = make_grid(DatasetConfig.from_yaml(YAML).bounds, 0.5)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    ii, jj = np.where(tail)
    pts = []
    for ix, iy in zip(ii, jj):
        lat, lon = xy2ll(np.array([grid.x[ix]]), np.array([grid.y[iy]]), *grid.origin)
        e, n = tr.transform(lon[0], lat[0])
        pts.append((ix, iy, e, n, hot[ix, iy]))
    P = np.array([(p[2], p[3]) for p in pts])
    # count tail cells whose 4-neighbourhood also belongs to the tail
    nb = 0
    S = {(p[0], p[1]) for p in pts}
    for ix, iy, *_ in pts:
        nb += any((ix + dx, iy + dy) in S for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    print(f"  tail cells with a tail neighbour: {nb}/{len(pts)} "
          f"({100*nb/max(len(pts),1):.0f}%)  -- high = coherent patch, low = salt-and-pepper")
    print("\n  ix  iy      Easting     Northing   maxU   res    unc")
    for ix, iy, e, n, u in sorted(pts, key=lambda p: -p[4])[:15]:
        print(f"  {ix:3d} {iy:3d}  {e:11.0f}  {n:11.0f}  {u:5.2f}  "
              f"{rm[ix,iy]:.3f}  {um[ix,iy]:.4f}")

    # figure ---------------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.2))
    ax = axs[0]
    if Vo is not None:
        ax.scatter(hot[rest], omed[rest], s=6, color="0.6", label="rest")
        ax.scatter(hot[tail], omed[tail], s=22, color="tab:red", label=f"tail (U > {UHI})")
        lim = [1.5, 5.0]
        ax.plot(lim, lim, "k--", lw=1.2, label="fundamental = overtone")
        ax.set_xlim(1.5, 4.2); ax.set_ylim(1.5, 5.0)
        ax.set_xlabel("fundamental max U in band [km/s]")
        ax.set_ylabel("overtone median U in band [km/s]")
        ax.set_title("1. overtone leakage?\npoints ON the dashed line = the two modes merge",
                     fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axs[1]
    ax.scatter(um[rest], hot[rest], s=6, color="0.6", label="rest")
    ax.scatter(um[tail], hot[tail], s=22, color="tab:red", label="tail")
    ax.axhline(UHI, color="k", ls="--", lw=1.0)
    ax.set_xlabel("median unc_s in band"); ax.set_ylabel("fundamental max U [km/s]")
    ax.set_title("3. thin data?\n(elevated unc_s would mean weakly constrained)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axs[2]
    for t in Tf:
        pass
    prof_tail = np.array([np.nanmedian(Vf[k][tail]) for k in range(len(Tf))])
    prof_rest = np.array([np.nanmedian(Vf[k][rest]) for k in range(len(Tf))])
    Tall, Vall = band_stack(F, 0.0, 99.0)
    pt = np.array([np.nanmedian(Vall[k][tail]) for k in range(len(Tall))])
    pr = np.array([np.nanmedian(Vall[k][rest]) for k in range(len(Tall))])
    ax.plot(Tall, pr, "-", color="0.4", lw=2.0, label="rest of the map")
    ax.plot(Tall, pt, "-", color="tab:red", lw=2.4, label="tail cells")
    if Vo is not None:
        Toa, Voa = band_stack(O, 0.0, 99.0)
        ax.plot(Toa, [np.nanmedian(Voa[k][tail]) for k in range(len(Toa))], "--",
                color="tab:blue", lw=1.8, label="overtone, at the tail cells")
    ax.axvspan(*TBAND, color="k", alpha=0.07)
    ax.axhline(UHI, color="k", ls=":", lw=1.0)
    ax.set_xlim(0.4, 6.6); ax.set_xlabel("period [s]"); ax.set_ylabel("U [km/s]")
    ax.set_title("the tail cells' own dispersion curve", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"Origin of the S9 tail — {tail.sum()} cells above {UHI} km/s "
                 f"at {TBAND[0]}-{TBAND[1]} s", fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(a.out, "S9_tail_origin.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); print("\nwrote", p)


if __name__ == "__main__":
    main()
