#!/usr/bin/env python
"""Quantitative misfit of riehen Vs volumes against the Michel (2016) in-situ Vs log.

This is the project's only true Vs ground truth: aargau's overlays are interval Vp divided by
an assumed Vp/Vs (1.73-2.50), a band far too wide to score a 0.2 km/s difference against.
Scored only over each cell's own reliable depth window, and reported alongside the number of
depth samples used, because an arm that resolves less depth is not automatically "better" for
having a smaller misfit over fewer points.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from well_vs_qc import WELLS, overlay_curves                     # noqa: E402

E = "/Users/genevievesavard/Codes/extract_higher_modes"


def nearest(vol, lon, lat):
    ll = vol["lonlat"]
    d = np.hypot((ll[:, 0] - lon) * np.cos(np.deg2rad(lat)) * 111.32,
                 (ll[:, 1] - lat) * 111.32)
    i = int(np.argmin(d))
    return i, d[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="riehen")
    ap.add_argument("--arms", nargs="+", required=True, help="name:volume_file.npz")
    ap.add_argument("--max-km", type=float, default=2.0)
    a = ap.parse_args()

    B = f"{E}/Projects/{a.net}/tomo/2_vs_depth_inversion/vs_prod3"
    for w in WELLS[a.net]:
        name, la, lo = w[0], w[1], w[2]
        cur = [c for c in overlay_curves(a.net, name) if "in-situ" in c[2]]
        if not cur:
            continue
        gv, gz = np.asarray(cur[0][0], float), np.asarray(cur[0][1], float)
        print(f"\n=== {name}  (Michel in-situ Vs, {len(gv)} pts to {gz.max():.2f} km) ===")
        print(f"{'arm':16s} {'dist':>6s} {'n_z':>5s} {'window':>13s} {'RMS':>7s} {'bias':>7s}")
        for spec in a.arms:
            arm, vf = spec.split(":")
            f = f"{B}/{arm}/{vf}"
            if not os.path.exists(f):
                print(f"{arm:16s}  (missing)"); continue
            vol = np.load(f, allow_pickle=True)
            i, dist = nearest(vol, lo, la)
            if dist > a.max_km:
                print(f"{arm:16s} {dist:6.2f}  cell too far"); continue
            z = vol["depth"]; vs = vol["vs_median"][i]
            z0, z1 = vol["z_reliable_min"][i], vol["z_reliable_max"][i]
            m = np.isfinite(vs) & (z >= z0) & (z <= z1) & (z <= gz.max()) & (z >= gz.min())
            if m.sum() < 3:
                print(f"{arm:16s} {dist:6.2f} {m.sum():5d}  window too thin"); continue
            ref = np.interp(z[m], gz, gv)
            r = vs[m] - ref
            print(f"{arm:16s} {dist:6.2f} {m.sum():5d} "
                  f"{z0:5.2f}-{z1:5.2f} {np.sqrt(np.mean(r**2)):7.3f} {np.mean(r):+7.3f}")
    print("\nRMS/bias in km/s, over the cell's reliable window intersected with the log's range.")
    print("n_z differs between arms -- compare RMS only alongside it.")


main()
