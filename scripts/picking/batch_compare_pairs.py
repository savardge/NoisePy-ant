"""
Batch-run compare_picks_image.make_figure on the 100 largest-offset pairs of the dataset.
Threaded distance scan, sequential figure generation. Writes PNGs + a manifest into OUTDIR.

Usage:  python batch_compare_pairs.py [N=100] [outdir=/tmp/compare_batch_largest]
"""
import os
import sys
import glob
import csv
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from compare_picks_image import make_figure   # noqa: E402

logging.getLogger("findpeaks").setLevel(logging.ERROR)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else "/tmp/compare_batch_largest"
# optional: root stack dir, stations csv, pair glob (relative to root). Defaults = Riehen RI-RI.
ROOT = sys.argv[3] if len(sys.argv) > 3 else "/Volumes/Data/unige/riehen/crosscorrelations/STACK_CHRI_normZ"
STATIONS_CSV = sys.argv[4] if len(sys.argv) > 4 else "/Volumes/Data/unige/riehen/autocorrelations/stations_nodes_noisepy.csv"
PAIR_GLOB = sys.argv[5] if len(sys.argv) > 5 else "RI.*/*.h5"
STACK = "Allstack_pws"
COMPONENTS = ["ZZ", "RR", "RZ", "ZR", "TT"]   # Rayleigh (ZZ,RR,RZ,ZR) + Love (TT) per pair


def load_stations(path):
    """Return {NET.STA: (lat, lon)} from a station csv (BOM-safe; needs latitude,longitude cols)."""
    coords = {}
    with open(path, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            coords[f"{row['network']}.{row['station']}"] = (float(row["latitude"]),
                                                            float(row["longitude"]))
    return coords


def haversine(a, b):
    R = 6371.0
    lat1, lon1 = np.radians(a); lat2, lon2 = np.radians(b)
    dphi, dlam = lat2 - lat1, lon2 - lon1
    h = np.sin(dphi / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(h))


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    coords = load_stations(STATIONS_CSV)
    files = glob.glob(os.path.join(ROOT, PAIR_GLOB))
    print(f"computing distances for {len(files)} pairs from station coords ...", flush=True)
    res = []
    for f in files:
        s1, s2 = os.path.basename(f).replace(".h5", "").split("_")
        if s1 in coords and s2 in coords:
            res.append((f, haversine(coords[s1], coords[s2])))
    res.sort(key=lambda x: -x[1])
    top = res[:N]
    print(f"selected {len(top)} largest offsets: {top[0][1]:.1f} .. {top[-1][1]:.1f} km", flush=True)

    manifest = []
    for i, (f, dd) in enumerate(top):
        for comp in COMPONENTS:
            try:
                out, dist = make_figure(f, comp, STACK, OUTDIR)
                manifest.append((os.path.basename(f), comp, dist, out))
                print(f"[{i+1}/{len(top)}] {dist:6.1f} km {comp}  {os.path.basename(out)}", flush=True)
            except Exception as e:
                print(f"[{i+1}/{len(top)}] FAIL {os.path.basename(f)} {comp}: {e}", flush=True)

    with open(os.path.join(OUTDIR, "manifest.csv"), "w") as fh:
        fh.write("pair,component,dist_km,png\n")
        for p, comp, dd, o in manifest:
            fh.write(f"{p},{comp},{dd:.2f},{os.path.basename(o)}\n")
    print(f"DONE: {len(manifest)} figures in {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
