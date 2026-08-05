#!/usr/bin/env python3
"""Build a complete stations.csv for the tomography from a stack tree's own attrs.

The tomography needs every station referenced by the pick tables. The existing
`stations.csv` files predate the current pick set and are SHORT -- Aargau's lists 150
stations while the ts-PWS picks reference 190, and Riehen's lists 187 against 198 (that
one has a `stations_keepflag.csv` covering all 198; Aargau has no equivalent). Stations
absent from the file are silently unusable, so the maps quietly lose their paths.

Coordinates come from the stack files themselves: each pair carries latS/lonS (source) and
latR/lonR (receiver) on its Allstack_* ZZ dataset. Elevation is not stored, so it is
written as 0.0 -- matching the convention already in the existing station files.

Only a few files per station directory are opened: a pair names both its endpoints, so the
whole network is recovered without scanning all ~18k pairs.

Usage:
  python build_station_file.py --stack-root <STACK_...> --code AA --out stations_all.csv \
      [--per-dir 3] [--check-picks picks_fund_uni.csv]
"""
import argparse
import csv
import glob
import os

import h5py


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stack-root", required=True)
    ap.add_argument("--code", required=True, help="station-code prefix (RI/AA/SS)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-dir", type=int, default=3,
                    help="files opened per station directory (a pair names 2 stations, so "
                         "a few per directory recovers the whole network)")
    ap.add_argument("--check-picks", default=None,
                    help="optional pick CSV; reports stations it references that are "
                         "missing from the generated file")
    a = ap.parse_args()

    dirs = sorted(d for d in glob.glob(os.path.join(a.stack_root, "%s.*" % a.code))
                  if os.path.isdir(d))
    coords = {}
    for i, d in enumerate(dirs):
        for f in sorted(glob.glob(os.path.join(d, "%s.*_%s.*.h5" % (a.code, a.code))))[:a.per_dir]:
            base = os.path.basename(f)[:-3]
            try:
                s, r = base.split("_")
            except ValueError:
                continue
            if s in coords and r in coords:
                continue
            try:
                with h5py.File(f, "r") as g:
                    aux = g["AuxiliaryData"]
                    key = next((k for k in aux if k.startswith("Allstack")), None)
                    if key is None or "ZZ" not in aux[key]:
                        continue
                    at = aux[key]["ZZ"].attrs
                    coords.setdefault(s, (float(at["latS"]), float(at["lonS"])))
                    coords.setdefault(r, (float(at["latR"]), float(at["lonR"])))
            except Exception:
                continue
        if (i + 1) % 50 == 0:
            print("  %d/%d dirs | %d stations" % (i + 1, len(dirs), len(coords)), flush=True)

    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "latitude", "longitude", "elevation"])
        for sid in sorted(coords):
            lat, lon = coords[sid]
            w.writerow([sid, "%.8f" % lat, "%.8f" % lon, "0.0"])
    print("wrote %s with %d stations" % (a.out, len(coords)))

    if a.check_picks and os.path.exists(a.check_picks):
        import pandas as pd
        pk = pd.read_csv(a.check_picks)
        used = set(pk["stasrc"]) | set(pk["starcv"])
        missing = used - set(coords)
        print("  picks reference %d stations | missing from generated file: %d"
              % (len(used), len(missing)))
        if missing:
            print("  MISSING:", ", ".join(sorted(missing)[:10]))


if __name__ == "__main__":
    main()
