"""Quarantine riehen/R0g cells damaged by the 2026-08-15 duplicate-array collision.

Two arrays (4313649 manifested, 4313640 orphaned by an aborted submit_all run) inverted this
config concurrently into one directory with identical shard slices. Before work dirs were made
per task they shared work/<cell>/, and invert_one rmtree's that dir when its cell finishes --
pulling BayHunter's chain files out from under the other run, so the posterior was assembled
from fewer chains than requested.

Detection is exact: across 3821 single-array control cells (aargau/R0g, hautesorne/R0g,
riehen/L0p) n_chains_used is ALWAYS 24, so any shortfall is damage. n_models stays ~20000 (the
storage cap) and every file loads, so nothing else reveals it.

Moves (never deletes) the npz + its two PNGs into _collision_quarantine/, so the next resubmit
regenerates the cell while the evidence survives.

  python quarantine_collision.py <cells_dir> [--expect-chains 24] [--dry-run]
"""
import argparse
import glob
import os
import shutil
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cells_dir")
    ap.add_argument("--expect-chains", type=int, default=24)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    qdir = os.path.join(os.path.dirname(args.cells_dir.rstrip("/")), "_collision_quarantine")
    files = sorted(glob.glob(os.path.join(args.cells_dir, "cell_*.npz")))
    bad = []
    for f in files:
        try:
            with np.load(f, allow_pickle=True) as z:
                n = float(z["n_chains_used"]) if "n_chains_used" in z.files else float("nan")
        except Exception as e:                                        # noqa: BLE001
            print(f"UNREADABLE {os.path.basename(f)}: {e}")
            bad.append((f, -1))
            continue
        if not np.isfinite(n) or n < args.expect_chains:
            bad.append((f, int(n) if np.isfinite(n) else -1))

    print(f"{len(files)} cells scanned, {len(bad)} damaged "
          f"({100.0 * len(bad) / max(len(files), 1):.1f}%)")
    if not bad:
        return 0
    counts = {}
    for _, n in bad:
        counts[n] = counts.get(n, 0) + 1
    print("chains-used histogram of damaged cells:", dict(sorted(counts.items())))

    if args.dry_run:
        print("--dry-run: nothing moved")
        return 0

    os.makedirs(qdir, exist_ok=True)
    moved = 0
    with open(os.path.join(qdir, "MANIFEST.tsv"), "w") as fh:
        fh.write("file\tn_chains_used\n")
        for f, n in bad:
            base = os.path.basename(f)
            shutil.move(f, os.path.join(qdir, base))
            stem = base[:-len(".npz")]
            for png in (f"{stem}_diagnostics.png", f"{stem}_posterior.png"):
                src = os.path.join(args.cells_dir, png)
                if os.path.exists(src):
                    shutil.move(src, os.path.join(qdir, png))
            fh.write(f"{base}\t{n}\n")
            moved += 1
    print(f"quarantined {moved} cells -> {qdir}")
    print(f"remaining in cells/: {len(glob.glob(os.path.join(args.cells_dir, 'cell_*.npz')))}")
    print("they regenerate on the next resubmit (skip-if-exists sees no npz)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
