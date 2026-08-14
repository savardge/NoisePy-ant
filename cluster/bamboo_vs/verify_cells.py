"""Integrity sweep over a vs_prod3 run tree: find per-cell npz that cannot be read.

Both consumers treat "the file exists" as "this cell is done" -- grid_vs_inversion.py skips it
on every resubmit, and assemble_volume drops what it cannot load without a word -- so a
truncated npz (a task killed at walltime mid-write, before the atomic-write fix) becomes an
invisible hole in the volume rather than an error. This finds them, and with --delete removes
them so the next resubmit regenerates the cell.

Also removes stray *.tmp<pid>.npz left by a kill between savez and os.replace.

  python verify_cells.py /srv/beegfs/scratch/users/s/savardg/vs_prod3/runs [--delete]
"""
import argparse
import glob
import os
import sys

import numpy as np

# keys every result must carry; a file that loads but lacks these is truncated-but-parseable
REQUIRED = ("depth", "vs_median", "cell_ixiy")


def check(path):
    """-> None if healthy, else a short reason string."""
    try:
        with np.load(path, allow_pickle=True) as z:
            missing = [k for k in REQUIRED if k not in z.files]
            if missing:
                return "missing keys: " + ",".join(missing)
            # force a real read of the compressed payload, not just the header
            if not np.isfinite(np.asarray(z["vs_median"], float)).any():
                return "vs_median all non-finite"
    except Exception as e:                                            # noqa: BLE001
        return f"{type(e).__name__}: {e}"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", help="runs/ tree (scanned recursively for cells/*.npz)")
    ap.add_argument("--delete", action="store_true",
                    help="delete the bad files so a resubmit regenerates those cells")
    args = ap.parse_args()

    tmps = sorted(glob.glob(os.path.join(args.root, "**", "*.tmp*.npz"), recursive=True))
    files = [f for f in sorted(glob.glob(os.path.join(args.root, "**", "cells", "*.npz"),
                                         recursive=True))
             if ".tmp" not in os.path.basename(f)]
    print(f"scanning {len(files)} cell npz under {args.root}", flush=True)

    bad = []
    for i, f in enumerate(files, 1):
        why = check(f)
        if why:
            bad.append((f, why))
            print(f"BAD  {f}\n     {why}", flush=True)
        if i % 2000 == 0:
            print(f"  ...{i}/{len(files)} checked, {len(bad)} bad", flush=True)

    print(f"\n{len(files) - len(bad)}/{len(files)} healthy, {len(bad)} bad, "
          f"{len(tmps)} stray temp files")
    if args.delete:
        for f, _ in bad:
            os.remove(f)
        for f in tmps:
            os.remove(f)
        print(f"deleted {len(bad)} bad + {len(tmps)} temp files; resubmit to regenerate")
    elif bad or tmps:
        print("re-run with --delete to remove them, then resubmit the affected (net,cfg)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
