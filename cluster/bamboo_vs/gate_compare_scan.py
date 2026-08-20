#!/usr/bin/env python
"""Paired per-cell scan of an ungated arm against its mode-gated twin.

Writes one CSV row per cell present in BOTH arms, so every comparison is paired and no
cell-count difference can masquerade as a gate effect. Columns cover the two things the
gate is supposed to move -- how many dispersion samples it removed, and whether the chains
agree better once they are gone.
"""
import argparse, glob, os
import numpy as np

KEYS = ("chain_disagree", "n_chains_used", "n_layers")


def scan(root):
    """cell key -> flat dict of scalars, for every cell npz under <root>/cells."""
    out = {}
    for f in sorted(glob.glob(os.path.join(root, "cells", "*.npz"))):
        key = os.path.basename(f)[:-4]
        try:
            z = np.load(f, allow_pickle=True)
        except Exception:
            continue
        rec = {}
        for k in KEYS:
            if k in z.files:
                v = z[k]
                rec[k] = float(v) if v.ndim == 0 else float(np.nanmean(v))
        # number of dispersion samples actually inverted, per target
        for k in z.files:
            if k.startswith("nper_") or k in ("n_gate_dropped", "n_gate_judged"):
                v = z[k]
                rec[k] = float(v) if np.ndim(v) == 0 else float(np.size(v))
        if "vs_median" in z.files:
            rec["_vs"] = z["vs_median"]
        if "depth" in z.files:
            rec["_z"] = z["depth"]
        out[key] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="ungated arm dir")
    ap.add_argument("--gated", required=True, help="mode-gated arm dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    A, B = scan(a.base), scan(a.gated)
    common = sorted(set(A) & set(B))
    print(f"base {len(A)} cells, gated {len(B)} cells, paired {len(common)}")
    if not common:
        raise SystemExit("no paired cells")

    # Vs difference profile, on the shared depth axis
    z = A[common[0]].get("_z")
    dvs = []
    rows, cols = [], None
    for k in common:
        ra, rb = A[k], B[k]
        r = {"cell": k}
        for f in sorted(set(ra) | set(rb)):
            if f.startswith("_"):
                continue
            r["base_" + f] = ra.get(f, np.nan)
            r["gated_" + f] = rb.get(f, np.nan)
        rows.append(r)
        if cols is None:
            cols = list(r)
        va, vb = ra.get("_vs"), rb.get("_vs")
        if va is not None and vb is not None and va.shape == vb.shape:
            dvs.append(vb - va)

    with open(a.out, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print("wrote", a.out)

    if dvs:
        np.savez_compressed(a.out.replace(".csv", "_dvs.npz"),
                            depth=z, dvs=np.array(dvs), cells=np.array(common))
        print("wrote", a.out.replace(".csv", "_dvs.npz"), np.array(dvs).shape)


main()
