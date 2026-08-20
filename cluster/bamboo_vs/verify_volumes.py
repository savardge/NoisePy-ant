"""Acceptance check for the assembled vs_prod3 volumes.

Every defect this campaign hit produced plausible output rather than an error -- a volume with
no anisotropy arrays, an all-NaN misfit column, posteriors built from one chain. So check the
deliverable itself: completeness against the expected cell count, finite fractions of each
array, and the config-specific keys that must be populated.

  python verify_volumes.py [runs_dir]

Exit code 1 if any volume fails a check.
"""
import glob
import os
import sys

import numpy as np

from progress import EXPECTED   # same dir; single source of truth for cell counts

WAVESET = {"R0g": "fund", "R0p": "fund", "L0g": "love", "L0p": "love",
           "RLg_radial": "fundlove", "RLg_iso": "fundlove"}
# which chi_ column must carry numbers, per config
CHI = {"R0g": "chi_fund", "R0p": "chi_fund", "L0g": "chi_love", "L0p": "chi_love",
       "RLg_radial": "chi_fund", "RLg_iso": "chi_fund"}
ANISO = ("gamma_median", "gamma_p_pos", "zeta_median")


def check_one(path, net, cfg):
    problems = []
    exp = EXPECTED[(net, cfg)]
    with np.load(path, allow_pickle=True) as z:
        n = int(z["cells"].shape[0])
        if n < exp:
            problems.append(f"{exp - n} cells missing ({n}/{exp})")
        for k in ("vs_median", "vs_p16", "vs_p84", "lonlat", "depth"):
            if k not in z.files:
                problems.append(f"missing {k}")
                continue
            f = float(np.isfinite(z[k]).mean())
            if f < 0.999:
                problems.append(f"{k} only {f:.3f} finite")
        chi = CHI[cfg]
        if chi not in z.files:
            problems.append(f"missing {chi}")
        else:
            f = float(np.isfinite(z[chi]).mean())
            if f < 0.5:
                problems.append(f"{chi} only {f:.3f} finite")
        for k in ANISO:
            if k not in z.files:
                problems.append(f"missing {k}")
        if cfg == "RLg_radial" and "gamma_median" in z.files:
            g = z["gamma_median"]
            if not np.isfinite(g).any():
                problems.append("radial run but gamma all non-finite")
        vs = z["vs_median"] if "vs_median" in z.files else np.array([np.nan])
        info = (f"n={n}/{exp} vs={np.nanmin(vs):.2f}-{np.nanmax(vs):.2f}km/s")
    return problems, info


def main():
    runs = sys.argv[1] if len(sys.argv) > 1 else \
        "/srv/beegfs/scratch/users/s/savardg/vs_prod3/runs"
    vols = sorted(glob.glob(os.path.join(runs, "*", "*", "volume_*.npz")))
    if not vols:
        print("no volumes yet")
        return 0
    nfail = 0
    print(f"{len(vols)}/18 volumes present\n")
    for v in vols:
        cfg = os.path.basename(os.path.dirname(v))
        net = os.path.basename(os.path.dirname(os.path.dirname(v)))
        if (net, cfg) not in EXPECTED:
            print(f"?? {v} (unknown config)")
            continue
        probs, info = check_one(v, net, cfg)
        if probs:
            nfail += 1
            print(f"FAIL {net}/{cfg:<12} {info}")
            for p in probs:
                print(f"       - {p}")
        else:
            print(f"ok   {net}/{cfg:<12} {info}")
    print(f"\n{len(vols) - nfail} ok, {nfail} with problems")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
