#!/usr/bin/env python
"""Was a vs_max prior bound censoring the posterior, and at what depth does it stop mattering?

A posterior pressed against its prior ceiling is censored: the median is dragged down and the
upper credible interval means nothing. But a bound that only binds BELOW some depth is not a
prior problem at all -- it is the data ceasing to constrain, and raising the cap only inflates
those values. Reporting the depth profile of railing separates the two.
"""
import argparse, os
import numpy as np

E = "/Users/genevievesavard/Codes/extract_higher_modes"
DEPTHS = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


def rails(vol, cap, tol=0.05):
    med = vol["vs_median"]; z = vol["depth"]
    z0 = vol["z_reliable_min"][:, None]; z1 = vol["z_reliable_max"][:, None]
    m = np.isfinite(med) & (z[None, :] >= z0) & (z[None, :] <= z1)
    return med, z, m, med > cap - tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--base", required=True, help="e.g. L0p")
    ap.add_argument("--raised", required=True, help="e.g. L0p_vmax4.5")
    ap.add_argument("--volume", default="volume_love.npz")
    ap.add_argument("--cap-base", type=float, default=3.6)
    ap.add_argument("--cap-raised", type=float, default=4.5)
    a = ap.parse_args()

    B = f"{E}/Projects/{a.net}/tomo/2_vs_depth_inversion/vs_prod3"
    va = np.load(f"{B}/{a.base}/{a.volume}", allow_pickle=True)
    vb = np.load(f"{B}/{a.raised}/{a.volume}", allow_pickle=True)
    ka = {tuple(c): i for i, c in enumerate(va["cells"])}
    kb = {tuple(c): i for i, c in enumerate(vb["cells"])}
    com = sorted(set(ka) & set(kb))
    ia = np.array([ka[c] for c in com]); ib = np.array([kb[c] for c in com])
    print(f"=== {a.net}: {a.raised} vs {a.base}  ({len(com)} paired cells) ===")

    print(f"\n{'depth':>7s} {'%@bound '+str(a.cap_base):>14s} {'%@bound '+str(a.cap_raised):>14s} "
          f"{'medVs raised':>13s}")
    for nm, v, idx, cap in ((a.base, va, ia, a.cap_base), (a.raised, vb, ib, a.cap_raised)):
        pass
    meda, z, ma, ra = rails(va, a.cap_base)
    medb, _, mb, rb = rails(vb, a.cap_raised)
    for zt in DEPTHS:
        k = int(np.argmin(np.abs(z - zt)))
        sa, sb = ma[ia, k], mb[ib, k]
        if sa.sum() < 20 or sb.sum() < 20:
            print(f"{zt:6.1f} km   (too few resolved)"); continue
        print(f"{zt:6.1f} km {100*np.mean(ra[ia,k][sa]):13.1f}% {100*np.mean(rb[ib,k][sb]):13.1f}% "
              f"{np.median(medb[ib,k][sb]):13.2f}")

    d = medb[ib] - meda[ia]
    old_rail = np.array([(va["vs_p975"][i] > a.cap_base - 0.05).any() for i in ia])
    mx = np.nanmax(np.abs(d), axis=1)
    print(f"\ncells railing at {a.cap_base}: {old_rail.sum()}/{len(com)} "
          f"({100*old_rail.mean():.1f}%)")
    if old_rail.any() and (~old_rail).any():
        print(f"   median max|dVs| where railed  {np.nanmedian(mx[old_rail]):.3f} km/s")
        print(f"   median max|dVs| where not     {np.nanmedian(mx[~old_rail]):.3f} km/s")
    print("\nA bound that binds only at depth is missing data constraint, not a tight prior.")


main()
