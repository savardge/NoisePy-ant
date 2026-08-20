"""Period-MATCHED test: at each period, how do violating cells differ in the OVERTONE maps?

The earlier tests were weak for two avoidable reasons. They averaged the overtone maps over all
periods, washing out a signal that lives in a specific band; and their strongest hit was the
overtone/fundamental velocity ratio, which is circular -- a fundamental biased fast by leakage
lowers that ratio by construction, since U_fund appears in both the ratio and in U/c.

Here, at each period separately, cells are split by whether U>c at THAT period, and the two
groups are compared using overtone quantities that do NOT involve U_fund:

  ot_vel     overtone group velocity            (independent of the fundamental)
  ot_present whether the overtone was resolved at all (mask)
  ot_sig     overtone velocity uncertainty
  gap        U_ot - U_fund, reported but FLAGGED as circular

If overtone energy is leaking into the fundamental pick, violating cells should show the
overtone branch sitting closer to the fundamental -- i.e. an anomalously SLOW overtone, or an
overtone that fails to be separately resolved.

Note for Love: these maps are the RAYLEIGH overtone (the campaign has no Love overtone), so for
Love they measure general higher-mode excitation in the area, not the leaking branch itself.

  python uc_overtone_matched.py --net hautesorne --wave love --uc <uc_maps>
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vs_model_figures import E                                       # noqa: E402
from uc_overtone_correlation import PROD                             # noqa: E402


def load_period_maps(net, wave):
    """{rounded period: (vel, mask, unc_s)} for one wave."""
    d = f"{E}/{net}/tomo/1_velocity_maps/1_production/{PROD[net]}/production/{wave}"
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "map_T*.npz"))):
        z = np.load(f)
        out[round(float(z["period"]), 2)] = (z["vel"], z["mask"], z["unc_s"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", required=True)
    ap.add_argument("--wave", required=True, choices=("fund", "love"))
    ap.add_argument("--uc", required=True)
    ap.add_argument("--min-group", type=int, default=25)
    a = ap.parse_args()

    z = np.load(os.path.join(a.uc, f"uc_{a.net}_{a.wave}.npz"), allow_pickle=True)
    cells = z["cells"]
    T = np.asarray(z["T"], float)
    ratio = np.asarray(z["ratio"], float)

    ot = load_period_maps(a.net, "overtone")
    fu = load_period_maps(a.net, a.wave)
    if not ot:
        raise SystemExit("no overtone maps")

    print(f"{a.net}/{a.wave}: overtone maps at {len(ot)} periods "
          f"({min(ot):.2f}-{max(ot):.2f} s)")
    print("  For each period: cells WITH U>c vs cells WITHOUT, compared on the overtone maps.")
    print(f"\n{'T[s]':>6}{'%viol':>7}{'ot_vel viol':>13}{'ot_vel ok':>11}{'diff':>8}"
          f"{'ot_present v/ok':>18}{'gap viol':>10}{'gap ok':>8}")
    rows = []
    for k, t in enumerate(T):
        tt = round(float(t), 2)
        if tt not in ot:
            continue
        r = ratio[:, k]
        viol = r > 1
        ok = np.isfinite(r) & ~viol
        if viol.sum() < a.min_group or ok.sum() < a.min_group:
            continue
        ov, om, os_ = ot[tt]
        fv, fm, _ = fu.get(tt, (None, None, None))
        vals = {"ov": [], "om": [], "gap": []}
        for grp, sel in (("v", viol), ("o", ok)):
            vv, mm, gg = [], [], []
            for i, (ix, iy) in enumerate(cells):
                if not sel[i]:
                    continue
                ix, iy = int(ix), int(iy)
                if ix >= om.shape[0] or iy >= om.shape[1]:
                    continue
                present = bool(om[ix, iy])
                mm.append(present)
                if present:
                    vv.append(float(ov[ix, iy]))
                    if fv is not None and bool(fm[ix, iy]):
                        gg.append(float(ov[ix, iy]) - float(fv[ix, iy]))
            vals["ov"].append(np.array(vv)); vals["om"].append(np.array(mm))
            vals["gap"].append(np.array(gg))
        ovv, ovo = vals["ov"]; omv, omo = vals["om"]; gv, go = vals["gap"]
        if len(ovv) < 10 or len(ovo) < 10:
            continue
        d = np.median(ovv) - np.median(ovo)
        rows.append((tt, 100 * viol.mean(), np.median(ovv), np.median(ovo), d,
                     omv.mean(), omo.mean(),
                     np.median(gv) if len(gv) else np.nan,
                     np.median(go) if len(go) else np.nan))
        print(f"{tt:>6.2f}{100*viol.mean():>6.0f}%{np.median(ovv):>13.3f}{np.median(ovo):>11.3f}"
              f"{d:>+8.3f}{omv.mean():>10.2f}/{omo.mean():<7.2f}"
              f"{(np.median(gv) if len(gv) else np.nan):>10.3f}"
              f"{(np.median(go) if len(go) else np.nan):>8.3f}")
    if not rows:
        print("  (no period had both groups populated -- violation is not spatially split here)")
        return
    A = np.array([r[4] for r in rows])
    P = np.array([r[5] - r[6] for r in rows])
    print(f"\n  overtone velocity, violating minus non-violating: "
          f"median {np.median(A):+.3f} km/s over {len(A)} periods")
    print(f"  overtone DETECTION rate, violating minus non-violating: {np.median(P):+.3f}")
    print("  (leakage predicts a SLOWER overtone and/or LOWER detection in violating cells)")
    print("  gap columns use U_fund and are therefore circular -- context only")


if __name__ == "__main__":
    main()
