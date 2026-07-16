"""Three-station closure test for group-velocity picks (Luo et al. 2015, GJI 201, Sec. 4,
adapted from great circles to the local Cartesian LV95 frame).

For station triples a-b-c that are nearly aligned (b close to the straight a-c segment),
a surface wave from a to c should take the same time as a->b plus b->c. Following Luo et al.:
    Vac' = (Dab + Dbc) / (Dab/Vab + Dbc/Vbc)          combined velocity
    Vdif = Vac' - Vac                                  velocity closure residual
    Tdif = (Dab/Vab + Dbc/Vbc) - Dac/Vac               traveltime closure residual
    Tpdif = 100 * Tdif / (Dac/Vac)                     percent traveltime residual
If short legs are unbiased (only noisier), the residuals stay zero-mean in every distance
class. If group velocities on short paths are biased LOW, then Vac' < Vac systematically:
Vdif < 0 / Tpdif > 0, growing as the shortest leg drops below some r/lambda. The r/lambda
where the binned median residual becomes indistinguishable from the all-far control class is
the empirically supported distance cutoff for the picks.

Geometry criteria (paper uses detour <= 0.1 km and azimuth differences < 1 deg on 100s-km
paths; here scaled to array dimensions): detour Dab + Dbc - Dac <= max(0.1 km, 0.5% of Dac).
The long leg a-c must be >= CONTROL_LAMBDA wavelengths so the reference is itself far-field.

Run:  /opt/anaconda3/bin/python closure_test.py --net aargau [--wave fund]
Outputs: Projects/<net>/tomo/closure_test_<net>_<wave>.png + .csv (one row per triple/period).
"""
import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer

PROJROOT = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
DETOUR_KM = 0.1          # max extra path length of a-b-c vs a-c (or 0.5% of Dac if larger)
DETOUR_FRAC = 0.005
CONTROL_LAMBDA = 2.0     # long leg a-c must be at least this many wavelengths
RBINS = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 99.0]     # shortest-leg r/lambda classes
_TR = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)


def load(net, wave):
    st = pd.read_csv(os.path.join(PROJROOT, net, "tomo", "stations.csv"))
    e, n = _TR.transform(st["longitude"].values, st["latitude"].values)
    xy = {sid: (ee / 1e3, nn / 1e3) for sid, ee, nn in zip(st["id"], e, n)}
    p = pd.read_csv(os.path.join(PROJROOT, net, "tomo", f"picks_{wave}.csv"))
    return xy, p


def build_triples(net="aargau", wave="fund"):
    xy, picks = load(net, wave)
    dist = {}

    def D(s1, s2):
        k = (s1, s2) if s1 < s2 else (s2, s1)
        if k not in dist:
            (x1, y1), (x2, y2) = xy[k[0]], xy[k[1]]
            dist[k] = float(np.hypot(x1 - x2, y1 - y2))
        return dist[k]

    rows = []
    for T, sub in picks.groupby("inst_period"):
        lam = float(sub["group_velocity"].median() * T)
        # pair -> vg lookup and station adjacency at this period
        vg = {}
        adj = defaultdict(set)
        for r in sub.itertuples():
            a, c = (r.stasrc, r.starcv) if r.stasrc < r.starcv else (r.starcv, r.stasrc)
            if a not in xy or c not in xy:
                continue
            vg[(a, c)] = r.group_velocity
            adj[a].add(c); adj[c].add(a)
        for (a, c), v_ac in vg.items():
            d_ac = D(a, c)
            if d_ac < CONTROL_LAMBDA * lam:            # long-leg control criterion
                continue
            for b in adj[a] & adj[c]:
                d_ab, d_bc = D(a, b), D(b, c)
                if d_ab >= d_ac or d_bc >= d_ac:       # b must lie between a and c
                    continue
                detour = d_ab + d_bc - d_ac
                if detour > max(DETOUR_KM, DETOUR_FRAC * d_ac):
                    continue
                v_ab = vg[(a, b) if a < b else (b, a)]
                v_bc = vg[(b, c) if b < c else (c, b)]
                t_comb = d_ab / v_ab + d_bc / v_bc
                t_ac = d_ac / v_ac
                v_comb = (d_ab + d_bc) / t_comb
                rows.append(dict(period=T, lam=lam, a=a, b=b, c=c,
                                 d_ac=d_ac, d_ab=d_ab, d_bc=d_bc,
                                 rmin_lam=min(d_ab, d_bc) / lam,
                                 rlong_lam=d_ac / lam,
                                 vdif=v_comb - v_ac,
                                 tpdif=100.0 * (t_comb - t_ac) / t_ac))
    return pd.DataFrame(rows)


def _boot_ci(x, n=2000, q=(2.5, 97.5), rng=np.random.default_rng(7)):
    if len(x) < 3:
        return np.nan, np.nan
    m = np.median(rng.choice(x, size=(n, len(x)), replace=True), axis=1)
    return tuple(np.percentile(m, q))


def report(tri, net, wave):
    out_png = os.path.join(PROJROOT, net, "tomo", f"closure_test_{net}_{wave}.png")
    tri.to_csv(out_png.replace(".png", ".csv"), index=False)
    print(f"\n== {net} {wave}: {len(tri)} triple-period measurements, "
          f"{tri['period'].nunique()} periods, long leg >= {CONTROL_LAMBDA:g} lambda ==")
    labs, meds, los, his, ns = [], [], [], [], []
    print(f"{'min-leg r/lam':>14} {'n':>6} {'med Tpdif%':>10} {'95% CI':>18} {'med Vdif m/s':>12}")
    for lo, hi in zip(RBINS[:-1], RBINS[1:]):
        s = tri[(tri.rmin_lam >= lo) & (tri.rmin_lam < hi)]
        if len(s) < 3:
            continue
        med = float(s["tpdif"].median())
        cl, ch = _boot_ci(s["tpdif"].values)
        labs.append(f"{lo:g}-{hi:g}" if hi < 99 else f">={lo:g}")
        meds.append(med); los.append(cl); his.append(ch); ns.append(len(s))
        print(f"{labs[-1]:>14} {len(s):>6d} {med:>+10.3f} [{cl:+.3f}, {ch:+.3f}]"
              f" {1e3*float(s['vdif'].median()):>+12.1f}")

    fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.2))
    a = axs[0]
    a.axhline(0, color="k", lw=0.8)
    xpos = np.arange(len(labs))
    a.errorbar(xpos, meds, yerr=[np.array(meds) - np.array(los),
                                 np.array(his) - np.array(meds)],
               fmt="o-", color="crimson", capsize=4, lw=1.5)
    for x, n_ in zip(xpos, ns):
        a.annotate(f"n={n_}", (x, a.get_ylim()[0]), xytext=(0, 6),
                   textcoords="offset points", ha="center", fontsize=8)
    a.set_xticks(xpos, labs)
    a.set_xlabel("shortest leg [wavelengths]")
    a.set_ylabel("median traveltime closure Tpdif [%]")
    a.set_title("closure residual vs shortest-leg distance\n(median, bootstrap 95% CI)")
    a = axs[1]
    a.axhline(0, color="k", lw=0.8)
    tmin = np.floor(tri["period"].min() * 2) / 2
    tmax = np.ceil(tri["period"].max() * 2) / 2
    tedges = np.arange(tmin, tmax + 0.25, 0.5)
    cmap = plt.get_cmap("plasma")
    norm = plt.Normalize(tedges[0], tedges[-1])
    for tlo, thi in zip(tedges[:-1], tedges[1:]):
        m_, x_ = [], []
        for lo, hi in zip(RBINS[:-1], RBINS[1:]):
            s = tri[(tri.rmin_lam >= lo) & (tri.rmin_lam < hi) &
                    (tri.period >= tlo) & (tri.period < thi)]
            if len(s) >= 15:
                x_.append(float(s["rmin_lam"].median())); m_.append(s["tpdif"].median())
        if len(x_) >= 2:
            a.plot(x_, m_, "o-", ms=4, lw=1.4, color=cmap(norm(0.5 * (tlo + thi))))
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=a,
                 label="period band centre [s] (0.5 s sub-bands)")
    a.set_xlabel("shortest leg [wavelengths]"); a.set_ylabel("median Tpdif [%]")
    a.set_title("by 0.5 s period sub-band (>=15 triples per point)")
    a = axs[2]
    redges = np.arange(1.0, 4.75, 0.25)
    H = np.full((len(tedges) - 1, len(redges) - 1), np.nan)
    for i, (tlo, thi) in enumerate(zip(tedges[:-1], tedges[1:])):
        s_t = tri[(tri.period >= tlo) & (tri.period < thi)]
        for j, (lo, hi) in enumerate(zip(redges[:-1], redges[1:])):
            s = s_t[(s_t.rmin_lam >= lo) & (s_t.rmin_lam < hi)]
            if len(s) >= 15:
                H[i, j] = s["tpdif"].median()
    vmax = np.nanpercentile(np.abs(H), 98) if np.isfinite(H).any() else 10.0
    pc = a.pcolormesh(redges, tedges, H, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      edgecolors="0.85", linewidth=0.3)
    fig.colorbar(pc, ax=a, label="median Tpdif [%]")
    for x in (1.5, 2.0, 2.5, 3.0):
        a.axvline(x, color="k", ls=":", lw=1.0)
    a.set_xlabel("shortest leg [wavelengths]"); a.set_ylabel("period [s]")
    a.set_title("median Tpdif in (shortest leg, period) cells\n"
                "(red = short legs slow; white/blue = closure holds)")
    fig.suptitle(f"{net.capitalize()} {wave} — three-station closure test "
                 f"(Luo et al. 2015 Sec. 4; positive Tpdif = short legs SLOW)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print("wrote", out_png)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", default="aargau", choices=("aargau", "riehen"))
    ap.add_argument("--wave", default="fund", choices=("fund", "overtone"))
    args = ap.parse_args()
    tri = build_triples(args.net, args.wave)
    if not len(tri):
        print("no aligned triples found — relax DETOUR_KM / CONTROL_LAMBDA")
        return
    report(tri, args.net, args.wave)


if __name__ == "__main__":
    main()
