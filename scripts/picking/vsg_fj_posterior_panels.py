#!/usr/bin/env python3
"""Per-network pair of panels for the F-J fundamental reference inversion:

  left  -- F-J dispersion image with the topology picks, the fitted fundamental (posterior median
           R0 fit and its 16-84 % predicted band from the stored ensemble), and R1-R3 forward-
           modelled from the median model; horizontal line = trapped-mode ceiling Vs(half-space).
  right -- the Vs posterior itself, drawn as a 2-D histogram of the stored ensemble
           (number of models per depth x Vs cell), median in black -- the usual
           "number of models" density figure.

One row per (network, run tag). Riehen may carry two rows: R0 alone (`_FJ`) and R0 with the B3
branch joined on as the fundamental (`_FJB3`).

Usage:
  python vsg_fj_posterior_panels.py --nets aargau,hautesorne,riehen
  python vsg_fj_posterior_panels.py --nets riehen --tags _FJ,_FJB3
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from disba import PhaseDispersion

P = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
BASE = "test_2026-08-16_vsg_reference_vmax5"


def result(net, suffix):
    f = f"{P}/{net}/tomo/2_vs_depth_inversion/tests/{BASE}{suffix}/R0p/bayhunter_result.npz"
    return np.load(f, allow_pickle=True) if os.path.exists(f) else None


def layered(d, v):
    e = np.arange(0, d.max() + .25, .25); mid = .5 * (e[:-1] + e[1:])
    vs = np.interp(mid, d, v); th = np.full(len(mid), .25); th[-1] = 100.
    return th, vs


def modes(th, vs, F, nm=4):
    vp = 1.73 * vs; rho = .32 * vp + .77; T = 1 / F
    out = {}
    for m in range(nm):
        try:
            dd = PhaseDispersion(th, vp, vs, rho)(T[::-1], mode=m, wave="rayleigh")
            if len(dd.velocity): out[m] = (1 / dd.period, dd.velocity)
        except Exception:
            pass
    return out


def draw_row(axL, axR, net, suffix, Fg):
    r = result(net, suffix)
    if r is None:
        for ax in (axL, axR): ax.text(.5, .5, f"{net}{suffix}: no result", ha="center", transform=ax.transAxes)
        return None
    # ---------------- left: F-J image + picks + fit ----------------
    z = np.load(f"{P}/{net}/vsg_modesep/vsg_fj_ZZ_sign+1.npz"); F, c, A = z["f"], z["vel"], z["FJ"]
    axL.pcolormesh(F, c, np.clip(A / np.percentile(A, 99.5), 0, 1), cmap="gray_r", shading="auto")
    D = pd.read_csv(f"{P}/{net}/vsg_modesep/fj_picks_ZZ.csv")
    cols = dict(zip(sorted(D.branch.unique()), plt.cm.tab10(np.linspace(0, .9, 10))))
    for b, g in D.groupby("branch"):
        axL.scatter(g.freq, g.c, s=7, color=cols[b], edgecolor="k", lw=.25, label=f"pick {b}", zorder=5)
    # observed curve actually inverted + predicted band from the ensemble
    T = r["obsT_fund_phase"]; obs = r["obs_fund_phase"]; pred = r["pred_fund_phase"]
    fo = 1 / T; o = np.argsort(fo)
    axL.errorbar(fo[o], obs[o], yerr=r["obssig_fund_phase"][o], fmt="s", ms=4, color="w", mec="k",
                 ecolor="k", elinewidth=.8, capsize=2, zorder=7, label="inverted curve ± σ")
    lo, md, hi = np.percentile(pred, [16, 50, 84], axis=0)
    axL.fill_between(fo[o], lo[o], hi[o], color="yellow", alpha=.45, zorder=6, label="R0 fit 16–84 %")
    axL.plot(fo[o], md[o], "-", color="gold", lw=2.2, zorder=7, label="R0 fit median")
    # forward modes from the median model
    th, vs = layered(r["depth"], r["vs_median"])
    for m, (ff, cc) in modes(th, vs, Fg).items():
        if m == 0:
            axL.plot(ff, cc, "--", color="gold", lw=1.2, zorder=6, label="R0 median model")
        else:
            axL.plot(ff, cc, "-", color="lime", lw=2.0 if m == 1 else 1.1, zorder=6, label=f"R{m} median model")
    axL.axhline(vs[-1], color="red", lw=1.6, zorder=6, label=f"ceiling Vs_hs {vs[-1]:.2f}")
    axL.set_xscale("log"); axL.set_xlim(0.18, 2.5); axL.set_ylim(0.5, 5); axL.grid(alpha=.25)
    axL.set_xlabel("frequency [Hz]"); axL.set_ylabel("phase velocity [km/s]")
    axL.set_title(f"{net} — R0p{suffix}: fit + forward modes  "
                  f"(chain_disagree {float(r['chain_disagree']):.2f}, chains ok {100*float(r['frac_chains_ok']):.0f} %)",
                  fontsize=9.5)
    axL.legend(fontsize=6.4, loc="upper right")
    # ---------------- right: Vs posterior density ----------------
    ens = r["ens_vs"]; d = r["depth"]
    ve = np.arange(0.5, 5.0 + 1e-9, 0.02)                    # Vs bins
    H = np.zeros((len(d), len(ve) - 1))
    for i in range(len(d)):
        H[i] = np.histogram(ens[:, i], bins=ve)[0]
    cmap = plt.cm.jet.copy(); cmap.set_under("white")
    pm = axR.pcolormesh(ve, np.r_[d, d[-1] + (d[-1] - d[-2])], H, cmap=cmap, vmin=0.5, shading="flat")
    axR.plot(r["vs_median"], d, "k-", lw=1.4, label="median")
    axR.plot(r["vs_p16"], d, "k--", lw=.7); axR.plot(r["vs_p84"], d, "k--", lw=.7, label="16–84 %")
    zr = float(r["z_reliable_max"])
    axR.axhline(zr, color="magenta", ls=":", lw=1.2, label=f"z_reliable_max {zr:.1f} km")
    axR.invert_yaxis(); axR.set_xlim(0.5, 5.0); axR.set_ylim(d.max(), 0)
    axR.set_xlabel("Vs [km/s]"); axR.set_ylabel("depth [km]"); axR.grid(alpha=.2)
    cb = plt.colorbar(pm, ax=axR, pad=.02); cb.set_label("number of models")
    axR.set_title(f"{net} — Vs posterior ({ens.shape[0]:,} models, {int(r['n_chains_kept'])}/{len(r['chains_kept'])} chains kept)",
                  fontsize=9.5)
    axR.legend(fontsize=7, loc="lower left")
    return dict(net=net, run=f"R0p{suffix}", n_pts=len(T), f_range=f"{fo.min():.2f}-{fo.max():.2f}",
                vs_hs=round(float(vs[-1]), 2), vs_hs_p16=round(float(r["vs_p16"][-1]), 2),
                vs_hs_p84=round(float(r["vs_p84"][-1]), 2), chain_disagree=round(float(r["chain_disagree"]), 3),
                frac_ok=round(float(r["frac_chains_ok"]), 2), z_rel=round(zr, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets", default="aargau,hautesorne,riehen")
    ap.add_argument("--tags", default="_FJ,_FJB3", help="run suffixes to look for per network")
    ap.add_argument("--out", default=f"{P}/_fj_fit_and_posterior.png")
    a = ap.parse_args()
    rows = [(n, s) for n in a.nets.split(",") for s in a.tags.split(",") if result(n, s) is not None]
    fig, axs = plt.subplots(len(rows), 2, figsize=(14, 5.6 * len(rows)), squeeze=False,
                            gridspec_kw=dict(width_ratios=[1.35, 1]))
    Fg = np.geomspace(0.18, 2.5, 140)
    tab = [draw_row(axL, axR, n, s, Fg) for (axL, axR), (n, s) in zip(axs, rows)]
    fig.tight_layout(); fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print(pd.DataFrame([t for t in tab if t]).to_string(index=False)); print("wrote", a.out)


if __name__ == "__main__":
    main()
