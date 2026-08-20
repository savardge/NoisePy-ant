#!/usr/bin/env python3
"""Redraw the F-J picks against forward modes from BOTH reference models -- the slant-stack R0
inversion (old) and the F-J R0 inversion (new) -- so the change the re-pick makes to the R1
prediction and to the trapped-mode ceiling is visible directly.

Reference model = the fundamental-only BayHunter posterior median (R0p) for each network,
resampled to 0.25 km layers, Vp/Vs = 1.73, rho by Brocher. "old" reads
tests/test_2026-08-16_vsg_reference{_vmax5}/R0p (fitted to ref_fundamental_phase.txt, the
slant-stack pick); "new" reads tests/test_2026-08-16_vsg_reference_vmax5_FJ/R0p (fitted to
ref_fundamental_phase_FJ.txt, the F-J topology pick).

Usage:
  python vsg_fj_modes_redraw.py --nets aargau,hautesorne,riehen
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from disba import PhaseDispersion

P = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
OLD_TAGS = ["test_2026-08-16_vsg_reference_vmax5", "test_2026-08-16_vsg_reference"]
NEW_TAG = "test_2026-08-16_vsg_reference_vmax5_FJ"


def layered(net, tags):
    for t in tags:
        f = f"{P}/{net}/tomo/2_vs_depth_inversion/tests/{t}/R0p/bayhunter_result.npz"
        if os.path.exists(f):
            z = np.load(f, allow_pickle=True); d, v = z["depth"], z["vs_median"]
            e = np.arange(0, d.max() + .25, .25); mid = .5 * (e[:-1] + e[1:])
            vs = np.interp(mid, d, v); th = np.full(len(mid), .25); th[-1] = 100.
            return th, vs, t
    return None, None, None


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets", default="aargau,hautesorne,riehen")
    a = ap.parse_args()
    nets = [n for n in a.nets.split(",") if n]
    fig, axs = plt.subplots(1, len(nets), figsize=(6.4 * len(nets), 7.2), squeeze=False)
    Fg = np.geomspace(0.18, 2.5, 140)
    rows = []
    for ax, net in zip(axs[0], nets):
        z = np.load(f"{P}/{net}/vsg_modesep/vsg_fj_ZZ_sign+1.npz"); F, c, A = z["f"], z["vel"], z["FJ"]
        ax.pcolormesh(F, c, np.clip(A / np.percentile(A, 99.5), 0, 1), cmap="gray_r", shading="auto")
        D = pd.read_csv(f"{P}/{net}/vsg_modesep/fj_picks_ZZ.csv")
        cols = dict(zip(sorted(D.branch.unique()), plt.cm.tab10(np.linspace(0, .9, 10))))
        for b, g in D.groupby("branch"):
            ax.scatter(g.freq, g.c, s=8, color=cols[b], edgecolor="k", lw=.25, label=f"pick {b}", zorder=5)
        for lab, tags, ls, lw in (("old (slant-stack R0 model)", OLD_TAGS, "--", 1.4),
                                  ("NEW (F-J R0 model)", [NEW_TAG], "-", 2.2)):
            th, vs, tag = layered(net, tags)
            if th is None:
                continue
            for m, (ff, cc) in modes(th, vs, Fg).items():
                ax.plot(ff, cc, ls, color="yellow" if m == 0 else "lime", lw=lw if m <= 1 else lw * .6,
                        label=f"R{m} {lab}" if m <= 1 else None, zorder=6)
            ax.axhline(vs[-1], color="red", ls=ls, lw=1.6, zorder=6,
                       label=f"ceiling {vs[-1]:.2f} {lab.split(' ')[0]}")
            rows.append(dict(net=net, model=lab.split(" ")[0], tag=tag, vs_hs=round(float(vs[-1]), 2),
                             vs_1km=round(float(np.interp(1.0, .5 * (np.arange(len(vs)) * .25 + (np.arange(len(vs)) + 1) * .25), vs)), 2)))
        ax.set_xscale("log"); ax.set_xlim(0.18, 2.5); ax.set_ylim(0.5, 5); ax.grid(alpha=.25)
        ax.set_xlabel("frequency [Hz]"); ax.set_ylabel("phase velocity [km/s]")
        ax.set_title(f"{net}: F-J picks vs modes — old vs NEW reference model", fontsize=10.5)
        ax.legend(fontsize=6.8, loc="upper right", ncol=1)
    fig.tight_layout()
    out = f"{P}/_fj_picks_vs_modes_oldnew.png"; fig.savefig(out, dpi=140, bbox_inches="tight")
    print(pd.DataFrame(rows).to_string(index=False)); print("wrote", out)


if __name__ == "__main__":
    main()
