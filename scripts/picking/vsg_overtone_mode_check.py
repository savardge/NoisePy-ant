#!/usr/bin/env python3
"""Is the VSG "overtone" reference ridge actually a Rayleigh higher mode?

A trapped Rayleigh mode of a layered half-space requires c < Vs(halfspace). That bound is set
by Vs; Vp enters only at second order, so a high Vp/Vs CANNOT lift it (verified for Riehen:
pushing Vp/Vs from 1.60 to 3.00, i.e. halfspace Vp 5.4 -> 10.1 km/s, moved R1 at 2 s by
0.10 km/s). If the picked ridge sits above every mode of a model that fits the FUNDAMENTAL,
it is not a Rayleigh overtone at all -- most plausibly body-wave energy (the Haute-Sorne
manuscript's own beamforming reports non-dispersive ~4.0 km/s energy it attributes to P).

For each network this forward-computes R0..R3 from the network-average Vs obtained by
inverting the FUNDAMENTAL reference alone (tests/<tag>/R0p), and compares with the picked
overtone ridge. Using the fundamental-only model is deliberate: it is the model the data
demand without ever having seen the overtone, so the test cannot be circular.

Usage:
  python vsg_overtone_mode_check.py --nets riehen,hautesorne,aargau
"""
import argparse, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from disba import PhaseDispersion

P = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
TAGS = ["test_2026-08-16_vsg_reference_vmax5", "test_2026-08-16_vsg_reference"]


def model_from(net):
    for t in TAGS:
        f = f"{P}/{net}/tomo/2_vs_depth_inversion/tests/{t}/R0p/bayhunter_result.npz"
        if os.path.exists(f):
            z = np.load(f, allow_pickle=True)
            d, v = z["depth"], z["vs_median"]
            e = np.arange(0, d.max() + 0.25, 0.25); mid = 0.5 * (e[:-1] + e[1:])
            vs = np.interp(mid, d, v); th = np.full(len(mid), 0.25); th[-1] = 100.0
            return th, vs, t
    return None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets", default="riehen,hautesorne,aargau")
    a = ap.parse_args()
    nets = [n.strip() for n in a.nets.split(",") if n.strip()]
    fig, axs = plt.subplots(1, len(nets), figsize=(6.0 * len(nets), 6.6), squeeze=False)
    print(f"{'net':<12}{'Vs_hs':>8}{'max c any R mode':>18}{'ridge c range':>16}{'ridge above bound':>19}")
    for i, net in enumerate(nets):
        th, vs, tag = model_from(net)
        if th is None:
            print(f"{net:<12}  no R0p inversion found"); continue
        vp = 1.73 * vs; rho = 0.32 * vp + 0.77
        o = np.loadtxt(f"{P}/{net}/vsg_modesep/ref_overtone_phase.txt")
        f0 = np.loadtxt(f"{P}/{net}/vsg_modesep/ref_fundamental_phase.txt")
        Tg = np.linspace(o[:, 0].min(), o[:, 0].max(), 24)
        ax = axs[0][i]
        best = 0.0
        for m, col in zip(range(4), ("tab:blue", "tab:orange", "tab:green", "tab:red")):
            try:
                d = PhaseDispersion(th, vp, vs, rho)(Tg, mode=m, wave="rayleigh")
                if len(d.velocity):
                    ax.plot(d.period, d.velocity, "-", color=col, lw=1.8, label=f"R{m} (model)")
                    best = max(best, float(np.nanmax(d.velocity)))
            except Exception:
                pass
        ax.plot(f0[:, 0], f0[:, 1], "k--", lw=1.4, label="picked R0 (VSG)")
        ax.plot(o[:, 0], o[:, 1], "k-", lw=2.4, label="picked 'overtone' (VSG)")
        ax.axhline(vs[-1], color="0.4", ls=":", lw=1.6,
                   label=f"Vs halfspace {vs[-1]:.2f} (mode ceiling)")
        ax.set_xlabel("period [s]"); ax.set_ylabel("phase velocity [km/s]")
        ax.set_title(f"{net}", fontsize=11); ax.grid(alpha=0.3); ax.legend(fontsize=7.5)
        ax.set_ylim(1.0, max(5.0, o[:, 1].max() + 0.4))
        frac = 100 * np.mean(np.interp(Tg, o[:, 0], o[:, 1]) > vs[-1])
        print(f"{net:<12}{vs[-1]:>8.2f}{best:>18.2f}"
              f"{f'{o[:,1].min():.2f}-{o[:,1].max():.2f}':>16}{f'{frac:.0f}%':>19}")
    fig.suptitle("Is the VSG 'overtone' ridge a Rayleigh higher mode?\n"
                 "modes forward-computed from the FUNDAMENTAL-ONLY inversion of each network",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    out = f"{P}/_vsg_overtone_mode_check.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
