"""Per-combo panels: Dinver (SWinvert pooled) vs BayHunter vs Michel log at the Riehen wells.

One figure per well, one panel per waveset combo, both engines overlaid -- the readable
counterpart of dinver_well_compare.py's all-in-one plot. Reads the same tests/dinver_swinvert
tree. Usage: python dinver_well_matrix_figure.py
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from noisepy import vs_inversion as vi
from dinver_well_compare import ROOT as TESTS, WELLS, michel

COMBOS = ["R0g", "R0p", "L0g", "L0p", "R0gR0p", "L0gL0p", "R0gL0g", "R0pL0p", "R0L0all",
          "R0gR1g", "R0pR1p", "R0R1"]
COL = {"dinver": "C0", "bayhunter": "C1"}


def main():
    zl, vl = michel()
    for well, ((ix, iy), zbas) in WELLS.items():
        d = os.path.join(TESTS, f"{well}_{ix}_{iy}")
        fig, axs = plt.subplots(3, 4, figsize=(16, 15), sharex=True, sharey=True)
        for ax, tag in zip(axs.ravel(), COMBOS):
            ax.step(vl, zl, where="post", color="k", lw=2, label="Michel log")
            ax.axhline(zbas, color="0.4", ls="--", lw=1)
            for eng in ("dinver", "bayhunter"):
                fs = glob.glob(os.path.join(d, f"{eng}_*_{tag}_result.npz"))
                if not fs:
                    continue
                r = vi.load_result(fs[0])
                z = r["depth"]
                ax.fill_betweenx(z, r["vs_p16"], r["vs_p84"], color=COL[eng], alpha=0.18)
                chi = vi.data_misfit(r)
                lab = "%s  χ=%s" % (eng, "/".join("%.1f" % v for v in chi.values()))
                ax.plot(r["vs_median"], z, color=COL[eng], lw=1.8, label=lab)
            ax.set_title(tag)
            ax.legend(fontsize=7, loc="lower left")
            ax.grid(alpha=0.3)
        axs[0, 0].set_ylim(5, 0); axs[0, 0].set_xlim(0.5, 4.3)
        for a in axs[-1]:
            a.set_xlabel("Vs [km/s]")
        for a in axs[:, 0]:
            a.set_ylabel("depth [km]")
        fig.suptitle(f"{well} cell ({ix},{iy}) — Dinver (SWinvert pooled) vs BayHunter per "
                     f"waveset; band = p16–p84; χ per target curve", y=0.995)
        fig.tight_layout()
        out = os.path.join(TESTS, f"well_matrix_{well}.png")
        fig.savefig(out, dpi=110)
        print("wrote", out)


if __name__ == "__main__":
    main()
