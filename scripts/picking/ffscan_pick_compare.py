"""ffscan: side-by-side pick-distribution comparison across the r/lambda thresholds.

One figure per (network, measure): rows = wave types (fund / overtone / love [/ love_ot]),
columns = the five thresholds. Each panel is the period x velocity density of the picks the
arm's PER-PICK cut keeps (distance >= X * v_row * T, recomputed from the base pool — bit-
identical to ffscan_filter_picks.py), drawn over the base pool in grey so the removed picks
stay visible. Period bins = the CWT scale grid (ffscan_common), velocity bins and color
scale shared across a row. Kept % annotated per panel.

Output: ffscan_logs/pick_comparison/{net}_{measure}.png

Usage: python ffscan_pick_compare.py [--nets riehen,aargau,hautesorne] [--measures group,phase]
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ffscan_common import scale_bin_edges

EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
OUTD = os.path.normpath(os.path.join(EHM, "..", "ffscan_logs", "pick_comparison"))
THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0]
WAVES = {"riehen": ("fund", "overtone", "love"),
         "aargau": ("fund", "overtone", "love"),
         "hautesorne": ("fund", "overtone", "love", "love_ot")}
TITLES = {"fund": "Rayleigh fund", "overtone": "Rayleigh overtone",
          "love": "Love fund", "love_ot": "Love overtone"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nets", default="riehen,aargau,hautesorne")
    ap.add_argument("--measures", default="group,phase")
    args = ap.parse_args()
    os.makedirs(OUTD, exist_ok=True)

    for net in args.nets.split(","):
        for measure in args.measures.split(","):
            tag = "_nf_phase" if measure == "phase" else "_nf"
            waves = WAVES[net]
            fig, axes = plt.subplots(
                len(waves), len(THRESHOLDS),
                figsize=(3.1 * len(THRESHOLDS) + 1.2, 2.6 * len(waves) + 0.8),
                sharex=True, squeeze=False)
            for iw, wave in enumerate(waves):
                fn = f"{EHM}/{net}/tomo/1_velocity_maps/inputs/ffscan/picks_{wave}_uni{tag}.csv"
                if not os.path.exists(fn):
                    for ax in axes[iw]:
                        ax.set_axis_off()
                    continue
                b = pd.read_csv(fn, usecols=["inst_period", "group_velocity", "distance"])
                T, V, R = (b.inst_period.values, b.group_velocity.values,
                           b.distance.values)
                rml = R / (V * T)                     # per-pick r/lambda, as in the filter
                tb = scale_bin_edges(net, T.min(), T.max())
                yb = np.histogram_bin_edges(V, bins=60)
                hb, _, _ = np.histogram2d(T, V, bins=[tb, yb])
                hb[hb == 0] = np.nan
                # linear count scale, saturated at the 99th percentile of occupied
                # cells so a few very dense short-period cells do not flatten the rest
                vmax = float(np.nanpercentile(hb, 99))
                for ix, X in enumerate(THRESHOLDS):
                    ax = axes[iw][ix]
                    keep = rml >= X
                    h, _, _ = np.histogram2d(T[keep], V[keep], bins=[tb, yb])
                    h[h == 0] = np.nan
                    ax.pcolormesh(tb, yb, hb.T, vmin=0, vmax=vmax,
                                  cmap="Greys", alpha=0.55)
                    im = ax.pcolormesh(tb, yb, h.T, vmin=0, vmax=vmax, cmap="viridis")
                    ax.set_title(f"r/λ ≥ {X:g}   ({100 * keep.mean():.0f}% kept)",
                                 fontsize=8.5)
                    if ix == 0:
                        ax.set_ylabel(f"{TITLES[wave]}\n{measure} vel. (km/s)",
                                      fontsize=8.5)
                    else:
                        ax.tick_params(labelleft=False)
                    if iw == len(waves) - 1:
                        ax.set_xlabel("period (s)", fontsize=8.5)
                fig.colorbar(im, ax=axes[iw].tolist(), shrink=0.9, pad=0.01,
                             extend="max", label="picks/cell")
            fig.suptitle(f"{net} — {measure}: per-pick far-field cut d ≥ X·v·T "
                         f"(grey = removed; CWT-scale period bins)", fontsize=11)
            out = os.path.join(OUTD, f"{net}_{measure}.png")
            fig.savefig(out, dpi=130, bbox_inches="tight")
            plt.close(fig)
            print(out)


if __name__ == "__main__":
    main()
