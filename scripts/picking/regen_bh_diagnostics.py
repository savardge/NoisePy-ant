"""Retroactively build convergence-diagnostic plots for a BayHunter run whose raw per-chain
files are still on disk under <outdir>/bh_results/data (SingleChain.save_finalmodels() output
is not cleaned up after a run). Useful for runs completed before noisepy.bh_diagnostics existed.

Usage (bayhunter env, needs PYTHONPATH=~/Codes/Noisepy-ant):
  python regen_bh_diagnostics.py <outdir> [depth_max] [title]
"""
import os
import sys

import numpy as np
from BayHunter import PlotFromStorage
from noisepy.bh_diagnostics import build_diagnostics, plot_diagnostics


def main():
    outdir = sys.argv[1]
    depth_max = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
    title = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(outdir.rstrip("/"))
    savepath = os.path.join(outdir, "bh_results")
    obj = PlotFromStorage(os.path.join(savepath, "data", "cell_config.pkl"))
    dep = np.linspace(0, depth_max, 121)
    diag = build_diagnostics(os.path.join(savepath, "data"), dep, obj.mantle)
    summ = plot_diagnostics(diag, os.path.join(outdir, "bayhunter_diagnostics.png"),
                            title=f"convergence diagnostics -- {title}")
    print(outdir, summ)


if __name__ == "__main__":
    main()
