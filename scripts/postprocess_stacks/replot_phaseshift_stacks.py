#!/usr/bin/env python
"""
Re-plot phase-shift stacked dispersion images from saved NPZ files with different x-axes.

Reads the stacks_<comp>.npz written by phaseshift_dispersion.py and regenerates the 2x2
stacking-method grid in four axis variants (no recompute needed):
  - frequency, linear x-axis
  - frequency, log x-axis
  - period, linear x-axis
  - period, log x-axis
capped at a chosen freqmax and over the requested period / frequency ranges.

Example
-------
    python replot_phaseshift_stacks.py ~/Data/riehen/phasevelocity_VSG \
        --freqmax 5 --pmin 0.1 --pmax 10
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from phaseshift_dispersion import plot_stack_grid, save_picks, configure_logging  # noqa: E402

STACK_KEYS = ("linear", "pws", "complex_pws", "root", "coverage")


def replot_one(npz_path, outdir, comp, freqmax, fmin, min_sources, pick_min_score=0.5):
    d = np.load(npz_path, allow_pickle=True)
    f = d["f"]
    vel = d["vel"]
    n_sources = int(d["n_sources"]) if "n_sources" in d else 0

    # Cap the frequency band (high end at freqmax, low end at fmin) by slicing columns.
    keep = (f >= fmin) & (f <= freqmax)
    f = f[keep]
    stacks = {k: np.asarray(d[k])[:, keep] for k in STACK_KEYS}

    # Fit axis limits to the actual data range (no blank padding).
    flim = (f.min(), f.max())
    plim = (1.0 / f.max(), 1.0 / f.min())
    variants = [
        ("freq", "linear", flim, "freqlin", " | linear f"),
        ("freq", "log", flim, "freqlog", " | log f"),
        ("period", "linear", plim, "perlin", " | linear T"),
        ("period", "log", plim, "perlog", " | log T"),
    ]
    picks = None
    for xaxis, xscale, xlim, tag, suffix in variants:
        out = os.path.join(outdir, f"stacked_{comp}_{tag}.png")
        # Picks are identical across axis variants; pass the cache so the (slow)
        # per-column persistence run happens only once.
        picks = plot_stack_grid(out, stacks, f, vel, xaxis, comp, n_sources, min_sources,
                                xscale=xscale, xlim=xlim, title_suffix=suffix,
                                pick_min_score=pick_min_score, picks_cache=picks)
    if picks is not None:
        save_picks(os.path.join(outdir, f"picks_{comp}.npz"), picks, f, vel, comp)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="Dir containing <comp>/stacks_<comp>.npz subfolders")
    p.add_argument("--components", default="ZZ,RZ,ZR,RR,TT",
                   help="Comma list of components to re-plot")
    p.add_argument("--freqmax", type=float, default=5.0, help="Max frequency [Hz]")
    p.add_argument("--fmin", type=float, default=0.0, help="Min frequency [Hz]")
    p.add_argument("--min-sources", type=int, default=3, help="Coverage mask threshold")
    p.add_argument("--pick-min-score", type=float, default=0.5,
                   help="Topology picking: min persistence score (default 0.5)")
    p.add_argument("--no-reference", action="store_true",
                   help="Do not overlay the reference dispersion curve")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    configure_logging(args.log_level)

    if args.no_reference:
        import phaseshift_dispersion
        phaseshift_dispersion._NO_REFERENCE = True

    comps = [c.strip().upper() for c in args.components.split(",") if c.strip()]
    for comp in comps:
        cdir = os.path.join(args.root, comp)
        npz = os.path.join(cdir, f"stacks_{comp}.npz")
        if not os.path.isfile(npz):
            print(f"[skip] {npz} not found")
            continue
        replot_one(npz, cdir, comp, args.freqmax, args.fmin, args.min_sources,
                   pick_min_score=args.pick_min_score)
    return 0


if __name__ == "__main__":
    sys.exit(main())
