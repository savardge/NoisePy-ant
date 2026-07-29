"""Shared helpers for the ffscan diagnostics: the CWT-scale-matched period binning.

Pick tables are exported on the picker's native CWT scale ladder (uneven, ~5.95%
spacing), so histograms must bin on that ladder: edges at the geometric midpoints
between neighbouring scales, one bin per rung. Uniform 0.1 s bins sawtooth badly
above ~2 s, where the ladder is coarser than 0.1 s and scale-less bins collect only
stragglers.
"""
import os

import numpy as np
import pandas as pd

EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"


def load_cwt_scales(net):
    """The network's discrete CWT scale set (s), cached next to the ffscan QC output."""
    cache = f"{EHM}/{net}/tomo/1_velocity_maps/inputs/ffscan/qc/cwt_scales.txt"
    if os.path.exists(cache):
        return np.loadtxt(cache)
    qc = f"{EHM}/{net}/tomo/1_velocity_maps/inputs/ffscan/qc/picks_unified_QCd.csv"
    u = np.unique(pd.read_csv(qc, usecols=["T_scale"]).T_scale.round(4).values)
    np.savetxt(cache, u, fmt="%.4f")
    return u


def scale_bin_edges(net, tmin, tmax, min_width=0.0):
    """Variable-width period-bin edges matching the CWT scale grid, covering
    [tmin, tmax]; geometric midpoints between scales, thinned to >= min_width.

    min_width 0 (default since the 2026-07-27 scale-native switch) = ONE BIN PER RUNG,
    which is exact now that the pick tables carry the CWT scale as their period. The old
    default of 0.1 existed only to stop bins going empty when the picks lived on the
    uniform 0.1 s nominal grid; applying it to scale-native data would re-merge rungs the
    export just took care to separate."""
    s = load_cwt_scales(net)
    s = s[(s >= tmin / 1.2) & (s <= tmax * 1.2)]
    edges = np.sqrt(s[:-1] * s[1:])
    edges = np.concatenate([[s[0] * np.sqrt(s[0] / s[1])], edges,
                            [s[-1] * np.sqrt(s[-1] / s[-2])]])
    # extend to cover the data range (linear-FTAN periods can exceed the scale set)
    while edges[0] > tmin:
        edges = np.concatenate([[edges[0] * s[0] / s[1]], edges])
    while edges[-1] < tmax:
        edges = np.concatenate([edges, [edges[-1] * s[-1] / s[-2]]])
    # merge edges closer than min_width (short-T scales are denser than the 0.1 s
    # rounding of the phase axis)
    keep = [0]
    for i in range(1, len(edges)):
        if edges[i] - edges[keep[-1]] >= min_width:
            keep.append(i)
    return edges[keep]


def populated_bin_edges(T, min_n=30, frac=0.05):
    """Geometric-midpoint bin edges around the ladder rungs ACTUALLY populated in `T`.

    Rungs between nominal periods hold 1-25 picks against ~10,000 on their neighbours
    (picks are emitted on the 0.1 s FTAN grid and snapped to the nearest CWT scale), so
    giving every rung a bin leaves near-empty white stripes and makes per-rung medians
    jitter. Binning on the populated rungs is the honest sampling of the data."""
    u, c = np.unique(np.round(np.asarray(T, float), 4), return_counts=True)
    # adaptive floor: a rung holding <5% of the typical rung population is a straggler,
    # not a sampled period (absolute floors leave 40-100 pick rungs next to 8,000 ones)
    thr = max(min_n, frac * np.median(c[c >= min_n]) if (c >= min_n).any() else min_n)
    u = u[c >= thr]
    if len(u) < 2:
        return None
    e = np.sqrt(u[:-1] * u[1:])
    return np.concatenate([[u[0] * np.sqrt(u[0] / u[1])], e,
                           [u[-1] * np.sqrt(u[-1] / u[-2])]])
