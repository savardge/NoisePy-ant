#!/usr/bin/env python3
"""
Derive network-average REFERENCE dispersion curves from slant-stack (phase-shift) stacks.

Input: stacks_<comp>.npz / picks_<comp>.npz files produced by phaseshift_dispersion.py.
Per wave type and branch (fundamental / first overtone), the STACKED dispersion image is
re-picked column by column: for each frequency, scipy.signal.find_peaks locates local
maxima of the (per-frequency normalised) stack amplitude, and the strongest peak inside
the branch window is kept -- a purely data-derived c(f) curve, no functional fit. After a
running-median continuity clean, the reference GROUP velocity follows by NUMERICAL
differentiation of the picked slowness curve (Bensen et al. 2007 eq. 7; Levshin et al.
1989 eq. 5.4):

    s_u = s_c + w * ds_c/dw  =  s_c + ds_c/d(ln f)        (x = ln f)

The derivative amplifies pick jitter, so the slowness series is lightly smoothed
(Savitzky-Golay over --smooth-cols columns; 0 disables) before np.gradient. Gaps wider
than --max-gap columns are kept as breaks (no interpolation across them).

Outputs (in <vsg_dir>/analysis/):
    reference_curves.csv        wave, branch, component, f, period, c_ref, U_ref
    reference_curves_<wave>.png picks + picked c per branch + derived U
    ftan_vs_reference.png       (optional, --ftan-csv) pairwise FTAN group-pick density
                                with U_ref overlaid -- the QC picture.

Example:
    python reference_disp_from_slantstack.py ~/Data/aargau/phasevelocity_VSG \\
        --ftan-csv /Volumes/T7Shield/aargau/dispersion-curves/picks_merged_V3_rma2_normZ_lambda1.5_SNR5.0.csv
"""
import argparse
import logging
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, medfilt, savgol_filter

logger = logging.getLogger("refdisp")

WAVES = {
    "rayleigh": ["ZZ", "RZ", "ZR", "RR"],
    "love": ["TT"],
}
# Per-branch acceptance windows (fmin [Hz], fmax [Hz], vmin [km/s], vmax [km/s]).
# Set from the Aargau stacked images: ridges are coherent only below ~1.5 Hz for the
# overtone and ~2.3 Hz for the fundamental; outside these windows picks are aliases.
BRANCH_WINDOWS = {
    ("rayleigh", "fundamental"): (0.16, 2.15, 1.20, 3.40),
    ("rayleigh", "overtone"): (0.42, 1.30, 2.60, 5.60),
    ("love", "fundamental"): (0.20, 2.20, 1.50, 3.40),
    ("love", "overtone"): (0.62, 1.15, 2.85, 4.00),
}
# Extra per-branch exclusions (f_lo, f_hi, v_lo, v_hi): image region REMOVED from the
# search. Used where fundamental and overtone windows would otherwise overlap.
BRANCH_EXCLUDE = {
    ("rayleigh", "fundamental"): [(0.42, 2.30, 2.75, 99.0)],
    ("love", "fundamental"): [(0.55, 1.30, 2.75, 99.0)],
}
BRANCH_NAMES = ["fundamental", "overtone"]


def load_stack(vsg_dir, comp, method):
    path = os.path.join(vsg_dir, comp, f"stacks_{comp}.npz")
    if not os.path.isfile(path):
        logger.warning("Missing %s", path)
        return None
    d = np.load(path, allow_pickle=True)
    if method not in d:
        logger.warning("Method %s not in %s", method, path)
        return None
    img = np.asarray(d[method], dtype=float)        # (nv, nf)
    peak = np.max(img, axis=0, keepdims=True)
    img = img / np.where(peak > 0, peak, 1.0)       # per-frequency normalised
    return {"img": img, "f": d["f"], "vel": d["vel"]}


def load_scatter_picks(vsg_dir, comp, method, min_score):
    """Raw topology picks from picks_<comp>.npz, for the context scatter only."""
    path = os.path.join(vsg_dir, comp, f"picks_{comp}.npz")
    if not os.path.isfile(path):
        return None
    d = np.load(path, allow_pickle=True)
    try:
        f, vel, score = d[f"{method}_f"], d[f"{method}_vel"], d[f"{method}_score"]
    except KeyError:
        return None
    good = (score >= min_score) & np.isfinite(vel)
    return f[good], vel[good]


def pick_branch_curve(stack, wave, branch, min_prom=0.05):
    """
    One pick per frequency column: the strongest scipy.signal.find_peaks local maximum
    of the normalised stack column that falls inside the branch window (exclusion zones
    removed). Returns c(f) on the full stack f axis, NaN where no acceptable peak.
    """
    f, vel, img = stack["f"], stack["vel"], stack["img"]
    fmin, fmax, vmin, vmax = BRANCH_WINDOWS[(wave, branch)]
    excludes = BRANCH_EXCLUDE.get((wave, branch), [])
    c = np.full(f.shape, np.nan)
    cols = np.where((f >= fmin) & (f <= fmax))[0]
    for j in cols:
        col = img[:, j]
        peaks, _ = find_peaks(col, prominence=min_prom)
        if peaks.size == 0:
            continue
        ok = (vel[peaks] >= vmin) & (vel[peaks] <= vmax)
        for xlo, xhi, vlo, vhi in excludes:
            if xlo <= f[j] <= xhi:
                ok &= ~((vel[peaks] >= vlo) & (vel[peaks] <= vhi))
        peaks = peaks[ok]
        if peaks.size == 0:
            continue
        c[j] = vel[peaks[np.argmax(col[peaks])]]
    return c


def continuity_clean(c, kernel=9, max_jump=0.15):
    """Reject picks deviating more than max_jump [km/s] from the running median."""
    out = c.copy()
    finite = np.isfinite(c)
    if finite.sum() < kernel:
        return out
    idx = np.where(finite)[0]
    med = medfilt(c[idx], kernel_size=kernel)
    bad = np.abs(c[idx] - med) > max_jump
    out[idx[bad]] = np.nan
    return out


def node_resample(f, c, n_nodes=45):
    """
    Aggregate column picks to log-spaced frequency nodes: node value = median of the
    picks in the node window after 3*MAD rejection. Purely data-derived (no fit).
    Returns (f_nodes, c_nodes) for nodes holding >= 3 picks.
    """
    finite = np.isfinite(c)
    if finite.sum() < 10:
        return np.array([]), np.array([])
    fmin, fmax = f[finite].min(), f[finite].max()
    edges = np.geomspace(fmin, fmax * 1.0001, n_nodes + 1)
    fn, cn = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = finite & (f >= lo) & (f < hi)
        if m.sum() < 3:
            continue
        v = c[m]
        med = np.median(v)
        mad = 1.4826 * np.median(np.abs(v - med))
        keep = np.abs(v - med) <= max(3.0 * mad, 0.05)
        if keep.sum() < 3:
            continue
        fn.append(np.exp(np.mean(np.log(f[m][keep]))))
        cn.append(np.median(v[keep]))
    return np.array(fn), np.array(cn)


def differentiate_nodes(fn, cn, smooth_nodes=7, max_node_gap=1.6):
    """
    U at the nodes by numerical differentiation: s_u = s_c + ds_c/d(ln f) with
    np.gradient on the node slowness series. Optional light Savitzky-Golay smoothing
    of the slowness series first (smooth_nodes=0 disables). Segments are broken where
    consecutive nodes are more than max_node_gap apart in frequency ratio.
    """
    U = np.full(cn.shape, np.nan)
    c_s = cn.copy()
    if fn.size < 5:
        return c_s, U
    ratio = fn[1:] / fn[:-1]
    breaks = np.where(ratio > max_node_gap)[0]
    segments = np.split(np.arange(fn.size), breaks + 1)
    for seg in segments:
        if seg.size < 5:
            continue
        x = np.log(fn[seg])
        s = 1.0 / cn[seg]
        if seg.size >= 5:
            s = medfilt(s, kernel_size=min(5, seg.size | 1))  # single-node outliers
        if smooth_nodes and seg.size >= smooth_nodes:
            win = min(smooth_nodes, seg.size) | 1
            s = savgol_filter(s, win, 2)
        s_u = s + np.gradient(s, x)
        with np.errstate(divide="ignore"):
            U[seg] = np.where(s_u > 0, 1.0 / s_u, np.nan)
            c_s[seg] = 1.0 / s
    return c_s, U


def process_wave(vsg_dir, wave, comps, args):
    """Pick + differentiate per component; pooled = per-column median across comps."""
    results = []   # (branch, comp_label, {f, c, U})
    raw_by_comp = {}
    curves = {b: {} for b in BRANCH_NAMES}
    f_axis = None
    for comp in comps:
        stack = load_stack(vsg_dir, comp, args.method)
        if stack is None:
            continue
        if f_axis is None:
            f_axis = stack["f"]
        elif stack["f"].shape != f_axis.shape or not np.allclose(stack["f"], f_axis):
            logger.warning("Frequency axis mismatch for %s; skipping.", comp)
            continue
        scatter = load_scatter_picks(vsg_dir, comp, args.method, args.min_score)
        if scatter is not None:
            raw_by_comp[comp] = scatter
        for branch in BRANCH_NAMES:
            c = pick_branch_curve(stack, wave, branch, min_prom=args.min_prom)
            c = continuity_clean(c, kernel=args.clean_kernel, max_jump=args.max_jump)
            if np.isfinite(c).sum() < 10:
                continue
            curves[branch][comp] = c
            fn, cn = node_resample(f_axis, c, args.nodes)
            c_s, U = differentiate_nodes(fn, cn, args.smooth_nodes)
            if fn.size:
                results.append((branch, comp, {"f": fn, "c": c_s, "U": U, "c_raw": cn}))
    for branch in BRANCH_NAMES:
        if not curves[branch]:
            continue
        allc = np.vstack(list(curves[branch].values()))
        with np.errstate(invalid="ignore"):
            pooled = np.nanmedian(allc, axis=0)
        pooled = continuity_clean(pooled, kernel=args.clean_kernel, max_jump=args.max_jump)
        fn, cn = node_resample(f_axis, pooled, args.nodes)
        c_s, U = differentiate_nodes(fn, cn, args.smooth_nodes)
        if fn.size:
            results.append((branch, "pooled", {"f": fn, "c": c_s, "U": U, "c_raw": cn}))
    return results, raw_by_comp


def plot_wave(outpath, wave, results, raw_by_comp):
    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    cmap = plt.get_cmap("tab10")
    colors = {c: cmap(i) for i, c in enumerate(sorted(raw_by_comp) + ["pooled"])}
    ax = axes[0]
    for comp, (f, vel) in raw_by_comp.items():
        ax.scatter(1.0 / f, vel, s=2, alpha=0.15, color=colors[comp], label=f"{comp} picks")
    for branch, comp, fit in results:
        ls = "-" if branch == "fundamental" else "--"
        lw = 2.5 if comp == "pooled" else 1.0
        m = np.isfinite(fit["c_raw"])
        ax.plot(1.0 / fit["f"][m], fit["c_raw"][m], ls, color=colors[comp], lw=lw,
                label=f"c {branch[:4]} {comp}" if comp == "pooled" else None)
    ax.set_ylabel("Phase velocity [km/s]")
    ax.set_title(f"{wave.capitalize()}: slant-stack picks and reference phase curves")
    ax.legend(fontsize=7, ncol=2, markerscale=4)
    ax = axes[1]
    for branch, comp, fit in results:
        ls = "-" if branch == "fundamental" else "--"
        lw = 2.5 if comp == "pooled" else 1.0
        m = np.isfinite(fit["U"])
        ax.plot(1.0 / fit["f"][m], fit["U"][m], ls, color=colors[comp], lw=lw,
                label=f"U {branch} {comp}")
    ax.set_xlabel("Period [s]")
    ax.set_ylabel("Group velocity [km/s]")
    ax.set_title("Derived reference group curves  (1/U = 1/c + d(1/c)/dln f, numerical)")
    ax.set_xscale("log")
    ax.legend(fontsize=7, ncol=2)
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
        ax.set_ylim(0.5, 4.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
    logger.info("Wrote %s", outpath)


def write_csv(outpath, all_results):
    with open(outpath, "w") as fh:
        fh.write("wave,branch,component,frequency,period,c_ref,U_ref\n")
        for wave, results in all_results.items():
            for branch, comp, fit in results:
                m = np.isfinite(fit["c"])
                for f, c, U in zip(fit["f"][m], fit["c"][m], fit["U"][m]):
                    fh.write(f"{wave},{branch},{comp},{f:.5f},{1.0/f:.5f},"
                             f"{c:.4f},{U:.4f}\n")
    logger.info("Wrote %s", outpath)


def ftan_comparison(outpath, csv_path, all_results, component="ZZ",
                    stack_method="pws", chunksize=2_000_000):
    """2D density of pairwise FTAN group picks vs the derived reference U curves."""
    import pandas as pd
    per_edges = np.geomspace(0.2, 6.0, 120)
    vel_edges = np.linspace(0.4, 4.5, 160)
    H = np.zeros((len(vel_edges) - 1, len(per_edges) - 1))
    usecols = ["inst_period", "group_velocity", "component", "stack_method"]
    nrows = 0
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        sel = chunk[(chunk.component == component) & (chunk.stack_method == stack_method)]
        if not len(sel):
            continue
        h, _, _ = np.histogram2d(sel.group_velocity, sel.inst_period,
                                 bins=[vel_edges, per_edges])
        H += h
        nrows += len(sel)
    logger.info("FTAN histogram from %d picks (%s, %s)", nrows, component, stack_method)
    fig, ax = plt.subplots(figsize=(9, 6))
    Hn = H / np.maximum(H.max(axis=0, keepdims=True), 1)  # per-period normalisation
    ax.pcolormesh(per_edges, vel_edges, Hn, cmap="Greys", vmax=1.0)
    wave_for_comp = "love" if component == "TT" else "rayleigh"
    for branch, comp, fit in all_results.get(wave_for_comp, []):
        if comp not in ("pooled", component):
            continue
        ls = "-" if comp == "pooled" else "--"
        color = "tab:red" if branch == "fundamental" else "tab:blue"
        mU, mc = np.isfinite(fit["U"]), np.isfinite(fit["c"])
        ax.plot(1.0 / fit["f"][mU], fit["U"][mU], ls, lw=2, color=color,
                label=f"U_ref {branch} ({comp})")
        ax.plot(1.0 / fit["f"][mc], fit["c"][mc], ":", lw=1.5, color=color,
                label=f"c_ref {branch} ({comp})")
    ax.set_xscale("log")
    ax.set_xlabel("Period [s]")
    ax.set_ylabel("Velocity [km/s]")
    ax.set_title(f"Pairwise FTAN group picks ({component}, {stack_method}) vs slant-stack reference")
    ax.set_ylim(0.4, 4.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
    logger.info("Wrote %s", outpath)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vsg_dir", help="phasevelocity_VSG directory (with ZZ/, RR/, ... subdirs)")
    ap.add_argument("--method", default="pws",
                    choices=["linear", "pws", "complex_pws", "root"],
                    help="Stack method (image re-picked column by column)")
    ap.add_argument("--min-prom", type=float, default=0.05,
                    help="find_peaks prominence on per-frequency normalised columns")
    ap.add_argument("--min-score", type=float, default=0.5,
                    help="Score cut for the context scatter (picks_<comp>.npz)")
    ap.add_argument("--clean-kernel", type=int, default=9,
                    help="Running-median kernel [columns] for continuity cleaning")
    ap.add_argument("--max-jump", type=float, default=0.15,
                    help="Reject picks deviating more than this [km/s] from running median")
    ap.add_argument("--nodes", type=int, default=35,
                    help="Log-spaced frequency nodes for median aggregation of column picks")
    ap.add_argument("--smooth-nodes", type=int, default=11,
                    help="Savitzky-Golay window [nodes] on slowness before np.gradient "
                         "(0 = raw numerical derivative)")
    ap.add_argument("--ftan-csv", default=None,
                    help="Optional merged pairwise FTAN picks CSV for the QC comparison plot")
    ap.add_argument("--ftan-component", default="ZZ")
    ap.add_argument("--ftan-stack", default="pws")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    outdir = os.path.join(args.vsg_dir, "analysis")
    os.makedirs(outdir, exist_ok=True)

    all_results = {}
    for wave, comps in WAVES.items():
        results, raw = process_wave(vsg_dir=args.vsg_dir, wave=wave, comps=comps, args=args)
        if not results:
            logger.warning("No curves for %s", wave)
            continue
        all_results[wave] = results
        plot_wave(os.path.join(outdir, f"reference_curves_{wave}.png"), wave, results, raw)
    write_csv(os.path.join(outdir, "reference_curves.csv"), all_results)
    if args.ftan_csv:
        ftan_comparison(os.path.join(outdir, "ftan_vs_reference.png"),
                        args.ftan_csv, all_results,
                        component=args.ftan_component, stack_method=args.ftan_stack)


if __name__ == "__main__":
    main()
