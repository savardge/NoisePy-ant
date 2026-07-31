#!/usr/bin/env python3
"""Census figure for the substack-jackknife uncertainties.

Summarises a `substack_jackknife_k2` tree -- the per-(pair, period) scatter of the group
velocity pick across independent substack blocks (Bensen-style repeatability). This sigma
is what feeds the Tarantola-Valette velocity-map inversion, whose posterior in turn feeds
BayHunter, so its magnitude and its period dependence matter downstream.

The same tree also carries the evidence against the stored time-domain `Allstack_pws`:
`flag_pws` marks cells where the pws pick sits more than 2 sigma from its own substack
consensus, and `U_pws - U_med_blocks` measures the bias directly against the linear stack.

Usage:
  python plot_sigma_census.py --jackknife <.../substack_jackknife_k2> --label riehen
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STREAMS = [("rayleigh_G_LR0", "Rayleigh fundamental (G_LR0)", "#1f77b4"),
           ("love_TT", "Love fundamental (TT)", "#d62728")]
COLS = ["stream", "period", "U_med_blocks", "sigma_mad", "n_blocks", "U_pws", "U_lin",
        "flag_pws"]


def load_tree(jk_dir, min_blocks):
    """Parse every *_jk.csv into flat arrays (pure-python split beats pandas per-file)."""
    files = sorted(glob.glob(os.path.join(jk_dir, "*", "*_jk.csv")))
    print("reading %s jackknife files ..." % format(len(files), ","), flush=True)
    st, per, umed, sig, nbl, upws, ulin, flag = [], [], [], [], [], [], [], []
    for i, fn in enumerate(files):
        try:
            with open(fn) as fh:
                head = fh.readline().rstrip("\n").split(",")
                if head != COLS:
                    continue
                for line in fh:
                    p = line.rstrip("\n").split(",")
                    if len(p) != 8:
                        continue
                    st.append(p[0])
                    per.append(p[1]); umed.append(p[2]); sig.append(p[3])
                    nbl.append(p[4]); upws.append(p[5]); ulin.append(p[6])
                    flag.append(p[7])
        except Exception:
            continue
        if (i + 1) % 5000 == 0:
            print("  %d/%d" % (i + 1, len(files)), flush=True)

    def f(a):
        return np.array([np.nan if x in ("", "nan", "NaN") else float(x) for x in a])

    d = {"stream": np.array(st), "period": f(per), "U_med": f(umed), "sigma": f(sig),
         "n_blocks": f(nbl), "U_pws": f(upws), "U_lin": f(ulin), "flag": f(flag)}
    keep = np.isfinite(d["sigma"]) & (d["n_blocks"] >= min_blocks)
    for k in d:
        d[k] = d[k][keep]
    print("  %s cells with n_blocks >= %d" % (format(keep.sum(), ","), min_blocks))
    return d


def band(ax, T, y, color, label, pct=(25, 75)):
    """Median line with an inter-quartile band, on the discrete period grid."""
    edges = np.unique(np.round(T, 2))
    med = np.array([np.nanmedian(y[np.round(T, 2) == e]) for e in edges])
    lo = np.array([np.nanpercentile(y[np.round(T, 2) == e], pct[0]) for e in edges])
    hi = np.array([np.nanpercentile(y[np.round(T, 2) == e], pct[1]) for e in edges])
    ax.fill_between(edges, lo, hi, color=color, alpha=0.20, lw=0)
    ax.plot(edges, med, color=color, lw=2.0, label=label)
    return edges, med


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jackknife", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--min-blocks", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    d = load_tree(a.jackknife, a.min_blocks)
    out = a.out or os.path.join(a.jackknife, "sigma_census.png")

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.2))
    fig.suptitle("Substack-jackknife uncertainty census -- %s  (%s pairs, %s cells, "
                 "n_blocks >= %d)"
                 % (a.label, format(len(glob.glob(os.path.join(a.jackknife, "*", "*_jk.csv"))), ","),
                    format(len(d["period"]), ","), a.min_blocks),
                 fontsize=13, y=0.98)

    # (a) sigma(T)
    ax = axes[0, 0]
    for key, name, c in STREAMS:
        m = d["stream"] == key
        if m.sum() > 100:
            band(ax, d["period"][m], d["sigma"][m], c, name)
    ax.set_xlabel("Period (s)"); ax.set_ylabel(r"$\sigma_U$ (km/s)")
    ax.set_title("(a) Pick uncertainty vs period\nmedian, IQR shaded")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (b) relative sigma
    ax = axes[0, 1]
    for key, name, c in STREAMS:
        m = (d["stream"] == key) & (d["U_med"] > 0)
        if m.sum() > 100:
            band(ax, d["period"][m], 100 * d["sigma"][m] / d["U_med"][m], c, name)
    ax.set_xlabel("Period (s)"); ax.set_ylabel(r"$\sigma_U / U$ (%)")
    ax.set_title("(b) Relative uncertainty vs period")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (c) sigma distribution.
    # sigma_mad is DISCRETE: the picks live on a 0.01 km/s velocity grid and the MAD is
    # scaled by 1.4826, so the achievable sigmas are multiples of ~0.0148 km/s. Bins of a
    # round width (0.02) alias against that ladder and produce a comb -- the same trap as
    # the period/velocity histogram edges. One bin per node, edges offset by half a node.
    # sigma_mad is quantized. The jackknife picks land on a 0.02 km/s velocity grid, so the
    # median absolute deviation is a multiple of 0.02 and sigma a multiple of
    # 1.4826 * 0.02 = 0.0297 km/s (measured: even multiples of 0.0148 outnumber odd ones
    # 3.5:1). Bin on that ladder by node INDEX -- not by value, because the CSV rounds to
    # 4 dp and fixed-width bins drift off the ladder within ~30 nodes.
    ax = axes[0, 2]
    step = 0.02 * 1.4826
    kmax = int(1.5 / step)
    bins = np.arange(-0.5, kmax + 1.5)
    for key, name, c in STREAMS:
        m = d["stream"] == key
        if m.sum() > 100:
            ax.hist(np.round(d["sigma"][m] / step), bins=bins, histtype="step", lw=1.8,
                    color=c,
                    label="%s\nmedian %.3f km/s" % (name, np.nanmedian(d["sigma"][m])))
    ticks = np.arange(0, kmax + 1, int(round(0.25 / step)))
    ax.set_xticks(ticks)
    ax.set_xticklabels(["%.2f" % (t * step) for t in ticks])
    ax.set_xlim(-0.5, kmax + 0.5)
    ax.set_xlabel(r"$\sigma_U$ (km/s)"); ax.set_ylabel("cells")
    ax.set_title("(c) Uncertainty distribution")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (d) flag_pws rate vs period -- stored time-domain PWS vs its own substack consensus
    ax = axes[1, 0]
    for key, name, c in STREAMS:
        m = d["stream"] == key
        if m.sum() > 100:
            edges = np.unique(np.round(d["period"][m], 2))
            rate = np.array([100 * np.nanmean(d["flag"][m][np.round(d["period"][m], 2) == e])
                             for e in edges])
            ax.plot(edges, rate, color=c, lw=2.0,
                    label="%s\noverall %.1f%%" % (name, 100 * np.nanmean(d["flag"][m])))
    ax.set_xlabel("Period (s)"); ax.set_ylabel(r"cells with $|U_{pws}-U_{cons}| > 2\sigma$ (%)")
    ax.set_title("(d) Stored time-domain PWS vs substack consensus")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (e) block coverage
    ax = axes[1, 1]
    nb = d["n_blocks"]
    ax.hist(nb, bins=np.arange(a.min_blocks - 0.5, min(np.nanmax(nb), 120) + 1.5, 2),
            color="#555555")
    ax.set_xlabel("substack blocks per cell"); ax.set_ylabel("cells")
    ax.set_title("(e) Jackknife coverage\nmedian %d blocks" % np.nanmedian(nb))
    ax.grid(alpha=0.3)

    # (f) stack bias: pws and linear against the substack consensus
    ax = axes[1, 2]
    bb = np.arange(-1.0, 1.005, 0.02)
    for tag, key, c, ls in (("time-domain PWS", "U_pws", "#e377c2", "-"),
                            ("linear", "U_lin", "#2ca02c", "--")):
        v = d[key] - d["U_med"]
        v = v[np.isfinite(v)]
        ax.hist(v, bins=bb, histtype="step", lw=1.8, color=c, ls=ls,
                label="%s\nmedian %+.3f, MAD %.3f"
                      % (tag, np.nanmedian(v),
                         np.nanmedian(np.abs(v - np.nanmedian(v)))))
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$U_{stack} - U_{consensus}$ (km/s)"); ax.set_ylabel("cells")
    ax.set_title("(f) Whole-stack pick vs substack consensus")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main()
