"""Diagnostic (user request 2026-07-26): recursive 3-sigma pick trimming, per period.

For each network's ffscan BASE fund-group pool (picks_fund_uni_nf.csv — 1-lambda floor, full
QC battery, NO far-field cut), per 0.1 s period bin: iterate {mean, std, drop |v-mean|>k*std}
until no pick is excluded (or --max-iter). This tests whether a plain recursive outlier
rejection can clean the long-period scatter the r/lambda thresholds address geometrically.

Output per net: ffscan_logs/sigma_trim_test/{net}_fund_group.png
  top    : period x velocity density — kept (viridis) over excluded (grey), final
           mean +/- k*sigma envelope (red), kept-median (white) vs base-median (cyan dashed)
  bottom : per-period removal fraction and iteration count
plus a printed per-net summary table.

Usage: python ffscan_sigma_trim_test.py [--nets riehen,aargau,hautesorne] [--k 3.0]
       [--wave fund] [--measure group]
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from ffscan_common import scale_bin_edges

EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
OUTD = os.path.normpath(os.path.join(EHM, "..", "ffscan_logs", "sigma_trim_test"))


def trim_period(v, k, max_iter):
    """Recursive mean/std k-sigma rejection; returns (keep_mask, n_iter)."""
    keep = np.ones(v.size, bool)
    for it in range(max_iter):
        m, s = v[keep].mean(), v[keep].std()
        if s == 0:
            return keep, it
        new = keep & (np.abs(v - m) <= k * s)
        if new.sum() == keep.sum():
            return keep, it
        keep = new
    return keep, max_iter


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nets", default="riehen,aargau,hautesorne")
    ap.add_argument("--k", type=float, default=3.0)
    ap.add_argument("--wave", default="fund")
    ap.add_argument("--measure", default="group", choices=("group", "phase"))
    ap.add_argument("--max-iter", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(OUTD, exist_ok=True)
    tag = "_nf_phase" if args.measure == "phase" else "_nf"

    for net in args.nets.split(","):
        fn = (f"{EHM}/{net}/tomo/1_velocity_maps/inputs/ffscan/"
              f"picks_{args.wave}_uni{tag}.csv")
        d = pd.read_csv(fn, usecols=["inst_period", "group_velocity"])
        T, V = d.inst_period.values, d.group_velocity.values
        # period bins matched to the picker's CWT scale grid (variable width; pools the
        # linear-FTAN 0.1 s periods that fall between scales -> continuous counts)
        tb = scale_bin_edges(net, T.min(), T.max())
        widths = np.diff(tb)
        centers = np.sqrt(tb[:-1] * tb[1:])
        bi = np.clip(np.digitize(T, tb) - 1, 0, len(centers) - 1)
        keep = np.ones(V.size, bool)
        rows = []
        for ib in range(len(centers)):
            sel = np.where(bi == ib)[0]
            if sel.size < 5:
                continue
            km, it = trim_period(V[sel], args.k, args.max_iter)
            keep[sel[~km]] = False
            vk = V[sel][km]
            rows.append(dict(T=centers[ib], T_lo=tb[ib], T_hi=tb[ib + 1],
                             width=widths[ib], n=sel.size, removed=int((~km).sum()),
                             frac=float((~km).mean()), iters=it,
                             mean0=float(V[sel].mean()), std0=float(V[sel].std()),
                             mean=float(vk.mean()), std=float(vk.std())))
        st = pd.DataFrame(rows)
        ktag = f"k{args.k:g}"
        st.to_csv(os.path.join(OUTD, f"{net}_{args.wave}_{args.measure}_{ktag}_stats.csv"),
                  index=False)
        # dedicated colorbar column so the strip axes align exactly with the map panel
        fig = plt.figure(figsize=(11, 8.6))
        gs = fig.add_gridspec(3, 2, width_ratios=[1, 0.025],
                              height_ratios=[3, 1, 1], hspace=0.14, wspace=0.04)
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
        ax3 = fig.add_subplot(gs[2, 0], sharex=ax)
        # velocity bins from ALL picks — deriving them from the rejected subset clipped
        # the kept-pick display whenever rejections spanned a narrow band (overtone bug)
        yb = np.histogram_bin_edges(V, bins=60)
        hb, _, _ = np.histogram2d(T[~keep], V[~keep], bins=[tb, yb])
        h, _, _ = np.histogram2d(T[keep], V[keep], bins=[tb, yb])
        # explicit norm limits: LogNorm autoscale breaks the colorbar on narrow-band
        # streams (e.g. love_ot, 4 period bins)
        norm = LogNorm(vmin=1, vmax=max(np.nanmax(h), np.nanmax(hb), 2))
        hb[hb == 0] = np.nan
        h[h == 0] = np.nan
        if (~keep).any():
            ax.pcolormesh(tb, yb, hb.T, norm=norm, cmap="Greys", alpha=0.7)
        im = ax.pcolormesh(tb, yb, h.T, norm=LogNorm(vmin=norm.vmin, vmax=norm.vmax),
                           cmap="viridis")
        fig.colorbar(im, cax=cax, label=f"picks / cell (grey = {args.k:g}σ-rejected)")
        ax.plot(st["T"], st["mean"], "r-", lw=1.4, label="converged mean")
        # keep-zone boundary as an OUTLINE, clipped to the pool's hard velocity bounds
        # (QC/export vbounds truncate the distribution; a boundary below them is
        # unphysical and visually inconsistent with the data floor)
        ax.plot(st["T"], np.clip(st["mean"] - args.k * st["std"], V.min(), V.max()),
                "r--", lw=1.1, label=f"±{args.k:g}σ keep-zone (clipped to vbounds)")
        ax.plot(st["T"], np.clip(st["mean"] + args.k * st["std"], V.min(), V.max()),
                "r--", lw=1.1)
        ax.plot(st["T"], st.mean0, "c--", lw=1.1, label="pre-trim mean")
        ax.set_ylabel(f"{args.measure} velocity (km/s)")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"{net}  {args.wave} {args.measure} — recursive {args.k:g}σ trim: "
                     f"{(~keep).sum():,}/{len(d):,} removed ({100 * (~keep).mean():.1f}%)\n"
                     f"pool = base export (_nf): NO far-field cut beyond the picker's "
                     f"per-pick ≥1λ floor (≈ ff1.0 arm); period bins = CWT scale grid",
                     fontsize=9.5)
        ax.tick_params(labelbottom=False)
        ax2.bar(st["T"], 100 * st.frac, width=0.9 * st.width, color="firebrick",
                label="% removed")
        ax2b = ax2.twinx()
        ax2b.plot(st["T"], st.iters, "k.-", ms=3, lw=0.8, label="iterations")
        ax2.set_ylabel("% removed"); ax2b.set_ylabel("iters")
        ax2.tick_params(labelbottom=False)
        ax3.bar(st["T"], st.n, width=0.9 * st.width, color="steelblue")
        ax3.set_yscale("log")
        ax3.set_ylabel("# picks / period")
        ax3.grid(alpha=0.3, axis="y")
        ax3.set_xlabel("period (s)")
        out = os.path.join(OUTD, f"{net}_{args.wave}_{args.measure}_{ktag}.png")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        hiT = st[st["T"] >= 2.5]
        print(f"{net:11s} {args.wave}/{args.measure}: removed {100 * (~keep).mean():4.1f}% "
              f"overall | T>=2.5 s: {100 * hiT.removed.sum() / hiT.n.sum():4.1f}% | "
              f"median iters {st.iters.median():.0f} | {out}")


if __name__ == "__main__":
    main()
