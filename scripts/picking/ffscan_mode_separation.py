"""Is the overtone branch statistically SEPARABLE from the fundamental, period by period?

Motivation (user, 2026-07-26): the overtone histograms still show occupied bins at
fundamental velocities. The existing `ot_res` gate is PER-PAIR (each overtone pick vs its
own pair's fundamental curve + a resolution-adaptive separation), so it cannot see whether
the two POPULATIONS overlap at a given period.

Test, per period bin (CWT-scale grid) and per measure:
  1. center/scale of the fundamental picks   -> mu0, sd0
  2. center/scale of the overtone picks      -> mu1, sd1
  (DEFAULTS: median/MAD, NO trimming -- measured 2026-07-27, trimming inflates every
   separation metric because it shrinks the scale by eating the shoulders: median d\u2032 on
   Haute-Sorne phase runs 4.26 raw -> 5.14 one-pass -> 8.31 recursive with sigma, i.e. the
   procedure removes the very overlap it is meant to quantify. MAD inflates far less
   (4.91 -> 6.38) and is the smaller scale on untrimmed data. Use --spread/--passes to
   reproduce the other combinations.)
  3. separability:
       gap      = mu1 - mu0
       dprime   = gap / sqrt(sd0^2 + sd1^2)      (population separation in sigmas)
       disjoint = (mu1 - k*sd1) > (mu0 + k*sd0)  (the k-sigma intervals do not overlap)
  4. proposed cut: keep overtone picks with U >= mu0 + k*sd0, and DROP periods that are
     not disjoint (there the "overtone" population is not distinguishable from the
     fundamental, so no per-pick rule can be trusted either).

CAVEAT, stated in the figure: sd is a NETWORK-WIDE spread, so it mixes lateral
heterogeneity (Riehen's graben-vs-basement contrast is ~0.8 km/s at 2 s) with mode
ambiguity. That makes this test CONSERVATIVE -- it can call a period inseparable where a
given pair's own two curves are cleanly resolved. Use it as a period-level screen ON TOP
of the per-pair ot_res, not as a replacement for it.

Output: ffscan_logs/mode_separation/{net}_{measure}.png + _stats.csv, and a printed summary.

Usage: python ffscan_mode_separation.py [--nets ...] [--measures group,phase] [--k 2]
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
OUTD = os.path.normpath(os.path.join(EHM, "..", "ffscan_logs", "mode_separation"))


def _center_scale(v, spread):
    """(center, scale) with either the mean/std pair or the robust median/MAD pair.
    MAD is rescaled by 1.4826 so the two are numerically comparable on Gaussian data.
    NOTE the limit of MAD here: it is robust to TAILS, not to genuine BIMODALITY -- a
    network whose picks are two lateral populations (Riehen graben vs basement) still
    yields a large MAD. Only a per-region split removes that."""
    if spread == "mad":
        c = np.median(v)
        return c, 1.4826 * np.median(np.abs(v - c))
    return v.mean(), v.std()


def trim(v, k, spread="sigma", passes=-1, max_iter=100):
    """passes = 0 -> NO trim (statistics of the full bin, the most honest population
    width); N > 0 -> exactly N trim iterations; -1 -> iterate to a fixed point.
    Recursion shrinks the scale by eating the shoulders, which inflates any separation
    metric built on it -- so 0 is the conservative reference, not the degenerate case."""
    keep = np.ones(v.size, bool)
    if passes == 0:
        m, s = _center_scale(v, spread)
        return m, s, int(v.size)
    n_it = max_iter if passes < 0 else passes
    for _ in range(n_it):
        m, s = _center_scale(v[keep], spread)
        if s == 0:
            break
        new = keep & (np.abs(v - m) <= k * s)
        if new.sum() == keep.sum():
            break
        keep = new
    m, s = _center_scale(v[keep], spread)
    return m, s, int(keep.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nets", default="riehen,aargau,hautesorne")
    ap.add_argument("--measures", default="group,phase")
    ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument("--overlap-tol", type=float, default=0.30,
                    help="allowed overlap of the two k-scale intervals, as a fraction of "
                         "their combined half-widths (0 = strict disjointness). Grazing "
                         "intervals are usable; 0.30 separates the three regimes measured "
                         "on aargau/hautesorne (interior 0.03-0.35 vs edges 0.4-1.5)")
    ap.add_argument("--max-in-fund", type=float, default=0.20,
                    help="max fraction of overtone picks allowed inside the fundamental "
                         "population (mu0 + k*s0) for a period to count as usable")
    ap.add_argument("--passes", type=int, default=0,
                    help="trim iterations: 0 = none (full-bin statistics), N = N passes, "
                         "-1 = recursive to a fixed point (default)")
    ap.add_argument("--spread", default="mad", choices=("sigma", "mad"),
                    help="dispersion estimator for the recursive trim and the interval: "
                         "sigma = mean/std, mad = median/1.4826*MAD (robust to tails, NOT "
                         "to bimodality -- see _center_scale)")
    args = ap.parse_args()
    os.makedirs(OUTD, exist_ok=True)
    k = args.k
    sp = args.spread
    pz = args.passes
    ptag = 'raw' if pz == 0 else ('rec' if pz < 0 else f'p{pz}')

    for net in args.nets.split(","):
        for measure in args.measures.split(","):
            tag = "_nf_phase" if measure == "phase" else "_nf"
            d = {}
            for wave in ("fund", "overtone"):
                fn = (f"{EHM}/{net}/tomo/1_velocity_maps/inputs/ffscan/"
                      f"picks_{wave}_uni{tag}.csv")
                d[wave] = pd.read_csv(fn, usecols=["inst_period", "group_velocity"])
            allT = np.concatenate([d[w].inst_period.values for w in d])
            tb = scale_bin_edges(net, allT.min(), allT.max())
            tc = np.sqrt(tb[:-1] * tb[1:])
            rows = []
            for ib in range(len(tc)):
                rec = dict(T=tc[ib], T_lo=tb[ib], T_hi=tb[ib + 1])
                ok = True
                for wave, sfx in (("fund", "0"), ("overtone", "1")):
                    x = d[wave]
                    v = x.group_velocity.values[(x.inst_period.values >= tb[ib])
                                                & (x.inst_period.values < tb[ib + 1])]
                    if v.size < args.min_n:
                        ok = False
                        break
                    m, s, n = trim(v, k, sp, pz)
                    rec[f"mu{sfx}"], rec[f"sd{sfx}"], rec[f"n{sfx}"] = m, s, n
                if not ok:
                    continue
                rec["gap"] = rec["mu1"] - rec["mu0"]
                rec["dprime"] = rec["gap"] / np.hypot(rec["sd0"], rec["sd1"])
                # strict disjointness is too brittle -- intervals that merely GRAZE
                # (relative overlap of a few %) mark perfectly usable periods. Score the
                # overlap as a fraction of the combined half-widths instead, and require
                # the physical criterion too: few overtone picks inside the fundamental.
                ov = ((rec["mu0"] + k * rec["sd0"]) - (rec["mu1"] - k * rec["sd1"]))
                rec["ov_rel"] = ov / (k * (rec["sd0"] + rec["sd1"]))
                rec["disjoint"] = bool(ov < 0)
                # overtone picks lying inside the fundamental population
                x = d["overtone"]
                sel = ((x.inst_period.values >= tb[ib]) & (x.inst_period.values < tb[ib + 1]))
                vo = x.group_velocity.values[sel]
                rec["frac_in_fund"] = float((vo < rec["mu0"] + k * rec["sd0"]).mean())
                rec["n_ot_raw"] = int(sel.sum())
                rec["usable"] = bool(rec["ov_rel"] <= args.overlap_tol
                                     and rec["frac_in_fund"] <= args.max_in_fund)
                rows.append(rec)
            st = pd.DataFrame(rows)
            st.to_csv(os.path.join(OUTD, f"{net}_{measure}_k{k:g}_{sp}_{ptag}_stats.csv"), index=False)

            fig, axes = plt.subplots(2, 1, figsize=(11, 7.4), sharex=True,
                                     gridspec_kw=dict(height_ratios=[2.4, 1]))
            ax = axes[0]
            x = d["overtone"]
            yb = np.histogram_bin_edges(x.group_velocity.values, bins=60)
            H, _, _ = np.histogram2d(x.inst_period.values, x.group_velocity.values,
                                     bins=[tb, yb])
            H[H == 0] = np.nan
            im = ax.pcolormesh(tb, yb, H.T, cmap="viridis", vmin=0,
                               vmax=float(np.nanpercentile(H, 99)))
            plt.colorbar(im, ax=ax, extend="max", label="overtone picks / cell")
            # fundamental = black, overtone = magenta: distinguishable against viridis
            # and from each other (red/orange were not), each with a white halo
            for y, col, ls, lab in (
                    (st.mu0, "k", "-", f"fundamental {'median' if sp == 'mad' else 'mean'} ({sp}, {ptag})"),
                    (st.mu0 + k * st.sd0, "k", "--",
                     f"fundamental +{k:g}·{sp}  = proposed overtone floor"),
                    (st.mu1, "magenta", "-", f"overtone {'median' if sp == 'mad' else 'mean'}"),
                    (st.mu1 - k * st.sd1, "magenta", "--", f"overtone −{k:g}·{sp}")):
                ax.plot(st["T"], y, ls, color="w", lw=3.4, solid_capstyle="round")
                ax.plot(st["T"], y, ls, color=col, lw=1.8, label=lab)
            # dark = rejected; light = kept but intervals graze (0 < overlap <= tol)
            for _, r in st.iterrows():
                if not r.usable:
                    ax.axvspan(r.T_lo, r.T_hi, color="0.25", alpha=0.34, lw=0)
                elif r.ov_rel > 0:
                    ax.axvspan(r.T_lo, r.T_hi, color="0.70", alpha=0.30, lw=0)
            ax.set_ylabel(f"{measure} velocity (km/s)")
            ax.legend(fontsize=8, loc="upper right")
            ax.set_title(f"{net} — {measure} overtone vs fundamental separation "
                         f"(k={k:g}, {sp}, trim={ptag})\n"
                         f"dark grey = REJECTED, light grey = kept but intervals graze; "
                         f"network-wide spread also carries lateral heterogeneity",
                         fontsize=9)
            ax2 = axes[1]
            ax2.plot(st["T"], st.dprime, "k.-", lw=1.1, ms=4, label=f"d′ = gap / √(s0²+s1²)  [{sp}]")
            ax2.axhline(2 * k, color="grey", ls=":", lw=1,
                        label=f"d′ = {2 * k:g} (intervals just disjoint)")
            ax2.set_ylabel("d′")
            ax2b = ax2.twinx()
            ax2b.bar(st["T"], 100 * st.frac_in_fund, width=0.9 * (st.T_hi - st.T_lo),
                     color="magenta", alpha=0.35)
            ax2b.set_ylabel("% overtone inside fund. population", color="magenta")
            ax2.set_xlabel("period (s)")
            ax2.legend(fontsize=8, loc="upper left")
            fig.tight_layout()
            out = os.path.join(OUTD, f"{net}_{measure}_k{k:g}_{sp}_{ptag}.png")
            fig.savefig(out, dpi=130)
            plt.close(fig)
            use = st[st.usable]
            if len(use):
                idx = np.flatnonzero(st.usable.to_numpy())
                contig = bool(np.all(np.diff(idx) == 1))
                band = (f"{use['T_lo'].min():.2f}-{use['T_hi'].max():.2f} s"
                        + ("" if contig else " (NON-contiguous)"))
            else:
                band = "NONE"
            print(f"{net:11s} {measure:5s} k={k:g} {sp:5s} {ptag:3s}: "
                  f"{len(use)}/{len(st)} bins usable (tol {args.overlap_tol:g}, "
                  f"in_fund <= {100 * args.max_in_fund:.0f}%) -> band {band} | "
                  f"median d\u2032={st.dprime.median():.2f} -> {os.path.basename(out)}")


if __name__ == "__main__":
    main()
