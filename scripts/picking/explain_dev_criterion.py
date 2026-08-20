#!/usr/bin/env python3
"""Why BayHunter's dev=0.05 outlier rejection discarded chains that agreed.

The criterion, verbatim from BayHunter/src/Plotting.py::get_outliers:

    maxlike = np.max(chainmedians)      # chainmedians are MEDIAN LOG-LIKELIHOODS
    if maxlike > 0:   scores = chainmedians / maxlike
    elif maxlike < 0: scores = maxlike / chainmedians
    outliers = chainidxs[(1 - scores) > dev]

It forms a RATIO OF TWO LOG-LIKELIHOODS. A relative tolerance is only meaningful for a quantity
with a natural zero, and a log-likelihood has none: its absolute value carries the data
normalisation. The absolute tolerance the criterion actually applies is

    |dlogL| = dev * |logL_best|,

so it is set by wherever logL happens to sit -- which depends on the number of data points, on
sigma, and even on the units the velocities are expressed in. Three figures:

  1. THE ARITHMETIC. What dev=0.05 means in absolute log-likelihood at these cells, next to the
     posterior's own spread; and the units demonstration -- the same inversion in m/s instead of
     km/s flips the code to its `maxlike < 0` branch and changes the tolerance by ~3x.
  2. WHAT THE TOLERANCE MEANS IN PRACTICE. Log-likelihood here tracks the fit closely
     (corr -0.97 with RMS misfit), so the tolerance converts directly into fit quality: chains
     are rejected once their median RMS misfit is ~4 m/s worse than the best chain's. That is
     6-10% of the noise level the sampler itself estimated for the data (sigma 42-63 m/s), and
     about one interquartile range of the posterior's own misfit spread.
  3. THE SAME CRITERION ON A CONVERGED RUN. Applied to our 24-chain section runs, whose chains
     agree on Vs (chain_disagree ~0.24), it still rejects most of them. That isolates the
     criterion from the data.

Plus the consequences across the grid: chains kept, its spatial pattern, and whether it imprints
on the model.
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
MB = f"{EHM}/hautesorne/tomo/2_vs_depth_inversion/manuscript_comparison_2026-08/0_manuscript_run"
VS = f"{EHM}/hautesorne/tomo/2_vs_depth_inversion/manuscript_comparison_2026-08/1_sections"
MS = f"{MB}/models2"
SEC = ["test_2026-08-08_AA_section_R0g", "test_2026-08-08_BB_section_R0g",
       "test_2026-08-16_CC_section_R0g"]
GVL = ["20.50x_10.50y", "20.50x_11.00y", "21.00x_10.50y", "21.00x_11.00y"]
LAB = {"20.50x_10.50y": "(20.5, 10.5)", "20.50x_11.00y": "(20.5, 11.0)",
       "21.00x_10.50y": "(21.0, 10.5)", "21.00x_11.00y": "(21.0, 11.0)"}
DEV = 0.05
NCH_MS = 30


def bayhunter_outliers(chainmedians, dev=DEV):
    """BayHunter's get_outliers, verbatim in its arithmetic."""
    cm = np.asarray(chainmedians, float)
    maxlike = np.nanmax(cm)
    if maxlike > 0:
        scores = cm / maxlike
    elif maxlike < 0:
        scores = maxlike / cm
    else:
        return np.zeros(cm.size, bool)
    return (1 - scores) > dev


def fig1(z, out):
    fig = plt.figure(figsize=(16.5, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.05], wspace=0.28)

    # -- A: the tolerance against the posterior's own log-likelihood spread ------------------
    ax = fig.add_subplot(gs[0])
    c = "21.00x_11.00y"
    L = z[f"L_{c}"]
    lo, hi = np.percentile(L, [5, 95])
    # every rejected chain's median logL is recoverable: logL_i = logL_best*(1-dev_i)
    d = z[f"dev_{c}"]
    Lbest = np.percentile(L, 95)          # kept chains only; an upper estimate of logL_best
    ax.hist(L, bins=90, color="0.75", edgecolor="none",
            label="posterior samples (kept chains)")
    tol = DEV * abs(np.median(L))
    ax.axvspan(np.median(L) - tol, np.median(L), color="tab:red", alpha=0.30,
               label=f"the dev=0.05 tolerance\n|dlogL| = {tol:.1f}")
    ax.axvline(lo, color="k", ls=":", lw=1.4)
    ax.axvline(hi, color="k", ls=":", lw=1.4)
    yt = ax.get_ylim()[1] * 0.60
    ax.annotate("", xy=(lo, yt), xytext=(hi, yt),
                arrowprops=dict(arrowstyle="<->", color="k", lw=1.4))
    ax.text((lo + hi) / 2, yt * 1.04, f"p5-p95 = {hi-lo:.1f}", ha="center", fontsize=9,
            bbox=dict(fc="w", ec="none", alpha=0.8, pad=1.5))
    ax.set_xlabel("log-likelihood"); ax.set_ylabel("posterior samples")
    ax.set_title(f"cell {LAB[c]} — the tolerance is narrower\nthan the posterior's own spread",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper right")

    # -- B: the units demonstration ----------------------------------------------------------
    ax = fig.add_subplot(gs[1])
    N = 60
    rows = []
    for cc in GVL:
        Lc = np.median(z[f"L_{cc}"])
        t_km = DEV * abs(Lc)                              # maxlike > 0 branch
        Lm = Lc - N * np.log(1000.0)                      # same models, velocities in m/s
        t_m = abs(Lm) * DEV / (1 - DEV)                   # maxlike < 0 branch
        rows.append((LAB[cc], Lc, t_km, Lm, t_m))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r[2] for r in rows], 0.4, color="tab:blue", label="velocities in km/s")
    ax.bar(x + 0.2, [r[4] for r in rows], 0.4, color="tab:orange", label="velocities in m/s")
    for i, r in enumerate(rows):
        ax.text(i + 0.2, r[4], f"{r[4]/r[2]:.1f}x", ha="center", va="bottom", fontsize=9,
                fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylabel("absolute tolerance |dlogL|")
    ax.set_title("the SAME inversion, only the units change\n"
                 "(m/s puts logL < 0, flipping the code's branch)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # -- C: what a relative cut on logL means as a likelihood ratio --------------------------
    ax = fig.add_subplot(gs[2])
    ll = np.linspace(1, 400, 400)
    ax.plot(ll, DEV * ll, "-", color="tab:red", lw=2.2,
            label="tolerance actually applied\n|dlogL| = dev x |logL|")
    ax.axhline(-np.log(1 - DEV), color="tab:green", lw=2.2, ls="--",
               label="a 5% cut on the LIKELIHOOD\n|dlogL| = 0.051")
    for cc in GVL:
        Lc = abs(np.median(z[f"L_{cc}"]))
        ax.plot([Lc], [DEV * Lc], "o", color="k", ms=7)
    ax.annotate("these cells", xy=(95, DEV * 95), xytext=(150, DEV * 30),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="k", lw=1.0))
    ax.set_yscale("log")
    ax.set_xlabel("|log-likelihood| of the best chain")
    ax.set_ylabel("absolute tolerance |dlogL|")
    ax.set_title("a relative cut on a LOG quantity\nis not a relative cut on the likelihood",
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.suptitle("1. dev=0.05 divides two LOG-likelihoods — the tolerance it applies depends on "
                 "where logL happens to sit", fontsize=13, fontweight="bold")
    p = os.path.join(out, "dev_1_arithmetic.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); print("wrote", p)
    for r in rows:
        print(f"  {r[0]}: logL {r[1]:+.1f} -> tol {r[2]:.2f}   |   in m/s: logL {r[3]:+.1f} "
              f"-> tol {r[4]:.2f}  ({r[4]/r[2]:.2f}x)")


def fig2(z, out):
    """Translate the tolerance into fit quality, and compare it with the data's own noise."""
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.2))
    ax = axs[0]
    stats = []
    for cc in GVL:
        L, M, S = z[f"L_{cc}"], z[f"M_{cc}"], z[f"S_{cc}"]
        k = np.isfinite(L) & np.isfinite(M)
        sl = np.polyfit(M[k], L[k], 1)[0]
        tol = DEV * abs(np.median(L))
        stats.append(dict(lab=LAB[cc], tol=tol, dm=tol / abs(sl),
                          r=np.corrcoef(M[k], L[k])[0, 1], sig=np.median(S),
                          iqr=np.subtract(*np.percentile(M[k], [75, 25])),
                          mis=np.median(M[k])))
        ax.scatter(M[k][::4], L[k][::4], s=2, alpha=0.22, label=LAB[cc])
    ax.set_xlabel("misfit — RMS of the dispersion fit [km/s]")
    ax.set_ylabel("log-likelihood")
    ax.set_title(f"log-likelihood tracks the FIT\n"
                 f"(corr {np.mean([s['r'] for s in stats]):+.2f})", fontsize=10)
    ax.legend(fontsize=8, markerscale=6); ax.grid(alpha=0.3)

    ax = axs[1]
    cc = "21.00x_11.00y"
    M = z[f"M_{cc}"]
    s = [q for q in stats if q["lab"] == LAB[cc]][0]
    ax.hist(M[np.isfinite(M)], bins=90, color="0.75", edgecolor="none",
            label="posterior misfit (kept chains)")
    m0 = np.median(M)
    ax.axvspan(m0, m0 + s["dm"], color="tab:red", alpha=0.35,
               label=f"dev=0.05 rejects beyond\n{s['dm']*1000:.1f} m/s of extra misfit")
    ax.annotate("", xy=(m0, ax.get_ylim()[1] * 0.72),
                xytext=(m0 + s["sig"], ax.get_ylim()[1] * 0.72),
                arrowprops=dict(arrowstyle="<->", color="tab:blue", lw=2.0))
    ax.text(m0 + s["sig"] / 2, ax.get_ylim()[1] * 0.76,
            f"the data's own noise\nsigma = {s['sig']*1000:.0f} m/s", ha="center",
            fontsize=9, color="tab:blue")
    ax.set_xlabel("misfit [km/s]"); ax.set_ylabel("posterior samples")
    ax.set_title(f"cell {LAB[cc]} — the rejection tolerance against\n"
                 f"the noise level the sampler itself estimated", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")

    ax = axs[2]
    x = np.arange(len(stats))
    ax.bar(x - 0.2, [q["dm"] * 1000 for q in stats], 0.4, color="tab:red",
           label="dev=0.05 tolerance")
    ax.bar(x + 0.2, [q["sig"] * 1000 for q in stats], 0.4, color="tab:blue",
           label="sigma estimated by the sampler")
    for i, q in enumerate(stats):
        ax.text(i - 0.2, q["dm"] * 1000, f"{100*q['dm']/q['sig']:.0f}%", ha="center",
                va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([q["lab"] for q in stats], fontsize=8)
    ax.set_ylabel("velocity [m/s]")
    ax.set_title("chains are separated at 6-10% of the\nuncertainty of the data they fit",
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    print("\n  tolerance translated into fit quality:")
    for q in stats:
        print(f"  {q['lab']}: |dlogL| {q['tol']:.2f} = {q['dm']*1000:.1f} m/s of extra RMS "
              f"misfit = {100*q['dm']/q['mis']:.1f}% of the median misfit, "
              f"{100*q['dm']/q['sig']:.0f}% of sigma, {100*q['dm']/q['iqr']:.0f}% of the "
              f"posterior misfit IQR")
    fig.suptitle("2. What the tolerance means in practice — chains are rejected for fit "
                 "differences far below the data's own noise", fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out, "dev_2_misfit.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); print("wrote", p)


def fig3(z, out):
    """Apply the identical criterion to our own converged 24-chain runs."""
    rows = []
    for tag in SEC:
        for f in sorted(glob.glob(f"{VS}/{tag}/cell_*/bayhunter_result.npz")):
            y = np.load(f, allow_pickle=True)
            cm = np.asarray(y["chain_loglike_med"], float)
            rej = bayhunter_outliers(cm)
            rows.append(dict(n=cm.size, rej=int(rej.sum()),
                             dis=float(y["chain_disagree"]),
                             kept_ours=int(y["n_chains_kept"]),
                             spread=float(np.nanmax(cm) - np.nanmin(cm)),
                             tol=DEV * abs(np.nanmax(cm))))
    if not rows:
        print("no section runs found"); return
    n = rows[0]["n"]
    rej = np.array([r["rej"] for r in rows])
    print(f"\n  our runs: {len(rows)} cells x {n} chains")
    print(f"  BayHunter dev=0.05 would reject {rej.mean():.1f} of {n} "
          f"({100*rej.mean()/n:.0f}%), leaving {n - rej.mean():.1f}")
    print(f"  our own convergence gate keeps "
          f"{np.mean([r['kept_ours'] for r in rows]):.1f} of {n}")
    print(f"  chain_disagree(Vs) median {np.median([r['dis'] for r in rows]):.3f} "
          f"(< 0.3 = chains agree on velocity)")

    fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.0))
    ax = axs[0]
    ax.hist(n - rej, bins=np.arange(-0.5, n + 1.5), color="tab:purple", alpha=0.85,
            label=f"our runs under dev=0.05\n(mean {n-rej.mean():.1f} of {n})")
    ax.hist(z["kept"], bins=np.arange(-0.5, NCH_MS + 1.5), color="tab:red", alpha=0.55,
            weights=np.full(z["kept"].size, len(rows) / z["kept"].size),
            label=f"manuscript run\n(mean {z['kept'].mean():.1f} of {NCH_MS})")
    ax.set_xlabel("chains surviving the criterion"); ax.set_ylabel("cells (rescaled)")
    ax.set_title("the criterion rejects most chains\neven when they agree", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axs[1]
    ax.scatter([r["dis"] for r in rows], [n - r["rej"] for r in rows], s=22,
               color="tab:purple")
    ax.axvline(0.3, color="k", ls="--", lw=1.2)
    ax.text(0.305, n * 0.95, "chains disagree on Vs →", fontsize=8)
    ax.set_xlabel("chain_disagree(Vs)  [mean |dVs| between chain medians, km/s]")
    ax.set_ylabel(f"chains surviving dev=0.05 (of {n})")
    rr = np.corrcoef([r["dis"] for r in rows], [n - r["rej"] for r in rows])[0, 1]
    ax.set_title(f"no relationship with actual Vs agreement\nbetween chains (corr {rr:+.2f})",
                 fontsize=10)
    print(f"  corr(chain_disagree, chains surviving dev=0.05) = {rr:+.3f}")
    ax.grid(alpha=0.3)

    ax = axs[2]
    d = z["dev"]
    ax.hist(np.clip(d, 1e-3, 100), bins=np.logspace(-1.5, 2, 70), color="tab:red", alpha=0.85)
    ax.axvline(DEV, color="k", lw=2.0, label="dev = 0.05 (the cut)")
    ax.axvline(1.0, color="tab:green", lw=2.0, ls="--",
               label="dev = 1 (an order of magnitude)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("stored deviation of the rejected chain")
    ax.set_ylabel("rejected chains")
    ax.set_title(f"manuscript run: {100*np.mean(d < 0.5):.1f}% of the {d.size:,} rejected\n"
                 f"chains are within 50% of the best", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.suptitle("3. The same criterion applied to a run whose chains demonstrably agree",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out, "dev_3_control.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); print("wrote", p)


def fig4(z, out):
    """Consequences: spatial pattern and imprint on the model."""
    import re
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.0))
    ax = axs[0]
    G = np.full((85, 53), np.nan)
    G[z["ix"], z["iy"]] = z["kept"]
    im = ax.pcolormesh(np.arange(86) * 0.5, np.arange(54) * 0.5, G.T, cmap="inferno",
                       vmin=0, vmax=30)
    ax.set_aspect("equal"); ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]")
    ax.set_title("chains kept, per cell", fontsize=10)
    plt.colorbar(im, ax=ax, label="chains kept of 30")

    # neighbour difference: if the criterion were tracking real convergence it would vary
    # smoothly; salt-and-pepper means it is not a property of the data
    ax = axs[1]
    dif = []
    K = {(i, j): k for i, j, k in zip(z["ix"], z["iy"], z["kept"])}
    for (i, j), k in K.items():
        for dd in ((1, 0), (0, 1)):
            n = K.get((i + dd[0], j + dd[1]))
            if n is not None:
                dif.append(abs(k - n))
    dif = np.array(dif)
    ax.hist(dif, bins=np.arange(-0.5, 26), color="tab:red", alpha=0.85)
    ax.set_xlabel("|difference in chains kept| between adjacent cells")
    ax.set_ylabel("cell pairs")
    ax.set_title(f"neighbouring cells differ by a median of {np.median(dif):.0f} chains\n"
                 f"({100*np.mean(dif >= 5):.0f}% differ by 5 or more)", fontsize=10)
    ax.grid(alpha=0.3)

    # does it imprint on the model?
    ax = axs[2]
    kept, vmax = [], []
    for i, j, k in zip(z["ix"], z["iy"], z["kept"]):
        f = f"{MS}/model_results_disp_2Dmap_ZZ_{0.5*i:.2f}x_{0.5*j:.2f}y.dat.txt"
        if not os.path.exists(f):
            continue
        a = np.loadtxt(f, skiprows=1)
        kept.append(k); vmax.append(np.nanmax(a[:, 2]))
    kept = np.array(kept); vmax = np.array(vmax)
    ax.scatter(kept, vmax, s=5, alpha=0.30, color="tab:red")
    b = np.arange(0, 31, 3)
    ctr, med = [], []
    for lo, hi in zip(b[:-1], b[1:]):
        m = (kept >= lo) & (kept < hi)
        if m.sum() > 10:
            ctr.append((lo + hi) / 2); med.append(np.median(vmax[m]))
    ax.plot(ctr, med, "o-", color="k", lw=2.0, ms=6, label="median")
    ax.axhline(5.0, color="tab:blue", ls="--", lw=1.5, label="Vs prior ceiling")
    ax.set_xlabel("chains kept of 30"); ax.set_ylabel("max median Vs in the cell [km/s]")
    ax.set_title(f"cells that kept fewer chains are faster\n"
                 f"(corr {np.corrcoef(kept, vmax)[0,1]:+.3f})", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print(f"\n  chains kept vs max median Vs: corr {np.corrcoef(kept, vmax)[0,1]:+.3f} "
          f"over {kept.size} cells")
    for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 31)):
        m = (kept >= lo) & (kept < hi)
        if m.sum():
            print(f"    {lo:2d}-{hi:2d} chains kept ({m.sum():4d} cells): "
                  f"median max Vs {np.median(vmax[m]):.2f} km/s, "
                  f"{100*np.mean(vmax[m] > 3.5):.0f}% above 3.5")
    fig.suptitle("4. Consequences — the number of surviving chains varies cell to cell, and "
                 "the model follows it", fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out, "dev_4_consequences.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); print("wrote", p)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=MB)
    a = ap.parse_args()
    z = np.load(f"{MB}/ivan_evidence.npz")
    fig1(z, a.out)
    fig2(z, a.out)
    fig3(z, a.out)
    fig4(z, a.out)


if __name__ == "__main__":
    main()
