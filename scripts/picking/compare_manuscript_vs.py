#!/usr/bin/env python3
"""Manuscript BayHunter Vs against BOTH of our arms, along the exact profiles, at true scale.

Ivan's run (yggdrasil /srv/beegfs/.../IVAN/Haute-Sorne/Rayleigh) uses the SAME grid as ours --
its files are disp_2Dmap_ZZ_{x}x_{y}y.dat with x = 0.5*ix km and y = 0.5*iy km -- so cells pair
exactly and no interpolation is needed. Its per-cell output (models2/) carries Depth, Vs_mean,
Vs_median, std_min, std_max, vs_min, vs_max and disc_prob on a 0-6 km / 0.1 km grid.

Five panels per profile, all drawn in elevation with `set_aspect("equal")` so there is no
vertical exaggeration:

    manuscript  |  ours untrimmed  |  ours trimmed  |  untrimmed - manuscript  |  trimmed - manuscript

Showing both of our arms matters because they bracket the question: the untrimmed arm uses the
SAME period band the manuscript did, so any difference there is down to the inversion setup
alone (chain selection, depth box, prior, weighting); the trimmed arm additionally removes the
periods where the maps are resolution-collapsed, which is the change we would actually
recommend. The manuscript panel simply stops at 6 km -- that is its `z = 0, 6` box, not missing
data, and it is left blank below so the limitation stays visible.

Geometry, cell selection and topography all come from section_figure.py, so this figure and the
section figures are guaranteed to be sampling the identical cells.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from section_figure import SECTIONS, load, sample_line, topo_of, TESTS  # noqa: E402

EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
MB = f"{EHM}/hautesorne/tomo/2_vs_depth_inversion/manuscript_comparison_2026-08/0_manuscript_run"
MS = f"{MB}/models2"
MS_ZMAX = 6.0          # their config.ini: z = 0, 6
PRIOR_MAX = 5.0        # their config.ini: vs = 0.5, 5.0
HEAD = 0.9
TICK_KM = 2.0


def load_ms(ix, iy):
    f = f"{MS}/model_results_disp_2Dmap_ZZ_{0.5*ix:.2f}x_{0.5*iy:.2f}y.dat.txt"
    if not os.path.exists(f):
        return None
    a = np.loadtxt(f, skiprows=1)
    return dict(z=a[:, 0], med=a[:, 2], vmax=a[:, 6])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--section", required=True, choices=list(SECTIONS))
    ap.add_argument("--out", default=MB)
    a = ap.parse_args()
    lend, rend, tag, P, Q = SECTIONS[a.section]

    A = load(tag)
    B = load(tag + "_trim")
    if not A or not B:
        raise SystemExit(f"missing runs for {a.section} (untrimmed {len(A)}, trimmed {len(B)})")
    keep = {(r["ix"], r["iy"]) for r in A} & {(r["ix"], r["iy"]) for r in B}
    A = [r for r in A if (r["ix"], r["iy"]) in keep]
    byk = {(r["ix"], r["iy"]): r for r in B}
    idx, t, perp, u, L = sample_line(A, P, Q)
    A = [A[i] for i in idx]
    o = np.argsort(t); t = t[o]
    A = [A[i] for i in o]
    B = [byk[(r["ix"], r["iy"])] for r in A]
    M = [load_ms(r["ix"], r["iy"]) for r in A]
    nmiss = sum(m is None for m in M)
    topo = topo_of(A) / 1000.0
    print(f"{a.section}: {len(A)} cells on the exact profile ({L:.2f} km, "
          f"bearing {np.degrees(np.arctan2(u[0], u[1]))%360:.2f} deg); "
          f"manuscript model missing at {nmiss}")

    zmax = float(A[0]["depth"].max())
    zel = np.linspace(topo.max() + HEAD, topo.min() - zmax, 300)
    zle = np.concatenate([[zel[0]+(zel[0]-zel[1])/2], 0.5*(zel[:-1]+zel[1:]),
                          [zel[-1]-(zel[0]-zel[1])/2]])
    te = np.concatenate([[t[0]-0.25], 0.5*(t[:-1]+t[1:]), [t[-1]+0.25]])

    def grid_of(S, zkey, vkey):
        G = np.full((len(zel), len(S)), np.nan)
        for i, r in enumerate(S):
            if r is None:
                continue
            ce = topo[i] - np.asarray(r[zkey], float)
            G[:, i] = np.interp(zel, ce[::-1], np.asarray(r[vkey], float)[::-1],
                                left=np.nan, right=np.nan)
        return G

    GM = grid_of(M, "z", "med")
    GA = grid_of(A, "depth", "vs")
    GB = grid_of(B, "depth", "vs")

    k = np.isfinite(GM) & np.isfinite(GA)
    print(f"  over the manuscript's own 0-{MS_ZMAX:.0f} km box:")
    print(f"    mean |dVs| untrimmed-manuscript {np.nanmean(np.abs(GA-GM)[k]):.3f} km/s "
          f"(bias {np.nanmean((GA-GM)[k]):+.3f})")
    kb = np.isfinite(GM) & np.isfinite(GB)
    print(f"    mean |dVs| trimmed-manuscript   {np.nanmean(np.abs(GB-GM)[kb]):.3f} km/s "
          f"(bias {np.nanmean((GB-GM)[kb]):+.3f})")
    print(f"    max Vs: manuscript {np.nanmax(GM):.2f}  untrimmed {np.nanmax(GA):.2f}  "
          f"trimmed {np.nanmax(GB):.2f}   (their prior ceiling {PRIOR_MAX})")
    rail = np.nanmean(np.nanmax(GM, axis=0) > 0.98 * PRIOR_MAX) * 100
    print(f"    columns whose manuscript median reaches 98% of the prior ceiling: {rail:.0f}%")

    span_x = te[-1] - te[0]; span_z = zel[0] - zel[-1]
    fig_w = 15.0
    axw = fig_w * 0.82 - 0.9
    ph = axw * span_z / span_x
    fig, axs = plt.subplots(5, 1, figsize=(fig_w, 5*ph + 4.4),
                            gridspec_kw={"height_ratios": [ph]*5})
    panels = [
        (GM, f"manuscript (Ivan) — 6 km box, {PRIOR_MAX:.0f} km/s prior, mean 9.5 of 30 chains",
         "RdYlBu", 1.5, 4.0, "Vs [km/s]"),
        (GA, "current workflow, SAME period band (T<=8.61 s) — 8 km box, all 24 chains",
         "RdYlBu", 1.5, 4.0, "Vs [km/s]"),
        (GB, "current workflow, res_diag-trimmed (T<=5.12 s)",
         "RdYlBu", 1.5, 4.0, "Vs [km/s]"),
        (GA - GM, "same band, ours − manuscript  (red = ours faster)", "bwr", -1.2, 1.2,
         "dVs [km/s]"),
        (GB - GM, "trimmed, ours − manuscript", "bwr", -1.2, 1.2, "dVs [km/s]"),
    ]
    for ax, (G, ttl, cm, vlo, vhi, lb) in zip(axs, panels):
        im = ax.pcolormesh(te, zle, G, cmap=cm, vmin=vlo, vmax=vhi, shading="flat")
        ax.plot(t, topo, "-", color="k", lw=1.1)
        # the base of their model box, so the reader can see where it stops
        ax.plot(t, topo - MS_ZMAX, "--", color="0.35", lw=1.0)
        ax.set_title(ttl, fontsize=10.5, fontweight="bold")
        ax.set_ylabel("elevation [km a.s.l.]")
        ax.set_xlim(te[0], te[-1]); ax.set_aspect("equal")
        # equal tick interval on both axes so the 1:1 scale is visible at a glance
        ax.xaxis.set_major_locator(MultipleLocator(TICK_KM))
        ax.yaxis.set_major_locator(MultipleLocator(TICK_KM))
        plt.colorbar(im, ax=ax, label=lb, fraction=0.026, pad=0.012)
        for xf, lbl, ha in ((0.0, lend, "left"), (1.0, rend, "right")):
            ax.annotate(lbl, (xf, 0.0), xycoords="axes fraction",
                        xytext=(8 if ha == "left" else -8, 8), textcoords="offset points",
                        ha=ha, va="bottom", fontsize=12, fontweight="bold",
                        bbox=dict(fc="w", alpha=0.85, ec="k", lw=0.5, pad=2))
    axs[-1].set_xlabel(f"distance along profile {lend}–{rend} [km]")
    fig.suptitle(f"Haute-Sorne {lend}-{rend} — manuscript inversion vs the current workflow "
                 f"(true scale, no vertical exaggeration)\n"
                 f"dashed line = base of the manuscript's 6 km model box",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out = os.path.join(a.out, f"{a.section}_manuscript_vs_current.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
