#!/usr/bin/env python3
"""Vs and interface-probability cross-sections for ANY profile orientation (A-A', B-B', C-C').

Generalises bb_section_figure.py / bb_section_interfaces.py, which assumed an east-west row
and used Easting as the horizontal axis. A-A' is north-south and C-C' is oblique (149 deg), so
the axis has to be DISTANCE ALONG THE PROFILE and the cells have to be ordered by projection
onto the profile direction rather than by a grid index.

Profile geometry comes from the EXACT LV95 endpoints in `SECTIONS` (see that table). The line is
walked at the 0.5 km grid spacing, the nearest cell centre is taken at each stop, and each kept
cell's along-coordinate is its own projection onto the segment. Cells outside the segment are
not drawn, so the figure spans the profile as defined rather than as far as cells happen to
exist.

Two renderings per profile, both with the reliability/prior caveats visible:
  --mode vs         Vs(z) untrimmed vs res_diag-trimmed + difference + %models>3.5 km/s
  --mode interface  ensemble interface probability P(z), same layout

Usage:
  python section_figure.py --section AA --mode vs
  python section_figure.py --section CC --mode interface
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pyproj import Transformer

EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
DEM = f"{EHM}/hautesorne/tomo/2_vs_depth_inversion/fig_assets_hautesorne_dem.npz"
TESTS = f"{EHM}/hautesorne/tomo/2_vs_depth_inversion/manuscript_comparison_2026-08/1_sections"
DZ = 0.05
# swisstopo "Deep wells" record for Glovelier-1 (= GVL-1), verified 2026-08-16:
#   E 2583491.08  N 1242491.79 (LV95); ground 494.2 m a.s.l.; 4041.5 m MD / 4005.9 m TVD.
# TVD is what the section's depth axis means, so the drawn stick uses TVD.
WELLS = [("GVL-1", 2583491.08, 1242491.79, 494.2, 4005.9)]
WELL_NEAR_KM = 1.5      # draw the well on a section passing within this distance
# name_left, name_right, tag, and the EXACT LV95 endpoints (start, end) of the profile, supplied
# by GS 2026-08-16. These are the reference geometry: do not re-derive them. Earlier versions
# took the axis from a PCA of the inverted cells with the sense pinned by hand, and read C-C''s
# start off Fig 8 -- that put C ~500 m too far west and the bearing ~1 deg out, and left B-B'
# covering only part of its true length.
SECTIONS = {
    "AA": ("A (N)", "A' (S)", "test_2026-08-08_AA_section_R0g",
           (2583493, 1254324), (2583493, 1237330)),
    "BB": ("B (W)", "B' (E)", "test_2026-08-08_BB_section_R0g",
           (2572477, 1242491), (2595079, 1242491)),
    "CC": ("C (NW)", "C' (SE)", "test_2026-08-16_CC_section_R0g",
           (2574985, 1256138), (2587558, 1235971)),
}
STEP_M = 500.0          # sample the line at the grid spacing, one cell per step
TICK_KM = 2.0           # same tick interval on both axes: the panels are 1:1


def load(tag, want_iface=False):
    from swtomotv.config import DatasetConfig
    from swtomotv.geometry import make_grid, xy2ll
    ds = DatasetConfig.from_yaml(glob.glob("/Users/genevievesavard/Codes/NoisePy-ant/"
                                           "param_files/cluster/tomo/"
                                           "hautesorne_tspws_group_scaled_lccov.yaml")[0])
    grid = make_grid(ds.bounds, 0.5)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    out = []
    for f in sorted(glob.glob(f"{TESTS}/{tag}/cell_*/bayhunter_result.npz")):
        b = os.path.basename(os.path.dirname(f)).split("_")
        ix, iy = int(b[1]), int(b[2])
        z = np.load(f, allow_pickle=True)
        lat, lon = xy2ll(np.array([grid.x[ix]]), np.array([grid.y[iy]]), *grid.origin)
        e, n = tr.transform(lon[0], lat[0])
        r = dict(ix=ix, iy=iy, E=e, N=n, lon=lon[0], lat=lat[0],
                 depth=z["depth"], vs=z["vs_median"])
        E = np.asarray(z["ens_vs"], float)
        m = (z["depth"] >= 2.24) & (z["depth"] <= 4.5)
        r["frac_hi"] = float(np.mean(E[:, m].max(axis=1) > 3.5))
        if want_iface and "iface_depths" in z:
            edges = np.arange(0, float(z["depth"].max()) + DZ, DZ)
            h, _ = np.histogram(np.asarray(z["iface_depths"], float), bins=edges)
            r["P"] = h / max(h.sum(), 1)
            r["ctr"] = 0.5 * (edges[:-1] + edges[1:])
        out.append(r)
    return out


def sample_line(recs, P, Q, step=STEP_M):
    """Pick the cells the exact profile passes through, and their distance along it.

    Walks the segment P->Q at `step` and takes the nearest cell centre at each stop, dropping
    consecutive repeats. Selecting instead by "within half a cell of the line" double-samples an
    oblique profile -- on C-C' it returned two cells at almost the same along-distance for half
    the stops, which would render as duplicated columns.

    Each kept cell's along-coordinate is its OWN projection onto the segment, not the stop that
    found it, so the column spacing stays honest. Distances come back in km (LV95 is in metres).
    """
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    d = Q - P
    L = float(np.hypot(*d)); u = d / L
    pos = np.array([[r["E"], r["N"]] for r in recs], float)
    picked, seen = [], None
    for s in np.arange(0.0, L + 1e-6, step):
        p = P + u * s
        k = int(np.argmin(((pos - p) ** 2).sum(axis=1)))
        if k != seen:
            picked.append(k); seen = k
    v = pos[picked] - P
    t = v @ u
    perp = np.abs(v[:, 0] * u[1] - v[:, 1] * u[0])
    return picked, t / 1000.0, perp, u, L / 1000.0


def topo_of(recs):
    dem = np.load(DEM, allow_pickle=True)
    elev, ext = dem["elev"].astype(float), dem["extent"]
    ny, nx = elev.shape
    out = []
    for r in recs:
        fx = (r["lon"] - ext[0]) / (ext[1] - ext[0]) * (nx - 1)
        fy = (r["lat"] - ext[3]) / (ext[2] - ext[3]) * (ny - 1)
        if not (0 <= fx < nx - 1 and 0 <= fy < ny - 1):
            out.append(np.nan); continue
        i0, j0 = int(fy), int(fx); a, b = fy - i0, fx - j0
        out.append(elev[i0, j0]*(1-a)*(1-b) + elev[i0+1, j0]*a*(1-b)
                   + elev[i0, j0+1]*(1-a)*b + elev[i0+1, j0+1]*a*b)
    t = np.array(out, float)
    if not np.isfinite(t).all():                 # boundless DEM -> NaN outside the tile
        g = np.isfinite(t)
        t = np.interp(np.arange(len(t)), np.flatnonzero(g), t[g]) if g.sum() >= 2 \
            else np.full(len(t), 500.0)
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--section", required=True, choices=list(SECTIONS))
    ap.add_argument("--mode", default="vs", choices=("vs", "interface"))
    a = ap.parse_args()
    lend, rend, tag, P, Q = SECTIONS[a.section]
    A = load(tag, want_iface=(a.mode == "interface"))
    B = load(tag + "_trim", want_iface=(a.mode == "interface"))
    if not A or not B:
        raise SystemExit(f"missing runs for {a.section} (untrimmed {len(A)}, trimmed {len(B)})")
    keep = {(r["ix"], r["iy"]) for r in A} & {(r["ix"], r["iy"]) for r in B}
    A = [r for r in A if (r["ix"], r["iy"]) in keep]
    B = [r for r in B if (r["ix"], r["iy"]) in keep]
    byk = {(r["ix"], r["iy"]): r for r in B}
    idx, t, perp, u, L = sample_line(A, P, Q)
    A = [A[i] for i in idx]
    B = [byk[(r["ix"], r["iy"])] for r in A]
    o = np.argsort(t)
    t = t[o]; perp = perp[o]
    A = [A[i] for i in o]; B = [B[i] for i in o]
    topo = topo_of(A)
    print(f"{a.section}: {len(A)} cells on the exact profile "
          f"({L:.2f} km, bearing {np.degrees(np.arctan2(u[0], u[1]))%360:.2f} deg); "
          f"cells lie {np.median(perp):.0f} m from the line (max {perp.max():.0f} m)")
    off = len(keep) - len(A)
    if off:
        print(f"  {off} inverted cells fall outside the exact segment and are not drawn")

    te = np.concatenate([[t[0]-0.25], 0.5*(t[:-1]+t[1:]), [t[-1]+0.25]])
    dep = A[0]["depth"] if a.mode == "vs" else A[0]["ctr"]
    # Everything vertical is in KILOMETRES so the section can be drawn at true scale: the
    # horizontal axis is km along the profile, so an equal aspect needs the same unit on both.
    topo = topo / 1000.0
    # headroom above the highest ground so the well label has somewhere to sit that is not on
    # top of the panel title
    HEAD = 0.9
    zel = np.linspace(topo.max() + HEAD, topo.min() - dep.max(), 300)
    zle = np.concatenate([[zel[0]+(zel[0]-zel[1])/2], 0.5*(zel[:-1]+zel[1:]),
                          [zel[-1]-(zel[0]-zel[1])/2]])
    key = "vs" if a.mode == "vs" else "P"

    def to_elev(S):
        M = np.full((len(zel), len(S)), np.nan)
        for i, r in enumerate(S):
            ce = topo[i] - (r["depth"] if a.mode == "vs" else r["ctr"])
            M[:, i] = np.interp(zel, ce[::-1], r[key][::-1], left=np.nan, right=np.nan)
        return M

    MA, MB = to_elev(A), to_elev(B)
    # size the figure so the true-scale panels fill it instead of leaving bands of whitespace
    span_x = te[-1] - te[0]
    span_z = zel[0] - zel[-1]
    fig_w = 15.0
    axw = fig_w * 0.82 - 0.9                      # usable axes width after margins + colorbar
    panel_h = axw * span_z / span_x               # true-scale height of one section panel
    prof_h = 1.5
    fig, axs = plt.subplots(4, 1, figsize=(fig_w, 3*panel_h + prof_h + 3.2),
                            gridspec_kw={"height_ratios": [panel_h]*3 + [prof_h]})
    if a.mode == "vs":
        cmap, vmin, vmax, lab = "RdYlBu", 1.5, 4.0, "Vs [km/s]"
    else:
        deep = dep >= 0.5
        cmap, vmin = "inferno", 0.0
        vmax = np.nanpercentile(np.concatenate([r["P"][deep] for r in A+B]), 99)
        lab = f"P(interface) / {DZ*1000:.0f} m bin"
    for k, (M, ttl) in enumerate(((MA, "v1 band (T<=8.61 s) — manuscript input config"),
                                  (MB, "res_diag-trimmed (T<=5.12 s)"))):
        ax = axs[k]
        im = ax.pcolormesh(te, zle, M, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
        ax.plot(t, topo, "-", color="k" if a.mode == "vs" else "w", lw=1.2)
        ax.set_title(f"{a.section} — R0g, {ttl}", fontsize=11, fontweight="bold")
        ax.set_ylabel("elevation [km a.s.l.]"); ax.set_xlim(te[0], te[-1])
        ax.set_aspect("equal")
        # the panels are 1:1, so tick BOTH axes at the same interval -- with 5 km on x and 2 km
        # on y the labelled steps have very different lengths and the section reads as though it
        # were vertically exaggerated when it is not
        ax.xaxis.set_major_locator(MultipleLocator(TICK_KM))
        ax.yaxis.set_major_locator(MultipleLocator(TICK_KM))
        plt.colorbar(im, ax=ax, label=lab, fraction=0.026, pad=0.012)
        for xf, lb, ha in ((0.0, lend, "left"), (1.0, rend, "right")):
            ax.annotate(lb, (xf, 0.0), xycoords="axes fraction",
                        xytext=(8 if ha == "left" else -8, 8), textcoords="offset points",
                        ha=ha, va="bottom", fontsize=13, fontweight="bold",
                        bbox=dict(fc="w", alpha=0.85, ec="k", lw=0.5, pad=2))
    ax = axs[2]
    D = MB - MA
    m = np.nanpercentile(np.abs(D[np.isfinite(D)]), 99) if np.isfinite(D).any() else 1
    im = ax.pcolormesh(te, zle, D, cmap="bwr", vmin=-m, vmax=m, shading="flat")
    ax.set_title("trimmed − untrimmed", fontsize=11)
    ax.set_ylabel("elevation [km a.s.l.]"); ax.set_xlim(te[0], te[-1])
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(MultipleLocator(TICK_KM))
    ax.yaxis.set_major_locator(MultipleLocator(TICK_KM))
    plt.colorbar(im, ax=ax, label="difference", fraction=0.026, pad=0.012)
    # --- wells projected onto the profile -------------------------------------------------
    P = np.array([[r["E"], r["N"]] for r in A], float)
    for wnm, wE, wN, wZ, wTD in WELLS:
        dd = np.hypot(P[:, 0] - wE, P[:, 1] - wN)
        i = int(np.argmin(dd))
        if dd[i] / 1000.0 > WELL_NEAR_KM:
            continue
        for ax in axs[:3]:
            ax.plot([t[i], t[i]], [wZ/1000.0, (wZ - wTD)/1000.0], "-", color="k", lw=2.6,
                    zorder=8)
            ax.plot([t[i]], [wZ/1000.0], "v", mfc="k", mec="w", ms=9, zorder=9)
            ax.annotate(f"{wnm}\n{dd[i]/1000:.1f} km off-line", (t[i], wZ/1000.0),
                        xytext=(0, 12), textcoords="offset points", ha="center",
                        fontsize=8.5, fontweight="bold", zorder=10,
                        bbox=dict(fc="w", alpha=0.85, ec="k", lw=0.5, pad=1.5))
        print(f"  {wnm}: at {t[i]:.2f} km along, {dd[i]/1000:.2f} km off the line")

    ax = axs[3]
    ax.plot(t, [100*r["frac_hi"] for r in A], "o-", color="tab:purple", label="untrimmed")
    ax.plot(t, [100*r["frac_hi"] for r in B], "o-", color="tab:green", label="trimmed")
    ax.set_ylabel("% of models > 3.5 km/s\nat 2–4.5 km")
    ax.set_xlabel(f"distance along profile {lend}–{rend} [km]")
    ax.grid(alpha=0.3); ax.legend(fontsize=9); ax.set_xlim(te[0], te[-1])
    fig.suptitle(f"Haute-Sorne {lend}-{rend} section — R0g, "
                 f"{'Vs' if a.mode == 'vs' else 'interface probability'}",
                 fontsize=13, fontweight="bold")
    axs[3].set_aspect("auto")
    fig.tight_layout(rect=(0, 0, 1, 0.975))     # leave the suptitle clear of panel 1's title
    out = f"{TESTS}/{tag}/{a.section}_section_{a.mode}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
