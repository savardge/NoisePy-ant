#!/usr/bin/env python3
"""Are the manuscript's deep high-Vs bodies (a) fed by the S9 group-velocity outliers, or
(b) an artefact of the S10 chain-basin problem?

Two mechanisms, tested separately, on the CURRENT workflow's data.

A. THE S9 TAIL. SI Figure S9 histograms the Vg-map-derived dispersion curve of every covered
   cell; a plume of cells reaches 3.0-4.0 km/s group velocity at 2.8-4.2 s, far above the 95th
   percentile (~2.9). A cell whose observed U is 3.5 km/s at 3.5 s CANNOT be fit without a
   fast deep layer -- so if the deep high-Vs bodies sit on those cells, they are a faithful
   inversion of an outlier MEASUREMENT rather than an inversion pathology. This part rebuilds
   S9 from the prod3-k2 blanket maps, locates the outlier cells in space, and asks whether
   they carry low res_diag (i.e. whether the outlier is a resolution artefact of the map).

B. THE S10 BASINS. S10 reports 4-5 valid chains out of 30 and row (c) is visibly bimodal in
   both misfit and predicted curves. If the deep fast structure lives in a MINORITY of chains,
   the posterior median is a mixture and the body is a sampling artefact. Our runs store
   `chain_vs_profiles` (per-chain posterior median Vs), so the question is answerable directly:
   count how many of the 24 chains put >3.5 km/s in the 2.24-4.5 km window, and compare that
   with the ensemble fraction that the section figures plot.

The two are distinguishable. Data-driven (A): most chains agree, and the cell's own observed
curve is fast. Sampler-driven (B): the ensemble fraction is carried by a few chains while the
observed curve is unremarkable.
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer

EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
VS = f"{EHM}/hautesorne/tomo/2_vs_depth_inversion/manuscript_comparison_2026-08/1_sections"
MAPS = (f"{EHM}/hautesorne/tomo/1_velocity_maps/1_production/"
        f"tspws_group_blanket_dx0.5_prod3_k2/production/fund")
YAML = ("/Users/genevievesavard/Codes/NoisePy-ant/param_files/cluster/tomo/"
        "hautesorne_tspws_group_blanket.yaml")
# The S9 plume: the period window it occupies and the velocity it exceeds.
TBAND = (2.8, 4.2)
UHI = 3.5
# The depth window the section figures use for "% models > 3.5 km/s" (drilled basement to the
# bottom of the reliable band at GVL-1).
ZWIN = (2.24, 4.5)
SECTIONS = {"AA": "test_2026-08-08_AA_section_R0g",
            "BB": "test_2026-08-08_BB_section_R0g",
            "CC": "test_2026-08-16_CC_section_R0g"}


def cell_lv95(ix, iy, grid, tr):
    from swtomotv.geometry import xy2ll
    lat, lon = xy2ll(np.array([grid.x[ix]]), np.array([grid.y[iy]]), *grid.origin)
    return tr.transform(lon[0], lat[0])


def load_maps():
    """period -> (vel, mask, res_diag, unc_s), ascending in period."""
    out = {}
    for f in sorted(glob.glob(f"{MAPS}/map_T*.npz")):
        z = np.load(f, allow_pickle=True)
        out[float(z["period"])] = dict(vel=z["vel"], mask=z["mask"].astype(bool),
                                       res=z["res_diag"], unc=z["unc_s"])
    return dict(sorted(out.items()))


def part_a(M, grid, tr, outdir):
    T = np.array(list(M))
    # map arrays are indexed [ix, iy] -- grid.x is the long axis (85), grid.y 53.
    nx, ny = M[T[0]]["vel"].shape
    V = np.stack([M[t]["vel"] for t in T])          # (nT, nx, ny)
    K = np.stack([M[t]["mask"] for t in T])
    R = np.stack([M[t]["res"] for t in T])
    V = np.where(K, V, np.nan)

    # S9 clone -------------------------------------------------------------------------------
    covered = np.isfinite(V).any(axis=0)
    print(f"[A] {covered.sum()} covered cells (manuscript S9: 1126)")
    tb = (T >= TBAND[0]) & (T <= TBAND[1])
    band = V[tb]                                     # (nTb, ny, nx)
    hot = np.nanmax(band, axis=0)                    # per-cell max U inside the plume window
    nhot = int(np.nansum(hot > UHI))
    print(f"[A] cells with U > {UHI} km/s anywhere in {TBAND[0]}-{TBAND[1]} s: "
          f"{nhot}  ({100*nhot/max(covered.sum(),1):.2f}% of covered)")
    for thr in (3.0, 3.25, 3.5, 3.75):
        print(f"       U > {thr:.2f}: {int(np.nansum(hot > thr))} cells")

    fig = plt.figure(figsize=(16, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.0], wspace=0.26)

    ax = fig.add_subplot(gs[0])
    ve = np.arange(1.2, 4.35, 0.05)
    te = np.concatenate([[T[0] - 0.05], 0.5 * (T[:-1] + T[1:]), [T[-1] + 0.2]])
    H = np.zeros((len(ve) - 1, len(T)))
    for j in range(len(T)):
        v = V[j][np.isfinite(V[j])]
        H[:, j], _ = np.histogram(v, bins=ve)
    H[H == 0] = np.nan
    im = ax.pcolormesh(te, ve, H, cmap="inferno",
                       norm=matplotlib.colors.LogNorm(vmin=1, vmax=np.nanmax(H)))
    for q, st, lw in ((50, "-", 2.2), (25, "-", 1.0), (75, "-", 1.0),
                      (5, ":", 1.0), (95, ":", 1.0)):
        ax.plot(T, [np.nanpercentile(V[j], q) for j in range(len(T))], st,
                color="cyan", lw=lw)
    ax.axhline(UHI, color="w", ls="--", lw=1.3)
    ax.axvspan(*TBAND, color="w", alpha=0.12)
    ax.set_xlim(0.4, 6.5); ax.set_ylim(1.2, 4.3)
    ax.set_xlabel("period [s]"); ax.set_ylabel("Rayleigh group velocity [km/s]")
    ax.set_title(f"S9 clone — prod3 k2 blanket, {covered.sum()} covered cells\n"
                 f"cyan: median, IQR, 5-95%", fontsize=10)
    plt.colorbar(im, ax=ax, label="number of cells")

    # where are the outliers, and are they resolved? ------------------------------------------
    ax = fig.add_subplot(gs[1])
    E = np.full((nx, ny), np.nan); N = np.full((nx, ny), np.nan)
    for ix in range(nx):
        for iy in range(ny):
            if covered[ix, iy]:
                E[ix, iy], N[ix, iy] = cell_lv95(ix, iy, grid, tr)
    hm = np.where(covered, hot, np.nan)
    sc = ax.scatter(E / 1000, N / 1000, c=hm, s=9, cmap="inferno", vmin=2.0, vmax=4.0)
    o = covered & (hot > UHI)
    ax.scatter(E[o] / 1000, N[o] / 1000, s=34, facecolors="none", edgecolors="lime", lw=0.8)
    for sec, tag in SECTIONS.items():
        pts = []
        for f in sorted(glob.glob(f"{VS}/{tag}/cell_*/bayhunter_result.npz")):
            b = os.path.basename(os.path.dirname(f)).split("_")
            pts.append(cell_lv95(int(b[1]), int(b[2]), grid, tr))
        if pts:
            p = np.array(pts) / 1000
            ax.plot(p[:, 0], p[:, 1], "-", color="cyan", lw=1.6)
            ax.text(p[0, 0], p[0, 1], sec[0], color="cyan", fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlabel("Easting [km, LV95]"); ax.set_ylabel("Northing [km, LV95]")
    ax.set_title(f"max U in {TBAND[0]}-{TBAND[1]} s\ngreen ring = the S9 outlier population",
                 fontsize=10)
    plt.colorbar(sc, ax=ax, label="max U [km/s]")

    ax = fig.add_subplot(gs[2])
    rb = np.nanmax(np.where(np.isfinite(band), R[tb], np.nan), axis=0)
    ok = covered & np.isfinite(hot) & np.isfinite(rb)
    ax.scatter(rb[ok & ~o], hot[ok & ~o], s=6, color="0.6", label="rest")
    ax.scatter(rb[ok & o], hot[ok & o], s=14, color="tab:red", label=f"U > {UHI}")
    ax.axhline(UHI, color="k", ls="--", lw=1.0)
    ax.set_xlabel(f"max res_diag in {TBAND[0]}-{TBAND[1]} s")
    ax.set_ylabel("max U [km/s]")
    ax.set_title("is the outlier a RESOLUTION artefact?\n(low res_diag = prior-dominated)",
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    if o.sum():
        print(f"[A] outlier cells: median res_diag {np.nanmedian(rb[ok & o]):.3f} "
              f"vs {np.nanmedian(rb[ok & ~o]):.3f} for the rest")
    fig.suptitle("Mechanism A — the S9 group-velocity tail", fontsize=13, fontweight="bold")
    p = os.path.join(outdir, "S9_tail_check.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); print("wrote", p)
    return hot, covered, E, N


def part_b(outdir):
    """Per section cell: is the fast body carried by the DATA or by a few CHAINS?"""
    rows = []
    for sec, tag in SECTIONS.items():
        for f in sorted(glob.glob(f"{VS}/{tag}/cell_*/bayhunter_result.npz")):
            z = np.load(f, allow_pickle=True)
            b = os.path.basename(os.path.dirname(f)).split("_")
            d = z["depth"]; m = (d >= ZWIN[0]) & (d <= ZWIN[1])
            E = np.asarray(z["ens_vs"], float)
            frac = float(np.mean(E[:, m].max(axis=1) > UHI))
            C = np.asarray(z["chain_vs_profiles"], float)          # (nchain, nz)
            cf = float(np.mean(C[:, m].max(axis=1) > UHI))         # fraction of CHAINS
            T = np.asarray(z["obsT_fund"], float); U = np.asarray(z["obs_fund"], float)
            tb = (T >= TBAND[0]) & (T <= TBAND[1])
            rows.append(dict(sec=sec, ix=int(b[1]), iy=int(b[2]), frac=frac, chain_frac=cf,
                             umax=float(np.nanmax(U[tb])) if tb.any() else np.nan,
                             disagree=float(z["chain_disagree"]),
                             nkept=int(z["n_chains_kept"]), nchain=len(C)))
    if not rows:
        raise SystemExit("no section cells found")
    R = rows
    print(f"\n[B] {len(R)} section cells")
    hi = [r for r in R if r["frac"] > 0.10]
    print(f"[B] cells with >10% of models above {UHI} km/s: {len(hi)}")
    if hi:
        print(f"      their observed max U in band: {np.median([r['umax'] for r in hi]):.2f} "
              f"km/s (all cells: {np.median([r['umax'] for r in R]):.2f})")
        print(f"      chains carrying it: {np.median([r['chain_frac'] for r in hi])*100:.0f}% "
              f"of {R[0]['nchain']}")
    u = np.array([r["umax"] for r in R]); fr = np.array([r["frac"] for r in R])
    g = np.isfinite(u) & np.isfinite(fr)
    print(f"[B] corr(max observed U in band, frac models > {UHI}) = "
          f"{np.corrcoef(u[g], fr[g])[0,1]:+.3f}")
    print(f"[B] chains kept: median {np.median([r['nkept'] for r in R]):.0f}/"
          f"{R[0]['nchain']}   (manuscript S10: 4-5/30)")
    print(f"[B] chain_disagree: median {np.median([r['disagree'] for r in R]):.3f}")

    fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.0))
    col = {"AA": "tab:blue", "BB": "tab:red", "CC": "tab:green"}
    ax = axs[0]
    for s in SECTIONS:
        q = [r for r in R if r["sec"] == s]
        ax.scatter([r["umax"] for r in q], [100 * r["frac"] for r in q], s=26,
                   color=col[s], label=s)
    ax.set_xlabel(f"observed max U in {TBAND[0]}-{TBAND[1]} s [km/s]")
    ax.set_ylabel(f"% of models > {UHI} km/s at {ZWIN[0]}-{ZWIN[1]} km")
    ax.set_title("A: is the fast body demanded by the DATA?", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axs[1]
    for s in SECTIONS:
        q = [r for r in R if r["sec"] == s]
        ax.scatter([100 * r["chain_frac"] for r in q], [100 * r["frac"] for r in q], s=26,
                   color=col[s], label=s)
    ax.plot([0, 100], [0, 100], "k--", lw=1.0)
    ax.set_xlabel(f"% of CHAINS whose median exceeds {UHI} km/s")
    ax.set_ylabel(f"% of MODELS > {UHI} km/s")
    ax.set_title("B: is it carried by a chain minority?\n(on the 1:1 line = all chains agree)",
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axs[2]
    ax.hist([r["nkept"] for r in R], bins=np.arange(-0.5, R[0]["nchain"] + 1.5),
            color="tab:purple", alpha=0.85, label="this workflow (of 24)")
    ax.axvline(4.5, color="tab:red", lw=2.0, ls="--", label="manuscript S10 (4-5 of 30)")
    ax.set_xlabel("chains kept"); ax.set_ylabel("cells")
    ax.set_title("convergence, S10's own statistic", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Mechanism A vs B — does the deep fast body come from the data or the sampler?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(outdir, "S10_chain_vs_data.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); print("wrote", p)
    return R


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=f"{EHM}/hautesorne/tomo/2_vs_depth_inversion/manuscript_comparison_2026-08/"
                                     f"3_diagnostics/si_s9_s10_check")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from swtomotv.config import DatasetConfig
    from swtomotv.geometry import make_grid
    grid = make_grid(DatasetConfig.from_yaml(YAML).bounds, 0.5)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    part_a(load_maps(), grid, tr, a.out)
    part_b(a.out)


if __name__ == "__main__":
    main()
