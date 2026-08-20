#!/usr/bin/env python3
"""B-B' cross-section of INTERFACE PROBABILITY (not Vs), untrimmed vs res_diag-trimmed.

Same 30 cells and same elevation frame as bb_section_figure.py, but each column is the
ensemble interface probability P(z) -- the histogram of every layer boundary of every
posterior model in that cell, normalised to sum 1 down the column. This is the statistic the
manuscript uses in its Fig 6b, rendered as a section so it can be read against the dotted
"base of Mesozoic / top of crystalline basement" lines of Fig 8.

Panels
  1-2. P(z) untrimmed / trimmed. Colour scale is clipped at the p99 of the values BELOW
       0.5 km depth: the near-surface boundary peak is ~10x anything deeper and would
       otherwise saturate the scale and hide the basement structure entirely. The shallow
       band is therefore deliberately over-saturated -- it is not the subject here.
  3.   difference (trimmed - untrimmed).
  4.   per-cell elevation of the STRONGEST interface below 1 km depth = the section's own
       "top of basement" pick, directly comparable to the Fig-8 dotted line, plotted for
       both versions so trim sensitivity is visible.

Requires runs made with --save-ensemble.
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
DEM = f"{EHM}/hautesorne/tomo/2_vs_depth_inversion/fig_assets_hautesorne_dem.npz"
UNTRIM = "test_2026-08-08_BB_section_R0g"
TRIM = "test_2026-08-08_BB_section_R0g_trim"
NORTH = 1242491.0
IY = 21
DZ = 0.05
MARKS = [(2586000, "Dev. F."), (2597500, "Vic. F.")]


def load(tag, grid, tr_inv, zmax=8.0):
    from swtomotv.geometry import xy2ll
    edges = np.arange(0, zmax + DZ, DZ)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    out = []
    for f in sorted(glob.glob(f"{EHM}/hautesorne/tomo/2_vs_depth_inversion/tests/{tag}/"
                              f"cell_*_{IY}/bayhunter_result.npz")):
        ix = int(os.path.basename(os.path.dirname(f)).split("_")[1])
        z = np.load(f, allow_pickle=True)
        ifd = np.asarray(z["iface_depths"], float)
        h, _ = np.histogram(ifd, bins=edges)
        P = h / max(h.sum(), 1)
        lat, lon = xy2ll(np.array([grid.x[ix]]), np.array([grid.y[IY]]), *grid.origin)
        e, _n = tr_inv.transform(lon[0], lat[0])
        deep = ctr >= 1.0
        zbest = float(ctr[deep][np.argmax(P[deep])]) if deep.any() else np.nan
        out.append(dict(ix=ix, east=e, P=P, ctr=ctr, zbest=zbest))
    return sorted(out, key=lambda r: r["east"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=f"{EHM}/hautesorne/tomo/2_vs_depth_inversion/tests/"
                                     f"{UNTRIM}/BB_section_interface_probability.png")
    a = ap.parse_args()
    from swtomotv.config import DatasetConfig
    from swtomotv.geometry import make_grid
    ds = DatasetConfig.from_yaml(glob.glob("/Users/genevievesavard/Codes/NoisePy-ant/"
                                           "param_files/cluster/tomo/"
                                           "hautesorne_tspws_group_scaled_lccov.yaml")[0])
    grid = make_grid(ds.bounds, 0.5)
    tr_inv = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    tr_fwd = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    A, B = load(UNTRIM, grid, tr_inv), load(TRIM, grid, tr_inv)
    keep = {r["ix"] for r in A} & {r["ix"] for r in B}
    A = [r for r in A if r["ix"] in keep]; B = [r for r in B if r["ix"] in keep]
    print(f"{len(A)} paired cells, Easting {A[0]['east']:.0f} - {A[-1]['east']:.0f}")

    dem = np.load(DEM, allow_pickle=True)
    elev, ext = dem["elev"], dem["extent"]
    topo = []
    for r in A:
        lon, lat = tr_fwd.transform(r["east"], NORTH)
        j = int((lon - ext[0]) / (ext[1] - ext[0]) * (elev.shape[1] - 1))
        i = int((lat - ext[3]) / (ext[2] - ext[3]) * (elev.shape[0] - 1))
        topo.append(float(elev[np.clip(i, 0, elev.shape[0]-1), np.clip(j, 0, elev.shape[1]-1)]))
    topo = np.array(topo)
    east = np.array([r["east"] for r in A])
    ee = np.concatenate([[east[0] - 250], 0.5*(east[:-1]+east[1:]), [east[-1] + 250]])
    ctr = A[0]["ctr"]
    zel = np.linspace(topo.max(), topo.min() - ctr.max() * 1000.0, 260)
    zle = np.concatenate([[zel[0] + (zel[0]-zel[1])/2], 0.5*(zel[:-1]+zel[1:]),
                          [zel[-1] - (zel[0]-zel[1])/2]])

    def to_elev(S):
        M = np.full((len(zel), len(S)), np.nan)
        for i, r in enumerate(S):
            ce = topo[i] - r["ctr"] * 1000.0
            M[:, i] = np.interp(zel, ce[::-1], r["P"][::-1], left=np.nan, right=np.nan)
        return M

    MA, MB = to_elev(A), to_elev(B)
    deep = ctr >= 0.5
    vmax = np.nanpercentile(np.concatenate([[r["P"][deep] for r in A],
                                            [r["P"][deep] for r in B]]), 99)
    fig, axs = plt.subplots(4, 1, figsize=(13.5, 15),
                            gridspec_kw={"height_ratios": [1, 1, 1, 0.55]})
    for k, (M, lab) in enumerate(((MA, "R0g, v1 band (T<=8.61 s) — manuscript input config"),
                                  (MB, "R0g, res_diag-trimmed (T<=5.12 s)"))):
        ax = axs[k]
        im = ax.pcolormesh(ee, zle, M, cmap="inferno", vmin=0, vmax=vmax, shading="flat")
        ax.plot(east, topo, "-", color="w", lw=1.2)
        ax.set_title(lab + "   —   interface probability", fontsize=11, fontweight="bold")
        ax.set_ylabel("elevation [m a.s.l.]"); ax.set_xlim(ee[0], ee[-1])
        plt.colorbar(im, ax=ax, label=f"P(interface) / {DZ*1000:.0f} m bin  (clipped at p99 "
                                      f"below 0.5 km)")
        for e, nm in MARKS:
            if ee[0] <= e <= ee[-1]:
                ax.axvline(e, color="w", ls="--", lw=1.2)
                ax.text(e, zel[0], nm, rotation=90, va="top", fontsize=8, color="w")
    ax = axs[2]
    D = MB - MA
    m = np.nanpercentile(np.abs(D[np.isfinite(D)]), 99)
    im = ax.pcolormesh(ee, zle, D, cmap="bwr", vmin=-m, vmax=m, shading="flat")
    ax.set_title("trimmed − untrimmed  (red = trimming ADDED interface probability)",
                 fontsize=11)
    ax.set_ylabel("elevation [m a.s.l.]"); ax.set_xlim(ee[0], ee[-1])
    plt.colorbar(im, ax=ax, label="dP")
    ax = axs[3]
    for S, c, lab in ((A, "tab:purple", "untrimmed"), (B, "tab:green", "trimmed")):
        ax.plot([r["east"] for r in S],
                [topo[i] - S[i]["zbest"] * 1000.0 for i in range(len(S))],
                "o-", color=c, label=lab)
    ax.plot(east, topo, "-", color="k", lw=1.0, label="surface")
    ax.set_ylabel("elevation of strongest\ninterface below 1 km [m a.s.l.]")
    ax.set_xlabel("Easting [m, LV95]   —   profile B-B' at Northing 1,242,491")
    ax.grid(alpha=0.3); ax.legend(fontsize=9); ax.set_xlim(ee[0], ee[-1])
    fig.suptitle("Haute-Sorne B-B' — ensemble INTERFACE PROBABILITY (the Fig-6b statistic) "
                 "along the section", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
