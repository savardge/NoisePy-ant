#!/usr/bin/env python3
"""Fig-8 B-B' cross-section from the current workflow: R0g untrimmed vs res_diag-trimmed.

Renders the 30 inverted cells along Northing 1,242,491 (LV95) as a depth/elevation section in
the manuscript's own Fig-8 convention (red slow, blue fast, 1.5-4.0 km/s) so the panels can be
placed directly beside the published one. The question it answers: do the deep high-Vs
("purple") blobs at 2-4 km survive removal of the periods where the map cells are
resolution-collapsed?

Panels
  1. R0g with the v1 band (T <= 8.61 s)  = the manuscript's input configuration
  2. R0g trimmed to res_diag >= 0.05 (T <= 5.12 s)
  3. difference (trimmed - untrimmed)
  4. per-cell fraction of ENSEMBLE models exceeding 3.5 km/s at 2-4.5 km -- the quantitative
     version of "is that blob real". At GVL-1 this was 0.1% untrimmed.

Vertical axis is elevation in m a.s.l. (as in Fig 8), converted from inversion depth with the
DEM already cached in the tree; the topographic profile is drawn so the conversion is visible
rather than implicit.

NOTE the trim used here is a GLOBAL T <= 5.12 s cut taken from the GVL-1 cells. The production
criterion (`period_resolution.trim_reliable`, grid_vs_inversion --criterion) is PER CELL, and
the reliable limit across Haute-Sorne ranges 3.04 s (p5) to 5.75 s (p95) -- so this shows the
direction and rough magnitude of the effect, not the exact production result.
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
# LV95 eastings of the labelled structures on the manuscript panel (approximate, read off Fig 8)
MARKS = [(2586000, "Dev. F."), (2597500, "Vic. F.")]
PCT = [(2585500, "PCT-2"), (2594000, "PCT-3")]


def cell_easting(ix, tr_inv, grid):
    from swtomotv.geometry import xy2ll
    lat, lon = xy2ll(np.array([grid.x[ix]]), np.array([grid.y[IY]]), *grid.origin)
    e, n = tr_inv.transform(lon[0], lat[0])
    return e


def load(tag, grid, tr_inv):
    out = []
    for f in sorted(glob.glob(f"{EHM}/hautesorne/tomo/2_vs_depth_inversion/tests/{tag}/"
                              f"cell_*_{IY}/bayhunter_result.npz")):
        ix = int(os.path.basename(os.path.dirname(f)).split("_")[1])
        z = np.load(f, allow_pickle=True)
        E = np.asarray(z["ens_vs"], float)
        d = z["depth"]
        m = (d >= 2.24) & (d <= 4.5)
        out.append(dict(ix=ix, east=cell_easting(ix, tr_inv, grid), depth=d,
                        vs=z["vs_median"],
                        frac_hi=float(np.mean(E[:, m].max(axis=1) > 3.5))))
    return sorted(out, key=lambda r: r["east"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=f"{EHM}/hautesorne/tomo/2_vs_depth_inversion/tests/"
                                     f"{UNTRIM}/BB_section_R0g_trim_vs_untrim.png")
    a = ap.parse_args()
    from swtomotv.config import DatasetConfig
    from swtomotv.geometry import make_grid
    ds = DatasetConfig.from_yaml(glob.glob("/Users/genevievesavard/Codes/NoisePy-ant/"
                                           "param_files/cluster/tomo/"
                                           "hautesorne_tspws_group_scaled_lccov.yaml")[0])
    grid = make_grid(ds.bounds, 0.5)
    tr_inv = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    tr_fwd = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)

    A = load(UNTRIM, grid, tr_inv)
    B = load(TRIM, grid, tr_inv)
    if not A:
        raise SystemExit("no untrimmed cells yet")
    keep = {r["ix"] for r in A} & {r["ix"] for r in B}
    A = [r for r in A if r["ix"] in keep]
    B = [r for r in B if r["ix"] in keep]
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
    d = A[0]["depth"]
    ee = np.concatenate([[east[0] - 250], 0.5 * (east[:-1] + east[1:]), [east[-1] + 250]])
    # Common ELEVATION grid. Using a single mean topography for every column (as a first
    # version did) flattens ~250 m of relief into the section and mis-registers the shallow
    # part; each column must be mapped with ITS OWN surface height.
    zel = np.linspace(topo.max(), topo.min() - d.max() * 1000.0, 240)
    zle = np.concatenate([[zel[0] + (zel[0]-zel[1])/2], 0.5*(zel[:-1]+zel[1:]),
                          [zel[-1] - (zel[0]-zel[1])/2]])

    def to_elev(S):
        M = np.full((len(zel), len(S)), np.nan)
        for i, r in enumerate(S):
            col_elev = topo[i] - r["depth"] * 1000.0        # descending
            M[:, i] = np.interp(zel, col_elev[::-1], r["vs"][::-1],
                                left=np.nan, right=np.nan)
        return M

    fig, axs = plt.subplots(4, 1, figsize=(13.5, 15),
                            gridspec_kw={"height_ratios": [1, 1, 1, 0.5]})
    for k, (S, lab) in enumerate(((A, "R0g, v1 band (T<=8.61 s) — manuscript input config"),
                                  (B, "R0g, res_diag-trimmed (T<=5.12 s)"))):
        M = to_elev(S)
        ax = axs[k]
        im = ax.pcolormesh(ee, zle, M, cmap="RdYlBu", vmin=1.5, vmax=4.0, shading="flat")
        ax.plot(east, topo, "-", color="k", lw=1.2)
        ax.set_title(lab, fontsize=11, fontweight="bold")
        ax.set_ylabel("elevation [m a.s.l.]")
        plt.colorbar(im, ax=ax, label="Vs [km/s]")
        ax.set_xlim(ee[0], ee[-1])          # keep all panels on the SAME easting range
        for e, nm in MARKS:
            if ee[0] <= e <= ee[-1]:        # only annotate structures inside the data
                ax.axvline(e, color="k", ls="--", lw=1.3)
                ax.text(e, zel[0], nm, rotation=90, va="top", fontsize=8)
        for e, nm in PCT:
            if ee[0] <= e <= ee[-1]:
                ax.text(e, -1200, nm, fontsize=9, ha="center")
    D = to_elev(B) - to_elev(A)
    ax = axs[2]
    m = np.nanpercentile(np.abs(D), 99)
    im = ax.pcolormesh(ee, zle, D, cmap="bwr", vmin=-m, vmax=m, shading="flat")
    ax.set_title("trimmed − untrimmed  (red = trimming made it FASTER)", fontsize=11)
    ax.set_xlim(ee[0], ee[-1])
    ax.set_ylabel("elevation [m a.s.l.]")
    plt.colorbar(im, ax=ax, label="dVs [km/s]")
    ax = axs[3]
    ax.plot(east, [100*r["frac_hi"] for r in A], "o-", color="tab:purple", label="untrimmed")
    ax.plot(east, [100*r["frac_hi"] for r in B], "o-", color="tab:green", label="trimmed")
    ax.set_ylabel("% of models > 3.5 km/s\nat 2–4.5 km")
    ax.set_xlabel("Easting [m, LV95]   —   profile B-B' at Northing 1,242,491")
    ax.grid(alpha=0.3); ax.legend(fontsize=9); ax.set_xlim(ee[0], ee[-1])
    fig.suptitle("Haute-Sorne B-B' section, current workflow — do the deep high-Vs blobs "
                 "survive removal of resolution-collapsed periods?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
