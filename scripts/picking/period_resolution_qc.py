"""Diagnostics for the per-cell period-reliability criteria (noisepy.period_resolution).

Figures (clearly labelled per criterion A=combined / B=physical / C=tomographic):
  * reliability_<wellA>_vs_<wellB>.png — for two contrasting cells (edge vs centre): the
    dispersion curve, a keep-matrix (criteria × period), res_diag(T), wavelengths-inside-array
    d_edge/lambda(T), and kernel depth z_eff(T). Shows *why* long periods fail at the edge.
  * max_reliable_period_{A,B,C}.png — longest kept fund period per cell over the SRTM hillshade
    (bull's-eye: long periods only usable near the array centre).
  * res_diag_map_T{T}.png — resolution at a long period over the hillshade (edge collapse).

Run (bayesbay_dev env, has disba + swtomotv):
  PYTHONPATH=~/Codes/Noisepy-ant /opt/anaconda3/envs/bayesbay_dev/bin/python period_resolution_qc.py \
    --net aargau --wells Boettstein,Riniken
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from noisepy import vs_inversion as vi
from noisepy import period_resolution as pr
from dem_hillshade import hillshade_for_bbox, dem_for_net

PROJ = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
CRITS = [("combined", "A: combined"), ("physical", "B: physical"), ("tomographic", "C: tomographic")]
# well ix,iy per net (nearest inverted cell, from well_vs_qc mapping)
WELL_CELL = {"aargau": {"Boettstein": (13, 20), "Riniken": (10, 14)},
             "riehen": {"Basel-1": (9, 18), "Otterbach-2": (10, 17),
                        "Riehen-1": (17, 19), "Riehen-2": (18, 20)}}


def reliability_panels(net, wells, wave="fund", params=None, out=None):
    prod = f"{PROJ}/{net}/tomo/1_velocity_maps/_archive/swtomotv-output/production"
    yaml = f"{PROJ}/{net}/tomo/{net}_swtomotv.yaml"
    p = {**pr.DEFAULTS, **(params or {})}
    fig, axs = plt.subplots(5, len(wells), figsize=(6.2 * len(wells), 12), squeeze=False,
                            gridspec_kw=dict(height_ratios=[2, 1, 1.4, 1.4, 1.4]))
    for c, well in enumerate(wells):
        ix, iy = WELL_CELL[net][well]
        cell = vi.load_cell_curves(prod, ix, iy); vi.attach_cell_coords(cell, yaml)
        m = pr.metrics_table(cell, net, wave, p)
        T, U, S = cell.curves[wave]
        keeps = {k: pr.keep_mask(cell, net, wave, k, p) for k, _ in CRITS}

        a = axs[0][c]
        a.errorbar(T, U, yerr=S, fmt="o", ms=3, color="tab:blue")
        a.set(ylabel="U [km/s]", title=f"{net.capitalize()} — {well}  cell ({ix},{iy})  "
              f"d_edge={m['d_edge']:.1f} km")

        a = axs[1][c]                                   # keep-matrix
        M = np.vstack([keeps[k].astype(float) for k, _ in CRITS])
        a.imshow(M, aspect="auto", cmap="Greens", vmin=0, vmax=1,
                 extent=[T.min(), T.max(), len(CRITS) - 0.5, -0.5])
        a.set_yticks(range(len(CRITS))); a.set_yticklabels([lab for _, lab in CRITS], fontsize=8)
        a.set_ylabel("kept?")
        for k, (_, lab) in enumerate(CRITS):
            kk = keeps[CRITS[k][0]]
            if kk.any():
                a.text(0.99, k, f" {T[kk].min():.1f}-{T[kk].max():.1f}s", va="center",
                       ha="right", transform=a.get_yaxis_transform(), fontsize=7, color="0.2")

        a = axs[2][c]
        thr = max(p["R_min"], p["R_frac"] * np.nanmax(np.nan_to_num(m["res_diag"])))
        a.plot(T, m["res_diag"], "-o", ms=2, color="tab:purple")
        a.axhline(thr, color="k", ls="--", lw=0.8, label=f"C thresh {thr:.03f}")
        a.set(ylabel="res_diag"); a.legend(fontsize=7)

        a = axs[3][c]
        a.plot(T, m["n_lambda"], "-o", ms=2, color="tab:green")
        a.axhline(p["alpha"], color="k", ls="--", lw=0.8, label=f"B: alpha={p['alpha']}")
        a.set(ylabel="d_edge / λ"); a.legend(fontsize=7)

        a = axs[4][c]
        a.plot(T, m["z_eff"], "-o", ms=2, color="tab:red")
        a.axhline(p["beta"] * p["depth_max"], color="k", ls="--", lw=0.8,
                  label=f"B: {p['beta']}·depth_max")
        a.set(xlabel="period [s]", ylabel="z_eff [km]"); a.legend(fontsize=7)
    fig.suptitle(f"Period reliability — {' vs '.join(wells)}  ({wave})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = out or f"{PROJ}/{net}/tomo/vs_inversion/wells/period_qc/reliability_{'_vs_'.join(wells)}.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140); plt.close(fig); print("wrote", out)


def _load_maps(net, wave):
    prod = f"{PROJ}/{net}/tomo/1_velocity_maps/_archive/swtomotv-output/production"
    files = sorted(glob.glob(os.path.join(prod, wave, "map_T*.npz")),
                   key=lambda f: float(np.load(f)["period"]))
    T = np.array([float(np.load(f)["period"]) for f in files])
    res = np.stack([np.load(f)["res_diag"] for f in files], axis=-1)     # (nx,ny,nT)
    msk = np.stack([np.load(f)["mask"] for f in files], axis=-1)
    return T, res, msk


def _edge_grid(net, yaml):
    from swtomotv.config import DatasetConfig
    from swtomotv.geometry import make_grid
    grid = make_grid(DatasetConfig.from_yaml(yaml).bounds, DatasetConfig.from_yaml(yaml).dx_km)
    hull = pr._hull_xy(net)
    nx, ny = len(grid.x), len(grid.y)
    de = np.full((nx, ny), np.nan)
    for ix in range(nx):
        for iy in range(ny):
            de[ix, iy] = pr.edge_distance_xy(float(grid.x[ix] + grid.dx / 2),
                                             float(grid.y[iy] + grid.dx / 2), hull)
    return de, grid


def max_reliable_period_maps(net, wave="fund", params=None):
    yaml = f"{PROJ}/{net}/tomo/{net}_swtomotv.yaml"
    p = {**pr.DEFAULTS, **(params or {})}
    T, res, msk = _load_maps(net, wave)
    de, grid = _edge_grid(net, yaml)
    lam = pr.phase_wavelength(net, wave, T)              # (nT,)
    ze = pr.kernel_depth(wave, T, depth_max=p["depth_max"])
    nx, ny, nT = res.shape
    # per-criterion keep cube (nx,ny,nT)
    keep = {}
    tomo_thr = p["R_frac"] * np.nanmax(np.where(msk, res, np.nan), axis=2)   # (nx,ny) per-cell peak
    keep_tomo = msk & (res >= np.maximum(p["R_min"], tomo_thr)[:, :, None])
    keep_phys = msk & (de[:, :, None] >= p["alpha"] * lam[None, None, :]) & \
        (ze[None, None, :] <= p["beta"] * p["depth_max"])
    keep["tomographic"] = keep_tomo
    keep["physical"] = keep_phys
    keep["combined"] = keep_tomo & keep_phys

    # geography for pcolormesh + hillshade
    olat, olon = grid.origin
    R = 6371.0
    gx, gy = np.meshgrid(grid.x + grid.dx / 2, grid.y + grid.dx / 2, indexing="ij")
    lat2d = olat + gy / R * 180 / np.pi
    lon2d = olon + gx / (R * np.cos(np.radians((lat2d + olat) / 2))) * 180 / np.pi
    lo0, lo1 = np.nanmin(lon2d), np.nanmax(lon2d)
    la0, la1 = np.nanmin(lat2d), np.nanmax(lat2d)
    try:                                                # hillshade needs rasterio (base env only)
        hs, ext = hillshade_for_bbox(dem_for_net(net), lo0, lo1, la0, la1)
    except Exception as e:
        print(f"  (no hillshade: {e}); plain background")
        hs, ext = None, [lo0, lo1, la0, la1]

    for crit, lab in CRITS:
        mrp = np.full((nx, ny), np.nan)
        anyk = keep[crit].any(axis=2)
        Tcube = np.where(keep[crit], T[None, None, :], np.nan)
        mrp[anyk] = np.nanmax(Tcube, axis=2)[anyk]
        fig, ax = plt.subplots(figsize=(7, 6))
        if hs is not None:
            ax.imshow(hs, extent=ext, cmap="gray", origin="upper", aspect="auto", zorder=0)
        pc = ax.pcolormesh(lon2d, lat2d, mrp, cmap="viridis", alpha=0.7 if hs is not None else 1.0,
                           shading="nearest", zorder=1)
        plt.colorbar(pc, ax=ax, label="max reliable fund period [s]")
        for well, (wix, wiy) in WELL_CELL[net].items():
            ax.plot(lon2d[wix, wiy], lat2d[wix, wiy], "r^", ms=8, zorder=3)
            ax.annotate(well, (lon2d[wix, wiy], lat2d[wix, wiy]), fontsize=7, color="r")
        ax.set(xlim=(ext[0], ext[1]), ylim=(ext[2], ext[3]), xlabel="lon", ylabel="lat",
               title=f"{net.capitalize()} — max reliable period, {lab}")
        out = f"{PROJ}/{net}/tomo/vs_inversion/wells/period_qc/max_reliable_period_{crit}.png"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig); print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", required=True, choices=("aargau", "riehen"))
    ap.add_argument("--wells", default="Boettstein,Riniken")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--rfrac", type=float, default=0.5)
    ap.add_argument("--maps", action="store_true", help="also render max-reliable-period maps")
    args = ap.parse_args()
    params = {"alpha": args.alpha, "R_frac": args.rfrac}
    reliability_panels(args.net, [w.strip() for w in args.wells.split(",")], params=params)
    if args.maps:
        max_reliable_period_maps(args.net, params=params)
    print("done.")


if __name__ == "__main__":
    main()
