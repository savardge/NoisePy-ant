#!/usr/bin/env python
"""Sensitivity maps: max retained Rayleigh-fund GROUP period per cell vs trim parameters.

User request 2026-07-25 (deciding how to relax fundamental trimming): one figure per network
per filter family, each panel = a parameter variant, color = max period [s] surviving that
filter alone (>=4 points required, else the cell is shown as dropped). Figure A varies the
physical-trim edge factor alpha (C2 off); figure B varies the C2 near-field/azimuth parameters
(alpha off). Everything is read-time on the EXISTING maps -- no tomography is re-run.

Run in the bayesbay env (swtomotv coords):
  PYTHONPATH=~/Codes/NoisePy-ant python trim_sensitivity_maps.py --net riehen
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_vs_inversion import coverage_grid                      # noqa: E402
from noisepy import vs_inversion as vi, period_resolution as pr    # noqa: E402
from noisepy import curve_masks as cm                              # noqa: E402
from noisepy.lv95 import wgs84_to_lv95, extent_lv95_km             # noqa: E402
from vs_model_figures import _hillshade                            # noqa: E402

E = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
CFG = {
    "riehen": dict(
        gprod=f"{E}/riehen/tomo/1_velocity_maps/production/production_2026-07-14_uni_group_dx0.5/production",
        yaml=f"{E}/riehen/tomo/1_velocity_maps/inputs/riehen_unified_tomo.yaml"),
    "aargau": dict(
        gprod=f"{E}/aargau/tomo/1_velocity_maps/production/production_2026-07-14_uni_group_dx1.0/production",
        yaml=f"{E}/aargau/tomo/1_velocity_maps/inputs/aargau_unified_tomo.yaml"),
    "hautesorne": dict(
        gprod=f"{E}/hautesorne/tomo/1_velocity_maps/production/production_2026-07-25_uni_group_ffv2_dx0.5/production",
        yaml=f"{E}/hautesorne/tomo/1_velocity_maps/inputs/hautesorne_unified_tomo_ffv2.yaml",
        picks="picks_fund_uni_ffv2.csv"),   # el run ffv2 usa las tablas _ffv2
}
ALPHAS = [0.5, 0.4, 0.3, 0.2, 0.0]                       # 0.5 = current; 0.0 = rule off
C2SETS = [("C2 off", None),
          ("strict fn.4 az4 gap60", dict(frac_near_max=0.4, min_azi_bins=4, max_gap_deg=60.0)),
          ("DEFAULT fn.5 az3 gap90", dict(frac_near_max=0.5, min_azi_bins=3, max_gap_deg=90.0)),
          ("relaxed fn.7 az2 gap120", dict(frac_near_max=0.7, min_azi_bins=2, max_gap_deg=120.0)),
          ("ff2.0 (else default)", dict(farfield_factor=2.0, frac_near_max=0.5,
                                        min_azi_bins=3, max_gap_deg=90.0))]
MIN_PTS = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True, choices=("riehen", "aargau", "hautesorne"))
    a = ap.parse_args()
    net = a.net
    c = CFG[net]
    pr.PROD_ROOT[net] = c["gprod"]
    pr.CACHE_CSV[net] = os.path.join(os.path.dirname(c["gprod"]), "cache",
                                     "stations_in_grid.csv")
    cov = coverage_grid(c["gprod"], "fund")
    ixs, iys = np.where(cov >= c.get("cov_min", 25))
    cells = list(zip(ixs.tolist(), iys.tolist()))
    print(f"{net}: {len(cells)} cells")

    # load all fund curves + coords once
    curves, lonlat = {}, {}
    for ix, iy in cells:
        cell = vi.load_cell_curves(c["gprod"], ix, iy, waves=("fund",))
        if not cell.has("fund"):
            continue
        vi.attach_cell_coords(cell, c["yaml"])
        curves[(ix, iy)] = cell
        lonlat[(ix, iy)] = (cell.lon, cell.lat)
    keys = sorted(curves)
    ll = np.array([lonlat[k] for k in keys])

    dem = np.load(f"{E}/{net}/tomo/2_vs_depth_inversion/fig_assets_{net}_dem.npz")
    hs = _hillshade(dem["elev"].astype(float), dem["extent"])
    extent = extent_lv95_km(dem["extent"])          # mapas SIEMPRE en LV95 km, aspecto igual
    nx = max(k[0] for k in keys) + 1
    ny = max(k[1] for k in keys) + 1
    ixg = np.array([k[0] for k in keys], float)
    iyg = np.array([k[1] for k in keys], float)
    A = np.column_stack([np.ones_like(ixg), ixg, iyg])
    clon = np.linalg.lstsq(A, ll[:, 0], rcond=None)[0]
    clat = np.linalg.lstsq(A, ll[:, 1], rcond=None)[0]
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    B = np.column_stack([np.ones(gx.size), gx.ravel(), gy.ravel()])
    lon2d, lat2d = (B @ clon).reshape(nx, ny), (B @ clat).reshape(nx, ny)
    E2d, N2d = wgs84_to_lv95(lon2d, lat2d)
    e_km, n_km = E2d / 1e3, N2d / 1e3

    outdir = f"{E}/{net}/tomo/2_vs_depth_inversion/tests/test_2026-07-25_trim_sensitivity"
    os.makedirs(outdir, exist_ok=True)
    tmax_all = max(float(np.nanmax(curves[k].curves["fund"][0])) for k in keys)

    def draw(fig_variants, tmax_of, title, out):
        n = len(fig_variants)
        pw = 3.7
        ph = pw * (extent[3] - extent[2]) / (extent[1] - extent[0]) + 0.9
        fig, axs = plt.subplots(1, n, figsize=(pw * n, ph), squeeze=False)
        for j, (ax, (lbl, arg)) in enumerate(zip(axs.ravel(), fig_variants)):
            g = np.full((nx, ny), np.nan)
            n_drop = 0
            for k in keys:
                t = tmax_of(k, arg)
                if t is None:
                    n_drop += 1
                else:
                    g[k[0], k[1]] = t
            ax.imshow(hs, extent=extent, cmap="gray", origin="upper", zorder=0)
            pc = ax.pcolormesh(e_km, n_km, g, cmap="turbo", vmin=0.5, vmax=tmax_all,
                               alpha=0.8, shading="nearest", zorder=1)
            ax.set_title(f"{lbl}\n(dropped cells: {n_drop})", fontsize=9)
            ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)
            ax.set_xlabel("E [km LV95]", fontsize=7)
            if j == 0:
                ax.set_ylabel("N [km LV95]", fontsize=7)
        fig.colorbar(pc, ax=axs.ravel().tolist(), fraction=0.02, pad=0.01,
                     label="max retained fund-group period [s]")
        fig.suptitle(title, fontsize=12)
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print("wrote", out)

    # ---- figure A: alpha sweep (physical edge rule only; C2 off) ----
    def tmax_alpha(k, alpha):
        cell = curves[k]
        T = cell.curves["fund"][0]
        if alpha == 0.0:
            keep = np.ones(len(T), bool)
        else:
            keep = pr.keep_mask(cell, net, "fund", "physical",
                                {"alpha": alpha, "R_frac": 0.5, "depth_max": 6.0})
        return float(T[keep].max()) if keep.sum() >= MIN_PTS else None

    draw([(f"alpha = {al:g}" + (" (current)" if al == 0.5 else (" (off)" if al == 0 else "")),
           al) for al in ALPHAS],
         tmax_alpha, f"{net} — max fund-group period vs physical-trim alpha (C2 off)",
         os.path.join(outdir, f"tmax_vs_alpha_{net}.png"))

    # ---- figure B: C2 parameter sweep (alpha off) ----
    pin = f"{E}/{net}/tomo/1_velocity_maps/inputs"
    tables = {}
    for lbl, params in C2SETS:
        if params is None:
            continue
        tables[lbl] = cm.build_c2_table(
            os.path.join(pin, c.get("picks", "picks_fund_uni.csv")),
            os.path.join(pin, "stations.csv"),
            ll, cache=os.path.join(outdir, f"c2tab_{lbl.split()[0]}_{net}.npz"),
            verbose=False, **params)
    kidx = {k: i for i, k in enumerate(keys)}

    def tmax_c2(k, lbl):
        cell = curves[k]
        T = cell.curves["fund"][0]
        if lbl == "C2 off":
            keep = np.ones(len(T), bool)
        else:
            keep = cm.c2_keep_for_periods(tables[lbl], kidx[k], T)
        return float(T[keep].max()) if keep.sum() >= MIN_PTS else None

    draw([(lbl, lbl) for lbl, _ in C2SETS], tmax_c2,
         f"{net} — max fund-group period vs C2 parameters (alpha off)",
         os.path.join(outdir, f"tmax_vs_c2_{net}.png"))


if __name__ == "__main__":
    main()
