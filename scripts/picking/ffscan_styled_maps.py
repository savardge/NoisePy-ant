"""ffscan step 5: styled maps for ONE ffscan arm (iso NPZ stack + optional aniso .mat stack).

Generalizes styled_maps.py (whose static per-net overlays it imports) to an arbitrary
production run dir, a phase-aware colorbar label, and the uniform-2theta aniso product
(fast-axis glyph + iso-vs-aniso 3-panel, generalized from hautesorne map_figures.py).

Masking convention (user request 2026-07-26): iso maps draw the FULL coverage field
(vel_full) so dropped-cell velocities stay visible, with a white/black OUTLINE around the
res-confident cells (the npz `mask`); color limits come from the confident cells so wild
low-resolution values cannot blow the scale. The same outline is overlaid on the aniso and
comparison panels (whose .mat fields carry only the ray-count coverage mask).

The arm YAML (ffscan_make_yamls.py) already carries dem + tecto for every net, so ds needs
no overrides; rivers/towns/wells still come from styled_maps.NETS statics.

Figures -> {run_dir}/styled_maps/{wave}[/_aniso]/  (figures live inside their run dir).

Usage (base env): PYTHONPATH=~/Codes/NoisePy-ant:~/Codes/NoisePy-ant/scripts/picking:~/Codes/swtomotv/src \
  python3 ffscan_styled_maps.py --net riehen --yaml <arm.yaml> [--waves fund,love] [--tag "ff1.5 dx0.5"]
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

from swtomotv.config import DatasetConfig, MethodConfig
from swtomotv.geometry import make_grid, ll2xy
from swtomotv.legacy.matfiles import load_result
from swtomotv.products._shared import imshow_extent, vgmaps_dir
from swtomotv.products.figures import (build_hillshade, load_tecto, draw_tecto,
                                       draw_layer, masked)
from noisepy.lv95 import wgs84_to_lv95

from styled_maps import NETS, WELLS, TITLES, load_water, load_towns, draw_extras


def draw_fast_axis(ax, ext, fast_deg, a2_pct):
    """Corner glyph for the region-uniform 2-theta fast axis (deg E of N)."""
    cx = ext[0] + 0.90 * (ext[1] - ext[0])
    cy = ext[2] + 0.88 * (ext[3] - ext[2])
    L = 0.05 * (ext[1] - ext[0]) * max(0.5, min(a2_pct / 2.0, 2.5))
    th = np.radians(90.0 - fast_deg)
    dx, dy = L * np.cos(th), L * np.sin(th)
    for c, lw in (("k", 2.6), ("yellow", 1.4)):
        ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], "-", color=c, lw=lw,
                zorder=7 if c == "k" else 8, solid_capstyle="round")
    ax.annotate(f"fast N{fast_deg:.0f}E\nA2={a2_pct:.1f}%", (cx, cy),
                xytext=(0, -14), textcoords="offset points", ha="center",
                fontsize=6.5, zorder=8,
                path_effects=[pe.withStroke(linewidth=2, foreground="w")])


def conf_outline(ax, grid, show):
    """White-under-black boundary of the confident (unmasked) cell region."""
    if not show.any():
        return
    xc = grid.x + grid.dx / 2.0
    yc = grid.y + grid.dx / 2.0
    for c, lw in (("w", 2.4), ("k", 1.1)):
        ax.contour(xc, yc, show.T.astype(float), levels=[0.5], colors=c,
                   linewidths=lw, zorder=6)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", required=True, choices=tuple(NETS))
    ap.add_argument("--yaml", required=True, help="the ffscan arm YAML")
    ap.add_argument("--measure", default=None, choices=(None, "group", "phase"),
                    help="colorbar label; default inferred from the yaml name")
    ap.add_argument("--waves", default=None, help="comma list; default all in the run")
    ap.add_argument("--tag", default="", help="extra title context, e.g. 'r/lam>=1.5 dx0.5'")
    ap.add_argument("--skip-aniso", action="store_true")
    args = ap.parse_args()

    cfg = NETS[args.net]
    ds = DatasetConfig.from_yaml(args.yaml)
    measure = args.measure or ("phase" if "phase" in os.path.basename(args.yaml) else "group")
    vlabel = f"{measure} velocity (km/s)"
    method = MethodConfig()
    grid = make_grid(ds.bounds, ds.dx_km)
    ext = imshow_extent(grid)
    try:
        hs, hs_ext = build_hillshade(ds, method, grid)
    except Exception as e:
        print(f"{args.net}: no hillshade ({e})")
        hs, hs_ext = None, None
    tecto = load_tecto(ds, grid)
    water = load_water(ds, grid) if cfg["rivers"] else []
    towns = load_towns(ds, grid, cfg["towns"]) if cfg["towns"] else []
    wl = WELLS.get(args.net, [])
    wells_xy = []
    if wl:
        wx, wy = ll2xy(np.array([w[0] for w in wl]), np.array([w[1] for w in wl]),
                       *grid.origin)
        wells_xy = list(zip(wx, wy))
    E0, N0 = wgs84_to_lv95(grid.origin[1], grid.origin[0])
    E0_km, N0_km = float(E0) / 1e3, float(N0) / 1e3

    def style_axes(ax):
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_aspect("equal")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v + E0_km:.0f}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v + N0_km:.0f}"))
        ax.set_xlabel("E (km LV95)"); ax.set_ylabel("N (km LV95)")

    def decorate(ax, fast=None):
        draw_tecto(ax, tecto, legend=True)
        draw_extras(ax, ext, water, towns, wells_xy)
        if fast is not None:
            draw_fast_axis(ax, ext, *fast)
        style_axes(ax)

    outbase = os.path.join(str(ds.output_root), "styled_maps")
    prod = os.path.join(str(ds.output_root), "production")
    waves = (args.waves.split(",") if args.waves else
             sorted(w for w in os.listdir(prod) if w in TITLES)) if os.path.isdir(prod) else []
    for wave in waves:
        outdir = os.path.join(outbase, wave)
        os.makedirs(outdir, exist_ok=True)
        files = sorted(glob.glob(os.path.join(prod, wave, "map_T*.npz")))
        for f in files:
            d = np.load(f)
            T = float(d["period"])
            show = d["mask"].astype(bool)
            Vfull = d["vel_full"]
            if not np.isfinite(Vfull).any():
                continue
            Vc = np.where(show, Vfull, np.nan)
            ref = Vc if np.isfinite(Vc).any() else Vfull
            lo, hi = np.nanpercentile(ref, [2, 98])
            fig, ax = plt.subplots(figsize=(8.6, 6.2))
            draw_layer(ax, Vfull, "jet_r", vlabel, hs, hs_ext, ext, vmin=lo, vmax=hi)
            conf_outline(ax, grid, show)
            decorate(ax)
            ax.set_title(f"{args.net}  {TITLES[wave]}  T = {T:g} s  ({measure} iso"
                         f"{', ' + args.tag if args.tag else ''})\n"
                         f"N={int(d['N'])}  var_red={float(d['var_red']):.2f}  "
                         f"(outline = res-confident cells)", fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"map_T{ds.tfmt(T)}.png"), dpi=140)
            plt.close(fig)
        print(f"{args.net}/{wave}: {len(files)} iso maps -> {outdir}")

        if args.skip_aniso:
            continue
        ani_dir = vgmaps_dir(ds, wave, aniso=True)
        iso_dir = vgmaps_dir(ds, wave, topo=True)
        afiles = sorted(glob.glob(str(ani_dir / f"vgmap_final_{wave}_T*.mat")))
        if not afiles:
            print(f"{args.net}/{wave}: no aniso maps ({ani_dir})")
            continue
        outdir2 = os.path.join(outbase, f"{wave}_aniso")
        os.makedirs(outdir2, exist_ok=True)
        for f in afiles:
            T = float(os.path.basename(f).split("_T")[1][:-4])
            ma = load_result(f)
            Va = masked(ma)
            if not np.isfinite(Va).any():
                continue
            # the .mat mask is ray-count coverage only; the res-confident outline
            # comes from the matching iso npz when it exists
            npz_f = os.path.join(prod, wave, f"map_T{ds.tfmt(T)}.npz")
            show = np.load(npz_f)["mask"].astype(bool) if os.path.exists(npz_f) else None
            A2, fz = float(ma["A2_pct"]), float(ma["fast_az_deg"])
            lo, hi = np.nanpercentile(np.where(show, Va, np.nan), [2, 98]) \
                if show is not None and (show & np.isfinite(Va)).any() \
                else np.nanpercentile(Va, [2, 98])
            fig, ax = plt.subplots(figsize=(8.6, 6.2))
            draw_layer(ax, Va, "jet_r", vlabel, hs, hs_ext, ext, vmin=lo, vmax=hi)
            if show is not None:
                conf_outline(ax, grid, show)
            decorate(ax, fast=(fz, A2))
            ax.set_title(f"{args.net}  {TITLES[wave]}  T = {T:g} s  ({measure}, uniform 2θ"
                         f"{', ' + args.tag if args.tag else ''})", fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir2, f"map_T{ds.tfmt(T)}.png"), dpi=140)
            plt.close(fig)

            iso_f = iso_dir / os.path.basename(f)
            if not iso_f.exists():
                continue
            Vi = masked(load_result(iso_f))
            dV = Va - Vi
            dlim = max(float(np.nanpercentile(np.abs(dV), 98)), 1e-3)
            both = np.concatenate([Va.ravel(), Vi.ravel()])
            vmin, vmax = np.nanpercentile(both, [2, 98])
            fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))
            for ax, arr, ttl, cm, lim in [
                    (axes[0], Vi, "isotropic (topo kernels)", "jet_r",
                     dict(vmin=vmin, vmax=vmax)),
                    (axes[1], Va, f"uniform 2θ (A2={A2:.1f}%, fast N{fz:.0f}E)",
                     "jet_r", dict(vmin=vmin, vmax=vmax)),
                    (axes[2], dV, "difference (aniso − iso)", "RdBu_r",
                     dict(vmin=-dlim, vmax=dlim))]:
                draw_layer(ax, arr, cm, "km/s", hs, hs_ext, ext, **lim)
                if show is not None:
                    conf_outline(ax, grid, show)
                draw_tecto(ax, tecto)
                draw_extras(ax, ext, water, towns, wells_xy)
                style_axes(ax)
                ax.set_title(ttl, fontsize=9)
            draw_fast_axis(axes[1], ext, fz, A2)
            fig.suptitle(f"{args.net}  {TITLES[wave]}  T = {T:g} s  ({measure}"
                         f"{', ' + args.tag if args.tag else ''})", fontsize=11)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir2, f"cmp_T{ds.tfmt(T)}.png"), dpi=140)
            plt.close(fig)
        print(f"{args.net}/{wave}: {len(afiles)} aniso + cmp figs -> {outdir2}")


if __name__ == "__main__":
    main()
