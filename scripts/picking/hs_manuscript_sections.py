#!/usr/bin/env python
"""Haute-Sorne manuscript sections A-A', B-B', C-C' with the GVL-1 stratigraphic column.

Geometry and sampling are REUSED from section_figure.py rather than reimplemented. An earlier
version of this script rolled its own and got two things wrong that this one cannot:
  * it averaged every cell within 3.5 km of the line with a Gaussian kernel, which smeared the
    model laterally so the sections showed different features from the volume;
  * it interpolated scattered (distance, elevation) points, so cells at the same along-distance
    but different topography collided and left vertical stripes.
section_figure.sample_line walks the segment at the 500 m grid step and takes the single
nearest cell per stop (dropping repeats), which is the convention the manuscript figures use.

GVL-1 is drawn to its real total depth: 4041.5 m MD / 4006.12 m TVD from ground 494.20 m
(FWR / BH Directional EOW Report), i.e. the stick bottoms at -3512 m a.s.l. The stratigraphy
spreadsheet extrapolates its basement row to 6000 m, which would draw the well ~1.9 km too deep.
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator, FuncFormatter
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import RegularGridInterpolator
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from section_figure import SECTIONS, sample_line, topo_of, TICK_KM   # verified geometry

E = "/Users/genevievesavard/Codes/extract_higher_modes"
STRAT = "/Users/genevievesavard/Data/hautesorne/stratigraphy/GVL-1_Stratigraphy_v2.xlsx"
GVL1_E, GVL1_N, GVL1_GL = 2583491.08, 1242491.79, 494.20   # swisstopo, verified 2026-08-16
GVL1_TD_MD, GVL1_TD_TVD = 4041.5, 4006.12                  # FWR; well effectively vertical
MD_TO_TVD = GVL1_TD_TVD / GVL1_TD_MD
WELL_NEAR_M = 1500.0
CMAP, VMIN, VMAX = "RdYlBu", 1.5, 4.0                        # the manuscript scale


def resolve_cmap(name):
    """Accept a matplotlib name or a Crameri map as 'crameri:roma_r'.

    RdYlBu is the published scale but is NOT perceptually uniform: it has a bright band around
    2.5 km/s that reads as a boundary where the model is smooth, and it loses contrast at both
    ends. The Crameri maps are uniform in lightness, so apparent structure tracks Vs rather than
    the colormap.
    """
    if not isinstance(name, str):          # already a Colormap object
        return name
    if name == "fig8":
        # The colour scheme of the earlier manuscript figure (Fig. 8): dark red (slow) -> orange
        # -> yellow -> WHITE at the centre node -> light blue -> blue -> purple (fast). White sits
        # exactly at 0.5 so that --calibrate-basement (TwoSlopeNorm) puts it on the basement-top
        # Vs. Intended range 1.3-3.4 km/s (--vmin 1.3 --vmax 3.4).
        from matplotlib.colors import LinearSegmentedColormap
        nodes = [(0.00, "#7f0000"), (0.14, "#c81e14"), (0.28, "#f2612d"), (0.40, "#fbc02d"),
                 (0.47, "#fff3a0"), (0.50, "#ffffff"), (0.56, "#cfe9f7"), (0.68, "#78b4e6"),
                 (0.80, "#3f6fd0"), (0.90, "#4a3fb8"), (1.00, "#4b0f7a")]
        return LinearSegmentedColormap.from_list("fig8", nodes)
    # noisepy.colormaps carries the Crameri tables itself, so this works in the render envs
    # that have no cmcrameri (nant, bayhunter); it still prefers the package where installed.
    from noisepy.colormaps import get_cmap
    return get_cmap(name.split(":", 1)[1] if name.startswith("crameri:") else name)
LV95 = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
NORM = None


BASEMENT_MD = 2240.0        # GVL-1 basement top, m MD (FWR Table 4)


def basement_vs(arm, campaign):
    """The arm's own Vs where GVL-1 fixes the basement top, and the contrast across it.

    There is NO sonic log at GVL-1, so this is a DEPTH tie, not a velocity measurement: the well
    pins the interface depth independently and we read what the model puts there. Returns
    (vs_at_top, contrast) where contrast is Vs(top+400 m) - Vs(top-400 m); a small contrast means
    the arm does not resolve the interface and centring the colour scale on it would invent an
    edge that the model does not contain.
    """
    vf = [f for f in glob.glob(f"{campaign}/{arm}/volume_*.npz") if "gridded" not in f]
    v = np.load(vf[0], allow_pickle=True)
    xy = np.column_stack(LV95.transform(v["lonlat"][:, 0], v["lonlat"][:, 1]))
    i = int(np.argmin(np.hypot(xy[:, 0] - GVL1_E, xy[:, 1] - GVL1_N)))
    if np.hypot(*(xy[i] - [GVL1_E, GVL1_N])) > 1500:
        return None, None
    z, prof = v["depth"], v["vs_median"][i]
    zt = BASEMENT_MD * MD_TO_TVD / 1000.0
    at = float(np.interp(zt, z, prof))
    return at, float(np.interp(zt + 0.4, z, prof) - np.interp(zt - 0.4, z, prof))


def load_strat():
    g = pd.read_excel(STRAT, sheet_name="Groups").rename(
        columns={"MD Top [m]": "top_m", "MD Base [m]": "base_m", "Hex Color": "hex"})
    g = g[g.top_m < GVL1_TD_MD].copy()
    g["base_m"] = g.base_m.clip(upper=GVL1_TD_MD)          # do not draw past TD
    g["top_elev"] = GVL1_GL - g.top_m * MD_TO_TVD
    g["base_elev"] = GVL1_GL - g.base_m * MD_TO_TVD
    return g


def records(vol):
    """One record per inverted cell, in the shape section_figure's helpers expect."""
    lon, lat = vol["lonlat"][:, 0], vol["lonlat"][:, 1]
    ee, nn = LV95.transform(lon, lat)
    return [dict(ix=int(c[0]), iy=int(c[1]), E=ee[i], N=nn[i], lon=lon[i], lat=lat[i],
                 depth=vol["depth"], vs=vol["vs_median"][i], zmax=vol["z_reliable_max"][i])
            for i, c in enumerate(vol["cells"])]


def slice_3d(g, P, Q, step_km=0.05, method="linear"):
    """Slice the 3-D interpolated volume along a profile.

    Preferred over interpolating between 1-D columns: the smoothing has already been done ONCE
    in 3-D (tension-spline over the 0.5 km cell grid, refined to 0.125 km), so a section and a
    depth map of the same arm are guaranteed to show the same model. Sampling here is a lookup
    into that grid, not a second, different interpolation.

    The gridded volume is already masked to each cell's reliability window, so NaN regions are
    genuinely unresolved rather than hidden.
    """
    x, y, z = g["x_km"].astype(float), g["y_km"].astype(float), g["depth"].astype(float)
    V = np.asarray(g["vs"], float)                      # (nz, ny, nx)
    fn = RegularGridInterpolator((z, y, x), V, method=method,
                                 bounds_error=False, fill_value=np.nan)
    P = np.array(P, float) / 1000.0                     # LV95 m -> km
    Q = np.array(Q, float) / 1000.0
    L = float(np.hypot(*(Q - P)))
    n = max(int(L / step_km) + 1, 2)
    t = np.linspace(0.0, L, n)
    u = (Q - P) / L
    px, py = P[0] + t * u[0], P[1] + t * u[1]
    ZZ, TT = np.meshgrid(z, t, indexing="ij")
    PX = np.broadcast_to(px, ZZ.shape); PY = np.broadcast_to(py, ZZ.shape)
    M = fn(np.column_stack([ZZ.ravel(), PY.ravel(), PX.ravel()])).reshape(ZZ.shape)
    return t, z, M, px, py, L


PANEL_LOG = []


def panel(ax, g, key, strat, method="linear", step_km=0.05, datum="topography", topo_smooth_km=0.0):
    lend, rend, _tag, P, Q = SECTIONS[key]
    t, dep, M, px, py, L = slice_3d(g, P, Q, step_km, method)

    inv = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    lon, lat = inv.transform(px * 1000.0, py * 1000.0)
    topo_true = np.array(topo_of([dict(lon=a, lat=b) for a, b in zip(lon, lat)])) / 1000.0
    if datum == "flat":
        # No topographic hang: the gridded volume is indexed by depth BELOW SURFACE, so this is
        # the model as inverted, with every profile starting at a common datum. The DEM profile
        # is drawn ABOVE that datum as true-scale relief (axes are 1:1, no exaggeration), so the
        # reader still sees where the ridges and valleys are without the model being draped.
        topo = np.zeros_like(px)
        relief = topo_true - np.nanmin(topo_true)
    else:
        topo = topo_true
        relief = None
    # Hang the model on a SMOOTHED surface (Gaussian, sigma = topo_smooth_km). The model's depth axis
    # is depth below a surface that the surface waves only feel averaged over their lateral
    # footprint (~2 km here), so hanging every 50 m column on the raw DEM imprints sub-resolution
    # ridges and valleys on the model. The real topography is still drawn as the black line.
    hang = topo
    if datum != "flat" and topo_smooth_km > 0:
        from scipy.ndimage import gaussian_filter1d
        hang = gaussian_filter1d(topo, topo_smooth_km / step_km, mode="nearest")
        PANEL_LOG.append((key, float(np.max(np.abs(hang - topo))) * 1000.0,
                          float(np.interp(np.hypot(*(np.array([GVL1_E, GVL1_N]) - np.array(P, float))) / 1000.0,
                                          t, hang - topo)) * 1000.0))

    head = (float(np.nanmax(relief)) + 0.6) if datum == "flat" else 0.9
    zel = np.linspace(topo.max() + head, min(topo.min(), hang.min()) - dep.max(), 420)
    E2 = np.full((len(zel), len(t)), np.nan)
    for i in range(len(t)):                              # per-column: no cross-column mixing
        ce = hang[i] - dep
        col = M[:, i]
        ok = np.isfinite(col)
        if ok.sum() >= 2:
            E2[:, i] = np.interp(zel, ce[ok][::-1], col[ok][::-1], left=np.nan, right=np.nan)
            if topo[i] > hang[i] and ok[0]:              # ridge above the smoothed surface: extend
                E2[(zel > hang[i]) & (zel <= topo[i]), i] = col[0]   # the top value up to the ground
        E2[zel > topo[i], i] = np.nan                    # never paint above the real ground

    if NORM is not None:
        im = ax.pcolormesh(t, zel, E2, cmap=resolve_cmap(CMAP), norm=NORM, shading="gouraud")
    else:
        im = ax.pcolormesh(t, zel, E2, cmap=resolve_cmap(CMAP), vmin=VMIN, vmax=VMAX,
                           shading="gouraud")
    ax.plot(t, topo, "-", color="k", lw=1.2)
    if datum == "flat":
        ax.fill_between(t, 0.0, relief, color="0.88", lw=0, zorder=2)
        ax.plot(t, relief, "-", color="0.25", lw=0.9, zorder=3)
        ax.text(t[0] + 0.01 * (t[-1] - t[0]), float(np.nanmax(relief)) + 0.05,
                f"topography, true scale (relief above {np.nanmin(topo_true)*1000:.0f} m a.s.l.)",
                fontsize=6.5, color="0.3", va="bottom", zorder=4)
        ax.set_ylabel("Depth below surface (m)", fontsize=8)

    Pa, Qa = np.array(P, float), np.array(Q, float)
    uu = (Qa - Pa) / np.hypot(*(Qa - Pa))
    rel = np.array([GVL1_E, GVL1_N]) - Pa
    sg, pp = rel @ uu, abs(rel[0]*uu[1] - rel[1]*uu[0])
    if pp < WELL_NEAR_M and 0 <= sg/1000.0 <= L:
        w = L * 0.011
        for _, r in strat.iterrows():
            if datum == "flat":       # depth below ground, not elevation
                top_y, bot_y = -r.top_m * MD_TO_TVD / 1000.0, -r.base_m * MD_TO_TVD / 1000.0
            else:
                top_y, bot_y = r.top_elev / 1000.0, r.base_elev / 1000.0
            ax.add_patch(Rectangle((sg/1000.0 - w/2, bot_y), w, top_y - bot_y,
                                   facecolor=r.hex, edgecolor="0.25", lw=0.4, zorder=6))
        ax.text(sg/1000.0, (float(np.nanmax(relief)) + 0.25) if datum == "flat" else (topo.max() + 0.35), "GVL-1", ha="center", fontsize=8.5,
                weight="bold", zorder=7)
    ax.set_xlim(t[0], t[-1]); ax.set_aspect("equal")
    ax.xaxis.set_major_locator(MultipleLocator(TICK_KM))
    ax.yaxis.set_major_locator(MultipleLocator(TICK_KM))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{abs(v)*1000:.0f}" if datum == "flat" else f"{v*1000:.0f}"))
    if datum != "flat":
        ax.set_ylabel("Elevation (m a.s.l.)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.text(0.0, 1.02, lend, transform=ax.transAxes, fontsize=9, weight="bold", va="bottom")
    ax.text(1.0, 1.02, rend, transform=ax.transAxes, fontsize=9, weight="bold",
            va="bottom", ha="right")
    return im, int(np.isfinite(E2).sum()), L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--interp", default="linear", choices=("linear", "cubic"),
                    help="order of the 3-D grid lookup")
    ap.add_argument("--step-km", type=float, default=0.05)
    ap.add_argument("--cmap", default="RdYlBu",
                    help="Crameri name (\"roma\", also accepted as \"crameri:roma\") or any "
                         "matplotlib name. Default `roma` since 2026-09-06 (user preference); "
                         "--cmap RdYlBu reproduces the earlier published scale.")
    ap.add_argument("--datum", default="topography", choices=("topography", "flat"),
                    help="flat = no topographic hang; y is depth below surface")
    ap.add_argument("--field", default="vs", choices=("vs", "vs_mean", "width", "iface", "zeta", "zeta_shrink"),
                    help="vs = posterior median (volume_<ws>_gridded.npz); vs_mean = posterior mean "
                         "(…_gridded_vs_mean.npz, from grid_model_surface.py --field vs_mean), drawn on "
                         "the same scale as vs; width = half the p16-p84 "
                         "posterior width (…_gridded_width.npz, from grid_model_surface.py --field width); "
                         "iface = ensemble interface probability P(z) (…_gridded_iface.npz, from "
                         "iface_volume.py + grid_model_surface.py --field iface); zeta = radial anisotropy "
                         "median; zeta_shrink = prior-to-posterior shrinkage of gamma (1 resolved, 0 prior)")
    ap.add_argument("--mask-shrink", type=float, default=None, metavar="S",
                    help="with --field zeta: blank nodes whose gamma shrinkage (…_gridded_zeta_shrink.npz) "
                         "is below S, i.e. show zeta only where the data resolved it (suggested 0.5)")
    ap.add_argument("--ws", default="*", help="waveset of the gridded volume to slice (fund, fundlove, ...); default any")
    ap.add_argument("--smooth-km", default=None, metavar="H,V",
                    help="3-D Gaussian smoothing of the gridded volume BEFORE slicing, sigma in km "
                         "horizontally and vertically (e.g. 0.75,0.10); NaN-aware (normalised convolution)")
    ap.add_argument("--vmin", type=float, default=None, help="colour-scale minimum (default 1.5)")
    ap.add_argument("--vmax", type=float, default=None, help="colour-scale maximum (default 4.0)")
    ap.add_argument("--calibrate-basement", action="store_true",
                    help="centre the colour scale on the arm's Vs at GVL-1's basement top")
    ap.add_argument("--topo-smooth-km", type=float, default=0.0, metavar="SIGMA",
                    help="hang the model on the topography smoothed by a Gaussian of this sigma (km) "
                         "while still drawing the real topography; 0 = raw DEM (old behaviour). ~1 km "
                         "matches the model's lateral resolution")
    ap.add_argument("--suffix", default="",
                    help="appended to the output filename so variants never overwrite")
    ap.add_argument("--endpoints", action="append", default=[], metavar="KEY=E1,N1,E2,N2",
                    help="override a section's LV95 endpoints (repeatable), e.g. "
                         "--endpoints BB=2568000,1242491,2598000,1242491. Default = the verified "
                         "SECTIONS table in section_figure.py (GS 2026-08-16).")
    ap.add_argument("--campaign",
                    default=f"{E}/Projects/hautesorne/tomo/2_vs_depth_inversion/vs_prod3")
    a = ap.parse_args()
    for spec in a.endpoints:
        k, v = spec.split("=", 1); E1, N1, E2, N2 = (float(x) for x in v.split(","))
        SECTIONS[k] = SECTIONS[k][:3] + ((E1, N1), (E2, N2))
        print(f"  endpoint override {k}: ({E1:.0f},{N1:.0f}) -> ({E2:.0f},{N2:.0f})")
    suffix = "" if a.field == "vs" else f"_{a.field}"
    if a.field != "vs" and not a.suffix:
        a.suffix = f"_{a.field}"
    vf = sorted(glob.glob(f"{a.campaign}/{a.arm}/volume_{a.ws}_gridded{suffix}.npz"))
    if not vf:
        raise SystemExit(f"no 3-D gridded volume for {a.arm} "
                         f"(run smooth_maps.py first; sections are sliced from the 3-D model)")
    g = np.load(vf[0], allow_pickle=True)
    if a.mask_shrink is not None:
        if a.field != "zeta":
            raise SystemExit("--mask-shrink only applies to --field zeta")
        sf = vf[0].replace("_gridded_zeta.npz", "_gridded_zeta_shrink.npz")
        if not os.path.exists(sf):
            raise SystemExit(f"no shrinkage volume {sf} (run grid_model_surface.py --field zeta_shrink)")
        S = np.asarray(np.load(sf, allow_pickle=True)["vs"], float)
        gg = dict(g); Z = np.asarray(gg["vs"], float).copy()
        drop = np.isfinite(Z) & ~(S >= a.mask_shrink)
        Z[drop] = np.nan; gg["vs"] = Z; g = gg
        print(f"  zeta masked where gamma shrinkage < {a.mask_shrink}: {drop.sum()} of {np.isfinite(Z).sum() + drop.sum()} nodes blanked")
        a.suffix = a.suffix + f"_masked{a.mask_shrink:g}"
    if a.smooth_km:
        from scipy.ndimage import gaussian_filter
        sh, sv = (float(x) for x in a.smooth_km.split(","))
        gg = dict(g); Vraw = np.asarray(gg["vs"], float)
        dz = float(np.median(np.diff(np.asarray(gg["depth"], float))))
        dx = float(np.median(np.diff(np.asarray(gg["x_km"], float))))
        sig = (sv / dz, sh / dx, sh / dx)                       # (nz, ny, nx)
        ok = np.isfinite(Vraw); num = gaussian_filter(np.where(ok, Vraw, 0.0), sig)
        den = gaussian_filter(ok.astype(float), sig)
        Vs = np.where(ok & (den > 0.05), num / np.maximum(den, 1e-9), np.nan)   # keep the original NaN mask
        gg["vs"] = Vs; g = gg
        print(f"  3-D Gaussian smoothing: sigma_h {sh} km ({sh/dx:.1f} nodes), sigma_v {sv} km ({sv/dz:.1f} nodes)")
    global CMAP, NORM, VMIN, VMAX
    CMAP = a.cmap
    if a.field == "width":
        CMAP, VMIN, VMAX = ("magma_r" if a.cmap in ("RdYlBu", "fig8", "roma") else a.cmap), 0.0, 0.35
    elif a.field == "iface":
        CMAP, VMIN, VMAX = ("Greys" if a.cmap in ("RdYlBu", "fig8", "roma") else a.cmap), 0.0, None
    elif a.field == "zeta":
        CMAP, VMIN, VMAX = ("RdBu_r" if a.cmap in ("RdYlBu", "fig8", "roma") else a.cmap), -0.3, 0.3
    elif a.field == "zeta_shrink":
        CMAP, VMIN, VMAX = ("viridis" if a.cmap in ("RdYlBu", "fig8", "roma") else a.cmap), 0.0, 1.0
    a.cmap = CMAP                                       # the title names the map actually used
    if a.vmin is not None: VMIN = a.vmin
    if a.vmax is not None: VMAX = a.vmax
    cal = ""
    if a.calibrate_basement:
        vc, contrast = basement_vs(a.arm, a.campaign)
        if vc is None:
            print("  GVL-1 too far from any cell -- not calibrating")
        elif abs(contrast) < 0.15:
            print(f"  contrast across the basement top is only {contrast:+.2f} km/s -- "
                  f"this arm does not resolve it; NOT calibrating")
        else:
            NORM = TwoSlopeNorm(vmin=VMIN, vcenter=vc, vmax=VMAX)
            cal = (f"; pale band calibrated to GVL-1 basement top "
                   f"Vs={vc:.2f} km/s (contrast {contrast:+.2f}) - NOTE non-linear scale")
            print(f"  basement top Vs = {vc:.2f} km/s, contrast {contrast:+.2f} km/s")
    strat = load_strat()

    fig, axs = plt.subplots(3, 1, figsize=(13.5, 12.5))
    for ax, key in zip(axs, ("AA", "BB", "CC")):
        im, n, mp = panel(ax, g, key, strat, a.interp, a.step_km, a.datum, a.topo_smooth_km)
        print(f"  {key}: {mp:.2f} km profile, {n} filled pixels")
    axs[-1].set_xlabel("distance along profile (km)", fontsize=8)
    cb = fig.colorbar(im, ax=axs, fraction=0.020, pad=0.012)
    cb.set_label({"vs": "Vs (km/s)", "vs_mean": "Vs (km/s) — posterior mean",
                  "width": "posterior half-width (p84−p16)/2 (km/s)", "iface": "interface probability P(z) (per 0.1 km, z ≥ 0.5 km)", "zeta": "radial anisotropy ζ = (Vsh−Vsv)/V_Voigt (median)" + (f"; blank = γ shrinkage < {a.mask_shrink:g} (prior-dominated)" if a.mask_shrink is not None else ""), "zeta_shrink": "γ shrinkage 1 − Var_post/Var_prior (1 = resolved, 0 = prior)"}[a.field], fontsize=8); cb.ax.tick_params(labelsize=7)
    h = [plt.Rectangle((0, 0), 1, 1, fc=r.hex, ec="0.3", lw=.4) for _, r in strat.iterrows()]
    fig.legend(h, list(strat.Group), loc="lower center", ncol=5, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.44, 0.02))
    fig.suptitle(f"Haute-Sorne — {a.arm}   (sliced from the 3-D tension-spline model, "
                 f"{a.interp}; {a.cmap}; datum={a.datum}{cal}"
                 + (f"; model hung on topography smoothed σ={a.topo_smooth_km:g} km, black line = real "
                    f"topography" if a.topo_smooth_km > 0 and a.datum != "flat" else "")
                 + "; blank = outside reliable window)",
                 fontsize=11, y=0.97)
    od = f"{a.campaign}/{a.arm}/figures/sections_AA_BB_CC"   # where the manuscript set now lives
    out = f"{od if os.path.isdir(od) else a.campaign + '/' + a.arm}/manuscript_sections{a.suffix}.png"
    for key_, mx, gv in PANEL_LOG:
        print(f"  {key_}: hang surface vs real DEM: max |shift| {mx:.0f} m, at GVL-1 {gv:+.0f} m")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
