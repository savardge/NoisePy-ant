#!/usr/bin/env python
"""Vs model figures from a production volume: 500 m depth-slice maps + 1 km-spaced sections.

Design (user spec 2026-07-22):
  MAPS   one figure per depth 0.5..6.0 km step 0.5: Vs mesh (alpha) over DEM hillshade,
         GK500 faults/fold-axes, stations, wells, city labels, scalebar. Color limits are
         per-depth and SHARED across arms via vlims.json written by the first (reference) run,
         so reference-vs-otexempt maps are directly comparable.
  SECTIONS every 1 km along easting (fixed northing / iy) and northing (fixed ix):
         each cell's 1-D profile is snapped to its local DEM surface elevation and the section
         is plotted against ELEVATION (m a.s.l.) -- Y[k,j] = elev_j - 1000*z_k (2-D pcolormesh).
         An exaggerated topographic profile rides on top (own panel, shared x); geophones within
         200 m of the profile are surface triangles; wells within 200 m are vertical lines to
         their true vertical depth below surface with a name label on top; a small inset map
         shows the section line on the hillshade.

Assets from prep_fig_assets.py (DEM + GK500 as plain npz) -- this script needs only
numpy/matplotlib and runs in the bayhunter env:
  PYTHONPATH=~/Codes/NoisePy-ant /opt/anaconda3/envs/bayhunter/bin/python vs_model_figures.py \
      --griddir .../production_2026-07-17_hybrid_recipe --net riehen
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

E = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from well_vs_qc import WELLS                                     # noqa: E402
from noisepy.lv95 import wgs84_to_lv95, extent_lv95_km          # noqa: E402
from noisepy.colormaps import get_cmap                           # noqa: E402

MAP_DEPTHS = np.arange(0.5, 6.01, 0.5)
NEAR_KM = 0.2                                                    # stations/wells within 200 m
CITIES = {
    "riehen": [("Basel", 7.588, 47.560), ("Riehen", 7.651, 47.583),
               ("Lörrach", 7.664, 47.615), ("Weil a. Rhein", 7.620, 47.593)],
    "aargau": [("Brugg", 8.208, 47.481), ("Baden", 8.306, 47.473),
               ("Waldshut", 8.214, 47.623), ("Frick", 8.024, 47.507)],
    # Jura / Delemont valley, inside the hautesorne grid bounds
    "hautesorne": [("Delemont", 7.344, 47.364), ("Glovelier", 7.211, 47.341),
                   ("Bassecourt", 7.244, 47.339), ("Moutier", 7.370, 47.279),
                   ("Saignelegier", 6.996, 47.256), ("Porrentruy", 7.075, 47.417)],
}
SECT_STEP_KM = 1.0


# ------------------------------------------------------------------ small helpers
def bilinear(elev, extent, lon, lat):
    """Sample DEM (row 0 = top/north) at lon/lat arrays."""
    lo0, lo1, la0, la1 = extent
    ny, nx = elev.shape
    fx = (np.asarray(lon) - lo0) / (lo1 - lo0) * (nx - 1)
    fy = (la1 - np.asarray(lat)) / (la1 - la0) * (ny - 1)
    x0 = np.clip(np.floor(fx).astype(int), 0, nx - 2)
    y0 = np.clip(np.floor(fy).astype(int), 0, ny - 2)
    tx, ty = np.clip(fx - x0, 0, 1), np.clip(fy - y0, 0, 1)
    z = (elev[y0, x0] * (1 - tx) * (1 - ty) + elev[y0, x0 + 1] * tx * (1 - ty)
         + elev[y0 + 1, x0] * (1 - tx) * ty + elev[y0 + 1, x0 + 1] * tx * ty)
    return z


def km_xy(lon, lat, lat0=None):
    """LV95 (EPSG:2056) E/N en km — regla fija 2026-07-25: todos los mapas en LV95,
    aspecto igual. lat0 se conserva por compatibilidad de firma (sin uso)."""
    E_, N_ = wgs84_to_lv95(lon, lat)
    return E_ / 1e3, N_ / 1e3


def dist_to_polyline_km(px, py, qx, qy):
    """Min distance from points (px,py) to the polyline (qx,qy), all in km."""
    d = np.full(np.shape(px), np.inf)
    for a in range(len(qx) - 1):
        ax, ay, bx, by = qx[a], qy[a], qx[a + 1], qy[a + 1]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = np.clip(((px - ax) * dx + (py - ay) * dy) / max(L2, 1e-12), 0, 1)
        d = np.minimum(d, np.hypot(px - (ax + t * dx), py - (ay + t * dy)))
    return d


def _hillshade(elev, extent):
    ny, nx = elev.shape
    latm = 0.5 * (extent[2] + extent[3])
    dy = (extent[3] - extent[2]) / max(ny, 1) * 111_000.0
    dx = (extent[1] - extent[0]) / max(nx, 1) * 111_000.0 * np.cos(np.deg2rad(latm))
    z = np.nan_to_num(elev, nan=np.nanmin(elev))
    return LightSource(azdeg=315, altdeg=45).hillshade(z, vert_exag=3.0, dx=dx, dy=dy)


def _tecto(ax, gk, lw=1.0, zorder=3):
    """GK500 lines from the flat (verts, offsets, kindcode) layout: 0/1=fault/thrust, 2=axis."""
    if gk is None:          # not every network has a GK500 asset (hautesorne has DEM only)
        return
    verts, offs, kc = gk["verts"], gk["offsets"], gk["kindcode"]
    for i in range(len(kc)):
        xy = verts[offs[i]:offs[i + 1]]
        gx_, gy_ = km_xy(xy[:, 0], xy[:, 1])
        if kc[i] == 2:
            ax.plot(gx_, gy_, color="royalblue", lw=lw, ls="-.", zorder=zorder)
        else:
            ax.plot(gx_, gy_, color="darkred", lw=lw, zorder=zorder)


def load_zrel(griddir, cells, waveset="fundotlove"):
    """Per-cell z_reliable_max [km] from cells/*.npz, aligned to `cells` order (nan if absent).

    Only a fallback now: assemble_volume carries z_reliable_max in the volume, so a griddir
    holding just the volume npz (e.g. one pulled off the cluster without its per-cell tree)
    masks correctly without this.

    THE RING FIX (user report 2026-07-23): below a cell's data reach the trans-D posterior
    parsimoniously EXTENDS the last constrained velocity downward, so short-reach (rim) cells
    paint an unphysical slow ring at depth. Measured on the riehen reference: cells beyond
    their z_reliable_max are 0.3-0.8 km/s slower than reached cells at the same depth, deficit
    growing downward; at 2 km 56% of cells are beyond reach. Everything below z_reliable_max
    (chain-agreement AND wavelength floor, the wells-validated product of vs_reliability) is
    therefore MASKED in maps and sections -- prior fill is not a measurement.
    """
    out = np.full(len(cells), np.nan)
    for i, (cix, ciy) in enumerate(cells):
        fp = os.path.join(griddir, "cells", f"cell_{int(cix)}_{int(ciy)}_{waveset}.npz")
        if os.path.exists(fp):
            r = np.load(fp, allow_pickle=True)
            if "z_reliable_max" in r.files:
                out[i] = float(r["z_reliable_max"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--griddir", required=True)
    ap.add_argument("--net", required=True, choices=("riehen", "aargau", "hautesorne"))
    ap.add_argument("--waveset", default="fundotlove",
                    help="which volume_<waveset>.npz to render: fund (Rayleigh fundamental), "
                         "love, fundlove, fundotlove. The vs_prod3 single-target configs are "
                         "fund (R0g/R0p) and love (L0g/L0p).")
    ap.add_argument("--label", default=None,
                    help="suffix for figure filenames + titles, e.g. the config name R0g. "
                         "Lets several configs of one network write into distinct files.")
    ap.add_argument("--vlims-mode", default="auto", choices=("auto", "own", "shared"),
                    help="'own' = per-ARM per-depth p2/p98 (never saturates, but arms are not "
                         "directly comparable); 'shared' = reuse --vlims-from; 'auto' (default) "
                         "= shared when --vlims-from is given, own otherwise. A shared scale "
                         "calibrated on a GROUP arm saturates a PHASE arm badly -- phase Vs "
                         "runs ~0.5 km/s faster, so its median can exceed the group p98.")
    ap.add_argument("--gridded-maps", action="store_true",
                    help="draw the MAPS from volume_<ws>_gridded.npz (the 3-D tension-spline "
                         "field) instead of the raw cell nodes. Cells are inverted "
                         "INDEPENDENTLY in 1-D with no lateral coupling, so a per-cell map "
                         "steps abruptly wherever neighbouring columns place an interface at "
                         "different depths -- a fixed-depth slice then crosses that interface "
                         "laterally and paints a sharp-edged 'anomaly'. Sections already use "
                         "the spline (--gridded), so without this, maps and sections of the "
                         "SAME arm show different fields.")
    ap.add_argument("--vlims-pct", default="2,98",
                    help="percentiles for the 'own' colour scale, 'lo,hi'. The 2,98 default "
                         "CLIPS 4%% of cells: harmless when the arm spans a wide Vs range, but "
                         "on a narrow one (hautesorne aniso2t at 1 km spans 1.72-2.44 km/s) the "
                         "clipped tails are large contiguous patches and the map reads as "
                         "saturated. Use 0,100 for a scale that clips nothing.")
    ap.add_argument("--vlims-from", default=None,
                    help="reference griddir whose figures/vlims.json fixes the color scale "
                         "(default: use/create this griddir's own)")
    ap.add_argument("--unmasked", action="store_true",
                    help="disable the below-reach reliability mask (artifact demo only)")
    ap.add_argument("--also-unmasked", action="store_true",
                    help="ALSO write vs_map_z*km_nomask.png showing the prior-filled values "
                         "the mask hides, with the reach boundary outlined. A fully blank "
                         "masked map (e.g. riehen R0g at 4 km, where 0%% of cells reach) is "
                         "indistinguishable from a broken figure without this companion.")
    ap.add_argument("--topo-smooth-km", type=float, default=1.0, metavar="SIGMA",
                    help="sections: hang the columns on the DEM smoothed with a 2-D Gaussian of this sigma "
                         "[km] (0 = raw). The top strip still draws the real ground; the hang is the dashed "
                         "line. 1.0 per the Haute-Sorne imprint test (2026-09-25).")
    ap.add_argument("--maps-only", action="store_true",
                    help="skip the section figures (much faster when only maps changed)")
    ap.add_argument("--disagree-ref", default=None,
                    help="griddir of a REFERENCE arm (typically the phase twin). Renders an "
                         "extra copy of every section with hatching where this arm's Vs and "
                         "the reference's disagree beyond their combined 1-sigma posteriors -- "
                         "|mA-mB| > sqrt(sigA^2+sigB^2), sig=(p84-p16)/2, both depths inside "
                         "each arm's own reliable_mask. Diagnosed on the raw cells (measurement "
                         "resolution), drawn on top of whatever the section shows.")
    ap.add_argument("--disagree-waveset", default=None,
                    help="waveset of the reference volume (default: same as --waveset)")
    ap.add_argument("--disagree-suffix", default="gpdisagree",
                    help="filename suffix for the overlay figures: sect_*_<suffix>.png")
    ap.add_argument("--disagree-min-sigma", type=float, default=2.0,
                    help="hatch only where |dVs| exceeds this many combined sigmas")
    ap.add_argument("--disagree-min-relgap", type=float, default=0.25,
                    help="AND exceeds this fraction of the mean Vs. Needed because the group "
                         "arms are biased slow nearly everywhere (median gap 3.4 sigma, 21%% "
                         "relative vs R0p): a sigma cut alone hatches the whole section, "
                         "hiding the structural contradictions inside a pervasive offset.")
    ap.add_argument("--anomaly", action="store_true",
                    help="draw SECTIONS as the Vs anomaly in %% relative to the network median Vs at "
                         "the SAME depth, instead of absolute Vs (Lehujeur et al. 2018, Soultz). The "
                         "reference is the median over all reliable cells at each depth, so the "
                         "strong vertical gradient -- which otherwise makes the whole shallow "
                         "Mesozoic red and hides everything else -- is removed and LATERAL structure "
                         "(a graben, a basement high) is what the colour shows. Writes into "
                         "sections_anomaly/ so the absolute set is never overwritten.")
    ap.add_argument("--alims", type=float, default=None, metavar="PCT",
                    help="symmetric colour limit for --anomaly in %% (default: the 98th percentile "
                         "of |anomaly| over the whole reliable volume, rounded up to 5%%)")
    ap.add_argument("--anomaly-cmap", default="RdBu",
                    help="diverging colormap for --anomaly (default RdBu: red = SLOW, blue = fast, "
                         "the house convention for anomaly maps)")
    ap.add_argument("--vs-field", default="vs_median", choices=("vs_median", "vs_mean"),
                    help="which posterior point estimate to draw. vs_mean routes every output to "
                         "figures/maps_mean and figures/sections[_anomaly]_mean so the median and "
                         "mean figure sets stay distinct, and (with --gridded/--gridded-maps) reads "
                         "volume_<ws>_gridded_vs_mean.npz from grid_model_surface.py --field vs_mean. "
                         "Pair a vs_mean run with --vlims-from <this griddir> so both sets share the "
                         "median run's color limits.")
    ap.add_argument("--gridded", action="store_true",
                    help="draw SECTIONS from volume_<ws>_gridded.npz (GMT surface tension "
                         "spline) instead of raw cell columns. Each cell is one measurement at a "
                         "discrete node, so raw sections are blocky at the parameterisation "
                         "spacing; build the file with grid_model_surface.py (pygmt env).")
    ap.add_argument("--cmap", default="RdYlBu",
                    help="colour scale for Vs on maps and sections. A Crameri name "
                         "(\"roma\", \"roma_r\", \"cmc.roma\") or any matplotlib name. Default "
                         "`roma` since 2026-09-06 (user preference), replacing `RdYlBu`: RdYlBu "
                         "is not perceptually uniform -- its bright band near the middle of the "
                         "scale reads as an interface where the model is smooth. Pass "
                         "--cmap RdYlBu to reproduce the earlier figures.")
    ap.add_argument("--rail-tol", type=float, default=0.05,
                    help="report cells within this of the volume-wide Vs min/max as "
                         "PRIOR-RAILED: at those depths the map shows the prior, and no "
                         "colour scale can fix it")
    ap.add_argument("--measure", choices=("group", "phase"), default=None,
                    help="which measure this arm inverted; with --period-ranges it puts the "
                         "inverted period band in every title. Inferred from --label when it "
                         "ends in g/p (R0g -> group, R0p -> phase).")
    ap.add_argument("--period-ranges", default=f"{E}/method_tests/5_vs_inversion/period_band_selection/"
                                               "period_ranges_DECISIONS_v1.csv",
                    help="the CSV the inversion was trimmed with; its band is annotated on "
                         "every figure. The band is not cosmetic -- a map at 0.5 km made from "
                         "data starting at 0.72 s has no short-period constraint on that depth "
                         "and is showing downward-extended structure, so the reader needs it.")
    a = ap.parse_args()
    net = a.net
    vcmap = get_cmap(a.cmap)
    # period band actually inverted, for the titles
    measure = a.measure
    if measure is None and a.label:
        measure = {"g": "group", "p": "phase"}.get(a.label.strip()[-1:].lower())
    tband = ""
    if measure and a.period_ranges and os.path.exists(a.period_ranges):
        import csv as _csv
        # waveset -> the waves it actually inverts. The old {"fund","love"} lookup returned
        # None for "fundot" and the fallback listed love in a fund+overtone arm's titles.
        want = {"fund": ["fund"], "fundot": ["fund", "overtone"], "love": ["love"],
                "fundlove": ["fund", "love"],
                "fundotlove": ["fund", "overtone", "love"]}.get(a.waveset,
                                                                ["fund", "overtone", "love"])
        bits = []
        for row in _csv.DictReader(open(a.period_ranges)):
            if (row.get("net") or "").strip() != net:
                continue
            if (row.get("measure") or "").strip() != measure:
                continue
            if (row.get("wave") or "").strip() not in want:
                continue
            lo, hi = (row.get("T_valid_min") or "").strip(), (row.get("T_valid_max") or "").strip()
            if lo or hi:
                bits.append(f"{row['wave']} {lo or 'open'}-{hi or 'open'} s")
        if bits:
            tband = f"{measure} T: " + ", ".join(bits)
    v = np.load(os.path.join(a.griddir, f"volume_{a.waveset}.npz"), allow_pickle=True)
    cells, lonlat, z = v["cells"], v["lonlat"], np.asarray(v["depth"], float)
    if a.vs_field not in v.files:
        raise SystemExit(f"--vs-field {a.vs_field}: not in volume_{a.waveset}.npz -- volumes "
                         f"assembled before 2026-09-11 lack vs_mean (re-assemble, or stack it "
                         f"from the cells as in the D-prime arm's add_vs_mean.py)")
    vs = np.asarray(v[a.vs_field], float)                        # (ncell, ndepth)
    STAT = "median" if a.vs_field == "vs_median" else "mean"
    # output-dir suffix keeping the mean set distinct from the (default) median set
    _statsfx = "" if a.vs_field == "vs_median" else "_mean"
    _gridsfx = "" if a.vs_field == "vs_median" else "_vs_mean"   # grid_model_surface naming
    vs_raw = vs.copy()                                           # pre-mask, for --also-unmasked
    zrel = np.full(len(cells), np.nan)
    if not a.unmasked:
        # prefer the copy carried in the volume; fall back to the per-cell tree
        if "z_reliable_max" in v.files:
            zrel = np.asarray(v["z_reliable_max"], float)
        else:
            zrel = load_zrel(a.griddir, cells, a.waveset)
        # The SHALLOW limit matters as much as the deep one. Vantassel & Cox (2021) put the
        # thinnest resolvable layer at lam_min/3, so a band starting at 1.08 s (lam ~1.95 km in
        # aargau) resolves nothing above ~650 m -- whereas phase from 0.30 s reaches ~140 m.
        # vs_reliability already computes this as z_reliable_min, and it is NOT zero: 20% of
        # aargau R0g cells have it above 0.5 km (p90 1.10 km). Showing Vs above it presents
        # parameterisation as measurement, exactly the error the deep mask exists to prevent.
        zmin = (np.asarray(v["z_reliable_min"], float) if "z_reliable_min" in v.files
                else np.zeros(len(cells)))
        # Prefer the per-depth reliable_mask when the volume carries it. z_reliable_min/max is
        # only the LONGEST contiguous reliable run, so a cell reliable both shallow AND deep
        # with an unresolved band between loses the shorter band -- 13-17% of aargau cells, and
        # in 5-9% the lost band is the SHALLOW one, which is what made sections show a masked
        # shallow column sitting on unmasked deep rock. The interval remains the fallback for
        # volumes assembled before the mask was carried.
        relmask = (np.asarray(v["reliable_mask"], bool)
                   if "reliable_mask" in v.files
                   and np.shape(v["reliable_mask"]) == (len(cells), len(z))
                   else None)
        n_masked = n_shallow = 0
        for i in range(len(cells)):
            if relmask is not None:
                m = ~relmask[i]
            else:
                m = np.zeros(len(z), bool)
                if np.isfinite(zrel[i]):
                    m |= z > zrel[i] + 1e-9
                if np.isfinite(zmin[i]) and zmin[i] > 0:
                    m |= z < zmin[i] - 1e-9
            if np.isfinite(zmin[i]) and zmin[i] > 0:
                n_shallow += int(np.isfinite(vs[i, m & (z < zmin[i] - 1e-9)]).sum())
            n_masked += int(np.isfinite(vs[i, m]).sum())
            vs[i, m] = np.nan
        how = ("per-depth reliable_mask" if relmask is not None
               else "per-cell z_reliable_min..max interval")
        print(f"reliability mask: NaN outside {how} "
              f"({n_masked} samples masked across {len(cells)} cells; "
              f"{n_shallow} of them ABOVE z_reliable_min)")
    dem = np.load(f"{E}/{net}/tomo/2_vs_depth_inversion/fig_assets_{net}_dem.npz")
    elev, extent = dem["elev"].astype(float), dem["extent"]
    gk_path = f"{E}/{net}/tomo/2_vs_depth_inversion/fig_assets_{net}_gk500.npz"
    gk = np.load(gk_path, allow_pickle=True) if os.path.exists(gk_path) else None
    if gk is None:
        print(f"note: no GK500 asset for {net} -- maps drawn without faults/fold axes")
    # The 2026-08-05 reorg moved inputs/ -> 0_inputs/configs/; try the current layout first and
    # fall back to the legacy one so this works against either tree.
    _st_cands = [f"{E}/{net}/tomo/1_velocity_maps/0_inputs/configs/stations.csv",
                 f"{E}/{net}/tomo/1_velocity_maps/inputs/stations.csv"]
    _st = next((p for p in _st_cands if os.path.exists(p)), None)
    if _st is None:
        raise SystemExit("no stations.csv found; tried:\n  " + "\n  ".join(_st_cands))
    st = np.genfromtxt(_st, delimiter=",", names=True)
    hs = _hillshade(elev, extent)
    import dem_hillshade as DH
    elev_hang = DH.smooth_dem(elev, extent, a.topo_smooth_km)     # datum for the section columns
    # optional Geo2Riehen seismic horizons (riehen; prep_fig_assets --stage geo2riehen)
    hz_path = f"{E}/{net}/tomo/2_vs_depth_inversion/fig_assets_{net}_horizons.npz"
    horizons = np.load(hz_path) if os.path.exists(hz_path) else None
    figdir = os.path.join(a.griddir, "figures")
    MAPDIR = "maps" + _statsfx
    os.makedirs(os.path.join(figdir, MAPDIR), exist_ok=True)
    # Clear this arm's OWN map outputs before rewriting them. Without this, an arm re-run
    # against a shallower volume leaves the previous run's deeper maps in place: riehen/R0g
    # carried vs_map_z4.0-5.5km.png from an earlier, deeper volume while the current one
    # reaches only 3.5 km, so four figures showed a masked -- i.e. apparently measured --
    # field at depths no cell reaches. Only files this script writes are removed.
    stale = sorted(glob.glob(os.path.join(figdir, MAPDIR, "vs_map_z*km.png"))
                   + glob.glob(os.path.join(figdir, MAPDIR, "vs_map_z*km_nomask.png")))
    for f_ in stale:
        os.remove(f_)
    if stale:
        print(f"cleared {len(stale)} existing map figure(s) before rewriting")
    SECDIR = ("sections_anomaly" if a.anomaly else "sections") + _statsfx
    os.makedirs(os.path.join(figdir, SECDIR), exist_ok=True)
    arm = a.label or os.path.basename(os.path.normpath(a.griddir))
    lat0 = float(np.mean(lonlat[:, 1]))

    _PLO, _PHI = (float(x) for x in a.vlims_pct.split(","))
    # shared color limits (reference writes, comparison arms reuse)
    use_shared = (a.vlims_mode == "shared" or (a.vlims_mode == "auto" and a.vlims_from))
    vl_path = os.path.join((a.vlims_from or a.griddir), "figures", "vlims.json")
    if use_shared and os.path.exists(vl_path):
        vlims = {float(k): tuple(vv) for k, vv in json.load(open(vl_path)).items()}
        print(f"vlims from {vl_path}")
    else:
        vlims = {}
        for d in MAP_DEPTHS:
            k = int(np.argmin(np.abs(z - d)))
            col = vs[:, k][np.isfinite(vs[:, k])]
            # a depth below EVERY cell's reach is fully masked -- no map to draw there, and
            # percentile of an empty slice raises. Skip it; the map loop skips it too.
            if col.size == 0:
                print(f"  depth {d:g} km: no cell reaches it (fully masked) -- skipped")
                continue
            vlims[float(d)] = (float(np.percentile(col, _PLO)), float(np.percentile(col, _PHI)))
        os.makedirs(os.path.dirname(vl_path), exist_ok=True)
        vl_out = os.path.join(a.griddir, "figures", "vlims.json")
        os.makedirs(os.path.dirname(vl_out), exist_ok=True)
        json.dump({str(k): vv for k, vv in vlims.items()}, open(vl_out, "w"))
        print(f"vlims written -> {vl_out}")

    # SATURATION GUARD. A colour scale borrowed from another arm can clip most of the map
    # while the figure still looks plausible -- exactly what happened to hautesorne/L0p under
    # L0g's limits (L0p median 2.88 > L0g p98 2.84, so >50% of cells were pinned).
    worst = 0.0
    for d, (lo_, hi_) in sorted(vlims.items()):
        k = int(np.argmin(np.abs(z - d)))
        col = vs[:, k][np.isfinite(vs[:, k])]
        if col.size == 0:
            continue
        f = float(np.mean((col < lo_) | (col > hi_)))
        worst = max(worst, f)
        # OWN limits are p2/p98, so ~4% lies outside BY CONSTRUCTION -- warning at 5% fired on
        # every arm and meant nothing. Only flag a genuine excess, and never tell the user to
        # "use own limits" when own limits are already in use.
        thresh = 0.12 if not use_shared else 0.05
        if f > thresh:
            print(f"  !! depth {d:g} km: {100*f:.0f}% of cells OUTSIDE the colour range "
                  f"[{lo_:.2f}, {hi_:.2f}] (data {col.min():.2f}-{col.max():.2f})")
    if worst > (0.12 if not use_shared else 0.05):
        if use_shared:
            print(f"  !! worst-depth saturation {100*worst:.0f}% against the BORROWED scale "
                  f"from {os.path.basename(os.path.dirname(vl_path))} -- rerun with "
                  f"--vlims-mode own")
        else:
            print(f"  !! worst-depth saturation {100*worst:.0f}% even with this arm's OWN "
                  f"p2/p98 limits -- the distribution is strongly skewed (check for prior "
                  f"railing below)")

    # Limits from the UNMASKED field, used only for the _nomask companion at depths where the
    # mask removes every cell -- otherwise those depths disappear from the output and "show me
    # the velocities there" is unanswerable.
    vlims_raw = {}
    for d in MAP_DEPTHS:
        k = int(np.argmin(np.abs(z - d)))
        col = vs_raw[:, k][np.isfinite(vs_raw[:, k])]
        if col.size:
            vlims_raw[float(d)] = (float(np.percentile(col, _PLO)), float(np.percentile(col, _PHI)))

    # PRIOR-RAIL REPORT. Distinct from saturation: when most cells sit within rail-tol of the
    # volume-wide Vs min/max, the posterior has hit the PRIOR BOUND and the map is showing the
    # prior, not the data. No colour scale can repair that -- the depth simply is not resolved.
    _f = vs_raw[np.isfinite(vs_raw)]
    if _f.size:
        _lo, _hi = _f.min(), _f.max()
        for d in sorted(vlims):
            k = int(np.argmin(np.abs(z - d)))
            col = vs[:, k][np.isfinite(vs[:, k])]        # masked values = what is plotted
            if col.size == 0:
                continue
            fr = float(np.mean(col > _hi - a.rail_tol))
            if fr > 0.20:
                print(f"  !! depth {d:g} km: {100*fr:.0f}% of PLOTTED cells are within "
                      f"{a.rail_tol} km/s of the Vs prior ceiling {_hi:.2f} -- this slice is "
                      f"prior-railed, not resolved")

    # 3-D tension-spline field for --gridded-maps. x_km/y_km are LV95 KILOMETRES, the same
    # frame the map axes use, so the slice drops onto the map with no re-projection.
    _GM = None
    if a.gridded_maps:
        _gf = os.path.join(a.griddir, f"volume_{a.waveset}_gridded{_gridsfx}.npz")
        if not os.path.exists(_gf):
            raise SystemExit(f"--gridded-maps: {_gf} not found -- build it with "
                             f"grid_model_surface.py (pygmt env) first")
        _g = np.load(_gf, allow_pickle=True)
        _GM = {"z": np.asarray(_g["depth"], float), "vs": np.asarray(_g["vs"], float),
               "x": np.asarray(_g["x_km"], float), "y": np.asarray(_g["y_km"], float)}
        print(f"maps drawn from the 3-D spline: {os.path.basename(_gf)} {_GM['vs'].shape}")

    # affine node grid (regular in ix/iy)
    ix, iy = cells[:, 0].astype(float), cells[:, 1].astype(float)
    A = np.column_stack([np.ones_like(ix), ix, iy])
    clon = np.linalg.lstsq(A, lonlat[:, 0], rcond=None)[0]
    clat = np.linalg.lstsq(A, lonlat[:, 1], rcond=None)[0]
    nx, ny = int(ix.max()) + 1, int(iy.max()) + 1
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    B = np.column_stack([np.ones(gx.size), gx.ravel(), gy.ravel()])
    lon2d, lat2d = (B @ clon).reshape(nx, ny), (B @ clat).reshape(nx, ny)
    e2d, n2d = km_xy(lon2d, lat2d)                               # LV95 km (regla de mapas)
    extent_km = extent_lv95_km(extent)
    stx, sty = km_xy(st["longitude"], st["latitude"])
    dx_km = abs(clon[1]) * 111.0 * np.cos(np.deg2rad(lat0))      # grid step along ix [km]
    dy_km = abs(clat[2]) * 111.0                                 # grid step along iy [km]

    wells = [(nm, la, lo, dep) for nm, la, lo, dep in WELLS.get(net, [])]

    # ------------------------------------------------------------------ A. depth-slice maps
    def _grid(vals):
        g = np.full((nx, ny), np.nan)
        for (cix, ciy), val in zip(cells, vals):
            g[int(cix), int(ciy)] = val
        return g

    def _reach_grid(d):
        """1 where the cell's data reach this depth, 0 where the mask hides it."""
        r = np.full((nx, ny), np.nan)
        for i, (cix, ciy) in enumerate(cells):
            if np.isfinite(zrel[i]):
                r[int(cix), int(ciy)] = 1.0 if zrel[i] >= d - 1e-9 else 0.0
        return r

    for d in MAP_DEPTHS:
        if float(d) not in vlims and float(d) not in vlims_raw:
            continue
        k = int(np.argmin(np.abs(z - d)))
        g = _grid(vs[:, k])
        g_map = gx_ = gy_ = None
        if _GM is not None:
            kg = int(np.argmin(np.abs(_GM["z"] - d)))
            g_map, gx_, gy_ = _GM["vs"][kg], _GM["x"], _GM["y"]
        reach = _reach_grid(d)
        vmin, vmax = vlims.get(float(d), vlims_raw.get(float(d)))
        if float(d) not in vlims:
            # nothing survives the mask here: emit only the _nomask companion below
            if a.also_unmasked:
                gr = _grid(vs_raw[:, k])
                figx, axx = plt.subplots(figsize=(7.6, 7.2))
                axx.imshow(hs, extent=extent_km, cmap="gray", origin="upper", zorder=0)
                pcx = axx.pcolormesh(e2d, n2d, gr, cmap=vcmap, vmin=vmin, vmax=vmax,
                                     alpha=0.68, shading="nearest", zorder=1)
                _tecto(axx, gk)
                axx.plot(stx, sty, ".", color="k", ms=1.6, alpha=0.55, zorder=4)
                axx.set_xlim(extent_km[0], extent_km[1]); axx.set_ylim(extent_km[2], extent_km[3])
                axx.set_aspect("equal")
                axx.set_xlabel("E [km LV95]"); axx.set_ylabel("N [km LV95]")
                axx.set_title(f"{net} Vs {STAT} at {d:g} km depth — {arm}\n"
                              f"UNMASKED — NO cell reaches this depth: 100% prior fill, "
                              f"not a measurement"
                              + (f"\n{tband}" if tband else ""), fontsize=10)
                plt.colorbar(pcx, ax=axx, fraction=0.04, pad=0.02).set_label("Vs [km/s]")
                ox = os.path.join(figdir, MAPDIR, f"vs_map_z{d:03.1f}km_nomask.png")
                figx.tight_layout(); figx.savefig(ox, dpi=150); plt.close(figx)
            continue
        fig, ax = plt.subplots(figsize=(7.6, 7.2))
        ax.imshow(hs, extent=extent_km, cmap="gray", origin="upper", zorder=0)
        pc = ax.pcolormesh(*((gx_, gy_, g_map) if g_map is not None else (e2d, n2d, g)),
                           cmap=vcmap, vmin=vmin, vmax=vmax,
                           alpha=0.68, shading="nearest", zorder=1)
        _tecto(ax, gk)
        ax.plot(stx, sty, ".", color="k", ms=1.6, alpha=0.55, zorder=4)
        for nm, la, lo, dep in wells:
            wx, wy = km_xy(lo, la)
            ax.plot(wx, wy, "s", mfc="k", mec="w", ms=7, zorder=5)
            ax.annotate(nm, (wx, wy), xytext=(4, 4), textcoords="offset points",
                        fontsize=7.5, fontweight="bold", zorder=6,
                        bbox=dict(fc="w", alpha=0.65, ec="none", pad=1))
        for nm, lo, la in CITIES[net]:
            cx, cy = km_xy(lo, la)
            ax.plot(cx, cy, "o", mfc="w", mec="k", ms=4, zorder=5)
            ax.annotate(nm, (cx, cy), xytext=(4, -8), textcoords="offset points",
                        fontsize=7, style="italic", zorder=6)
        # 2 km scalebar, bottom left (ejes ya en km, largo literal)
        sb_x = extent_km[0] + 0.08 * (extent_km[1] - extent_km[0])
        sb_y = extent_km[2] + 0.05 * (extent_km[3] - extent_km[2])
        ax.plot([sb_x, sb_x + 2.0], [sb_y, sb_y], "k-", lw=3, zorder=6)
        ax.annotate("2 km", (sb_x + 1.0, sb_y), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8, zorder=6)
        ax.set_xlim(extent_km[0], extent_km[1]); ax.set_ylim(extent_km[2], extent_km[3])
        ax.set_aspect("equal")
        ax.set_xlabel("E [km LV95]"); ax.set_ylabel("N [km LV95]")
        ax.set_title(f"{net} Vs {STAT} at {d:g} km depth — {arm}"
                     + (f"\n{tband}" if tband else ""), fontsize=11)
        plt.colorbar(pc, ax=ax, fraction=0.04, pad=0.02).set_label("Vs [km/s]")
        if np.isfinite(reach).any() and 0 < np.nanmean(reach) < 1:
            ax.contour(e2d, n2d, np.nan_to_num(reach), levels=[0.5], colors="k",
                       linewidths=1.4, linestyles="--", zorder=4)
        out = os.path.join(figdir, MAPDIR, f"vs_map_z{d:03.1f}km.png")
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)

        if a.also_unmasked:
            gr = _grid(vs_raw[:, k])
            fig2, ax2 = plt.subplots(figsize=(7.6, 7.2))
            ax2.imshow(hs, extent=extent_km, cmap="gray", origin="upper", zorder=0)
            pc2 = ax2.pcolormesh(e2d, n2d, gr, cmap=vcmap, vmin=vmin, vmax=vmax,
                                 alpha=0.68, shading="nearest", zorder=1)
            _tecto(ax2, gk)
            if np.isfinite(reach).any():
                ax2.contour(e2d, n2d, np.nan_to_num(reach), levels=[0.5], colors="k",
                            linewidths=1.6, linestyles="--", zorder=4)
            ax2.plot(stx, sty, ".", color="k", ms=1.6, alpha=0.55, zorder=4)
            ax2.set_xlim(extent_km[0], extent_km[1]); ax2.set_ylim(extent_km[2], extent_km[3])
            ax2.set_aspect("equal")
            ax2.set_xlabel("E [km LV95]"); ax2.set_ylabel("N [km LV95]")
            frac = float(np.nanmean(reach)) if np.isfinite(reach).any() else float("nan")
            ax2.set_title(f"{net} Vs {STAT} at {d:g} km depth — {arm}\n"
                          f"UNMASKED (dashed = reliability boundary; "
                          f"{100*frac:.0f}% of cells reach this depth)", fontsize=10)
            plt.colorbar(pc2, ax=ax2, fraction=0.04, pad=0.02).set_label("Vs [km/s]")
            o2 = os.path.join(figdir, MAPDIR, f"vs_map_z{d:03.1f}km_nomask.png")
            fig2.tight_layout(); fig2.savefig(o2, dpi=150); plt.close(fig2)
    print(f"maps: {len(MAP_DEPTHS)} -> {figdir}/{MAPDIR}/"
          + ("  (+ _nomask companions)" if a.also_unmasked else ""))
    if a.maps_only:
        print("maps-only: sections skipped")
        return

    # ------------------------------------------------------------------ B. sections
    GRID = None
    if a.gridded:
        gf = os.path.join(a.griddir, f"volume_{a.waveset}_gridded{_gridsfx}.npz")
        if not os.path.exists(gf):
            raise SystemExit(f"--gridded: {gf} not found. Build it with:\n"
                             f"  /opt/anaconda3/envs/pygmt-env/bin/python "
                             f"scripts/picking/grid_model_surface.py --griddir {a.griddir} "
                             f"--waveset {a.waveset}")
        G = np.load(gf, allow_pickle=True)
        if not np.allclose(np.asarray(G["depth"], float), z):
            raise SystemExit("--gridded: depth axis differs from the raw volume; regrid")
        from pyproj import Transformer as _T
        GRID = dict(x=np.asarray(G["x_km"], float), y=np.asarray(G["y_km"], float),
                    vs=np.asarray(G["vs"], float),
                    tr=_T.from_crs("EPSG:4326", "EPSG:2056", always_xy=True),
                    tension=float(G["tension"]), cell=float(G["cell_km"]),
                    refine=int(G["refine"]))
        print(f"sections from GMT-surface grid: tension {GRID['tension']}, "
              f"{GRID['cell']/GRID['refine']:.3f} km nodes")

    def sample_grid(plon, plat, n_along):
        """Vs (ndepth, n_along) sampled from the tension-spline grid along a profile."""
        from scipy.interpolate import RegularGridInterpolator
        t = np.linspace(0, 1, n_along)
        t0 = np.linspace(0, 1, len(plon))
        f_lon, f_lat = np.interp(t, t0, plon), np.interp(t, t0, plat)
        gx, gy = GRID["tr"].transform(f_lon, f_lat)
        pts = np.c_[np.asarray(gy) / 1000.0, np.asarray(gx) / 1000.0]
        out = np.full((len(z), n_along), np.nan)
        for k in range(len(z)):
            sl = GRID["vs"][k]
            if not np.isfinite(sl).any():
                continue
            # NaN outside support must STAY NaN: interpolate the validity mask alongside the
            # values and re-blank, else the spline's blanked halo bleeds back in as real data.
            fi = RegularGridInterpolator((GRID["y"], GRID["x"]),
                                         np.where(np.isfinite(sl), sl, 0.0),
                                         bounds_error=False, fill_value=np.nan)
            mi = RegularGridInterpolator((GRID["y"], GRID["x"]),
                                         np.isfinite(sl).astype(float),
                                         bounds_error=False, fill_value=0.0)
            out[k] = np.where(mi(pts) > 0.98, fi(pts), np.nan)
        return out, f_lon, f_lat

    DIS = None                      # (ncells, ndepth) bool: this arm vs reference disagree
    if a.disagree_ref:
        rws = a.disagree_waveset or a.waveset
        rf = os.path.join(a.disagree_ref, f"volume_{rws}.npz")
        Rv = np.load(rf, allow_pickle=True)
        if not np.allclose(np.asarray(Rv["depth"], float), z):
            raise SystemExit("--disagree-ref: depth axis differs")
        rkey = {tuple(c): i for i, c in enumerate(Rv["cells"])}
        rmed = np.asarray(Rv["vs_median"], float)
        rsig = 0.5 * (np.asarray(Rv["vs_p84"], float) - np.asarray(Rv["vs_p16"], float))
        rrel = (np.asarray(Rv["reliable_mask"], bool) if "reliable_mask" in Rv.files
                else np.isfinite(rmed))
        amed = np.asarray(v["vs_median"], float)
        asig = 0.5 * (np.asarray(v["vs_p84"], float) - np.asarray(v["vs_p16"], float))
        arel = (np.asarray(v["reliable_mask"], bool) if "reliable_mask" in v.files
                else np.isfinite(amed))
        DIS = np.zeros(amed.shape, bool)
        n_pair = 0
        for i, c in enumerate(cells):
            j = rkey.get(tuple(c))
            if j is None:
                continue
            n_pair += 1
            both = arel[i] & rrel[j] & np.isfinite(amed[i]) & np.isfinite(rmed[j])
            gap = np.abs(amed[i] - rmed[j])
            tol = np.sqrt(asig[i] ** 2 + rsig[j] ** 2)
            relg = gap / np.maximum(0.5 * (amed[i] + rmed[j]), 1e-6)
            DIS[i] = (both & (gap > a.disagree_min_sigma * tol)
                      & (relg > a.disagree_min_relgap))
        frac = DIS.sum() / max((arel & np.isfinite(amed)).sum(), 1)
        dis_label = os.path.basename(a.disagree_ref.rstrip("/"))
        print(f"disagreement overlay vs {dis_label}: {n_pair} paired cells, "
              f"{100*frac:.1f}% of reliable samples flagged "
              f"(>{a.disagree_min_sigma:g} sigma AND >{100*a.disagree_min_relgap:.0f}% rel gap)")

    st_elev = bilinear(elev, extent, st["longitude"], st["latitude"])
    global_vmin = min(vv[0] for vv in vlims.values())
    global_vmax = max(vv[1] for vv in vlims.values())

    # ---- anomaly reference: the network median Vs at each depth, over RELIABLE cells only ----
    # Prior fill below a cell's reach would drag the reference toward the prior, so it is excluded.
    REF = None
    if a.anomaly:
        # `vs` is already NaN outside each cell's reliable window (see the reliability-mask
        # block above), so finiteness IS the reliability mask here.
        vmask = np.isfinite(vs)
        REF = np.array([np.nanmedian(vs[vmask[:, k], k]) if vmask[:, k].any() else np.nan
                        for k in range(len(z))])
        # fill any depth with no reliable cell so the division never yields NaN stripes
        good = np.isfinite(REF)
        if good.sum() < 2:
            raise SystemExit("--anomaly: no reliable cells to build a reference profile")
        REF = np.interp(z, z[good], REF[good])
        anom_all = 100.0 * (np.where(vmask, vs, np.nan) - REF[None, :]) / REF[None, :]
        ALIM = a.alims if a.alims else max(5.0, 5.0 * np.ceil(
            np.nanpercentile(np.abs(anom_all), 98) / 5.0))
        print(f"--anomaly: reference = network median Vs(z) over reliable cells "
              f"({REF[0]:.2f} km/s at {z[0]:.2f} km -> {REF[-1]:.2f} at {z[-1]:.2f}); "
              f"colour limit +-{ALIM:.0f}%  (p98 of |anomaly| = "
              f"{np.nanpercentile(np.abs(anom_all), 98):.1f}%)")

    def one_section(axis, line_idx, name=None):
        """axis='EW': fixed iy, along ix. axis='NS': fixed ix, along iy.

        `name` appends a label to the filename (used for the through-well sections)."""
        if axis == "EW":
            sel = np.where(cells[:, 1] == line_idx)[0]
            order = np.argsort(cells[sel, 0])
        else:
            sel = np.where(cells[:, 0] == line_idx)[0]
            order = np.argsort(cells[sel, 1])
        sel = sel[order]
        if len(sel) < 5:
            return False
        plon, plat = lonlat[sel, 0], lonlat[sel, 1]
        px, py = km_xy(plon, plat, lat0)
        along = np.concatenate([[0], np.cumsum(np.hypot(np.diff(px), np.diff(py)))])
        surf = bilinear(elev_hang, extent, plon, plat)           # hang datum per column (smoothed DEM)
        # The DEM is read BOUNDLESS (see dem_hillshade.load_dem), so ground outside the SRTM
        # tile is NaN. pcolormesh rejects non-finite coordinate arrays outright, so a profile
        # crossing the uncovered strip used to abort the whole figure. Interpolate the surface
        # across the gap from the covered part instead; bail only if nothing is covered.
        if not np.isfinite(surf).all():
            good = np.isfinite(surf)
            if good.sum() < 2:
                print(f"  section {axis} {line_idx}: no DEM coverage on this line -- skipped")
                plt.close(fig) if "fig" in dir() else None
                return False
            surf = np.interp(along, along[good], surf[good])
        if GRID is not None:
            # resample the profile at the grid spacing so the section shows the spline, not the
            # cell columns; geometry (along, topography) is unchanged
            n_f = max(int(along[-1] / (GRID["cell"] / GRID["refine"])) + 1, len(sel))
            Vs, f_lon2, f_lat2 = sample_grid(plon, plat, n_f)
            along_f = np.linspace(along[0], along[-1], n_f)
            surf_f = bilinear(elev_hang, extent, f_lon2, f_lat2)
            if not np.isfinite(surf_f).all():
                gg = np.isfinite(surf_f)
                surf_f = (np.interp(along_f, along_f[gg], surf_f[gg]) if gg.sum() >= 2
                          else np.full_like(surf_f, np.nanmean(surf)))
            X = np.tile(along_f, (len(z), 1))
            Y = surf_f[None, :] - 1000.0 * z[:, None]
        else:
            Vs = vs[sel].T                                       # (ndepth, ncol)
            X = np.tile(along, (len(z), 1))
            Y = surf[None, :] - 1000.0 * z[:, None]              # m a.s.l.

        fig = plt.figure(figsize=(max(9, 0.55 * len(sel)) + 2.2, 7.4))
        # Colorbar gets its OWN column. Attaching it with colorbar(ax=ax) shrinks only the
        # main axes, leaving the topography strip one colorbar-width too wide (measured: 203
        # px of a 4369 px figure) -- and set_position() after the fact does not survive the
        # savefig re-layout. A dedicated column makes the two panels share a width by
        # construction.
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 4.6], width_ratios=[1, 0.018],
                              hspace=0.06, wspace=0.012)
        ax_t = fig.add_subplot(gs[0, 0])
        ax = fig.add_subplot(gs[1, 0], sharex=ax_t)
        cax = fig.add_subplot(gs[1, 1])

        # fine topo profile (50 m sampling), exaggerated by the short top panel
        f_along = np.linspace(0, along[-1], max(int(along[-1] / 0.05), 2))
        f_lon = np.interp(f_along, along, plon); f_lat = np.interp(f_along, along, plat)
        f_topo = bilinear(elev, extent, f_lon, f_lat)
        if not np.isfinite(f_topo).all():                        # same gap, finer sampling
            g2 = np.isfinite(f_topo)
            f_topo = (np.interp(f_along, f_along[g2], f_topo[g2]) if g2.sum() >= 2
                      else np.full_like(f_topo, np.nanmean(surf)))
        ax_t.fill_between(f_along, f_topo, f_topo.min() - 10, color="0.75", lw=0)
        ax_t.plot(f_along, f_topo, "k-", lw=0.9)
        if a.topo_smooth_km > 0:                                   # the hang datum, for the record
            f_hang = bilinear(elev_hang, extent, f_lon, f_lat)
            ax_t.plot(f_along, f_hang, "k--", lw=0.7, alpha=0.8)
        ax_t.set_ylabel("m a.s.l.", fontsize=8)
        ax_t.tick_params(labelbottom=False, labelsize=7)
        ax_t.set_title(f"{net} — {axis} section "
                       f"{'lat' if axis == 'EW' else 'lon'}="
                       f"{np.mean(plat if axis == 'EW' else plon):.4f} — {arm}"
                       + (f"   [{tband}]" if tband else ""), fontsize=10)

        if REF is not None:
            Vs = 100.0 * (Vs - REF[:, None]) / REF[:, None]
            pc = ax.pcolormesh(X, Y, Vs, cmap=get_cmap(a.anomaly_cmap), vmin=-ALIM, vmax=ALIM,
                               shading="nearest")
        else:
            pc = ax.pcolormesh(X, Y, Vs, cmap=vcmap, vmin=global_vmin, vmax=global_vmax,
                               shading="nearest")
        if DIS is not None:
            # Hatch where this arm and the reference disagree beyond combined 1-sigma.
            # Drawn on the RAW cell columns whatever the section itself shows -- the
            # diagnosis lives at measurement resolution, and smoothing it with the spline
            # would blur exactly the boundary it exists to mark.
            D = DIS[sel].T.astype(float)                         # (ndepth, ncol)
            Xr = np.tile(along, (len(z), 1))
            Yr = surf[None, :] - 1000.0 * z[:, None]
            ax.contourf(Xr, Yr, D, levels=[0.5, 1.5], colors="none",
                        hatches=["////"], zorder=4)
            ax.contour(Xr, Yr, D, levels=[0.5], colors="k", linewidths=0.7, zorder=5)
            frac_here = D.mean()
            ax.annotate(f"hatched: vs {dis_label}, |ΔVs| > {a.disagree_min_sigma:g}σ and "
                        f"> {100*a.disagree_min_relgap:.0f}% of mean ({100*frac_here:.0f}% of "
                        f"this section)",
                        (0.99, 1.01), xycoords="axes fraction", ha="right",
                        va="bottom", fontsize=8, style="italic")
        ax.plot(along, surf, "k-", lw=1.0)                       # surface
        # geophones within 200 m
        dsta = dist_to_polyline_km(stx, sty, px, py)
        near = dsta <= NEAR_KM
        if near.any():
            # along-position = along-coordinate of the nearest profile vertex (adequate at
            # the 0.5-1 km column spacing; stations are <=200 m off the line by selection)
            vi_near = [int(np.argmin(np.hypot(px - stx[i], py - sty[i])))
                       for i in np.where(near)[0]]
            ax.plot(along[vi_near], st_elev[near] + 15, "v", mfc="lime", mec="k",
                    ms=6, zorder=6, label=f"geophones ≤{int(NEAR_KM*1000)} m")
            ax_t.plot(along[vi_near], st_elev[near], "v", mfc="lime", mec="k", ms=5, zorder=6)
        # wells within 200 m: vertical line to TVD below surface, label on top
        for nm, wla, wlo, wdep in wells:
            wx, wy = km_xy(wlo, wla, lat0)
            if dist_to_polyline_km(np.array([wx]), np.array([wy]), px, py)[0] > NEAR_KM:
                continue
            wa = along[int(np.argmin(np.hypot(px - wx, py - wy)))]
            wel = float(bilinear(elev, extent, np.array([wlo]), np.array([wla]))[0])
            ax.plot([wa, wa], [wel, wel - wdep], color="k", lw=2.2, zorder=7)
            ax.annotate(nm, (wa, wel), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=8.5, fontweight="bold", zorder=8,
                        bbox=dict(fc="w", alpha=0.75, ec="k", lw=0.4, pad=1.5))
        # Geo2Riehen seismic horizons: points within 200 m of the profile, projected to the
        # along-coordinate and linearly interpolated (drawn only across the span where points
        # exist -- no extrapolation beyond the constrained reach of the 3D survey).
        if horizons is not None:
            hv, ho = horizons["verts"], horizons["offsets"]
            hstyle = {"TKris": ("m", "-"), "TMusch": ("darkgreen", "--"),
                      "Base Mesozoic": ("saddlebrown", ":")}
            for hi, hname in enumerate(horizons["names"]):
                pts3 = hv[ho[hi]:ho[hi + 1]]
                hx, hy = km_xy(pts3[:, 0], pts3[:, 1], lat0)
                dh = dist_to_polyline_km(hx, hy, px, py)
                m = dh <= NEAR_KM
                if m.sum() < 2:
                    continue
                # project: along-coordinate of the nearest profile vertex per point, then bin
                # to a fine along-grid and take the median z per bin (robust to the swath of
                # 3D-survey points collapsing onto one profile coordinate)
                ai = np.array([along[int(np.argmin(np.hypot(px - hx[j], py - hy[j])))]
                               for j in np.where(m)[0]])
                zi = pts3[m, 2].astype(float)
                order = np.argsort(ai)
                ai, zi = ai[order], zi[order]
                bins = np.arange(ai.min(), ai.max() + 0.26, 0.25)
                bc, bz = [], []
                for b0, b1 in zip(bins[:-1], bins[1:]):
                    sel_b = (ai >= b0) & (ai < b1)
                    if sel_b.any():
                        bc.append(0.5 * (b0 + b1)); bz.append(np.median(zi[sel_b]))
                if len(bc) < 2:
                    continue
                key = next((k for k in hstyle if str(hname).startswith(k)), "TKris")
                col, ls = hstyle[key]
                ax.plot(bc, bz, color=col, ls=ls, lw=2.0, zorder=6,
                        label=str(hname).split(" (")[0] + " (Geo2Riehen)")
        ax.set_xlabel("distance along profile [km]")
        ax.set_ylabel("elevation [m a.s.l.]")
        ax.set_xlim(0, along[-1])
        ax.set_ylim(np.nanmin(Y) - 100, np.nanmax(surf) + 150)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7, loc="lower left")
        plt.colorbar(pc, cax=cax).set_label(
            f"Vs anomaly vs the network median at the same depth [%]" if REF is not None
            else "Vs median [km/s]")

        # ORIENTATION: state which end is which, read from the coordinates rather than
        # assumed. EW profiles are built by sorting on ix (grid east) and NS on iy (grid
        # north), but the label must reflect the actual first/last vertex.
        if abs(plon[-1] - plon[0]) >= abs(plat[-1] - plat[0]):
            lend, rend = ("W", "E") if plon[-1] > plon[0] else ("E", "W")
        else:
            lend, rend = ("S", "N") if plat[-1] > plat[0] else ("N", "S")
        # bottom corners: the top-right corner is occupied by the inset locator map, which
        # would sit on top of the "E"/"N" label.
        for xf, lab, ha in ((0.0, lend, "left"), (1.0, rend, "right")):
            ax.annotate(lab, (xf, 0.0), xycoords="axes fraction",
                        xytext=(8 if ha == "left" else -8, 8), textcoords="offset points",
                        ha=ha, va="bottom", fontsize=13, fontweight="bold", zorder=9,
                        bbox=dict(fc="w", alpha=0.85, ec="k", lw=0.5, pad=2))
        # GEOGRAPHIC CONTEXT: project the named places onto the profile. Without these the
        # along-distance axis is unreadable -- a well at "16 km" cannot be checked against
        # "GVL-1 is near Glovelier" by eye, which is exactly how a correct marker comes to
        # look misplaced.
        for cnm, clo, cla in CITIES.get(net, []):
            cx, cy = km_xy(clo, cla, lat0)
            if dist_to_polyline_km(np.array([cx]), np.array([cy]), px, py)[0] > 2.0:
                continue
            ca = along[int(np.argmin(np.hypot(px - cx, py - cy)))]
            ax_t.axvline(ca, color="0.35", lw=0.8, ls=":", zorder=5)
            ax_t.annotate(cnm, (ca, ax_t.get_ylim()[1]), xytext=(0, -2),
                          textcoords="offset points", rotation=90, ha="center", va="top",
                          fontsize=6.5, style="italic", color="0.25", zorder=6)

        # inset map with the section line (LV95 km, aspecto igual)
        axi = fig.add_axes([0.795, 0.70, 0.185, 0.24])
        axi.imshow(hs, extent=extent_km, cmap="gray", origin="upper")
        nkx, nky = km_xy(lonlat[:, 0], lonlat[:, 1])
        axi.plot(nkx, nky, ".", color="0.55", ms=0.8)
        axi.plot(px, py, "r-", lw=2)
        axi.set_aspect("equal")
        axi.set_xticks([]); axi.set_yticks([])
        for sp in axi.spines.values():
            sp.set_linewidth(1.4)

        tag = f"iy{line_idx:02d}" if axis == "EW" else f"ix{line_idx:02d}"
        if name:
            tag = f"{tag}_{name}"
        if DIS is not None:
            tag = f"{tag}_{a.disagree_suffix}"                   # never overwrite the plain set
        out = os.path.join(figdir, SECDIR, f"sect_{axis}_{tag}.png")
        fig.savefig(out, dpi=145, bbox_inches="tight"); plt.close(fig)
        return True

    step_iy = max(1, int(round(SECT_STEP_KM / dy_km)))
    step_ix = max(1, int(round(SECT_STEP_KM / dx_km)))
    n_ew = sum(one_section("EW", l) for l in range(0, ny, step_iy))
    n_ns = sum(one_section("NS", l) for l in range(0, nx, step_ix))
    print(f"sections: {n_ew} EW + {n_ns} NS -> {figdir}/{SECDIR}/ "
          f"(steps iy={step_iy}, ix={step_ix}; grid d={dx_km:.2f}/{dy_km:.2f} km)")

    # Sections THROUGH each well. The regular grid steps rarely land within the 200 m
    # well-annotation radius (Basel-1 misses its nearest riehen line by ~300 m), so a well
    # would never actually appear on a section unless we cut one at its own row/column.
    n_w = 0
    for nm, wla, wlo, _dep in wells:
        # nearest inverted cell to the well, in the volume's own lon/lat
        d2 = (lonlat[:, 0] - wlo) ** 2 + ((lonlat[:, 1] - wla) * 1.4) ** 2
        if not len(d2):
            continue
        wix, wiy = (int(v) for v in cells[int(np.argmin(d2))])
        safe = nm.replace(" ", "").replace("/", "-")
        n_w += int(one_section("EW", int(wiy), name=safe))
        n_w += int(one_section("NS", int(wix), name=safe))
    if wells:
        print(f"well sections: {n_w} -> {figdir}/{SECDIR}/ "
              f"(EW+NS through each of {len(wells)} wells)")


if __name__ == "__main__":
    main()
