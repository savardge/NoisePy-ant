#!/usr/bin/env python
"""One-time extraction of figure assets (DEM elevation + GK500 tectonic lines) to plain npz.

No single conda env here has BOTH rasterio and geopandas, so the Vs-model figure script
(vs_model_figures.py, bayhunter env, numpy+matplotlib only) consumes pre-extracted assets:

  --stage dem    (env with rasterio, e.g. masw-das):  elev grid + extent for the net bbox
  --stage gk500  (env with geopandas, e.g. map):      fault/thrust/axes polylines in WGS84

Output: Projects/<net>/tomo/2_vs_depth_inversion/fig_assets_<net>_<stage>.npz  (shared by arms).
Bbox is taken from the production volume's cell lonlat + pad.
"""
import argparse
import os

import numpy as np

E = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
GK500 = "/Users/genevievesavard/Data/hautesorne/GIS/GK-500-V-4/GK500_V1_4_FR/Shapes_WGS84"


def net_bbox(net, pad=0.03):
    v = np.load(f"{E}/{net}/tomo/2_vs_depth_inversion/production/"
                f"production_2026-07-17_hybrid_recipe/volume_fundotlove.npz", allow_pickle=True)
    ll = v["lonlat"]
    return (ll[:, 0].min() - pad, ll[:, 0].max() + pad,
            ll[:, 1].min() - pad, ll[:, 1].max() + pad)


def stage_dem(net):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dem_hillshade import load_dem, dem_for_net
    lo0, lo1, la0, la1 = net_bbox(net)
    elev, extent = load_dem(dem_for_net(net), bbox=(lo0, lo1, la0, la1), pad=0.02)
    out = f"{E}/{net}/tomo/2_vs_depth_inversion/fig_assets_{net}_dem.npz"
    np.savez_compressed(out, elev=elev.astype(np.float32), extent=np.array(extent, float))
    print(f"wrote {out}: elev {elev.shape}, extent {np.round(extent, 3)}")


def stage_gk500(net):
    import geopandas as gpd
    from shapely.geometry import box
    lo0, lo1, la0, la1 = net_bbox(net)
    bb = box(lo0, la0, lo1, la1)
    out_lines, out_type = [], []
    # LI_Accident_tecto = faults/thrusts (attribute TYPE_AT distinguishes), LI_Axes = fold axes
    for shp, kind_default, attr in (("LI_Accident_tecto_wgs84.shp", "fault", "TYPE_AT"),
                                    ("LI_Axes_de_struct_wgs84.shp", "axis", None)):
        fp = os.path.join(GK500, shp)
        if not os.path.exists(fp):
            print(f"  missing {fp}; skipped")
            continue
        g = gpd.read_file(fp)
        g = g[g.intersects(bb)]
        print(f"  {shp}: {len(g)} features in bbox; columns: {list(g.columns)[:8]}")
        for _, row in g.iterrows():
            kind = kind_default
            if attr and attr in row and isinstance(row[attr], str):
                t = row[attr].lower()
                kind = "thrust" if ("chevauch" in t or "thrust" in t) else "fault"
            geom = row.geometry.intersection(bb)
            geoms = getattr(geom, "geoms", [geom])
            for ln in geoms:
                xy = np.asarray(ln.coords, float)
                if len(xy) >= 2:
                    out_lines.append(xy.astype(np.float32))
                    out_type.append(kind)
    # FLAT, pickle-free layout (object arrays pickled by numpy 2.x are unreadable in the
    # bayhunter env's numpy 1.x): all vertices concatenated + per-line offsets + kind codes.
    out = f"{E}/{net}/tomo/2_vs_depth_inversion/fig_assets_{net}_gk500.npz"
    verts = (np.concatenate(out_lines, axis=0) if out_lines
             else np.zeros((0, 2), np.float32))
    offs = np.concatenate([[0], np.cumsum([len(l) for l in out_lines])]).astype(np.int64)
    kindcode = np.array([{"fault": 0, "thrust": 1, "axis": 2}[k] for k in out_type], np.int8)
    np.savez_compressed(out, verts=verts, offsets=offs, kindcode=kindcode)
    print(f"wrote {out}: {len(out_lines)} polylines "
          f"({sum(k=='fault' for k in out_type)} faults, "
          f"{sum(k=='thrust' for k in out_type)} thrusts, "
          f"{sum(k=='axis' for k in out_type)} axes)")


GEO2RIEHEN = "/Volumes/T7Shield/riehen/geo2riehen_data"
HORIZON_PATTERNS = {          # filename substring (case-insensitive) -> canonical horizon name
    "tkris": "TKris (top crystalline basement)",
    "tmusch": "TMusch (top Muschelkalk)",
    "meso": "Base Mesozoic",
}


def _ch_to_wgs84(e, n):
    """Swisstopo approximate formulas (accurate ~1 m). Accepts LV95 (2.6e6/1.2e6) or LV03."""
    e = np.asarray(e, float); n = np.asarray(n, float)
    if np.nanmedian(e) > 2e6:                                  # LV95 -> LV03
        e = e - 2_000_000.0; n = n - 1_000_000.0
    y = (e - 600_000.0) / 1e6; x = (n - 200_000.0) / 1e6
    lon = (2.6779094 + 4.728982 * y + 0.791484 * y * x + 0.1306 * y * x**2
           - 0.0436 * y**3) * 100.0 / 36.0
    lat = (16.9023892 + 3.238272 * x - 0.270978 * y**2 - 0.002528 * x**2
           - 0.0447 * y**2 * x - 0.0140 * x**3) * 100.0 / 36.0
    return lon, lat


def stage_geo2riehen(net):
    """Geo2Riehen 3D-seismic horizon points (x, y, z) -> flat npz for the section overlays.

    Format is sniffed per file: delimiter (comma/semicolon/whitespace), header lines skipped,
    coordinates auto-detected (LV95 / LV03 / lon-lat by magnitude), z reported as-read with
    diagnostics printed so the m a.s.l. vs depth convention can be VERIFIED on first run
    (vertical sign errors are silent killers on section overlays).
    """
    assert net == "riehen", "Geo2Riehen horizons are a riehen-only product"
    import glob as _g
    pts, names = [], []
    files = sorted(_g.glob(os.path.join(GEO2RIEHEN, "*")))
    if not files:
        raise SystemExit(f"no files under {GEO2RIEHEN} (volume unreadable or empty?)")
    for fp in files:
        base = os.path.basename(fp).lower()
        name = next((v for k, v in HORIZON_PATTERNS.items() if k in base), None)
        if name is None or not os.path.isfile(fp):
            print(f"  skip (no horizon pattern): {os.path.basename(fp)}")
            continue
        raw = open(fp, errors="ignore").read()
        delim = ";" if ";" in raw.splitlines()[max(0, min(5, len(raw.splitlines()) - 1))] \
            else ("," if "," in raw.splitlines()[0] else None)
        arr = np.genfromtxt(fp, delimiter=delim, comments="#", invalid_raise=False)
        if arr.ndim == 1:
            arr = arr[None, :]
        arr = arr[np.isfinite(arr).all(axis=1)]
        if arr.shape[1] < 3:
            print(f"  skip ({arr.shape[1]} cols): {os.path.basename(fp)}")
            continue
        x, y, zv = arr[:, 0], arr[:, 1], arr[:, 2]
        if np.nanmedian(np.abs(x)) < 90:                       # already lon/lat
            lon, lat = x, y
        else:
            lon, lat = _ch_to_wgs84(x, y)
        print(f"  {os.path.basename(fp)} -> {name}: {len(lon)} pts | lon {np.nanmin(lon):.4f}"
              f"..{np.nanmax(lon):.4f} lat {np.nanmin(lat):.4f}..{np.nanmax(lat):.4f} | "
              f"z {np.nanmin(zv):.0f}..{np.nanmax(zv):.0f} (VERIFY: m a.s.l. expected, "
              f"negative below sea level)")
        pts.append(np.column_stack([lon, lat, zv]).astype(np.float32))
        names.append(name)
    if not pts:
        raise SystemExit("no horizon files matched TKris/TMusch/Meso patterns")
    verts = np.concatenate(pts, axis=0)
    offs = np.concatenate([[0], np.cumsum([len(p) for p in pts])]).astype(np.int64)
    out = f"{E}/{net}/tomo/2_vs_depth_inversion/fig_assets_{net}_horizons.npz"
    np.savez_compressed(out, verts=verts, offsets=offs,
                        names=np.array(names, dtype="U64"))
    print(f"wrote {out}: {len(names)} horizons, {len(verts)} points")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True, choices=("riehen", "aargau"))
    ap.add_argument("--stage", required=True, choices=("dem", "gk500", "geo2riehen"))
    a = ap.parse_args()
    {"dem": stage_dem, "gk500": stage_gk500, "geo2riehen": stage_geo2riehen}[a.stage](a.net)
