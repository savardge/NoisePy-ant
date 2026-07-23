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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True, choices=("riehen", "aargau"))
    ap.add_argument("--stage", required=True, choices=("dem", "gk500"))
    a = ap.parse_args()
    (stage_dem if a.stage == "dem" else stage_gk500)(a.net)
