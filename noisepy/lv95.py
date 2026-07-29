"""WGS84 <-> LV95 (CH1903+/EPSG:2056) via the swisstopo approximate formulas (~1 m).

Pure numpy so it works in EVERY env (bayhunter's has no pyproj). Standing figure rule
(2026-07-25): ALL map-view figures plot in LV95 Cartesian with ax.set_aspect("equal") —
never raw lon/lat axes (1 deg lon ~ 0.68 x 1 deg lat at 47.4N distorts shapes).

Ref: swisstopo, "Approximate formulas for the transformation between Swiss
projection coordinates and WGS84" (2016).
"""
import numpy as np


def wgs84_to_lv95(lon, lat):
    """lon/lat [deg] -> (E, N) in meters, LV95 (2.6e6 / 1.2e6 false origin)."""
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    lp = (lon * 3600.0 - 26782.5) / 10000.0
    pp = (lat * 3600.0 - 169028.66) / 10000.0
    E = (2600072.37 + 211455.93 * lp - 10938.51 * lp * pp
         - 0.36 * lp * pp**2 - 44.54 * lp**3)
    N = (1200147.07 + 308807.95 * pp + 3745.25 * lp**2
         + 76.63 * pp**2 - 194.56 * lp**2 * pp + 119.79 * pp**3)
    return E, N


def lv95_to_wgs84(E, N):
    """(E, N) LV95 meters -> lon/lat [deg]. Accepts LV03 too (auto-detected)."""
    E = np.asarray(E, float)
    N = np.asarray(N, float)
    if np.nanmedian(E) > 2e6:
        E = E - 2_000_000.0
        N = N - 1_000_000.0
    y = (E - 600_000.0) / 1e6
    x = (N - 200_000.0) / 1e6
    lon = (2.6779094 + 4.728982 * y + 0.791484 * y * x + 0.1306 * y * x**2
           - 0.0436 * y**3) * 100.0 / 36.0
    lat = (16.9023892 + 3.238272 * x - 0.270978 * y**2 - 0.002528 * x**2
           - 0.0447 * y**2 * x - 0.0140 * x**3) * 100.0 / 36.0
    return lon, lat


def extent_lv95_km(extent_ll):
    """(lon0, lon1, lat0, lat1) -> (E0, E1, N0, N1) in km. For imshow backgrounds
    (hillshade): affine corner mapping, adequate over ~50 km boxes."""
    lon0, lon1, lat0, lat1 = extent_ll
    latm = 0.5 * (lat0 + lat1)
    lonm = 0.5 * (lon0 + lon1)
    (e0, _), (e1, _) = [wgs84_to_lv95(lo, latm) for lo in (lon0, lon1)]
    (_, n0), (_, n1) = [wgs84_to_lv95(lonm, la) for la in (lat0, lat1)]
    return (e0 / 1e3, e1 / 1e3, n0 / 1e3, n1 / 1e3)
