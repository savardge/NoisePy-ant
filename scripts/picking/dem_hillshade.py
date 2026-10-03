"""Load an SRTM/DHM DEM (GeoTIFF or .hgt, EPSG:4326) and build a hillshade for a lon/lat
bbox, for use as a semi-transparent-velocity-mesh background in the Vs map figures.

`hillshade_for_bbox` returns (hs, extent) where hs is a 0..1 grayscale array and
extent = [lon0, lon1, lat0, lat1] ready for `ax.imshow(hs, extent=extent, cmap="gray")`.
"""
import numpy as np


def smooth_dem(elev, extent, sigma_km):
    """Gaussian-smoothed copy of a DEM (row 0 = north), sigma in km, NaN-aware (normalised
    convolution). This is the HANG datum for the 1-D Vs columns: the imprint test
    (hautesorne/.../tests/test_2026-09-25_topographic_imprint) found the data referenced to the
    surface smoothed at sigma ~1 km, not to 30 m relief. sigma_km <= 0 returns elev unchanged."""
    if sigma_km is None or sigma_km <= 0:
        return elev
    from scipy.ndimage import gaussian_filter
    ny, nx = elev.shape
    lon0, lon1, lat0, lat1 = extent
    latm = 0.5 * (lat0 + lat1)
    dy_m = abs(lat1 - lat0) / max(ny, 1) * 111_000.0
    dx_m = abs(lon1 - lon0) / max(nx, 1) * 111_000.0 * np.cos(np.deg2rad(latm))
    sig = (sigma_km * 1e3 / dy_m, sigma_km * 1e3 / dx_m)
    ok = np.isfinite(elev)
    num = gaussian_filter(np.where(ok, elev, 0.0), sig, mode="nearest")
    den = gaussian_filter(ok.astype(float), sig, mode="nearest")
    out = np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)
    return np.where(ok, out, np.nan)


def load_dem(path, bbox=None, pad=0.02):
    """Return (elev, extent) from a rasterio-readable DEM, optionally cropped to
    bbox=(lon0,lon1,lat0,lat1) with `pad` degrees margin. extent=[lon0,lon1,lat0,lat1]."""
    import rasterio
    from rasterio.windows import from_bounds
    with rasterio.open(path) as d:
        if bbox is not None:
            lon0, lon1, lat0, lat1 = bbox
            win = from_bounds(lon0 - pad, lat0 - pad, lon1 + pad, lat1 + pad, d.transform)
            # BOUNDLESS. A plain read() silently CLIPS a window that runs off the raster
            # (Haute-Sorne asks for lon 6.87 but tile N47E007 starts at 7.00, so col_off was
            # -400 and 1599 of 1999 columns came back) while window_transform() still returns
            # the transform of the FULL requested window -- the short array then gets labelled
            # with the wide extent and every feature is displaced. Measured: 0.111 deg =
            # 8.35 km of eastward shift in the hautesorne hillshade. Reading boundless keeps
            # array and extent consistent and marks the uncovered strip as nodata.
            b = d.bounds
            if (lon0 - pad < b.left or lon1 + pad > b.right
                    or lat0 - pad < b.bottom or lat1 + pad > b.top):
                print(f"  load_dem: requested bbox exceeds {path} "
                      f"({b.left:.3f}..{b.right:.3f}, {b.bottom:.3f}..{b.top:.3f}); "
                      f"reading boundless, uncovered area = nodata")
            elev = d.read(1, window=win, boundless=True,
                          fill_value=(d.nodata if d.nodata is not None else -32768)
                          ).astype(float)
            t = d.window_transform(win)
            h, w = elev.shape
            left, top = t * (0, 0)
            right, bottom = t * (w, h)
        else:
            elev = d.read(1).astype(float)
            left, bottom, right, top = d.bounds
        nod = d.nodata
    if nod is not None:
        elev[elev == nod] = np.nan
    elev[elev < -1e4] = np.nan
    return elev, [left, right, bottom, top]


def hillshade_for_bbox(dem_path, lon0, lon1, lat0, lat1, pad=0.02,
                       azdeg=315, altdeg=45, vert_exag=3.0):
    """(hs 0..1, extent) hillshade for the bbox, meters-aware spacing at the bbox mid-lat."""
    from matplotlib.colors import LightSource
    elev, extent = load_dem(dem_path, bbox=(lon0, lon1, lat0, lat1), pad=pad)
    latm = 0.5 * (lat0 + lat1)
    ny, nx = elev.shape
    dy = abs(extent[3] - extent[2]) / max(ny, 1) * 111_000.0
    dx = abs(extent[1] - extent[0]) / max(nx, 1) * 111_000.0 * np.cos(np.deg2rad(latm))
    z = np.nan_to_num(elev, nan=np.nanmin(elev))
    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    hs = ls.hillshade(z, vert_exag=vert_exag, dx=dx, dy=dy)
    return hs, extent


def dem_for_net(net):
    """Path to the prepared DEM for a network."""
    import glob
    base = f"/Users/genevievesavard/Codes/extract_higher_modes/Projects/{net}/tomo/dem"
    tif = glob.glob(f"{base}/*.tif")
    if tif:
        return tif[0]
    hgt = glob.glob(f"{base}/*.hgt")
    if hgt:
        return hgt[0]
    raise FileNotFoundError(f"no DEM under {base}")
