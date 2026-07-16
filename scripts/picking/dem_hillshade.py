"""Load an SRTM/DHM DEM (GeoTIFF or .hgt, EPSG:4326) and build a hillshade for a lon/lat
bbox, for use as a semi-transparent-velocity-mesh background in the Vs map figures.

`hillshade_for_bbox` returns (hs, extent) where hs is a 0..1 grayscale array and
extent = [lon0, lon1, lat0, lat1] ready for `ax.imshow(hs, extent=extent, cmap="gray")`.
"""
import numpy as np


def load_dem(path, bbox=None, pad=0.02):
    """Return (elev, extent) from a rasterio-readable DEM, optionally cropped to
    bbox=(lon0,lon1,lat0,lat1) with `pad` degrees margin. extent=[lon0,lon1,lat0,lat1]."""
    import rasterio
    from rasterio.windows import from_bounds
    with rasterio.open(path) as d:
        if bbox is not None:
            lon0, lon1, lat0, lat1 = bbox
            win = from_bounds(lon0 - pad, lat0 - pad, lon1 + pad, lat1 + pad, d.transform)
            elev = d.read(1, window=win).astype(float)
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
