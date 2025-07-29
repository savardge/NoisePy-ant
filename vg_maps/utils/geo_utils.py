import numpy as np

def ll2xy_sphericalproj(lat, lon, grid_origin):
    """
    Convert geographic coordinates to local Cartesian x, y coordinates (in km)
    using a spherical Earth projection centered at grid_origin.

    Parameters:
        lat (array-like): Latitudes
        lon (array-like): Longitudes
        grid_origin (tuple): (min_lat, min_lon)

    Returns:
        x (ndarray): x coordinates in km
        y (ndarray): y coordinates in km
    """
    R_earth = 6371  # Earth's radius in km

    lat = np.asarray(lat)
    lon = np.asarray(lon)
    ref_lat = np.full_like(lat, grid_origin[0])
    ref_lon = np.full_like(lon, grid_origin[1])

    avg_lat_rad = np.deg2rad((lat + ref_lat) / 2)
    dlon_rad = np.deg2rad(lon - ref_lon)
    dlat_rad = np.deg2rad(lat - ref_lat)

    x = R_earth * np.cos(avg_lat_rad) * dlon_rad
    y = R_earth * dlat_rad

    return x, y
