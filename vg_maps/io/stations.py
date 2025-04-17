import pandas as pd
import numpy as np
from utils.geo_utils import ll2xy_sphericalproj

def get_stations_extent(bounds, station_file):
    """
    Compute the grid extent and convert station lat/lon to local cartesian coordinates.

    Parameters:
        bounds (list): [min_lat, max_lat, min_lon, max_lon]
        station_file (str): Path to station CSV file with at least 'latitude', 'longitude'

    Returns:
        grid_origin (tuple): (min_lat, min_lon)
        extent (tuple): (xmax, ymax) in km
        stations (DataFrame): Filtered stations with 'xstat', 'ystat', and 'id'
    """
    min_lat, max_lat, min_lon, max_lon = bounds
    grid_origin = (min_lat, min_lon)

    xmax, ymax = ll2xy_sphericalproj(max_lat, max_lon, grid_origin)
    extent = (xmax, ymax)
    print(f"Grid origin (SW corner) lat,long: ({min_lat:.4f}, {min_lon:.4f})")
    print(f"Grid extent: {xmax:.2f} km by {ymax:.2f} km")

    stations = pd.read_csv(station_file)
    if 'channel' in stations.columns:
        stations = stations[stations['channel'].str.contains('Z')]

    nsta0 = len(stations)
    xstat, ystat = ll2xy_sphericalproj(stations.latitude.values, stations.longitude.values, grid_origin)
    stations['xstat'] = xstat
    stations['ystat'] = ystat

    # Filter stations within grid
    stations = stations[(xstat > 0) & (xstat < xmax) & (ystat > 0) & (ystat < ymax)]
    if 'id' not in stations.columns:
        stations['id'] = stations['network'].astype(str) + '.' + stations['station'].astype(str)

    nsta = len(stations)
    print(f"Number of stations inside grid: {nsta} ({nsta0 - nsta} removed).")
    return grid_origin, extent, stations
