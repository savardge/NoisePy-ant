import numpy as np
from utils.geo_utils import ll2xy_sphericalproj

def get_data_kernel(x_grid, y_grid, data, stations):
    """
    Construct the data kernel matrix G where each row corresponds to a ray path
    and each column to a grid cell. G[i, j] is the distance ray i travels in cell j.

    Parameters:
        x_grid (np.ndarray): 1D array of grid x-coordinates (km)
        y_grid (np.ndarray): 1D array of grid y-coordinates (km)
        data (pd.DataFrame): Table of dispersion picks with 'stasrc', 'starcv'
        stations (pd.DataFrame): Station metadata with 'id', 'xstat', 'ystat'

    Returns:
        G (np.ndarray): (n_rays x n_cells) data kernel
        mask (np.ndarray): (nx x ny) binary mask of cells with sufficient ray density
    """
    dx_grid = x_grid[1] - x_grid[0]
    dy_grid = y_grid[1] - y_grid[0]
    nx, ny = len(x_grid), len(y_grid)
    nb_cell = nx * ny
    nb_ray = len(data)

    G = np.zeros((nb_ray, nb_cell))
    dl = 1e-3  # km step along ray path

    grid_indices = np.arange(nb_cell).reshape((nx, ny))

    for iray, row in data.iterrows():
        src = row['stasrc']
        rcv = row['starcv']
        x0 = stations.loc[stations['id'] == src, 'xstat'].values[0]
        y0 = stations.loc[stations['id'] == src, 'ystat'].values[0]
        x1 = stations.loc[stations['id'] == rcv, 'xstat'].values[0]
        y1 = stations.loc[stations['id'] == rcv, 'ystat'].values[0]

        dx, dy = x1 - x0, y1 - y0
        dist = np.sqrt(dx ** 2 + dy ** 2)
        if dist == 0:
            continue

        ux, uy = dx / dist, dy / dist
        steps = int(np.ceil(dist / dl)) + 1
        x_ray = x0 + np.arange(steps) * dl * ux
        y_ray = y0 + np.arange(steps) * dl * uy

        ix = ((x_ray - x_grid[0]) / dx_grid).astype(int)
        iy = ((y_ray - y_grid[0]) / dy_grid).astype(int)

        valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        linear_idx = grid_indices[ix[valid], iy[valid]]

        for ind in np.unique(linear_idx):
            G[iray, ind] += dl

    # Compute density mask
    thres_dist = 10e-3
    min_density = 3
    G3D = G.T.reshape((nx, ny, nb_ray))
    ray_density = np.sum(G3D > thres_dist, axis=2)
    mask = np.full((nx, ny), np.nan)
    mask[ray_density > min_density] = 1.0

    return G, mask
