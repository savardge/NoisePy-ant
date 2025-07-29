import numpy as np
from scipy.sparse import diags
from scipy.linalg import inv
from scipy.spatial.distance import cdist

def TV_inversion(x_grid, y_grid, sigma, LC, TAU, v_prior, G):
    dx_grid = x_grid[1] - x_grid[0]
    dy_grid = y_grid[1] - y_grid[0]
    L0 = np.sqrt(dx_grid**2 + dy_grid**2)

    N_m = len(x_grid) * len(y_grid)
    x_cell, y_cell = np.meshgrid(x_grid, y_grid, indexing='ij')
    x_flat = x_cell.flatten()
    y_flat = y_cell.flatten()
    DIST_CELL = cdist(np.stack([x_flat, y_flat], axis=1), np.stack([x_flat, y_flat], axis=1))

    d = TAU
    rel_err = 0.10
    Cd_vec = (rel_err * d) ** 2
    CD_inv = diags(1.0 / Cd_vec)

    CM = (sigma) ** 2 * np.exp(-DIST_CELL / LC)
    CM_inv = inv(CM)

    m_prior = 1.0 / v_prior
    d_prior = G @ m_prior

    A = G.T @ CD_inv @ G + CM_inv
    b = G.T @ CD_inv @ (d - d_prior)
    m_est = m_prior + np.linalg.solve(A, b)
    d_post = G @ m_est

    var_prior = np.var(d - d_prior)
    var_post = np.var(d - d_post)
    var_red = 1 - var_post / var_prior
    restit_prior = np.sqrt(np.mean(((d - d_prior) / d) ** 2)) * 100
    restit_post = np.sqrt(np.mean(((d - d_post) / d) ** 2)) * 100

    V_map = 1.0 / m_est.reshape((len(x_grid), len(y_grid)))
    stats = {
        'var_prior': var_prior,
        'var_post': var_post,
        'var_red': var_red,
        'restit_prior': restit_prior,
        'restit_post': restit_post,
        'misfit_prior': d - d_prior,
        'misfit_post': d - d_post,
    }
    return V_map, stats


def TV_inversion_2step(x_grid, y_grid, sigma, LC, TAU, v_prior, G):
    dx_grid = x_grid[1] - x_grid[0]
    dy_grid = y_grid[1] - y_grid[0]
    L0 = np.sqrt(dx_grid**2 + dy_grid**2)

    N_m = len(x_grid) * len(y_grid)
    x_cell, y_cell = np.meshgrid(x_grid, y_grid, indexing='ij')
    x_flat = x_cell.flatten()
    y_flat = y_cell.flatten()
    DIST_CELL = cdist(np.stack([x_flat, y_flat], axis=1), np.stack([x_flat, y_flat], axis=1))

    d = TAU
    rel_err = 0.10
    Cd_vec1 = (rel_err * d) ** 2
    CD_inv1 = diags(1.0 / Cd_vec1)
    CM = (sigma) ** 2 * np.exp(-DIST_CELL / LC)
    CM_inv = inv(CM)

    m_prior1 = 1.0 / v_prior
    d_prior1 = G @ m_prior1
    A1 = G.T @ CD_inv1 @ G + CM_inv
    b1 = G.T @ CD_inv1 @ (d - d_prior1)
    m_est1 = m_prior1 + np.linalg.solve(A1, b1)
    d_post1 = G @ m_est1
    misfit1 = d - d_post1

    Cd_vec2 = np.copy(Cd_vec1)
    ioutliers = np.abs(misfit1 - misfit1.mean()) > 2 * misfit1.std()
    Cd_vec2[ioutliers] *= np.exp((np.abs(misfit1[ioutliers]) / (2 * misfit1.std())) - 1)
    CD_inv2 = diags(1.0 / Cd_vec2)

    m_prior2 = m_est1
    d_prior2 = G @ m_prior2
    A2 = G.T @ CD_inv2 @ G + CM_inv
    b2 = G.T @ CD_inv2 @ (d - d_prior2)
    m_est2 = m_prior2 + np.linalg.solve(A2, b2)
    d_post2 = G @ m_est2

    var_prior = np.var(d - d_prior1)
    var_post = np.var(d - d_post2)
    var_red = 1 - var_post / var_prior
    restit_prior = np.sqrt(np.mean(((d - d_prior1) / d) ** 2)) * 100
    restit_post = np.sqrt(np.mean(((d - d_post2) / d) ** 2)) * 100

    V_map = 1.0 / m_est2.reshape((len(x_grid), len(y_grid)))
    stats = {
        'var_prior': var_prior,
        'var_post': var_post,
        'var_red': var_red,
        'restit_prior': restit_prior,
        'restit_post': restit_post,
        'misfit_prior': d - d_prior1,
        'misfit_post': d - d_post2,
    }
    return V_map, stats
