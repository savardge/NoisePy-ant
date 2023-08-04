# Code below in progress, doesn't work yet

def invert_TV():
    import numpy.matlib
    import time

    # Inversion Tarantola-Valette

    # Read in data
    Tc = 1.0
    output_dir_kern = "/home/users/s/savardg/riehen/vg_maps/kernels"
    fname = os.path.join(output_dir_kern, f"data_and_kern_T{Tc:.1f}_ZZ.npz")
    npzfile = np.load(fname)
    d = npzfile['TAU']
    G_mat = npzfile['G_mat']
    v_moy = npzfile['v_moy']

    fname = os.path.join(output_folder, "kernel.npz")
    npzfile = np.load(fname)
    X_GRID = npzfile['X_GRID']
    Y_GRID = npzfile['Y_GRID']
    x_grid = npzfile['x_grid']
    y_grid = npzfile['y_grid']
    dx_grid = npzfile['dx_grid']
    dy_grid = npzfile['dy_grid']

    N_d = G_mat.shape[0]  # number of data points = number of ray paths
    N_m = G_mat.shape[1]  # number of model cells
    print(f"number of data: {N_d}, number of cells: {N_m}")

    # Grid info
    L0 = np.sqrt(dx_grid ** 2 + dy_grid ** 2)  # Size of a model cell (diagonal)
    x_cell = X_GRID.ravel('F')  # same as np.reshape(X_GRID,(nb_cell,), order='F')
    y_cell = Y_GRID.ravel('F')
    X_CELL = np.matlib.repmat(x_cell, N_m, 1).T
    Y_CELL = np.matlib.repmat(y_cell, N_m, 1).T
    DIST_CELL = np.sqrt((X_CELL - X_CELL.T) ** 2 + (Y_CELL - Y_CELL.T) ** 2)

    # Regularization params
    rel_err = 15 / 100.  # % relative error on data; could try other values or affect a varying number depending on pick confidence
    Lc = 4
    sigma = 4

    # Calculate prior data covariance matrix
    Cd_diag = (rel_err * d) ** 2
    Cd = np.diag(Cd_diag)
    Cd_inv = np.diag(1 / Cd_diag)

    # Calculate prior model
    s_moy = 1 / v_moy  # average slowness
    m_prior = s_moy * np.ones(shape=(N_m,))

    # Calculate prior model covariance matrix
    Cm = (sigma * L0 / Lc) ** 2 * np.exp(- DIST_CELL / Lc)  # Thomas' code
    # Cm = (sigma)**2 * np.exp( - DIST_CELL / Lc)  # litterature (Montagner 1986)
    Cm_inv = numpy.linalg.inv(Cm)

    # Inversion
    # m_est = m_prior + np.linalg.inv(G_mat.T @ Cd_inv @ G_mat + Cm_inv) @ G_mat.T @ Cd_inv @ (d - G_mat @  m_prior)

    def get_m_hat(dict_m, dict_d, G):
        m0, C_m0 = dict_m['m0'], dict_m['C_m0']
        d0, C_d0 = dict_d['d0'], dict_d['C_d0']

        residuals = d0 - np.dot(G, m0)
        M = C_d0 + np.dot(np.dot(G, C_m0), np.transpose(G))

        return m0 + np.dot(np.dot(C_m0, np.transpose(G)), np.dot(np.linalg.inv(M), residuals))

    dict_m = {'m0': m_prior, 'C_m0': Cm}
    dict_d = {'d0': d, 'C_d0': Cd}
    t0 = time.time()
    m_est = get_m_hat(dict_m, dict_d, G_mat)
    print(f"Time elapsed: {time.time() - t0} s")
    # Reshape
    S_map = np.reshape(m_est, (len(x_grid), len(y_grid)), order='F')
    V_map = 1 / S_map


    fname = os.path.join(output_folder, "kernel.npz")
    npzfile = np.load(fname)
    x_stat = npzfile['x_stat']
    y_stat = npzfile['y_stat']

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.pcolor(X_GRID, Y_GRID, V_map,
                   cmap='RdBu',
                   vmin=1.5, vmax=4.)
    ax.set_title(f"Vg_map at T = {Tc}")
    ax.scatter(x_stat, y_stat, c='k', s=16, marker='v')
    ax.plot(X_GRID, Y_GRID, '+b')
    fig.colorbar(im)
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    plt.show()
    plt.close()