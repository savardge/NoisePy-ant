import numpy as np
import pandas as pd
from scipy.io import savemat
import os
import glob
import pyasdf
import yaml
import sys
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
Logger = logging.getLogger(__name__)

def make_stat_list():
    # Make stat_list_merged.mat:
    # -----------------------------------------------------
    # 'stat_list_merged','lat_merged','lon_merged','elev_merged','fs_new'

    # Get list of stations contained in stack files
    stackfiles = glob.glob(os.path.join(STACK_DIR, "*", "*.h5"))
    stalst_h5 = []
    for f in stackfiles:
        ff = os.path.split(f)[1]
        sta1 = ff.split(".h5")[0].split("_")[0].split(".")[1]
        if sta1 not in stalst_h5:
            stalst_h5.append(sta1)
        sta2 = ff.split(".h5")[0].split("_")[1].split(".")[1]
        if sta2 not in stalst_h5:
            stalst_h5.append(sta2)

    # Read station csv file used for noisepy
    stadf = pd.read_csv(stacsv)
    stadf = stadf.drop(columns="channel").drop_duplicates()  # remove duplicate rows
    stadf = stadf[stadf['station'].isin(stalst_h5)]  # Keep station actually used for stacking
    net_list = stadf["network"].values
    stat_list = stadf["station"].values
    stat_lat = stadf["latitude"].values
    stat_lon = stadf["longitude"].values
    stat_elev = stadf["elevation"].values
    mdict = {
        'stat_list_merged': stat_list,
        'net_list_merged': net_list,
        'lat_merged': stat_lat,
        'lon_merged': stat_lon,
        'elev_merged': stat_elev,
        'fs_new': fs
    }
    fname = os.path.join(output_folder, "stat_list_merged.mat")
    savemat(fname, mdict=mdict)
    Logger.info(f"Wrote file {fname}")
    fname = os.path.join(output_folder, "stat_list_merged.npz")
    np.savez(fname,
             stat_list_merged=stat_list,
             net_list_merged=net_list,
             lat_merged=stat_lat,
             lon_merged=stat_lon,
             elev_merged=stat_elev,
             fs_new=fs
             )
    Logger.info(f"Wrote file {fname}")


def make_dist_stat():
    # Make dist_stat.mat
    # -----------------------------------------------------
    # 'DIST_mat','x_stat','y_stat','stat_list','x_max','y_max','SW_corner','SE_corner','NW_corner','NE_corner'
    nb_stat = len(stat_list)
    SW_corner = [min_lat, min_lon]
    SE_corner = [min_lat, max_lon]
    NW_corner = [max_lat, min_lon]
    NE_corner = [max_lat, max_lon]
    ref_lat_glob = min_lat * np.ones(shape=stat_lat.shape)  # south west corner chosen as grid origin
    ref_lon_glob = min_lon * np.ones(shape=stat_lon.shape)

    # Transform lat,long to x,y
    x_max = R_earth * np.cos((max_lat + min_lat) / 2 * np.pi / 180) * (
                max_lon - min_lon) * np.pi / 180  # nb: by definition xmin = 0 and ymin = 0
    y_max = R_earth * (max_lat - min_lat) * np.pi / 180

    x_stat = R_earth * np.cos((stat_lat + ref_lat_glob) / 2 * np.pi / 180) * (stat_lon - ref_lon_glob) * np.pi / 180
    y_stat = R_earth * (stat_lat - ref_lat_glob) * np.pi / 180

    DIST_mat = np.zeros(shape=(nb_stat, nb_stat))
    for ind_stat1 in range(nb_stat):
        for ind_stat2 in range(nb_stat):
            dx = x_stat[ind_stat2] - x_stat[ind_stat1]
            dy = y_stat[ind_stat2] - y_stat[ind_stat1]

            DIST_mat[ind_stat1, ind_stat2] = np.sqrt(dx ** 2 + dy ** 2)
    mdict = {
        'DIST_mat': DIST_mat,
        'x_stat': x_stat,
        'y_stat': y_stat,
        'stat_list': stat_list,
        'x_max': x_max,
        'y_max': y_max,
        'SW_corner': SW_corner,
        'SE_corner': SE_corner,
        'NW_corner': NW_corner,
        'NE_corner': NE_corner
    }
    fname = os.path.join(output_folder, "dist_stat.mat")
    savemat(fname, mdict=mdict)
    Logger.info(f"Wrote file {fname}")
    fname = os.path.join(output_folder, "dist_stat.npz")
    np.savez(fname,
             DIST_mat=DIST_mat,
             x_stat=x_stat,
             y_stat=y_stat,
             stat_list=stat_list,
             x_max=x_max,
             y_max=y_max,
             SW_corner=SW_corner,
             SE_corner=SE_corner,
             NW_corner=NW_corner,
             NE_corner=NE_corner
             )
    Logger.info(f"Wrote file {fname}")


def make_stat_grid():
    # Make stat_grid.mat
    # -----------------------------------------------------
    x_grid = np.arange(0, x_max, dx_grid)
    y_grid = np.arange(0, y_max, dy_grid)
    X_GRID, Y_GRID = np.meshgrid(x_grid, y_grid, indexing='ij')
    mdict = {
        'X_GRID': X_GRID,
        'Y_GRID': Y_GRID,
        'x_stat': x_stat,
        'y_stat': y_stat,
        'x_grid': x_grid,
        'y_grid': y_grid,
        'dx_grid': dx_grid,
        'dy_grid': dy_grid,
        'x_max': x_max,
        'y_max': y_max
    }
    fname = os.path.join(output_folder, "stat_grid.mat")
    savemat(fname, mdict=mdict)
    Logger.info(f"Wrote file {fname}")
    fname = os.path.join(output_folder, "stat_grid.npz")
    np.savez(fname,
             X_GRID=X_GRID,
             Y_GRID=Y_GRID,
             x_stat=x_stat,
             y_stat=y_stat,
             x_grid=x_grid,
             y_grid=y_grid,
             dx_grid=dx_grid,
             dy_grid=dy_grid,
             x_max=x_max,
             y_max=y_max
             )
    Logger.info(f"Wrote file {fname}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(X_GRID, Y_GRID, "b+")
    ax.scatter(x_stat, y_stat, c="r", s=20, marker="^")
    ax.set_xlim((0, x_max))
    ax.set_ylim((0, y_max))
    plt.show()
    plt.close()


def make_kernel():
    # Make kernel.mat
    # -----------------------------------------------------
    dl = 0.005  # 5m, minimum length of ray in cell
    nb_stat = len(x_stat)
    nb_cell = int(X_GRID.size)
    nb_ray = int(nb_stat * (nb_stat - 1) / 2)

    ray_mat = {}
    G_mat = np.zeros(shape=(nb_ray, nb_cell))
    IND_LIN_GRID = np.reshape(np.arange(0, nb_cell, dtype=np.int16), (len(x_grid), len(y_grid)),
                              order='F')  # np.reshape(...order='F') = matlab reshape
    IND_S1 = np.zeros(shape=(nb_ray,), dtype=np.int16)  # to retrieve station from ray index
    IND_S2 = np.zeros(shape=(nb_ray,), dtype=np.int16)
    ind_ray = 0

    # construct kernel G
    # G_ij = distance traveled by ray i in cell j
    for s1 in range(nb_stat - 1):
        if s1 % 10 == 0: print(f"{s1}/{nb_stat}")
        ssta = stat_list[s1]
        ray_mat[ssta] = {}
        for s2 in np.arange(s1 + 1, nb_stat):
            rsta = stat_list[s2]
            delta_x = x_stat[s2] - x_stat[s1]
            delta_y = y_stat[s2] - y_stat[s1]
            dist = np.sqrt(delta_x ** 2 + delta_y ** 2)
            ux_ray, uy_ray = delta_x / dist, delta_y / dist
            ray_x = x_stat[s1] + np.arange(0, dist, dl) * ux_ray
            ray_y = y_stat[s1] + np.arange(0, dist, dl) * uy_ray
            ray_mat[ssta][rsta] = np.vstack([ray_x, ray_y])

            # G_mat
            IND_S1[ind_ray], IND_S2[ind_ray] = s1, s2  # to retrieve station from ray index
            x_ind = np.int16(
                np.floor((ray_x - x_grid[0]) / dx_grid))  # x ind of cell it falls on (if always positive values?)
            y_ind = np.int16(
                np.floor((ray_y - y_grid[0]) / dy_grid))  # y ind of cell it falls on (if always positive values?)

            for rr in range(len(ray_x)):  # points along the ray
                G_mat[ind_ray][IND_LIN_GRID[x_ind[rr], y_ind[rr]]] += dl

            ind_ray += 1

            # # plot
            # fig, ax = plt.subplots(1,1,figsize=(10,10))
            # ax.plot(X_GRID,Y_GRID,"b+")
            # ax.scatter(x_stat,y_stat,c="r",s=60, marker="^")
            # ax.plot(ray_x, ray_y, "k-") #, marker=".", markerfacecolor="r", ms=6)
            # ax.plot(x_grid[x_ind], y_grid[y_ind], "go")
            # ax.set_title(f"{ssta} - {rsta}, {dist:.0f} km apart")
            # ax.set_xlim((0, x_max))
            # ax.set_ylim((0, y_max))
            # plt.show()
            # plt.close()

    print(ind_ray)

    mdict = {
        'G_mat': G_mat,
        'dx_grid': dx_grid,
        'dy_grid': dy_grid,
        'X_GRID': X_GRID,
        'Y_GRID': Y_GRID,
        'x_grid': x_grid,
        'y_grid': y_grid,
        'x_stat': x_stat,
        'y_stat': y_stat
    }
    fname = os.path.join(output_folder, "kernel.mat")
    savemat(fname, mdict=mdict)
    Logger.info(f"Wrote file {fname}")
    fname = os.path.join(output_folder, "kernel.npz")
    np.savez(fname,
             G_mat=G_mat,
             dx_grid=dx_grid,
             dy_grid=dy_grid,
             X_GRID=X_GRID,
             Y_GRID=Y_GRID,
             x_grid=x_grid,
             y_grid=y_grid,
             x_stat=x_stat,
             y_stat=y_stat
             )
    Logger.info(f"Wrote file {fname}")

    # Plot a few random kernels
    # --------------------------
    # import matplotlib.pyplot as plt
    # # Plot kernel
    # x_plot = x_grid + dx_grid / 2
    # y_plot = y_grid + dy_grid / 2

    # for ray in np.random.randint(0, nb_ray, size=10):  #nb_ray):

    #     s1, s2 = IND_S1[ray], IND_S2[ray]
    #     x_ray_vec, y_ray_vec = ray_mat[stat_list[s1]][stat_list[s2]]

    #     G_slice = np.reshape(G_mat[ray,:],(len(x_grid), len(y_grid)), order='F')

    #     # Plot
    #     fig, ax = plt.subplots(1,1,figsize=(10,10))
    #     # ax.imshow(G_slice,
    #     #           extent=[min(x_plot), max(x_plot), min(y_plot), max(y_plot)],
    #     #           origin='lower',
    #     #           cmap='Greys',
    #     #           vmin=0, vmax=np.sqrt(dx_grid**2 + dy_grid**2))
    #     ax.pcolor(X_GRID, Y_GRID, G_slice,
    #               cmap='Greys',
    #               vmin=0, vmax=np.sqrt(dx_grid**2 + dy_grid**2))
    #     ax.set_title(f"Distance traveled in cell (km) by ray {ray}")
    #     ax.scatter(x_stat, y_stat, c='k', s=16, marker='v')
    #     ax.plot(x_ray_vec, y_ray_vec, 'k-')
    #     ax.plot(X_GRID,Y_GRID,'+b')
    #     ax.scatter(x_stat[s1], y_stat[s1], c='r', s=60, marker='v')
    #     ax.text(x_stat[s1], y_stat[s1], stat_list[s1], fontsize=14)
    #     ax.scatter(x_stat[s2], y_stat[s2], c='r', s=60, marker='v')
    #     ax.text(x_stat[s2], y_stat[s2], stat_list[s2], fontsize=14)
    #     ax.set_xlabel('X (km)')
    #     ax.set_ylabel('Y (km)')
    #     plt.show()
    #     plt.close()


def make_pick_cell():
    # Make all_picks.mat (PICK_CELL)
    # -----------------------------------------------------
    import pandas as pd
    import os
    import glob
    import numpy as np
    import sys
    import pickle

    fname = os.path.join(output_folder, "stat_list_merged.npz")
    npzfile = np.load(fname)
    stat_list = npzfile['stat_list_merged']
    net_list = npzfile['net_list_merged']

    comp = "ZZ"
    # dispdir = f"/home/users/s/savardg/scratch/aargau/STACK_CH-AA/dispersion_topology/pws/vg_{comp}"
    dispdir = f"/home/users/s/savardg/scratch/riehen/STACK_CHRI_norm/dispersion/pws/vg_{comp}"
    dispfiles = glob.glob(os.path.join(dispdir, "*", "*lagsym.csv"))

    PICK_CELL = {}
    for ss in range(nb_stat - 1):
        if ss % 50 == 0: print(f"{ss}/{nb_stat}")
        snet = net_list[ss]
        ssta = stat_list[ss]
        PICK_CELL[ssta] = {}
        for rr in np.arange(ss + 1, nb_stat):
            rnet = net_list[rr]
            rsta = stat_list[rr]
            dispfile = os.path.join(dispdir, f"{snet}.{ssta}", f"{snet}.{ssta}_{rnet}.{rsta}_group_{comp}_lagsym.csv")
            if os.path.exists(dispfile):
                picks = pd.read_csv(dispfile)
                # picks = picks.loc[ (picks.score == 1.0) & (picks.snr_nbG > 5) & (picks.ratio_d_lambda > 3.0), :] # Apply criteria
                picks = picks.loc[(picks.snr_nbG > 5) & (picks.ratio_d_lambda > 1.5), :]  # Apply criteria
                if len(picks.inst_period.values) > 0:
                    data = np.float32(np.vstack([picks.inst_period.values, picks.group_velocity.values]))
                    PICK_CELL[ssta][rsta] = data

    mdict = {'PICK_CELL': PICK_CELL}
    fname = os.path.join(output_folder, f"all_picks_{comp}_lamb15.mat")
    savemat(fname, mdict=mdict)
    Logger.info(f"Wrote file {fname}")
    fname = os.path.join(output_folder, f"all_picks_{comp}_lamb15.pkl")
    with open(fname, 'wb') as output:
        # Pickle dictionary using protocol 0.
        pickle.dump(PICK_CELL, output)


def make_data_kernels():
    # Make data_and_kern_Tx.mat
    # --------------------------------------

    fname = os.path.join(output_folder, "dist_stat.npz")
    npzfile = np.load(fname)
    DIST_mat = npzfile['DIST_mat']
    stat_list = npzfile['stat_list']
    nb_stat = len(stat_list)

    fname = os.path.join(output_folder, "kernel.npz")
    npzfile = np.load(fname)
    G_mat = npzfile['G_mat']
    nb_ray = G_mat.shape[0]

    fname = os.path.join(output_folder, f"all_picks_{comp}_lamb3.pkl")
    with open(fname, "rb") as fp:
        PICK_CELL = pickle.load(fp)

    # kernel for each period according to pick data
    output_dir_kern = "/home/users/s/savardg/riehen/vg_maps/kernels"
    if not os.path.exists(output_dir_kern):
        os.mkdir(output_dir_kern)

    Tc_min = 0.6
    Tc_max = 5.0
    Tc_dt = 0.1
    Tc_list = np.arange(Tc_min, Tc_max, Tc_dt)
    nb_cpl = nb_ray  # number of station pairs (= # of rays)

    num_picks = np.zeros(shape=(len(Tc_list),))

    # Iterate over periods: one kernel per period
    for i, Tc in enumerate(Tc_list):

        TAU = np.zeros(shape=(nb_cpl,))
        V_dat = np.zeros(shape=(nb_cpl,))
        bool_nodata = np.zeros(shape=(nb_cpl,))
        G_mat_Tc = G_mat.copy()

        # Iterate over pairs/rays
        cpl = 0
        N = 0
        for ss in range(nb_stat - 1):
            ssta = stat_list[ss]
            for rr in np.arange(ss + 1, nb_stat):
                rsta = stat_list[rr]

                # Get pick data for pair
                try:
                    T_list, V_list = PICK_CELL[ssta][rsta]
                except KeyError:  # no data for this pair
                    bool_nodata[cpl] = 1
                    cpl += 1
                    continue

                # Check for data at period Tc
                ind = np.where(np.abs(T_list - Tc) < 0.01)
                if ind[0].size == 0:  # no data for this pair at this specific period
                    bool_nodata[cpl] = 1
                elif ind[0].size > 1:
                    raise ValueError("Can't have more than 2 picks!")
                elif ind[0].size == 1:
                    # Compile data
                    TAU[cpl] = DIST_mat[ss, rr] / V_list[ind]
                    V_dat[cpl] = V_list[ind]
                    N += 1

                cpl += 1

        print("size of TAU before delete: ", TAU.shape)
        print("size of G_mat before delete: ", G_mat.shape)

        # Now trim arrays where there is no data
        indx = np.where(bool_nodata == 1)
        if indx[0].size > 0:
            list_exclude = indx[0]
            print("size of list_exclude:", len(list_exclude))
            G_mat_Tc = np.delete(G_mat_Tc, list_exclude, axis=0)
            TAU = np.delete(TAU, list_exclude, axis=0)
            V_dat = np.delete(V_dat, list_exclude, axis=0)
        else:
            print(f"no data excluded for T = {Tc}!?")
        v_moy = np.mean(V_dat)
        num_picks[i] = V_dat.shape[0]
        print(f"T = {Tc:.1f} s: # picks: {N}, {V_dat.shape[0]}")

        # Check if any problem occured
        if np.where(TAU == 0)[0].size > 0:
            raise ValueError("There are times = 0!")
        print("size of TAU: ", TAU.shape)
        print("size of G_mat: ", G_mat_Tc.shape)
        if G_mat_Tc.shape[0] != TAU.shape[0]:
            raise ValueError("Shape of TAU don't match G")

        # Save
        mdict = {'TAU': TAU,
                 'G_mat': G_mat_Tc,
                 'bool_nodata': bool_nodata,
                 'v_moy': v_moy,
                 'Tc': Tc
                 }
        fname = os.path.join(output_dir_kern, f"data_and_kern_T{Tc:.1f}_{comp}.mat")
        savemat(fname, mdict=mdict)
        Logger.info(f"Wrote file {fname}")
        fname = os.path.join(output_dir_kern, f"data_and_kern_T{Tc:.1f}_{comp}.npz")
        np.savez(fname,
                 TAU=TAU,
                 G_mat=G_mat_Tc,
                 bool_nodata=bool_nodata,
                 v_moy=v_moy,
                 Tc=Tc
                 )
        Logger.info(f"Wrote file {fname}")

    # Plot number of picks per period
    # fig, ax = plt.subplots(1,1,figsize=(12,6))
    # ax.stem(Tc_list[num_picks>0],num_picks[num_picks>0])
    # ax.set_title("Number of picks per period")
    # ax.set_xlabel("Period [s]")
    # plt.show()
    # plt.close()

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

