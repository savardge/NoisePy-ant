from scipy.io import savemat
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import numpy as np
import pickle
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
Logger = logging.getLogger(__name__)


def make_stat_list(STACK_DIR, station_file, fs, output_folder, save_mat=True, save_python=True):
    """
    Create "stat_list_merged.mat" file for stations with NCCF data
    Args:
        STACK_DIR: Parent directory of stack output of NoisePy
        station_file: CSV file with columns: network, station, channel, latitude, longitude, elevation.
            Same station file as used by NoisePy
        fs: sampling rate of stacked noise CCFs
        output_folder: parent directory where to write "stat_list_merged"
        save_mat: save in Matlab format
        save_python: save in .npz format

    Returns: dictionary of 'stat_list_merged','lat_merged','lon_merged','elev_merged','fs_new'

    """

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
    stadf = pd.read_csv(station_file)
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
    if save_mat:
        fname = os.path.join(output_folder, "stat_list_merged.mat")
        savemat(fname, mdict=mdict)
        Logger.info(f"Wrote file {fname}")
    if save_python:
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
    return mdict


def make_dist_stat(bounds, stainfo, output_folder, R_earth=6371, save_mat=True, save_python=True):
    """
    Make dist_stat.mat
    Args:
        bounds: dictionary of geographic bounds with keys: min_lat, min_lon, max_lat, max_lon
        stainfo: dictionary of station info with keys: stat_list, stat_lat, stat_lon
        output_folder: parent folder where to write dist_stat files
        R_earth: radius of earth used for transformation to cartesian coordinates
        save_mat: save in matlab format
        save_python: save in .npz format

    Returns: dictionary with 'DIST_mat','x_stat','y_stat','stat_list','x_max','y_max','SW_corner','SE_corner','NW_corner','NE_corner'

    """

    stat_list = stainfo['stat_list']
    stat_lat = stainfo['stat_lat']
    stat_lon = stainfo['stat_lon']
    nb_stat = len(stat_list)

    min_lat = bounds['min_lat']
    max_lat = bounds['max_lat']
    min_lon = bounds['min_lon']
    max_lon = bounds['max_lon']
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
    if save_mat:
        fname = os.path.join(output_folder, "dist_stat.mat")
        savemat(fname, mdict=mdict)
        Logger.info(f"Wrote file {fname}")
    if save_python:
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
    return mdict


def make_stat_grid(dist_stat, output_folder, make_plot=True, save_mat=True, save_python=True):
    """
    Make stat_grid.mat file
    Args:
        dist_stat: dictionary with output of make_dist_stat()
        output_folder: output path for stat_grid files
        make_plot: make a plot of the grid
        save_mat: save to matlab format
        save_python: save to .npz format

    Returns: dictionary with node positions and station locations in XY coordinates

    """
    # Make stat_grid.mat
    # -----------------------------------------------------
    x_max = dist_stat['x_max']
    y_max = dist_stat['y_max']
    dx_grid = dist_stat['dx_grid']
    dy_grid = dist_stat['dy_grid']
    x_stat = dist_stat['x_stat']
    y_stat = dist_stat['y_stat']

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
        'y_max': y_max,
        'stat_list': dist_stat['stat_list']
    }
    if save_mat:
        fname = os.path.join(output_folder, "stat_grid.mat")
        savemat(fname, mdict=mdict)
        Logger.info(f"Wrote file {fname}")
    if save_python:
        fname = os.path.join(output_folder, "stat_grid.npz")
        np.savez(fname,
                 stat_list=dist_stat['stat_list'],
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

    if make_plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot(X_GRID, Y_GRID, "b+")
        ax.scatter(x_stat, y_stat, c="r", s=20, marker="^")
        ax.set_xlim((0, x_max))
        ax.set_ylim((0, y_max))
        plt.savefig(os.path.join(output_folder, "stat_grid.png"))
        plt.close()

    return mdict


def make_kernel(stat_grid, output_folder, dl=0.005, plot_random_kernels=None, save_mat=True, save_python=True):
    """
    Make kernel.mat
    Args:
        stat_grid: dictionary with keys x_stat, y_stat, stat_list, X_GRID, Y_GRID, x_grid, y_grid, dx_grid, dy_grid
        output_folder: Path where to store kernel.mat
        dl: minimum length of ray in cell in km
        plot_random_kernels: number of random kernels to plot for checking
        save_mat: save to matlab format
        save_python: save to python .npz format

    Returns: dictionary of G_mat and grid info in XY coordinates

    """
    # Make kernel.mat
    # -----------------------------------------------------
    x_stat = stat_grid['x_stat']
    y_stat = stat_grid['y_stat']
    x_grid = stat_grid['x_grid']
    y_grid = stat_grid['y_grid']
    dx_grid = stat_grid['dx_grid']
    dy_grid = stat_grid['dy_grid']
    X_GRID = stat_grid['X_GRID']
    Y_GRID = stat_grid['Y_GRID']
    stat_list = stat_grid['stat_list']
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

    # print(ind_ray)

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
    if save_mat:
        fname = os.path.join(output_folder, "kernel.mat")
        savemat(fname, mdict=mdict)
        Logger.info(f"Wrote file {fname}")
    if save_python:
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

    if plot_random_kernels:
        # Plot a few random kernels
        for ray in np.random.randint(0, nb_ray, size=plot_random_kernels):
            s1, s2 = IND_S1[ray], IND_S2[ray]
            x_ray_vec, y_ray_vec = ray_mat[stat_list[s1]][stat_list[s2]]

            G_slice = np.reshape(G_mat[ray, :], (len(x_grid), len(y_grid)), order='F')

            # Plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.pcolor(X_GRID, Y_GRID, G_slice,
                      cmap='Greys',
                      vmin=0, vmax=np.sqrt(dx_grid ** 2 + dy_grid ** 2))
            ax.set_title(f"Distance traveled in cell (km) by ray {ray}")
            ax.scatter(x_stat, y_stat, c='k', s=16, marker='v')
            ax.plot(x_ray_vec, y_ray_vec, 'k-')
            ax.plot(X_GRID, Y_GRID, '+b')
            ax.scatter(x_stat[s1], y_stat[s1], c='r', s=60, marker='v')
            ax.text(x_stat[s1], y_stat[s1], stat_list[s1], fontsize=14)
            ax.scatter(x_stat[s2], y_stat[s2], c='r', s=60, marker='v')
            ax.text(x_stat[s2], y_stat[s2], stat_list[s2], fontsize=14)
            ax.set_xlabel('X (km)')
            ax.set_ylabel('Y (km)')
            plt.savefig(os.path.join(output_folder, f"kernel_ray{ray}.png"))
            plt.close()
    return mdict


def make_pick_cell(disp_dir, output_folder, lag="sym", comp="ZZ", topology=False, snr_nbG_thresh=5.,
                   d_lambda_thresh=1.5, save_mat=True, save_python=True):
    """
    Extract picks and make PICK_CELL
    Args:
        disp_dir: parent directory where pick files are.
            structure: (disp_dir, f"{snet}.{ssta}", f"{snet}.{ssta}_{rnet}.{rsta}_group_{comp}_lag{lag}.csv")
        output_folder: path where to write PICK_CELL
        lag: type of lay (default sym)
        comp: component (default ZZ)
        topology: if using topology method for picking
        snr_nbG_thresh: Threshold on SNR calculated with narrow-band gaussian filter
        d_lambda_thresh: Threshold on ratio of distance/wavelength
        save_mat: for to matlab format
        save_python: Save to pickle format

    Returns: dictionary with PICK_CELL

    """

    fname = os.path.join(output_folder, "stat_list_merged.npz")
    npzfile = np.load(fname)
    stat_list = npzfile['stat_list_merged']
    net_list = npzfile['net_list_merged']
    nb_stat = len(stat_list)

    PICK_CELL = {}
    for ss in range(nb_stat - 1):
        if ss % 50 == 0: print(f"{ss}/{nb_stat}")
        snet = net_list[ss]
        ssta = stat_list[ss]
        PICK_CELL[ssta] = {}
        for rr in np.arange(ss + 1, nb_stat):
            rnet = net_list[rr]
            rsta = stat_list[rr]
            dispfile = os.path.join(disp_dir, f"{snet}.{ssta}",
                                    f"{snet}.{ssta}_{rnet}.{rsta}_group_{comp}_lag{lag}.csv")
            if os.path.exists(dispfile):
                picks = pd.read_csv(dispfile)
                # Apply QC criteria
                if topology:
                    picks = picks.loc[(picks.score == 1.0) & (picks.snr_nbG > snr_nbG_thresh) & (
                                picks.ratio_d_lambda > d_lambda_thresh), :]
                else:
                    picks = picks.loc[(picks.snr_nbG > snr_nbG_thresh) & (picks.ratio_d_lambda > d_lambda_thresh), :]
                if len(picks.inst_period.values) > 0:
                    data = np.float32(np.vstack([picks.inst_period.values, picks.group_velocity.values]))
                    PICK_CELL[ssta][rsta] = data

    mdict = {'PICK_CELL': PICK_CELL}
    if save_mat:
        fname = os.path.join(output_folder, f"all_picks_{comp}_lamb15.mat")
        savemat(fname, mdict=mdict)
        Logger.info(f"Wrote file {fname}")
    if save_python:
        fname = os.path.join(output_folder, f"all_picks_{comp}_lamb15.pkl")
        with open(fname, 'wb') as output:
            # Pickle dictionary using protocol 0.
            pickle.dump(PICK_CELL, output)
    return mdict


def make_data_kernels(dist_stat, kernel, pick_cell, Tc_list, output_dir_kern, plot_num_picks=True, save_mat=True,
                      save_python=True):
    """
    Make kernels for each period matching existing picks (data_and_kern_TX.mat)
    Args:
        dist_stat: dictionary output of make_dist_stat()
        kernel: dictionary output of make_kernel()
        pick_cell: dictionary with key PICK_CELL
        Tc_list: List of periods at which to make kernels
        output_dir_kern: Output directory
        plot_num_picks: Make a plot of number of picks vs period
        save_mat: save to matlab format
        save_python: save to .npz format

    Returns: nothing

    """
    # Make data_and_kern_Tx.mat
    # kernel for each period according to pick data
    # --------------------------------------

    DIST_mat = dist_stat['DIST_mat']
    stat_list = dist_stat['stat_list']
    nb_stat = len(stat_list)
    G_mat = kernel['G_mat']
    nb_cpl = G_mat.shape[0]  # number of station pairs (= # of rays)

    PICK_CELL = pick_cell['PICK_CELL']

    if not os.path.exists(output_dir_kern):
        os.mkdir(output_dir_kern)

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

        # Now trim arrays where there is no data
        indx = np.where(bool_nodata == 1)
        if indx[0].size > 0:
            list_exclude = indx[0]
            G_mat_Tc = np.delete(G_mat_Tc, list_exclude, axis=0)
            TAU = np.delete(TAU, list_exclude, axis=0)
            V_dat = np.delete(V_dat, list_exclude, axis=0)
        else:
            Logger.info(f"no data excluded for T = {Tc}!?")
        v_moy = np.mean(V_dat)
        num_picks[i] = V_dat.shape[0]
        Logger.info(f"T = {Tc:.1f} s: # picks: {N}, {V_dat.shape[0]}")

        # Check if any problem occured
        if np.where(TAU == 0)[0].size > 0:
            raise ValueError("There are times = 0!")
        Logger.info("size of TAU: %d" % TAU.shape)
        Logger.info("size of G_mat: (%d,%d)" % (G_mat_Tc.shape[0], G_mat_Tc.shape[1]))
        if G_mat_Tc.shape[0] != TAU.shape[0]:
            Logger.error("Shape of TAU don't match G")

        # Save
        mdict = {'TAU': TAU,
                 'G_mat': G_mat_Tc,
                 'bool_nodata': bool_nodata,
                 'v_moy': v_moy,
                 'Tc': Tc
                 }
        if save_mat:
            fname = os.path.join(output_dir_kern, f"data_and_kern_T{Tc:.1f}.mat")
            savemat(fname, mdict=mdict)
            Logger.info(f"Wrote file {fname}")
        if save_python:
            fname = os.path.join(output_dir_kern, f"data_and_kern_T{Tc:.1f}.npz")
            np.savez(fname,
                     TAU=TAU,
                     G_mat=G_mat_Tc,
                     bool_nodata=bool_nodata,
                     v_moy=v_moy,
                     Tc=Tc
                     )
            Logger.info(f"Wrote file {fname}")

    if plot_num_picks:
        # Plot number of picks per period
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.stem(Tc_list[num_picks > 0], num_picks[num_picks > 0])
        ax.set_title("Number of picks per period")
        ax.set_xlabel("Period [s]")
        plt.savefig(os.path.join(output_dir_kern, "num_picks_per_period.png"))
        plt.close()
