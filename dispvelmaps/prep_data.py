"""
This module contains the functions necessary to convert station info and dispersion picks calculated with NoisePy-derived codes by G.S.
 to the inputs necessary for the ant_matlab inversion scripts. Specify input parameters and paths in a YAML file (see method prep_all and example config file).
 Creates the following files:
 - stat_list_merged.mat
 - dist_stat.mat
 - stat_grid.mat
 - kernel.mat
 - pick_cell.mat

Genevieve Savard @ UniGe (last updated 04.08.2023)
"""

from scipy.io import savemat
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import numpy as np
import pickle
import logging
import yaml
import time

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
    Logger.info(f"Number of stations in H5 stack files: {len(stalst_h5)}")

    # Read station csv file used for noisepy
    stadf = pd.read_csv(station_file)
    stadf.station = stadf.station.astype(str)
    stadf = stadf.drop(columns="channel").drop_duplicates()  # remove duplicate rows
    Logger.info(f"Number of stations in CSV station file: {len(stadf.station.values)}")
    stadf = stadf[stadf['station'].isin(stalst_h5)]  # Keep station actually used for stacking
    Logger.info(f"Number of stations in common: {len(stadf.station.values)}")
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
        save_mat: save in ant_matlab format
        save_python: save in .npz format

    Returns: dictionary with 'DIST_mat','x_stat','y_stat','stat_list','x_max','y_max','SW_corner','SE_corner','NW_corner','NE_corner'

    """

    stat_list = stainfo['stat_list']
    net_list = stainfo['net_list']
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
    ref_lat_glob = min_lat * np.ones(shape=stat_lat.shape)  # southwest corner chosen as grid origin
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
        'net_list': net_list,
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
                 net_list=net_list,
                 x_max=x_max,
                 y_max=y_max,
                 SW_corner=SW_corner,
                 SE_corner=SE_corner,
                 NW_corner=NW_corner,
                 NE_corner=NE_corner
                 )
        Logger.info(f"Wrote file {fname}")
    return mdict


def make_stat_grid(dist_stat, dx_grid, dy_grid, output_folder, make_plot=True, save_mat=True, save_python=True):
    """
    Make stat_grid.mat file
    Args:
        dist_stat: dictionary with output of make_dist_stat()
        output_folder: output path for stat_grid files
        make_plot: make a plot of the grid
        save_mat: save to ant_matlab format
        save_python: save to .npz format

    Returns: dictionary with node positions and station locations in XY coordinates

    """
    # Make stat_grid.mat
    # -----------------------------------------------------
    x_max = dist_stat['x_max']
    y_max = dist_stat['y_max']
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
        'stat_list': dist_stat['stat_list'],
        'net_list': dist_stat['net_list']
    }
    if save_mat:
        fname = os.path.join(output_folder, "stat_grid.mat")
        savemat(fname, mdict=mdict)
        Logger.info(f"Wrote file {fname}")
    if save_python:
        fname = os.path.join(output_folder, "stat_grid.npz")
        np.savez(fname,
                 stat_list=dist_stat['stat_list'],
                 net_list=dist_stat['net_list'],
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
        save_mat: save to ant_matlab format
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
    net_list = stat_grid['net_list']
    nb_stat = len(x_stat)
    nb_cell = int(X_GRID.size)
    nb_ray = int(nb_stat * (nb_stat - 1) / 2)

    ray_mat = {}
    G_mat = np.zeros(shape=(nb_ray, nb_cell))
    IND_LIN_GRID = np.reshape(np.arange(0, nb_cell, dtype=np.int16), (len(x_grid), len(y_grid)),
                              order='F')  # np.reshape(...order='F') = ant_matlab reshape
    IND_S1 = np.zeros(shape=(nb_ray,), dtype=np.int16)  # to retrieve station from ray index
    IND_S2 = np.zeros(shape=(nb_ray,), dtype=np.int16)
    ind_ray = 0

    # construct kernel G
    # G_ij = distance traveled by ray i in cell j
    for s1 in range(nb_stat - 1):
        if s1 % 10 == 0: print(f"{s1}/{nb_stat}")
        ssta = stat_list[s1]
        snet = net_list[s1]
        skey = f"{snet}_{ssta}"  # dictionary key for source station
        ray_mat[skey] = {}
        for s2 in np.arange(s1 + 1, nb_stat):
            rsta = stat_list[s2]
            rnet = net_list[s2]
            rkey = f"{rnet}_{rsta}"  # dictionary key for source station

            delta_x = x_stat[s2] - x_stat[s1]
            delta_y = y_stat[s2] - y_stat[s1]
            dist = np.sqrt(delta_x ** 2 + delta_y ** 2)
            ux_ray, uy_ray = delta_x / dist, delta_y / dist
            ray_x = x_stat[s1] + np.arange(0, dist, dl) * ux_ray
            ray_y = y_stat[s1] + np.arange(0, dist, dl) * uy_ray
            ray_mat[skey][rkey] = np.vstack([ray_x, ray_y])

            # G_mat
            IND_S1[ind_ray], IND_S2[ind_ray] = s1, s2  # to retrieve station from ray index
            x_ind = np.int16(
                np.floor((ray_x - x_grid[0]) / dx_grid))  # x index of cell it falls on (if always positive values?)
            y_ind = np.int16(
                np.floor((ray_y - y_grid[0]) / dy_grid))  # y index of cell it falls on (if always positive values?)

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
            skey = f"{net_list[s1]}_{stat_list[s1]}"
            rkey = f"{net_list[s2]}_{stat_list[s2]}"
            x_ray_vec, y_ray_vec = ray_mat[skey][rkey]

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


def make_pick_cell_from_pairwise_files(disp_dir, output_folder, lag="sym", comp="ZZ", topology=False, snr_nbG_thresh=5.,
                   d_lambda_thresh=1.5, topologyMinScore=0.5, save_mat=True, save_python=True):
    """
    Extract picks and make PICK_CELL from all station pair-wise CSV files.
    Args:
        disp_dir: parent directory where pick files are.
            structure: (disp_dir, f"{snet}.{ssta}", f"{snet}.{ssta}_{rnet}.{rsta}_group_{comp}_lag{lag}.csv")
        output_folder: path where to write PICK_CELL
        lag: type of lay (default sym)
        comp: component (default ZZ)
        topology: if using topology method for picking
        snr_nbG_thresh: Threshold on SNR calculated with narrowband gaussian filter
        d_lambda_thresh: Threshold on ratio of distance/wavelength
        topologyMinScore: Minimum score of peaks retained for the topology peak detection method
        save_mat: for to ant_matlab format
        save_python: Save to pickle format

    Returns: dictionary with PICK_CELL

    """

    fname = os.path.join(output_folder, "stat_list_merged.npz")
    npzfile = np.load(fname, allow_pickle=True)
    stat_list = npzfile['stat_list_merged']
    net_list = npzfile['net_list_merged']
    nb_stat = len(stat_list)
    Logger.info(f"Number of stations: {nb_stat}")

    PICK_CELL = {}
    Ntot = 0
    for ss in range(nb_stat - 1):  # Iterate over virtual sources
        if ss % 50 == 0: print(f"{ss}/{nb_stat}")
        snet = net_list[ss]  # network for source station
        ssta = stat_list[ss]  # source station name
        skey = f"{snet}_{ssta}"  # key name for source station
        PICK_CELL[skey] = {}
        for rr in np.arange(ss + 1, nb_stat):  # Iterate over virtual receivers
            rnet = net_list[rr]
            rsta = stat_list[rr]
            rkey = f"{rnet}_{rsta}"
            dispfile = os.path.join(disp_dir, f"{snet}.{ssta}",
                                    f"{snet}.{ssta}_{rnet}.{rsta}_group_{comp}_lag{lag}.csv")
            if os.path.exists(dispfile):
                picks = pd.read_csv(dispfile)
                # Apply QC criteria
                if topology:
                    picks = picks.loc[(picks.score >= topologyMinScore) & (picks.snr_nbG > snr_nbG_thresh) & (
                            picks.ratio_d_lambda > d_lambda_thresh), :]
                else:
                    picks = picks.loc[(picks.snr_nbG > snr_nbG_thresh) & (picks.ratio_d_lambda > d_lambda_thresh), :]
                if len(picks.inst_period.values) > 0:
                    Ntot += len(picks.inst_period.values)
                    data = np.float32(np.vstack([picks.inst_period.values, picks.group_velocity.values]))
                    PICK_CELL[skey][rkey] = data
    Logger.info(f"Number of picks added: {Ntot}")
    mdict = {'PICK_CELL': PICK_CELL}
    if save_mat:
        fname = os.path.join(output_folder, f"all_picks_{comp}_lamb{d_lambda_thresh}.mat")
        savemat(fname, mdict=mdict)
        Logger.info(f"Wrote file {fname}")
    if save_python:
        fname = os.path.join(output_folder, f"all_picks_{comp}_lamb{d_lambda_thresh}.pkl")
        with open(fname, 'wb') as output:
            # Pickle dictionary using protocol 0.
            pickle.dump(PICK_CELL, output)
        Logger.info(f"Wrote file {fname}")
    return mdict


def picks_recursive_filtering(picks, multiplier=2):
    """
    Remove recursively picks outside a multiplier of the standard deviation around the mean at each period.
    Loop stops when all picks are within the defined boundaries.

    Args:
        picks: pandas.DataFrame of group velocity picks for different periods with at least the following columns:
        'inst_period', 'group_velocity'
        multiplier: Filtering threshold in terms of multiple of standard deviation. e.g. for multiplier=2, all picks
        outside two standard deviation from the mean are removed.

    Returns: filtered picks [pandas.DataFrame]

    """
    df = picks.copy()
    Nremoved = 1000  # dummy initialization
    while Nremoved > 10:
        # Filter within 2 standard deviations
        groups_byt_gv = df.groupby('inst_period')['group_velocity']
        gv_mean = groups_byt_gv.transform('mean')
        gv_std = groups_byt_gv.transform('std')
        ikeep = df['group_velocity'].between(gv_mean.sub(gv_std.mul(multiplier)),
                                             gv_mean.add(gv_std.mul(multiplier)), inclusive=False)
        Nremoved = df.shape[0] - ikeep.sum()
        df = df.loc[ikeep, :]

    return df


def make_pick_cell_from_dataframe(picks, station_fname, output_fname, save_mat=True, save_python=True):
    """
    Extract picks and make PICK_CELL
    Args:
        picks: pandas.DataFrame with selected picks
        station_fname: full path of station file "stat_list_merged.npz"
        output_fname: path where to write PICK_CELL without the extension.
        save_mat: save to ant_matlab ".mat"
        save_python: save to pickle ".pkl"

    Returns: dictionary with PICK_CELL
    """

    npzfile = np.load(station_fname, allow_pickle=True)
    stat_list = npzfile['stat_list_merged']
    net_list = npzfile['net_list_merged']
    nb_stat = len(stat_list)
    Logger.info(f"Number of stations: {nb_stat}")

    PICK_CELL = {}
    Ntot = 0
    ts = time.time()
    for ss in range(nb_stat - 1):  # Iterate over virtual sources
        if ss % 50 == 0:
            print(f"{ss}/{nb_stat}: {time.time() - ts} s elapsed")
        snet = net_list[ss]  # network for source station
        ssta = stat_list[ss]  # source station name
        skey = f"{snet}_{ssta}"  # key name for source station

        PICK_CELL[skey] = {}
        for rr in np.arange(ss + 1, nb_stat):  # Iterate over virtual receivers
            rnet = net_list[rr]
            rsta = stat_list[rr]
            rkey = f"{rnet}_{rsta}"

            tmp = picks.loc[(picks.stasrc == f"{snet}.{ssta}") & (picks.starcv == f"{rnet}.{rsta}"), :].copy()
            # Ensure no duplicates
            tmp.sort_values(by="group_velocity", inplace=True)
            tmp.drop_duplicates(subset="inst_period", keep="last", inplace=True)
            tmp.sort_values(by="inst_period", inplace=True)

            periods = tmp["inst_period"].values
            group_velocity = tmp["group_velocity"].values
            snr = tmp["score"].values

            if len(periods) > 0:
                Ntot += len(periods)
                data = np.float32(np.vstack([periods, group_velocity]))
                PICK_CELL[skey][rkey] = data

    Logger.info(f"Number of picks added: {Ntot}")
    mdict = {"PICK_CELL": PICK_CELL}
    if save_mat:
        fname = output_fname + ".mat"
        savemat(fname, mdict=mdict)
        Logger.info(f"Wrote file {fname}")
    if save_python:
        fname = output_fname + ".pkl"
        with open(fname, 'wb') as output:  # Pickle dictionary using protocol 0.
            pickle.dump(PICK_CELL, output)
        Logger.info(f"Wrote file {fname}")
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
        save_mat: save to ant_matlab format
        save_python: save to .npz format

    Returns: nothing

    """
    # Make data_and_kern_Tx.mat
    # kernel for each period according to pick data
    # --------------------------------------

    DIST_mat = dist_stat['DIST_mat']
    stat_list = dist_stat['stat_list']
    net_list = dist_stat['net_list']
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
            snet = net_list[ss]
            skey = f"{snet}_{ssta}"  # key name for virtual source
            for rr in np.arange(ss + 1, nb_stat):
                rsta = stat_list[rr]
                rnet = net_list[rr]
                rkey = f"{rnet}_{rsta}"  # key name for virtual receiver
                # Get pick data for pair
                try:
                    T_list, V_list = PICK_CELL[skey][rkey]
                    V_list /= 1e3  # Need km (ref. Thomas' codes)
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


def prep_all(config_file, ccomp_list, method="pws", save_mat=True, save_python=True):
    """
    Prepare station list, inversion grid, and pick data for group velocity map inversion
    Args:
        config_file: YAML parameter file
        ccomp_list: List of cross-component for which to extract picks
        method: stack method used for the picks. Used to get path to dispersion pick files
        save_mat: save outputs to ant_matlab variable files
        save_python: save outputs to numpy pickle files

    Returns:

    """
    # Extract parameters
    with open(config_file, 'r') as file:
        params = yaml.safe_load(file)
    Logger.info(f"Parameters read from file {config_file}")

    # Make stat_list_merged.mat:
    Logger.info("Making stat_list_merged.mat")
    stat_list_merged = make_stat_list(params["nccf"]["STACK_DIR"],
                                      params["station_csv_file"],
                                      params["nccf"]["fs"],
                                      params["output_folder"],
                                      save_mat=save_mat,
                                      save_python=save_python
                                      )

    # Make dist_stat.mat
    Logger.info("Making dist_stat.mat")
    bounds = {
        "min_lat": params["map_grid"]["min_lat"],
        "max_lat": params["map_grid"]["max_lat"],
        "min_lon": params["map_grid"]["min_lon"],
        "max_lon": params["map_grid"]["max_lon"]
    }
    stainfo = {
        "stat_list": stat_list_merged["stat_list_merged"],
        "net_list": stat_list_merged["net_list_merged"],
        "stat_lat": stat_list_merged["lat_merged"],
        "stat_lon": stat_list_merged["lon_merged"]
    }
    dist_stat = make_dist_stat(bounds,
                               stainfo,
                               params["output_folder"],
                               save_mat=save_mat,
                               save_python=save_python
                               )

    # Make stat_grid.mat
    Logger.info("Making stat_grid.mat")
    dgrid = make_stat_grid(dist_stat,
                           dx_grid=params["map_grid"]["dx_grid"],
                           dy_grid=params["map_grid"]["dy_grid"],
                           output_folder=params["output_folder"],
                           make_plot=True,
                           save_mat=save_mat,
                           save_python=save_python
                           )

    # Make kernel.mat
    Logger.info("Making kernel.mat")
    _ = make_kernel(dgrid,
                    output_folder=params["output_folder"],
                    dl=0.005,  # minimum distance for the ray to travel in a cell in km to count the cell
                    plot_random_kernels=3,
                    save_mat=save_mat,
                    save_python=save_python
                    )

    # Make PICK_CELL.mat
    # ----------------------------------

    # EXAMPLE 1: Make PICK_CELL from individual pair-wise CSV file with dispersion picks
    # for comp in ccomp_list:
    #     disp_dir = os.path.join(params["nccf"]["STACK_DIR"], "dispersion", f"vg_{comp}")
    #     Logger.info(f"Extracting picks for component {comp} from directory {disp_dir}")
    #     dpicks = make_pick_cell(disp_dir,
    #                             output_folder=params["output_folder"],
    #                             lag=params["picks"]["lag"],
    #                             comp=comp,
    #                             topology=params["picks"]["topology"],
    #                             snr_nbG_thresh=params["picks"]["snr_nbG_thresh"],
    #                             d_lambda_thresh=params["picks"]["d_lambda_thresh"],
    #                             save_mat=save_mat,
    #                             save_python=save_python
    #                             )

    # EXAMPLE 2: Make PICK_CELL from merged CSV dataframe (example)
    # picks = pd.read_csv("/media/genevieve/sandisk4TB/aargau-data/picks_merged_CHAA_V2_normZ.csv")
    # for comp in ["ZZ"]:  # "ZR", "RZ","ZZ-ZR","RR","all4","RR-RZ"]:
    #     print(comp)
    #     lag = "sym"  # "sym"
    #     pick_method = "topology"
    #     stack_method = "pws"
    #     score_thresh = 1.0
    #     multiplier = 2
    #     ratio_d_lambda = 1.5
    #     # picks = pd.read_csv("/media/savardg/sandisk4TB/riehen-data/picks_merged_CHRI_V2.csv")
    #     if comp == "all4":
    #         df = picks.loc[(picks.pick_method == pick_method) &
    #                        (picks.score >= score_thresh) &
    #                        #                          (picks.stack_method==stack_method) &
    #                        #                          (picks.snr_nbG >= 5) &
    #                        #                          (picks.ratio_d_lambda >= ratio_d_lambda) &
    #                        (picks.component == comp), :]
    #     else:
    #         df = picks.loc[(picks.pick_method == pick_method) &
    #                        (picks.score >= 1) &
    #                        (picks.component == comp) &
    #                        (picks.stack_method == stack_method) &
    #                        (picks.lag == lag) &
    #                        (picks.snr_nbG >= 5) &
    #                        (picks.ratio_d_lambda >= ratio_d_lambda), :]
    #     if multiplier > 0:
    #         # Filter within X standard deviations
    #         df = picks_recursive_filtering(df, multiplier=multiplier)
    #
    #     # Define paths
    #     output_folder = f"/media/genevieve/sandisk4TB/aargau-data/vg-maps/picks_V2_{pick_method}_{stack_method}"
    #     if not os.path.exists(output_folder):
    #         os.mkdir(output_folder)
    #     output_fname = os.path.join(output_folder, f"all_picks_{comp}_lamb{ratio_d_lambda}_mul{multiplier}")
    #     station_fname = os.path.join("/media/genevieve/sandisk4TB/aargau-data/vg-maps/stat_list_merged.npz")

    # Make PICK_CELL and save
    # t0 = time.time()
    # mdict = make_pick_cell_from_dataframe(picks, station_fname, output_fname, save_mat=True, save_python=True)
    # print(f"Elapsed time: {time.time() - t0} s.")
