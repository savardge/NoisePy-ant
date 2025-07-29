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

# User inputs
# -----------------------------------------------------
config_file = sys.argv[1]  # Input parameter file as first argument
with open(config_file, 'r') as file:
    para = yaml.safe_load(file)

STACK_DIR = para["STACK_DIR"]
fs = para["fs"]
stacsv = para["stacsv"]
comp = para["comp"]
stack_type = para["stack_type"]

# For grid:
min_lat = para["min_lat"]
max_lat = para["max_lat"]
min_lon = para["min_lon"]
max_lon = para["max_lon"]
R_earth = para["R_earth"]

# output path
output_folder = para["output_folder"]
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Make stat_list_merged.mat:
# -----------------------------------------------------
# 'stat_list_merged','lat_merged','lon_merged','elev_merged','fs_new'

# Get list of stations contained in stack files
stackfiles = glob.glob(os.path.join(STACK_DIR, "*", "*.h5"))
stalst_h5 = []
for f in stackfiles:
    ff = os.path.split(f)[1]
    sta1 =ff.split(".h5")[0].split("_")[0].split(".")[1]
    if sta1 not in stalst_h5:
        stalst_h5.append(sta1)
    sta2 = ff.split(".h5")[0].split("_")[1].split(".")[1]
    if sta2 not in stalst_h5:
        stalst_h5.append(sta2)
Logger.info(f"Number of stations from stack H5 files: {len(stalst_h5)}")

# Read station csv file used for noisepy
stadf = pd.read_csv(stacsv)
stadf.station = stadf.station.astype(str)
stadf = stadf.drop(columns="channel").drop_duplicates() # remove duplicate rows
Logger.info(f"Number of stations from csv station file: {stadf.shape[0]}")
stadf = stadf[stadf['station'].isin(stalst_h5)] # Keep station actually used for stacking
Logger.info(f"Number of stations after merge: {stadf.shape[0]}")
stat_list = stadf["station"].values
stat_lat = stadf["latitude"].values
stat_lon = stadf["longitude"].values
stat_elev = stadf["elevation"].values
mdict = {
    'stat_list_merged': stat_list,
    'lat_merged': stat_lat,
    'lon_merged': stat_lon,
    'elev_merged': stat_elev,
    'fs_new': fs
}
fname = os.path.join(output_folder, "stat_list_merged.mat")
savemat(fname, mdict=mdict)
Logger.info(f"Wrote file {fname}")

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
x_max = R_earth * np.cos( (max_lat+min_lat)/2 * np.pi/180) * (max_lon-min_lon) * np.pi/180 # nb: by definition xmin = 0 and ymin = 0
y_max = R_earth * (max_lat-min_lat) * np.pi/180

x_stat = R_earth * np.cos( (stat_lat+ref_lat_glob)/2 * np.pi/180) * (stat_lon-ref_lon_glob) * np.pi/180
y_stat = R_earth * (stat_lat-ref_lat_glob) * np.pi/180

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

# Make stack_select_ZZ.mat
# -----------------------------------------------------
# 'STACK', 'STACK_FULL', 'stat_list_merged','nb_samp_avg'
# nb_samp_CCF=length(lags); clear lags;
# nb_samp_avg=(nb_samp_CCF-1)/2+1; % number of samp after causal-anticausal average

# Get nb_samp:
with pyasdf.ASDFDataSet(stackfiles[0], mode="r") as ds:
    path_list_type = ds.auxiliary_data.list()
    path_list_comp = ds.auxiliary_data[path_list_type[0]].list()
    nb_samp_CCF = ds.auxiliary_data[path_list_type[0]][path_list_comp[0]].data.shape[0]
    nb_samp_avg = int((nb_samp_CCF - 1) / 2 + 1)

STACK_full = np.zeros(shape=(nb_samp_CCF, nb_stat, nb_stat)) # full CCFs
STACK = np.zeros(shape=(nb_samp_avg, nb_stat, nb_stat)) # symmetric CCFs
Nadded = 0
for ss in range(nb_stat):
    stat_src = stat_list[ss]
    for rr in np.arange(ss+1, nb_stat):
        if ss == rr:
            continue
        stat_rec = stat_list[rr]
        Logger.info(f"Processing pair {stat_src}_{stat_rec}.")
        sfile = [f for f in stackfiles if stat_src in os.path.split(f)[1].split("_")[0] and stat_rec in os.path.split(f)[1].split("_")[1]]
        if len(sfile) == 0:
            Logger.info(f"Could not find stack file for pair {stat_src}_{stat_rec}.")
            continue
        elif len(sfile) == 1:
            sfile = sfile[0]
        else:
            Logger.error(f"More than 1 stack file for pair {stat_src}_{stat_rec} ?!? Check data.")

        with pyasdf.ASDFDataSet(sfile, mode="r") as ds:
            if stack_type in ds.auxiliary_data.list():
                if comp in ds.auxiliary_data[stack_type].list():

                    tdata = ds.auxiliary_data[stack_type][comp].data
                    tdata_sym = (tdata[nb_samp_avg-1:] + tdata[:nb_samp_avg]) / 2
                    STACK_full[:, ss, rr] = tdata
                    STACK[:, ss, rr] = tdata_sym
                    Nadded += 1
                else:
                    Logger.info(f"Component {comp} not found for {stack_type} and pair {stat_src}_{stat_rec}.")
            else:
                Logger.info(f"Stack type {stack_type} not found for pair {stat_src}_{stat_rec}.")
Logger.info(f"Read in {Nadded} CCFs out of {len(stackfiles)} stack files.")
mdict = {
    'stat_list_merged': stat_list,
    'STACK_full': STACK_full,
    'STACK': STACK,
    'nb_samp_avg': nb_samp_avg,
    'stack_type': stack_type, # added for more details
    'comp': comp, # added for more details
    'freq_norm': para["freq_norm"], # added for more details
    'time_norm': para["freq_norm"], # added for more details
    'cc_method': para["cc_method"], # added for more details
    'normz': para["normz"]
}

fname = os.path.join(output_folder, f"stack_select_{comp}.mat")
savemat(fname, mdict=mdict)
Logger.info(f"Wrote file {fname}")