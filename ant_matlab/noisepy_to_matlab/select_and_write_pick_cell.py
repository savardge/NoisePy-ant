import pandas as pd
import numpy as np
import os
import pandas
import numpy as np
from scipy.io import savemat
import logging
import pickle
import time
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
Logger = logging.getLogger(__name__)

def picks_recursive_filtering(picks, multiplier=2):
    """ 
    Remove recursively picks outside a multiplier of the standard deviation around the mean at each period.
    Loop stops when all picks are within the defined boundaries.
    """    
    df = picks.copy()
    Nremoved = 1000 # dummy initialization
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

    fname = os.path.join(output_folder, "stat_list_merged.npz")
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
            print(f"{ss}/{nb_stat}: {time.time()-ts} s elapsed")
        snet = net_list[ss]  # network for source station
        ssta = stat_list[ss]  # source station name
        skey = f"{snet}_{ssta}"  # key name for source station

        PICK_CELL[skey] = {}
        for rr in np.arange(ss + 1, nb_stat):  # Iterate over virtual receivers
            rnet = net_list[rr]
            rsta = stat_list[rr]
            rkey = f"{rnet}_{rsta}"

            tmp = df.loc[(df.stasrc == f"{snet}.{ssta}") & (df.starcv == f"{rnet}.{rsta}"),:].copy() 
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
        with open(fname, 'wb') as output: # Pickle dictionary using protocol 0.            
            pickle.dump(PICK_CELL, output)
        Logger.info(f"Wrote file {fname}")
    return mdict

if __name__ == "__main__":
    picks = pd.read_csv("/media/genevieve/sandisk4TB/aargau-data/picks_merged_CHAA_V2_normZ.csv")

    # Select picks
    for comp in ["ZZ"]: #"ZR", "RZ","ZZ-ZR","RR","all4","RR-RZ"]:
        print(comp)
        lag = "sym" #"sym"
        pick_method = "topology"
        stack_method = "pws"
        score_thresh = 1.0
        multiplier = 2
        ratio_d_lambda = 1.5
        # picks = pd.read_csv("/media/savardg/sandisk4TB/riehen-data/picks_merged_CHRI_V2.csv")
        if comp == "all4":
            df = picks.loc[(picks.pick_method==pick_method) & 
                             (picks.score >= score_thresh) & 
    #                          (picks.stack_method==stack_method) & 
    #                          (picks.snr_nbG >= 5) &
    #                          (picks.ratio_d_lambda >= ratio_d_lambda) &
                             (picks.component==comp), :]
        else:
            df = picks.loc[(picks.pick_method==pick_method) & 
                            (picks.score >= 1) & 
                            (picks.component==comp) &
                            (picks.stack_method==stack_method) &                     
                            (picks.lag==lag) &
                            (picks.snr_nbG >= 5) &
                            (picks.ratio_d_lambda >= ratio_d_lambda) , :]
        if multiplier > 0:
            # Filter within X standard deviations
            df = picks_recursive_filtering(df, multiplier=multiplier)

        # Define paths
        output_folder = f"/media/genevieve/sandisk4TB/aargau-data/vg-maps/picks_V2_{pick_method}_{stack_method}"
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        output_fname = os.path.join(output_folder, f"all_picks_{comp}_lamb{ratio_d_lambda}_mul{multiplier}")
        station_fname = os.path.join("/media/genevieve/sandisk4TB/aargau-data/vg-maps/stat_list_merged.npz")

        # Make PICK_CELL and save
        t0 = time.time()
        mdict= make_pick_cell_from_dataframe(picks, station_fname, output_fname, save_mat=True, save_python=True)
        print(f"Elapsed time: {time.time() - t0} s.")