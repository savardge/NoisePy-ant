import sys
import glob
import os,gc
import obspy
import time
import pyasdf
import numpy as np
import pandas as pd
from mpi4py import MPI

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
os.system('export HDF5_USE_FILE=FALSE')
from noisepy import preprocess_h5

#######################################################
################PARAMETER SECTION######################
#######################################################
tt0=time.time()

# data/file paths
rootpath  = '/home/users/s/savardg/aargau_ant'                           # absolute path for your project
RAWDATA   = '/home/users/s/savardg/scratch/aargau/RAW_DATA' #os.path.join(rootpath,'RAW_DATA')                           # dir where mseed/SAC files are located
DATADIR   = '/home/users/s/savardg/scratch/aargau/CLEAN_DATA_H5_10sps'                         # dir where cleaned data in ASDF format are going to be outputted
locations = '/home/users/s/savardg/aargau_ant/station_locations_noisepy_cleaned_NEZ.csv'       # station info including network,station,channel,latitude,longitude,elevation

# useful parameters for cleaning the data
input_fmt = 'sac'                                                       # input file format between 'sac' and 'mseed' 
samp_freq = 10                                                          # targeted sampling rate
stationxml= False                                                       # station.XML file exists or not
rm_resp   = 'no'                                                        # select 'no' to not remove response and use 'inv','spectrum','RESP', or 'polozeros' to remove response
respdir   = os.path.join(rootpath,'resp')                               # directory where resp files are located (required if rm_resp is neither 'no' nor 'inv')
freqmin   = 0.05                                                        # pre filtering frequency bandwidth
freqmax   = 4.5                                                           # note this cannot exceed Nquist freq
flag      = True                                                       # print intermediate variables and computing time

# having this file saves a tons of time: see L95-126 for why
wiki_file = os.path.join(rootpath,'allfiles_time.csv')                  # file containing the path+name for all sac/mseed files and its start-end time      
allfiles_path = os.path.join(RAWDATA, "300*", "*." + input_fmt)                   # make sure all sac/mseed files can be found through this format
messydata = True                                                       # set this to False when daily noise data directory is stored in sub-directory of Event_year_month_day 
ncomp     = 3

# targeted time range # Dec 5 to Jan 5
start_date = ['2020_12_05_0_0_0']                                       # start date of local data
end_date   = ['2021_01_05_0_0_0']                                       # end date of local data
inc_hours  = 8                                                          # sac/mseed file length for a continous recording

# get rough estimate of memory needs to ensure it now below up in S1
cc_len    = 1800                                                        # basic unit of data length for fft (s)
step      = 450                                                         # overlapping between each cc_len (s)
MAX_MEM   = 8.0                                                         # maximum memory allowed per core in GB

##################################################
# we expect no parameters need to be changed below

# assemble parameters for data pre-processing
prepro_para = {'RAWDATA':DATADIR,
               'wiki_file':wiki_file,
               'messydata':messydata,
               'input_fmt':input_fmt,
               'stationxml':stationxml,
               'rm_resp':rm_resp,
               'respdir':respdir,
               'freqmin':freqmin,
               'freqmax':freqmax,
               'samp_freq':samp_freq,
               'inc_hours':inc_hours,
               'start_date':start_date,
               'end_date':end_date,
               'allfiles_path':allfiles_path,
               'cc_len':cc_len,
               'step':step,
               'ncomp':ncomp,
               'MAX_MEM':MAX_MEM}
metadata = os.path.join(DATADIR,'download_info.txt') 

##########################################################
#################PROCESSING SECTION#######################
##########################################################

#---------MPI-----------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(f"Rank {rank}, size {size}")
#-----------------------

if rank == 0:
    
    # assemble timestamp info
#     allfiles = glob.glob(allfiles_path)
    all_stimes, allfiles = preprocess_h5.make_timestamps(prepro_para)

    # all time chunk for output: loop for MPI
    all_chunk = preprocess_h5.get_event_list(start_date[0],end_date[0],inc_hours)
    splits     = len(all_chunk)-1
    if splits<1:raise ValueError('Abort! no chunk found between %s-%s with inc %s'%(start_date[0],end_date[0],inc_hours))

else:
    splits,all_chunk,all_stimes,allfiles = [None for _ in range(4)]

# broadcast the variables
splits     = comm.bcast(splits,root=0)
all_chunk = comm.bcast(all_chunk,root=0)
all_stimes = comm.bcast(all_stimes,root=0)
allfiles   = comm.bcast(allfiles,root=0)

# MPI: loop through each time-chunk
for ick in range(rank,splits,size):
    t0=time.time()

    # time window defining the time-chunk
    s1=obspy.UTCDateTime(all_chunk[ick])
    s2=obspy.UTCDateTime(all_chunk[ick+1])     
    time1=s1-obspy.UTCDateTime(1970,1,1)
    time2=s2-obspy.UTCDateTime(1970,1,1)     
    
    # find all data pieces having data of the time-chunk
    indx1 = np.where((time1>=all_stimes[:,0]) & (time1<all_stimes[:,1]))[0]
    indx2 = np.where((time2>all_stimes[:,0]) & (time2<=all_stimes[:,1]))[0]
    indx3 = np.where((time1<=all_stimes[:,0]) & (time2>=all_stimes[:,1]))[0]
    indx4 = np.concatenate((indx1,indx2,indx3))
    indx  = np.unique(indx4)

    print(f"Rank {rank}, size {size}\n\tstarttime = {s1} (time1 = {time1}), Endtime = {s2} (time2 = {time2})\n\t# files to process: {len(indx)}")

tt1=time.time()
print('step0B takes '+str(tt1-tt0)+' s')

comm.barrier()
if rank == 0:
    sys.exit()
