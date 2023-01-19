"""
Auto-correlation stacking script of NoisePy to:
    1) load auto-correlation data for sub-stacking (if needed) and all-time average;
    2) stack data with either linear or phase weighted stacking (pws) methods (or both);
    3) save outputs in ASDF or SAC format depend on user's choice (for latter option, find the script of write_sac
       in the folder of application_modules;
"""
import sys
import time
import pyasdf
import os
import glob
import datetime
import numpy as np
import yaml
import pandas as pd
from mpi4py import MPI
from noisepy import stacking

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
Logger = logging.getLogger(__name__)

tt0 = time.time()

# PARAMETER SECTION

config_file = sys.argv[1]  # Input parameter file as first argument
with open(config_file, 'r') as file:
    stack_para = yaml.safe_load(file)

# maximum memory allowed per core in GB
MAX_MEM = stack_para['MAX_MEM']

# absolute path parameters
rootpath = stack_para['rootpath']  # root path for this data processing
CCFDIR = stack_para['CCFDIR']  # dir where CC data is stored
STACKDIR = stack_para['STACKDIR']  # dir where stacked data is going to
locations = stack_para['locations']  # station info including network,station,channel,latitude,longitude,elevation
if not os.path.isfile(locations):
    raise ValueError('Abort! station info is needed for this script')

# define new stacking para
keep_substack = stack_para['keep_substack']  # keep all sub-stacks in final ASDF file
flag = stack_para['flag']  # output intermediate args for debugging
stack_method = stack_para['stack_method']  # linear, pws, robust, nroot, selective, auto_covariance or all
ncomp = stack_para['ncomp']
overwrite = stack_para['overwrite']
enz_system_option = stack_para['enz_system_option']

##################################################

# load fc_para parameters from Step1
fc_metadata = os.path.join(CCFDIR, 'fft_cc_data.yaml')
with open(fc_metadata, "r") as file:
    fc_para = yaml.safe_load(file)
ncomp = fc_para['ncomp']
samp_freq = fc_para['samp_freq']
start_date = fc_para['start_date']
end_date = fc_para['end_date']
inc_hours = fc_para['inc_hours']
cc_len = fc_para['cc_len']
step = fc_para['step']
maxlag = fc_para['maxlag']
substack = fc_para['substack']
substack_len = fc_para['substack_len']

# Add fc_para to stack_para
stack_para.update(fc_para)

# cross component info
if ncomp == 1:
    enz_system = ['ZZ']
else:
    if enz_system_option == 1:
        enz_system = ['EE','EN','EZ','NN','NZ','ZZ']
    elif enz_system_option == 2 :
        enz_system = ['EE', 'NE', 'NN', 'ZE', 'ZN', 'ZZ']
    else:
        msg = f"enz_system_option parameter must be 1 or 2."
        raise ValueError(msg)

# save fft metadata for future reference
stack_metadata = os.path.join(STACKDIR, 'stack_data.yaml')
if os.path.exists(stack_metadata):
    stack_metadata = stack_metadata.replace(".yaml", "_.yaml")

# --------MPI---------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:

    if not os.path.isdir(STACKDIR):
        os.mkdir(STACKDIR)

    # save metadata
    with open(stack_metadata, 'w') as file:
        yaml.dump(stack_para, file, sort_keys=False)

    # cross-correlation files
    ccfiles = sorted(glob.glob(os.path.join(CCFDIR, '*.h5')))

    # load station info
    tlocs = pd.read_csv(locations)
    tlocs.network = tlocs.network.fillna('')  # GS fix
    tlocs.station = tlocs.station.astype(str)  # GS fix
    sta = sorted(np.unique(tlocs['network'] + '.' + tlocs['station']))
    print(sta)
    for ii in range(len(sta)):
        tmp = os.path.join(STACKDIR, sta[ii])
        if not os.path.isdir(tmp):
            if flag: Logger.info(f"Creating directory {tmp}")
            os.mkdir(tmp)

    # station-pairs
    pairs_all = []
    for ii in range(len(sta)):
        pairs_all.append(sta[ii] + '_' + sta[ii])
    print(pairs_all)
    splits = len(pairs_all)
    if len(ccfiles) == 0 or splits == 0:
        raise IOError('Abort! no available CCF data for stacking')

else:
    splits, ccfiles, pairs_all = [None for _ in range(3)]

# broadcast the variables
splits = comm.bcast(splits, root=0)
ccfiles = comm.bcast(ccfiles, root=0)
pairs_all = comm.bcast(pairs_all, root=0)

# MPI loop: loop through each user-defined time chunck
for ipair in range(rank, splits, size):
    t0 = time.time()

    if flag: Logger.info(f"{ipair}th path for station-pair {pairs_all[ipair]}")
    # source folder
    ttr = pairs_all[ipair].split('_')
    snet, ssta = ttr[0].split('.')
    rnet, rsta = ttr[1].split('.')
    idir = ttr[0]
    # check if it is auto-correlation
    if ssta != rsta:
        continue  # Skip cross-correlations

    # continue when file is done
    toutfn = os.path.join(STACKDIR, idir + '/' + pairs_all[ipair] + '.tmp')
    if os.path.isfile(toutfn) and not overwrite:
        Logger.info(f"tmp file {toutfn} already processed. Next.")
        continue

    if flag: Logger.info(f"{ipair}th path for station-pair {pairs_all[ipair]}")

    # crude estimation on memory needs (assume float32)
    nccomp = len(enz_system)
    num_chunck = len(ccfiles) * nccomp
    num_segmts = 1
    if substack:  # things are difference when do substack
        if substack_len == cc_len:
            num_segmts = int(np.floor((inc_hours * 3600 - cc_len) / step))
        else:
            num_segmts = int(inc_hours / (substack_len / 3600))
    npts_segmt = int(2 * maxlag * samp_freq) + 1
    memory_size = num_chunck * num_segmts * npts_segmt * 4 / 1024 ** 3

    if memory_size > MAX_MEM:
        raise ValueError(
            'Require %5.3fG memory but only %5.3fG provided)! Cannot load cc data all once!' % (memory_size, MAX_MEM))
    if flag:
        print('Good on memory (need %5.2f G and %s G provided)!' % (memory_size, MAX_MEM))

    # allocate array to store fft data/info
    cc_array = np.zeros((num_chunck * num_segmts, npts_segmt), dtype=np.float32)
    cc_time = np.zeros(num_chunck * num_segmts, dtype=np.float)
    cc_ngood = np.zeros(num_chunck * num_segmts, dtype=np.int16)
    cc_comp = np.chararray(num_chunck * num_segmts, itemsize=2, unicode=True)

    # loop through all time-chuncks
    iseg = 0
    dtype = pairs_all[ipair]
    for ifile in ccfiles:

        # load the data from daily compilation
        ds = pyasdf.ASDFDataSet(ifile, mpi=False, mode='r')
        try:
            path_list = ds.auxiliary_data[dtype].list()
            tparameters = ds.auxiliary_data[dtype][path_list[0]].parameters
        except Exception:
            if flag: print('continue! no pair of %s in %s' % (dtype, ifile))
            continue

        # Check number of cross-components
        if ncomp == 3 and len(path_list) < 6:
            if flag:
                Logger.warning('continue! not enough components (counted %d) for auto-correlation for pair %s in %s' % (
                len(path_list), dtype, ifile))
            continue
        if len(path_list) > 6:
            # raise ValueError('more than 6 cross-component exists for %s %s! please double check'%(ifile,dtype))
            Logger.warning('more than 6 cross-component exists for %s %s! Removing redundant cross-components' % (ifile, dtype))
            premove = []
            for path in path_list:
                dum1, dum2 = path.split("_")
                if dum1[0] != dum2[0]:  # Skip mix of channel BHx with HHx
                    premove.append(path)
                elif dum1[0] == "B" or dum2[0] == "B":  # Favor HHx component over BHx
                    premove.append(path)
            path_list = [p for p in path_list if p not in premove]
            Logger.warning("New path_list is:", path_list)

        if flag: Logger.info(
            f"All checks passed for number of components. Now starting loading of 9-component data from {ifile}")

        # load the 6-component data, which is in order in the ASDF
        for tpath in path_list:
            cmp1 = tpath.split('_')[0]
            cmp2 = tpath.split('_')[1]
            tcmp1 = cmp1[-1];
            tcmp2 = cmp2[-1]

            # read data and parameter matrix
            tdata = ds.auxiliary_data[dtype][tpath].data[:]
            ttime = ds.auxiliary_data[dtype][tpath].parameters['time']
            tgood = ds.auxiliary_data[dtype][tpath].parameters['ngood']
            if substack:
                for ii in range(tdata.shape[0]):
                    cc_array[iseg] = tdata[ii]
                    cc_time[iseg] = ttime[ii]
                    cc_ngood[iseg] = tgood[ii]
                    cc_comp[iseg] = tcmp1 + tcmp2
                    iseg += 1
            else:
                cc_array[iseg] = tdata
                cc_time[iseg] = ttime
                cc_ngood[iseg] = tgood
                cc_comp[iseg] = tcmp1 + tcmp2
                iseg += 1

    t1 = time.time()
    if flag: Logger.info(f"loading CCF data from {ifile} takes {t1-t0:.2f} s")

    # continue when there is no data or for auto-correlation
    if iseg <= 1:
        if flag:
            Logger.warning(f"Stop! Not processing this pair because no data in file {ifile}.")
        continue

    # Output file
    outfn = pairs_all[ipair] + '.h5'
    stack_h5 = os.path.join(STACKDIR, idir + '/' + outfn)
    if flag: Logger.info(f"Stack output file: {stack_h5}")

    # loop through cross-component for stacking
    iflag = 1
    for icomp in range(nccomp):
        comp = enz_system[icomp]
        indx = np.where(cc_comp == comp)[0]

        # jump if there are not enough data
        if len(indx) < 2:
            Logger.warning(f"Not enough matching components for {comp} were loaded from {ifile}?")
            iflag = 0
            continue # or break??

        t2 = time.time()

        # output stacked data
        cc_final, ngood_final, stamps_final, allstacks1, allstacks2, allstacks3, allstacks4, allstacks5, nstacks = stacking.stacking(
            cc_array[indx], cc_time[indx], cc_ngood[indx], stack_para)  # GS
        if not len(allstacks1):
            Logger.warning(f"returned empty stack for {ssta}.{comp}")
            continue

        # write stacked data into ASDF file
        with pyasdf.ASDFDataSet(stack_h5, mpi=False) as ds:
            tparameters['time'] = stamps_final[0]
            tparameters['ngood'] = nstacks
            tparameters['comp'] = comp
            if stack_method != 'all':
                data_type = 'Allstack_' + stack_method
                ds.add_auxiliary_data(data=allstacks1,
                                      data_type=data_type,
                                      path=comp,
                                      parameters=tparameters)
            else:  # all methods: linear, pws, robust, nroot, selective, auto_covariance
                ds.add_auxiliary_data(data=allstacks1,
                                      data_type='Allstack_linear',
                                      path=comp,
                                      parameters=tparameters)
                ds.add_auxiliary_data(data=allstacks2,
                                      data_type='Allstack_pws',
                                      path=comp,
                                      parameters=tparameters)
                ds.add_auxiliary_data(data=allstacks3,
                                      data_type='Allstack_robust',
                                      path=comp,
                                      parameters=tparameters)
                ds.add_auxiliary_data(data=allstacks4,
                                      data_type='Allstack_auto_covariance',
                                      path=comp,
                                      parameters=tparameters)
                ds.add_auxiliary_data(data=allstacks5,
                                      data_type='Allstack_nroot',
                                      path=comp,
                                      parameters=tparameters)

        # keep a track of all sub-stacked data from S1
        if keep_substack:
            for ii in range(cc_final.shape[0]):
                with pyasdf.ASDFDataSet(stack_h5, mpi=False) as ds:
                    tparameters['time'] = stamps_final[ii]
                    tparameters['ngood'] = ngood_final[ii]
                    tparameters['comp'] = comp
                    data_type = 'T' + str(int(stamps_final[ii]))
                    ds.add_auxiliary_data(data=cc_final[ii],
                                          data_type=data_type,
                                          path=comp,
                                          parameters=tparameters)

        Logger.info(f"Wrote component {comp} to output file: {stack_h5}")

        t3 = time.time()
        if flag:
            Logger.info('takes %6.2fs to stack one component with %s stacking method' % (t3 - t1, stack_method))

    t4 = time.time()
    Logger.info('takes %6.2fs to stack all station pairs %s' % (t4 - t1, pairs_all[ipair]))

    # write file stamps 
    ftmp = open(toutfn, 'w')
    ftmp.write('done')
    ftmp.close()

tt1 = time.time()
Logger.info('it takes %6.2fs to process step 2 in total' % (tt1 - tt0))
comm.barrier()

# merge all path_array and output
if rank == 0:
    sys.exit()
