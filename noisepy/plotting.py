'''
Ensembles of plotting functions to display intermediate/final waveforms from the NoisePy package.
Originally by Chengxin Jiang @Harvard (May.04.2019)
Modified by Genevieve Savard @UniGe (2023)
'''

import os
import sys
import glob
import obspy
import scipy
import pyasdf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, DayLocator, MonthLocator
from scipy.fftpack import next_fast_len
from obspy.signal.filter import bandpass
from matplotlib.ticker import MultipleLocator
import pandas as pd

#############################################################################
###############PLOTTING FUNCTIONS FOR FILES FROM S0##########################
#############################################################################
def plot_availability(wdir, title, plot=False, savefig=False, figname=None):
    """
    Make a bar plot of data availability based on folder with H5 files for raw data created with S0B script
    Args:
        wdir: Path to the H5 files created with script S0B [str]
        title: Plot title [str]
        plot: Whether to make the plot or not [bool]
        savefig: Whether to save the figure or not [bool]
        figname: Path where to save figure [str]

    Returns:
        Pandas.DataFrame with data availability for each station

    """

    # Get files
    sfiles = glob.glob(os.path.join(wdir, "*.h5"))
    sfiles.sort()

    # Extract availability and Build dataframe
    dfa = []
    for _i, sf in enumerate(sfiles):
        #print(sf)
        if _i % 200 == 0: print(f"Reading files {_i+1}/{len(sfiles)}...")
        t1 = os.path.split(sf)[1].split("T")[0].strip("_")
        t2 = os.path.split(sf)[1].split("T")[1].strip("_").strip(".h5")
        st1 = obspy.UTCDateTime(t1)
        st2 = obspy.UTCDateTime(t2)

        with pyasdf.ASDFDataSet(sf, mode="r") as ds:
            stalst = ds.waveforms.list()
            for sta in stalst:
                num = len(ds.waveforms[sta].list())
                dum = pd.DataFrame(data={"Station": [sta],
                                    "Start": [st1._get_datetime()],
                                    "End": [st2._get_datetime()],
                                         "num_comps": [num]
                                   }, index=None)
                dfa.append(dum)

    dfa = pd.concat(dfa, axis=0)

    if plot:
        station, start, stop, num_comps = dfa["Station"], dfa["Start"], dfa["End"], dfa["num_comps"]

        # Unique stations
        stalst, unique_idx, stalst_inv = np.unique(station, 1, 1)
        #print(stalst)

        #Build y values from the number of unique stations
        y = (stalst_inv + 1) / float(len(stalst) + 1)

        # Create fig handles
        fig, ax = plt.subplots(1,1,figsize=(14,0.5*len(stalst)))

        # Plot availability
        for b, nc, start1, stop1 in zip(y, num_comps, start, stop):
            if nc < 4:
                ax.hlines(b, start1, stop1, color="r", lw=8)
            elif nc == 4:
                ax.hlines(b, start1, stop1, color="g", lw=8)
            else:
                ax.hlines(b, start1, stop1, color="b", lw=8)
        # X axis
        ax.xaxis_date()
        myFmt = DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_formatter(myFmt)

        delta = (stop.max() - start.min())/20
        num_xticks = 20
        minterval = (int(np.floor((stop.max() - start.min()).days/num_xticks)) // 7 ) * 7
        if int(np.floor((stop.max() - start.min()).days/num_xticks)) > 30:
            minterval = minterval // 30
            ax.xaxis.set_major_locator(MonthLocator(interval=minterval)) # used to be SecondLocator(0, interval=20)
        else:
            ax.xaxis.set_major_locator(DayLocator(interval=minterval)) # used to be SecondLocator(0, interval=20)
        if int(np.floor((stop.max() - start.min()).days/num_xticks)) > 30:
            sinterval = 1
            ax.xaxis.set_minor_locator(MonthLocator(interval=sinterval))
        else:
            sinterval = 7
            ax.xaxis.set_minor_locator(DayLocator(interval=sinterval))
        plt.xticks(rotation=90)
        ax.set_xlim(start.min()-delta, stop.max()+delta)
        # ax.set_xlabel('Time')
        ax.tick_params(which='major', length=7, width=2)

        # Y axis
        plt.yticks(y[unique_idx], stalst)
        ax.set_ylim(0,1)

        # Title
        ax.set_title(title)

        # Grid
        plt.grid(b=True, which='both', color='0.65', linestyle='-')
        # plt.grid(b=True, which='minor', color='0.65', linestyle='-')

        # Save or show
        if savefig and figname:
            plt.savefig(figname, format="PNG", dpi=300)
        else:
            plt.show()
        plt.close()

    return dfa


def plot_waveform(sfile, net, sta, comp, freqmin, freqmax, savefig=False, sdir=None):
    '''
    display the downloaded waveforms for given station "sta"

    PARAMETERS:
    -----------------------
    sfile: file containing all waveform data for a time-chunck in ASDF format
    net,sta: network, station name
    freqmin: min frequency to be filtered
    freqmax: max frequency to be filtered
    savefig: whether to save figure or not
    sdir: directory where to save figure.

    USAGE: 
    -----------------------
    plot_waveform('temp.h5','CI','BLC',0.01,0.5)
    '''
    # open pyasdf file to read
    try:
        ds = pyasdf.ASDFDataSet(sfile, mode='r')
        sta_list = ds.waveforms.list()
    except Exception:
        print("exit! cannot open %s to read" % sfile)
        sys.exit()

    # check whether station exists
    tsta = net + '.' + sta
    if tsta not in sta_list:
        raise ValueError('no data for %s in %s' % (tsta, sfile))

    tcomp = ds.waveforms[tsta].get_waveform_tags()
    ncomp = len(tcomp)
    # Display all components
    dt = ds.waveforms[tsta][tcomp[0]][0].stats.delta
    npts = ds.waveforms[tsta][tcomp[0]][0].stats.npts
    starttime = ds.waveforms[tsta][tcomp[0]][0].stats.starttime
    tt = np.arange(0, npts) * dt # time lag vector
    data = np.zeros(shape=(ncomp, npts), dtype=np.float32)
    for ii in range(ncomp):
        data[ii] = ds.waveforms[tsta][tcomp[ii]][0].data
        data[ii] = bandpass(data[ii], freqmin, freqmax, int(1 / dt), corners=4, zerophase=True)

    fig, axs = plt.subplots(figsize=(12, 3*ncomp), sharex=True)
    for iax, ax in enumerate(axs):
        ax.plot(tt, data[iax], 'k-', linewidth=1, label=tcomp[0].split('_')[0].upper())
        ax.set_title('T\u2080:%s   %s.%s   @%5.3f-%5.2f Hz' % (starttime, net, sta, freqmin, freqmax))
        plt.legend(loc='upper left')

    plt.xlabel('Time [s]')
    plt.tight_layout()

    if savefig:
        if not os.path.isdir(sdir): os.mkdir(sdir)
        outfname = sdir + '/{0:s}_{1:s}.{2:s}.pdf'.format(sfile.split('.')[0], net, sta)
        plt.savefig(outfname, format='png', dpi=300)
        plt.close()
    else:
        plt.show()



###############PLOTTING FUNCTIONS FOR FILES FROM S1##########################
#############################################################################
def plot_substack_cc(sfile, freqmin, freqmax, disp_lag=None, savefig=True, sdir='./', dtype=None):
    '''
    display the 2D matrix of the cross-correlation functions for a certain time-chunck. 

    PARAMETERS:
    --------------------------
    sfile: cross-correlation functions outputed by S1
    freqmin: min frequency to be filtered
    freqmax: max frequency to be filtered
    disp_lag: time ranges for display

    USAGE: 
    --------------------------
    plot_substack_cc('temp.h5',0.1,1,100,True,'./')

    Note: IMPORTANT!!!! this script only works for cross-correlation with sub-stacks being set to True in S1.
    '''
    # open data for read
    if savefig:
        if sdir == None: print('no path selected! save figures in the default path')

    try:
        ds = pyasdf.ASDFDataSet(sfile, mode='r')
        # extract common variables
        spairs = ds.auxiliary_data.list()
        path_lists = ds.auxiliary_data[spairs[0]].list()
        flag = ds.auxiliary_data[spairs[0]][path_lists[0]].parameters['substack']
        dt = ds.auxiliary_data[spairs[0]][path_lists[0]].parameters['dt']
        maxlag = ds.auxiliary_data[spairs[0]][path_lists[0]].parameters['maxlag']
    except Exception:
        print("exit! cannot open %s to read" % sfile);
        sys.exit()

    # only works for cross-correlation with substacks generated
    if not flag:
        raise ValueError('seems no substacks have been done! not suitable for this plotting function')

    # lags for display   
    if not disp_lag: disp_lag = maxlag
    if disp_lag > maxlag: raise ValueError('lag excceds maxlag!')

    # t is the time labels for plotting
    t = np.arange(-int(disp_lag), int(disp_lag) + dt, step=int(2 * int(disp_lag) / 4))
    # windowing the data
    indx1 = int((maxlag - disp_lag) / dt)
    indx2 = indx1 + 2 * int(disp_lag / dt) + 1

    for spair in spairs:
        ttr = spair.split('_')
        net1, sta1 = ttr[0].split('.')
        net2, sta2 = ttr[1].split('.')
        for ipath in path_lists:
            if dtype:
                if ipath != dtype:
                    continue
            chan1, chan2 = ipath.split('_')
            try:
                dist = ds.auxiliary_data[spair][ipath].parameters['dist']
                ngood = ds.auxiliary_data[spair][ipath].parameters['ngood']
                ttime = ds.auxiliary_data[spair][ipath].parameters['time']
                timestamp = np.empty(ttime.size, dtype='datetime64[s]')
            except Exception:
                print('continue! something wrong with %s %s' % (spair, ipath))
                continue

            # cc matrix
            data = ds.auxiliary_data[spair][ipath].data[:, indx1:indx2]
            nwin = data.shape[0]
            amax = np.zeros(nwin, dtype=np.float32)
            if nwin == 0 or len(ngood) == 1: print('continue! no enough substacks!');continue

            tmarks = []
            # load cc for each station-pair
            for ii in range(nwin):
                data[ii] = bandpass(data[ii], freqmin, freqmax, int(1 / dt), corners=4, zerophase=True)
                amax[ii] = max(data[ii])
                data[ii] /= amax[ii]
                timestamp[ii] = obspy.UTCDateTime(ttime[ii])
                tmarks.append(obspy.UTCDateTime(ttime[ii]).strftime('%H:%M:%S'))

            # plotting
            if nwin > 10:
                tick_inc = int(nwin / 5)
            else:
                tick_inc = 2
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(211)
            ax.matshow(data, cmap='seismic', extent=[-disp_lag, disp_lag, nwin, 0], aspect='auto')
            ax.set_title('%s.%s.%s  %s.%s.%s  dist:%5.2fkm' % (net1, sta1, chan1, net2, sta2, chan2, dist))
            ax.set_xlabel('time [s]')
            ax.set_xticks(t)
            ax.set_yticks(np.arange(0, nwin, step=tick_inc))
            ax.set_yticklabels(timestamp[0::tick_inc])  # GS changed 0:-1:tick_inc
            ax.xaxis.set_ticks_position('bottom')
            ax1 = fig.add_subplot(413)
            ax1.set_title('stacked and filtered at %4.2f-%4.2f Hz' % (freqmin, freqmax))
            ax1.plot(np.arange(-disp_lag, disp_lag + dt, dt), np.mean(data, axis=0), 'k-', linewidth=1)
            ax1.set_xticks(t)
            ax2 = fig.add_subplot(414)
            ax2.plot(amax / min(amax), 'r-')
            ax2.plot(ngood, 'b-')
            ax2.set_xlabel('waveform number')
            ax2.set_xticks(np.arange(0, nwin, step=tick_inc))
            ax2.set_xticklabels(tmarks[0:nwin:tick_inc])
            # for tick in ax[2].get_xticklabels():
            #    tick.set_rotation(30)
            ax2.legend(['relative amp', 'ngood'], loc='upper right')
            fig.tight_layout()

            # save figure or just show
            if savefig:
                if sdir == None: sdir = sfile.split('.')[0]
                if not os.path.isdir(sdir): os.mkdir(sdir)
                outfname = sdir + '/{0:s}.{1:s}.{2:s}_{3:s}.{4:s}.{5:s}.png'.format(net1, sta1, chan1, net2, sta2,
                                                                                    chan2)
                fig.savefig(outfname, format='png', dpi=300)
                plt.close()
            else:
                plt.show()  # GS


def plot_substack_cc_spect(sfile, freqmin, freqmax, disp_lag=None, savefig=True, sdir='./'):
    '''
    display the 2D matrix of the cross-correlation functions for a time-chunck. 

    PARAMETERS:
    -----------------------
    sfile: cross-correlation functions outputed by S1
    freqmin: min frequency to be filtered
    freqmax: max frequency to be filtered
    disp_lag: time ranges for display

    USAGE: 
    -----------------------
    plot_substack_cc('temp.h5',0.1,1,200,True,'./')

    Note: IMPORTANT!!!! this script only works for the cross-correlation with sub-stacks in S1.
    '''
    # open data for read
    if savefig:
        if sdir == None: print('no path selected! save figures in the default path')

    try:
        ds = pyasdf.ASDFDataSet(sfile, mode='r')
        # extract common variables
        spairs = ds.auxiliary_data.list()
        path_lists = ds.auxiliary_data[spairs[0]].list()
        flag = ds.auxiliary_data[spairs[0]][path_lists[0]].parameters['substack']
        dt = ds.auxiliary_data[spairs[0]][path_lists[0]].parameters['dt']
        maxlag = ds.auxiliary_data[spairs[0]][path_lists[0]].parameters['maxlag']
    except Exception:
        print("exit! cannot open %s to read" % sfile);
        sys.exit()

    # only works for cross-correlation with substacks generated
    if not flag:
        raise ValueError('seems no substacks have been done! not suitable for this plotting function')

    # lags for display   
    if not disp_lag: disp_lag = maxlag
    if disp_lag > maxlag: raise ValueError('lag excceds maxlag!')
    t = np.arange(-int(disp_lag), int(disp_lag) + dt, step=int(2 * int(disp_lag) / 4))
    indx1 = int((maxlag - disp_lag) / dt)
    indx2 = indx1 + 2 * int(disp_lag / dt) + 1
    nfft = int(next_fast_len(indx2 - indx1))
    freq = scipy.fftpack.fftfreq(nfft, d=dt)[:nfft // 2]

    for spair in spairs:
        ttr = spair.split('_')
        net1, sta1 = ttr[0].split('.')
        net2, sta2 = ttr[1].split('.')
        for ipath in path_lists:
            chan1, chan2 = ipath.split('_')
            try:
                dist = ds.auxiliary_data[spair][ipath].parameters['dist']
                ngood = ds.auxiliary_data[spair][ipath].parameters['ngood']
                ttime = ds.auxiliary_data[spair][ipath].parameters['time']
                timestamp = np.empty(ttime.size, dtype='datetime64[s]')
            except Exception:
                print('continue! something wrong with %s %s' % (spair, ipath))
                continue

            # cc matrix
            data = ds.auxiliary_data[spair][ipath].data[:, indx1:indx2]
            nwin = data.shape[0]
            amax = np.zeros(nwin, dtype=np.float32)
            spec = np.zeros(shape=(nwin, nfft // 2), dtype=np.complex64)
            if nwin == 0 or len(ngood) == 1: print('continue! no enough substacks!');continue

            # load cc for each station-pair
            for ii in range(nwin):
                spec[ii] = scipy.fftpack.fft(data[ii], nfft, axis=0)[:nfft // 2]
                spec[ii] /= np.max(np.abs(spec[ii]), axis=0)
                data[ii] = bandpass(data[ii], freqmin, freqmax, int(1 / dt), corners=4, zerophase=True)
                amax[ii] = max(data[ii])
                data[ii] /= amax[ii]
                timestamp[ii] = obspy.UTCDateTime(ttime[ii])

            # plotting
            if nwin > 10:
                tick_inc = int(nwin / 5)
            else:
                tick_inc = 2
            fig, ax = plt.subplots(3, sharex=False)
            ax[0].matshow(data, cmap='seismic', extent=[-disp_lag, disp_lag, nwin, 0], aspect='auto')
            ax[0].set_title('%s.%s.%s  %s.%s.%s  dist:%5.2f km' % (net1, sta1, chan1, net2, sta2, chan2, dist))
            ax[0].set_xlabel('time [s]')
            ax[0].set_xticks(t)
            ax[0].set_yticks(np.arange(0, nwin, step=tick_inc))
            ax[0].set_yticklabels(timestamp[0::tick_inc])  # GS changed 0:-1:tick_inc
            ax[0].xaxis.set_ticks_position('bottom')
            ax[1].matshow(np.abs(spec), cmap='seismic', extent=[freq[0], freq[-1], nwin, 0], aspect='auto')
            ax[1].set_xlabel('freq [Hz]')
            ax[1].set_ylabel('amplitudes')
            ax[1].set_yticks(np.arange(0, nwin, step=tick_inc))
            ax[1].xaxis.set_ticks_position('bottom')
            ax[2].plot(amax / min(amax), 'r-')
            ax[2].plot(ngood, 'b-')
            ax[2].set_xlabel('waveform number')
            # ax[1].set_xticks(np.arange(0,nwin,int(nwin/5)))
            ax[2].legend(['relative amp', 'ngood'], loc='upper right')
            fig.tight_layout()

            # save figure or just show
            if savefig:
                if sdir == None: sdir = sfile.split('.')[0]
                if not os.path.isdir(sdir): os.mkdir(sdir)
                outfname = sdir + '/{0:s}.{1:s}.{2:s}_{3:s}.{4:s}.{5:s}.pdf'.format(net1, sta1, chan1, net2, sta2,
                                                                                    chan2)
                fig.savefig(outfname, format='png', dpi=300)
                plt.close()
            else:
                plt.show()  # GS


def plot_substack_cc_alltime(spair, ipath, sfiles, freqmin, freqmax, disp_lag=None, savefig=True, sdir='./',
                             figname=None):
    '''
    Plot all the substacks for a given pair for all time chunks in sfiles.
    '''

    ttr = spair.split('_')
    net1, sta1 = ttr[0].split('.')
    net2, sta2 = ttr[1].split('.')
    chan1, chan2 = ipath.split('_')

    # extract common variables
    for sfile in sfiles:
        ds = pyasdf.ASDFDataSet(sfile, mode='r')
        if spair not in ds.auxiliary_data.list(): continue
        if ipath not in ds.auxiliary_data[spair].list(): continue
        dt = ds.auxiliary_data[spair][ipath].parameters['dt']
        maxlag = ds.auxiliary_data[spair][ipath].parameters['maxlag']
        dist = ds.auxiliary_data[spair][ipath].parameters['dist']
        break

    # lags for display   
    if not disp_lag:
        disp_lag = maxlag
    if disp_lag > maxlag:
        raise ValueError('lag excceds maxlag!')

    # t is the time labels for plotting
    t = np.arange(-int(disp_lag), int(disp_lag) + dt, step=int(2 * int(disp_lag) / 4))
    # windowing the data
    indx1 = int((maxlag - disp_lag) / dt)
    indx2 = indx1 + 2 * int(disp_lag / dt) + 1

    # Read all time chunk files
    sfiles.sort()
    ngood_all = []
    ttime_all = []
    data_all = []
    for sfile in sfiles:
        #     print(f"Reading {os.path.split(sfile)[1]}")
        ds = pyasdf.ASDFDataSet(sfile, mode='r')
        if not spair in ds.auxiliary_data.list():
            # print(f"no data found for {spair} in {os.path.split(sfile)[1]}. Skip.")
            continue
        if ipath not in ds.auxiliary_data[spair].list():
            # print(f"no data found for {ipath} in {os.path.split(sfile)[1]}. Skip.")
            continue
        ngood = ds.auxiliary_data[spair][ipath].parameters['ngood']
        ttime = ds.auxiliary_data[spair][ipath].parameters['time']
        data = ds.auxiliary_data[spair][ipath].data[:, indx1:indx2]  # cc matrix
        if data.shape[0] == 0 or len(ngood) == 1: continue

        # Append
        data_all.append(data)
        ttime_all.append(ttime)
        ngood_all.append(ngood)

    # Concatenate lists into 2D matrices 
    data_all = np.vstack(data_all)
    ttime_all = np.hstack(ttime_all)
    ngood_all = np.hstack(ngood_all)

    # load cc for each station-pair
    nwin = data_all.shape[0]
    timestamp = np.empty(ttime_all.size, dtype='datetime64[s]')
    amax = np.zeros(nwin, dtype=np.float32)
    tmarks = []
    for ii in range(nwin):
        data_all[ii] = bandpass(data_all[ii], freqmin, freqmax, int(1 / dt), corners=4, zerophase=True)
        #     data_all[ii] = bandpass(data_all[ii],freqmin,freqmax,int(1/dt),corners=4, zerophase=False)
        amax[ii] = max(data_all[ii])
        data_all[ii] /= amax[ii]
        timestamp[ii] = obspy.UTCDateTime(ttime_all[ii])
        tmarks.append(obspy.UTCDateTime(ttime_all[ii]).strftime('%Y-%m-%d'))

    # plotting
    if nwin > 10:
        tick_inc = int(nwin / 10)
    else:
        tick_inc = 2
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(211)
    ax.matshow(data_all, cmap='seismic', extent=[-disp_lag, disp_lag, nwin, 0], aspect='auto')
    ax.set_title('%s.%s.%s  %s.%s.%s  dist:%5.2fkm' % (net1, sta1, chan1, net2, sta2, chan2, dist))
    ax.set_xlabel('time [s]')
    ax.set_xticks(t)
    ax.set_yticks(np.arange(0, nwin, step=tick_inc))
    ax.set_yticklabels(timestamp[0::tick_inc])  # GS changed 0:-1:tick_inc
    ax.xaxis.set_ticks_position('bottom')
    ax1 = fig.add_subplot(413)
    ax1.set_title('stacked and filtered at %4.2f-%4.2f Hz' % (freqmin, freqmax))
    ax1.plot(np.arange(-disp_lag, disp_lag + dt, dt), np.mean(data_all, axis=0), 'k-', linewidth=1)
    ax1.set_xticks(t)
    ax2 = fig.add_subplot(414)
    ax2.plot(amax / min(amax), 'r-')
    ax2.plot(ngood_all, 'b-')
    ax2.set_xlabel('waveform number')
    ax2.set_xticks(np.arange(0, nwin, step=tick_inc))
    ax2.set_xticklabels(tmarks[0:nwin:tick_inc])
    for tick in ax2.get_xticklabels():
        tick.set_rotation(30)
    ax2.legend(['relative amp', 'ngood'], loc='upper right')
    fig.tight_layout()

    # save figure or just show
    if savefig:
        if sdir == None: sdir = sfile.split('.')[0]
        if not os.path.isdir(sdir): os.mkdir(sdir)
        if not figname:
            figname = '{0:s}.{1:s}.{2:s}_{3:s}.{4:s}.{5:s}.png'.format(net1, sta1, chan1, net2, sta2, chan2)
        outfname = os.path.join(sdir, figname)
        fig.savefig(outfname, format='png', dpi=300)
        plt.close()
    else:
        plt.show()  # GS


#############################################################################
###############PLOTTING FUNCTIONS FOR FILES FROM S2##########################
#############################################################################

def plot_substack_all(sfile, freqmin, freqmax, ccomp, disp_lag=None, savefig=False, sdir=None, figsize=(14, 14)):
    '''
    display the 2D matrix of the cross-correlation functions stacked for all time windows.

    PARAMETERS:
    ---------------------
    sfile: cross-correlation functions outputed by S2
    freqmin: min frequency to be filtered
    freqmax: max frequency to be filtered
    disp_lag: time ranges for display
    ccomp: cross component of the targeted cc functions

    USAGE: 
    ----------------------
    plot_substack_all('temp.h5',0.1,1,'ZZ',50,True,'./')
    '''
    # open data for read
    if savefig:
        if sdir == None: print('no path selected! save figures in the default path')

    paths = ccomp
    try:
        ds = pyasdf.ASDFDataSet(sfile, mode='r')
        # extract common variables
        dtype_lists = ds.auxiliary_data.list()
        dt = ds.auxiliary_data[dtype_lists[0]][paths].parameters['dt']
        dist = ds.auxiliary_data[dtype_lists[0]][paths].parameters['dist']
        maxlag = ds.auxiliary_data[dtype_lists[0]][paths].parameters['maxlag']
    except Exception:
        print("exit! cannot open %s to read" % sfile);
        sys.exit()

    if len(dtype_lists) == 1:
        raise ValueError('Abort! seems no substacks have been done')

    # lags for display   
    if not disp_lag: disp_lag = maxlag
    if disp_lag > maxlag: raise ValueError('lag excceds maxlag!')
    t = np.arange(-int(disp_lag), int(disp_lag) + dt, step=int(2 * int(disp_lag) / 4))
    indx1 = int((maxlag - disp_lag) / dt)
    indx2 = indx1 + 2 * int(disp_lag / dt) + 1

    # other parameters to keep
    num_stacks = len([itype for itype in dtype_lists if "stack" in itype])
    nwin = len(dtype_lists) - num_stacks
    data = np.zeros(shape=(nwin, indx2 - indx1), dtype=np.float32)
    ngood = np.zeros(nwin, dtype=np.int16)
    ttime = np.zeros(nwin, dtype=np.int)
    timestamp = np.empty(ttime.size, dtype='datetime64[s]')
    amax = np.zeros(nwin, dtype=np.float32)

    for ii, itype in enumerate(dtype_lists[num_stacks:]):
        if "Allstack" in itype: continue
        timestamp[ii] = obspy.UTCDateTime(np.float(itype[1:]))
        try:
            ngood[ii] = ds.auxiliary_data[itype][paths].parameters['ngood']
            ttime[ii] = ds.auxiliary_data[itype][paths].parameters['time']
            # timestamp[ii] = obspy.UTCDateTime(ttime[ii])
            # cc matrix
            data[ii] = ds.auxiliary_data[itype][paths].data[indx1:indx2]
            data[ii] = bandpass(data[ii], freqmin, freqmax, int(1 / dt), corners=4, zerophase=True)
            amax[ii] = np.max(data[ii])
            data[ii] /= amax[ii]
        except Exception as e:
            #             print(e)
            continue

        if len(ngood) == 1:
            raise ValueError('seems no substacks have been done! not suitable for this plotting function')

    # plotting
    if nwin > 100:
        tick_inc = int(nwin / 10)
    elif nwin > 10:
        tick_inc = int(nwin / 5)
    else:
        tick_inc = 2
    fig, ax = plt.subplots(2, sharex=False, figsize=figsize)
    ax[0].matshow(data, cmap='seismic', extent=[-disp_lag, disp_lag, nwin, 0], aspect='auto')
    ax[0].set_title('%s dist:%5.2f km filtered at %4.2f-%4.2fHz' % (sfile.split('/')[-1], dist, freqmin, freqmax))
    ax[0].set_xlabel('time [s]')
    ax[0].set_ylabel('wavefroms')
    ax[0].set_xticks(t)
    ax[0].set_yticks(np.arange(0, nwin, step=tick_inc))
    ax[0].set_yticklabels(timestamp[0:nwin:tick_inc])
    ax[0].xaxis.set_ticks_position('bottom')
    ax[0].axvline(0, color="k", ls=":", lw=2)
    ax[1].plot(amax / max(amax), 'r-')
    ax2 = ax[1].twinx()
    ax2.plot(ngood, 'b-')
    ax2.set_ylabel('ngood', color="b")
    ax[1].set_ylabel('relative amp', color="r")
    ax[1].set_xlabel('waveform number')
    ax[1].set_xticks(np.arange(0, nwin, nwin // 5))
    ax[1].legend(['relative amp', 'ngood'], loc='upper right')
    # save figure or just show
    if savefig:
        if sdir == None: sdir = sfile.split('.')[0]
        if not os.path.isdir(sdir): os.mkdir(sdir)
        outfname = sdir + '/{0:s}_{1:4.2f}_{2:4.2f}Hz.pdf'.format(sfile.split('/')[-1], freqmin, freqmax)
        fig.savefig(outfname, format='png', dpi=300)
        plt.close()
    else:
        plt.show()  # GS


def plot_substack_all_spect(sfile, freqmin, freqmax, ccomp, disp_lag=None, savefig=False, sdir=None, figsize=(14, 14)):
    '''
    display the 2D matrix of the cross-correlation functions stacked for all time windows.

    PARAMETERS:
    -----------------------
    sfile: cross-correlation functions outputed by S2
    freqmin: min frequency to be filtered
    freqmax: max frequency to be filtered
    disp_lag: time ranges for display
    ccomp: cross component of the targeted cc functions

    USAGE: 
    -----------------------
    plot_substack_all('temp.h5',0.1,1,'ZZ',50,True,'./')
    '''
    # open data for read
    if savefig:
        if sdir == None: print('no path selected! save figures in the default path')

    paths = ccomp
    try:
        ds = pyasdf.ASDFDataSet(sfile, mode='r')
        # extract common variables
        dtype_lists = ds.auxiliary_data.list()
        dt = ds.auxiliary_data[dtype_lists[0]][paths].parameters['dt']
        dist = ds.auxiliary_data[dtype_lists[0]][paths].parameters['dist']
        maxlag = ds.auxiliary_data[dtype_lists[0]][paths].parameters['maxlag']
    except Exception:
        print("exit! cannot open %s to read" % sfile);
        sys.exit()

    if len(dtype_lists) == 1:
        raise ValueError('Abort! seems no substacks have been done')

    # lags for display   
    if not disp_lag: disp_lag = maxlag
    if disp_lag > maxlag: raise ValueError('lag excceds maxlag!')
    t = np.arange(-int(disp_lag), int(disp_lag) + dt, step=int(2 * int(disp_lag) / 4))
    indx1 = int((maxlag - disp_lag) / dt)
    indx2 = indx1 + 2 * int(disp_lag / dt) + 1
    nfft = int(next_fast_len(indx2 - indx1))
    freq = scipy.fftpack.fftfreq(nfft, d=dt)[:nfft // 2]

    # other parameters to keep
    num_stacks = len([itype for itype in dtype_lists if "stack" in itype])
    nwin = len(dtype_lists) - num_stacks
    data = np.zeros(shape=(nwin, indx2 - indx1), dtype=np.float32)
    spec = np.zeros(shape=(nwin, nfft // 2), dtype=np.complex64)
    ngood = np.zeros(nwin, dtype=np.int16)
    ttime = np.zeros(nwin, dtype=np.int)
    timestamp = np.empty(ttime.size, dtype='datetime64[s]')
    amax = np.zeros(nwin, dtype=np.float32)

    for ii, itype in enumerate(dtype_lists[num_stacks:]):
        if "stack" in itype: continue
        timestamp[ii] = obspy.UTCDateTime(np.float(itype[1:]))
        try:
            ngood[ii] = ds.auxiliary_data[itype][paths].parameters['ngood']
            ttime[ii] = ds.auxiliary_data[itype][paths].parameters['time']
            # timestamp[ii] = obspy.UTCDateTime(ttime[ii])
            # cc matrix
            tdata = ds.auxiliary_data[itype][paths].data[indx1:indx2]
            spec[ii] = scipy.fftpack.fft(tdata, nfft, axis=0)[:nfft // 2]
            spec[ii] /= np.max(np.abs(spec[ii]))
            data[ii] = bandpass(tdata, freqmin, freqmax, int(1 / dt), corners=4, zerophase=True)
            amax[ii] = np.max(data[ii])
            data[ii] /= amax[ii]
        except Exception as e:
            print(e)
            continue

        if len(ngood) == 1:
            raise ValueError('seems no substacks have been done! not suitable for this plotting function')

    # plotting
    if nwin > 100:
        tick_inc = int(nwin / 10)
    elif nwin > 10:
        tick_inc = int(nwin / 5)
    else:
        tick_inc = 2
    fig, ax = plt.subplots(3, sharex=False, figsize=figsize)
    ax[0].matshow(data, cmap='seismic', extent=[-disp_lag, disp_lag, nwin, 0], aspect='auto')
    ax[0].set_title('%s dist:%5.2f km' % (sfile.split('/')[-1], dist))
    ax[0].set_xlabel('time [s]')
    ax[0].set_ylabel('wavefroms')
    ax[0].set_xticks(t)
    ax[0].set_yticks(np.arange(0, nwin, step=tick_inc))
    ax[0].set_yticklabels(timestamp[0:nwin:tick_inc])
    ax[0].xaxis.set_ticks_position('bottom')
    ax[1].matshow(np.abs(spec), cmap='seismic', extent=[freq[0], freq[-1], nwin, 0], aspect='auto')
    ax[1].set_xlabel('freq [Hz]')
    ax[1].set_ylabel('amplitudes')
    ax[1].set_yticks(np.arange(0, nwin, step=tick_inc))
    ax[1].set_yticklabels(timestamp[0:nwin:tick_inc])
    ax[1].xaxis.set_ticks_position('bottom')
    ax[2].plot(amax / max(amax), 'r-')
    ax2 = ax[2].twinx()
    ax2.plot(ngood, 'b-')
    ax2.set_ylabel('ngood', color="b")
    ax[2].set_xlabel('Timestamp')
    ax[2].set_ylabel('relative amp', color="r")
    tick_inc = 4
    ax[2].set_xticks(np.arange(0, nwin, step=tick_inc))
    ax[2].set_xticklabels(timestamp[0:nwin:tick_inc])
#     ax[2].xaxis.set_major_locator(MultipleLocator(12))
    ax[2].set_xticklabels(timestamp[0:nwin:tick_inc])
    ax[2].xaxis.set_minor_locator(MultipleLocator(1))
    ax[2].xaxis.grid(True, which='major')
    ax[2].tick_params(axis='x', rotation=90)
    ax[2].tick_params(axis='x', which='major', length=7)
    ax[2].tick_params(axis='x', which='minor', length=3)
#     ax[2].set_xticks(np.arange(0, nwin, nwin // 5))
    ax[2].legend(['relative amp', 'ngood'], loc='upper right')
    # save figure or just show
    if savefig:
        if sdir == None: sdir = sfile.split('.')[0]
        if not os.path.isdir(sdir): os.mkdir(sdir)
#         outfname = sdir + '/{0:s}.png'.format(sfile.split('/')[-1])
        outfname = sdir + '/{0:s}_{1:s}_{2:4.2f}_{3:4.2f}Hz_spectra.png'.format(sfile.split('/')[-1], ccomp, freqmin, freqmax)
        fig.savefig(outfname, format='png') #, dpi=400)
        plt.close()
    else:
        plt.show()  # GS


def plot_all_moveout(sfiles, dtype, freqmin, freqmax, ccomp, dist_inc, disp_lag=None, savefig=False, sdir=None, figsize=(14,14)):
    '''
    display the moveout (2D matrix) of the cross-correlation functions stacked for all time chuncks.

    PARAMETERS:
    ---------------------
    sfile: cross-correlation functions outputed by S2
    dtype: datatype either 'Allstack0pws' or 'Allstack0linear'
    freqmin: min frequency to be filtered
    freqmax: max frequency to be filtered
    ccomp:   cross component
    dist_inc: distance bins to stack over
    disp_lag: lag times for displaying
    savefig: set True to save the figures (in pdf format)
    sdir: diresied directory to save the figure (if not provided, save to default dir)

    USAGE: 
    ----------------------
    plot_substack_moveout('temp.h5','Allstack_pws',0.1,0.2,1,'ZZ',200,True,'./temp')
    '''
    # open data for read
    if savefig:
        if sdir == None: print('no path selected! save figures in the default path')

    path = ccomp

    # extract common variables
    try:
        ds = pyasdf.ASDFDataSet(sfiles[0], mode='r')
        dt = ds.auxiliary_data[dtype][path].parameters['dt']
        maxlag = ds.auxiliary_data[dtype][path].parameters['maxlag']
        stack_method = dtype.split('0')[-1]
    except Exception:
        print("exit! cannot open %s to read" % sfiles[0]);
        sys.exit()

    # lags for display   
    if not disp_lag: disp_lag = maxlag
    if disp_lag > maxlag: raise ValueError('lag excceds maxlag!')
    t = np.arange(-int(disp_lag), int(disp_lag) + dt, step=(int(2 * int(disp_lag) / 4)))
    indx1 = int((maxlag - disp_lag) / dt)
    indx2 = indx1 + 2 * int(disp_lag / dt) + 1

    # cc matrix
    nwin = len(sfiles)
    data = np.zeros(shape=(nwin, indx2 - indx1), dtype=np.float32)
    dist = np.zeros(nwin, dtype=np.float32)
    ngood = np.zeros(nwin, dtype=np.int16)

    # load cc and parameter matrix
    for ii in range(len(sfiles)):
        sfile = sfiles[ii]

        ds = pyasdf.ASDFDataSet(sfile, mode='r')
        try:
            # load data to variables
            dist[ii] = ds.auxiliary_data[dtype][path].parameters['dist']
            ngood[ii] = ds.auxiliary_data[dtype][path].parameters['ngood']
            tdata = ds.auxiliary_data[dtype][path].data[indx1:indx2]
        except Exception:
            #  print("continue! cannot read %s "%sfile)
            continue

        data[ii] = bandpass(tdata, freqmin, freqmax, int(1 / dt), corners=4, zerophase=True)

    # average cc
    ntrace = int(np.round(np.max(dist) + 0.51) / dist_inc)
    ndata = np.zeros(shape=(ntrace, indx2 - indx1), dtype=np.float32)
    ndist = np.zeros(ntrace, dtype=np.float32)
    for td in range(0, ntrace - 1):
        tindx = np.where((dist >= td * dist_inc) & (dist < (td + 1) * dist_inc))[0]
        if len(tindx):
            ndata[td] = np.mean(data[tindx], axis=0)
            ndist[td] = (td + 0.5) * dist_inc

    # normalize waveforms 
    indx = np.where(ndist > 0)[0]
    ndata = ndata[indx]
    ndist = ndist[indx]
    for ii in range(ndata.shape[0]):
        #         print(ii,np.max(np.abs(ndata[ii])))
        ndata[ii] /= np.max(np.abs(ndata[ii]))

    # plotting figures
    fig, ax = plt.subplots(figsize=figsize)
    ax.matshow(ndata, cmap='seismic', extent=[-disp_lag, disp_lag, ndist[-1], ndist[0]], aspect='auto')
    lw = 1.5
    l1, = ax.plot([ndist[-1] / 0.5, 0], [ndist[-1], 0], "y-.", lw=lw)
    l2, = ax.plot([ndist[-1] / 1.0, 0], [ndist[-1], 0], c="orange", ls="-.", lw=lw)
    l3, = ax.plot([ndist[-1] / 1.5, 0], [ndist[-1], 0], "m-.", lw=lw)
    l4, = ax.plot([ndist[-1] / 2.0, 0], [ndist[-1], 0], "g-.", lw=lw)
    l5, = ax.plot([ndist[-1] / 3.0, 0], [ndist[-1], 0], "k-.", lw=lw)
    ax.plot([-ndist[-1] / 0.5, 0], [ndist[-1], 0], "y-.", lw=lw)
    ax.plot([-ndist[-1] / 1.0, 0], [ndist[-1], 0], c="orange", ls="-.", lw=lw)
    ax.plot([-ndist[-1] / 1.5, 0], [ndist[-1], 0], "m-.", lw=lw)
    ax.plot([-ndist[-1] / 2.0, 0], [ndist[-1], 0], "g-.", lw=lw)
    ax.plot([-ndist[-1] / 3.0, 0], [ndist[-1], 0], "k-.", lw=lw)
    ax.legend([l1, l2, l3, l4, l5], ["0.5 km/s", "1.0 km/s", "1.5 km/s", "2.0 km/s", "3.0 km/s"], loc='upper right')
    ax.set_title('%s %s @%5.3f-%5.2f Hz' % (ccomp, stack_method, freqmin, freqmax))
    ax.set_xlabel('time [s]')
    ax.set_ylabel('distance [km]')
    ax.set_xticks(t)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_ylim([ndist[-1], ndist[0]])
    ax.set_xlim([-disp_lag, disp_lag])
    # ax.text(np.ones(len(ndist))*(disp_lag-5),dist[ndist],ngood[ndist],fontsize=8)

    # save figure or show
    if savefig:
        outfname = sdir + '/moveout_allstack_' + str(stack_method) + '_' + str(ccomp) + '_bp' + str(
            freqmin) + '-' + str(freqmax) + '_' + str(dist_inc) + 'kmbin.pdf'
        fig.savefig(outfname, format='png', dpi=300)
        plt.close()
    else:
        plt.show()  # GS


#     return ndata, disp_lag, ndist, stack_method, freqmin, freqmax, t, dist, ngood


def plot_all_moveout_1D_1comp(sfiles, sta, dtype, freqmin, freqmax, ccomp, disp_lag=None, savefig=False, sdir=None,
                              figsize=(14, 11)):
    '''
    display the moveout waveforms of the cross-correlation functions stacked for all time chuncks.

    PARAMETERS:
    ---------------------
    sfile: cross-correlation functions outputed by S2
    sta: source station name
    dtype: datatype either 'Allstack0pws' or 'Allstack0linear'
    freqmin: min frequency to be filtered
    freqmax: max frequency to be filtered
    ccomp:   cross component
    disp_lag: lag times for displaying
    savefig: set True to save the figures (in pdf format)
    sdir: diresied directory to save the figure (if not provided, save to default dir)

    USAGE: 
    ----------------------
    plot_substack_moveout('temp.h5','Allstack0pws',0.1,0.2,'ZZ',200,True,'./temp')
    '''
    # open data for read
    if savefig:
        if sdir == None: print('no path selected! save figures in the default path')

    receiver = sta + '.h5'
    stack_method = dtype.split('_')[-1]

    # extract common variables
    try:
        ds = pyasdf.ASDFDataSet(sfiles[0], mode='r')
        dt = ds.auxiliary_data[dtype][ccomp].parameters['dt']
        maxlag = ds.auxiliary_data[dtype][ccomp].parameters['maxlag']
    except Exception:
        print("exit! cannot open %s to read" % sfiles[0]);
        sys.exit()

    # lags for display   
    if not disp_lag: disp_lag = maxlag
    if disp_lag > maxlag: raise ValueError('lag excceds maxlag!')
    tt = np.arange(-int(disp_lag), int(disp_lag) + dt, dt)
    indx1 = int((maxlag - disp_lag) / dt)
    indx2 = indx1 + 2 * int(disp_lag / dt) + 1

    # load cc and parameter matrix
    mdist = 0
    if not figsize:
        plt.figure()
    else:
        plt.figure(figsize=figsize)
    for ii in range(len(sfiles)):
        sfile = sfiles[ii]
        iflip = 0
        treceiver = sfile.split('_')[-1]
        if treceiver == receiver:
            iflip = 1

        ds = pyasdf.ASDFDataSet(sfile, mode='r')
        try:
            # load data to variables
            dist = ds.auxiliary_data[dtype][ccomp].parameters['dist']
            ngood = ds.auxiliary_data[dtype][ccomp].parameters['ngood']
            tdata = ds.auxiliary_data[dtype][ccomp].data[indx1:indx2]

        except Exception:
            print("continue! cannot read %s " % sfile);
            continue

        tdata = bandpass(tdata, freqmin, freqmax, int(1 / dt), corners=4, zerophase=True)
        tdata /= np.max(tdata, axis=0)

        if iflip:
            plt.plot(tt, np.flip(tdata, axis=0) + dist, 'k', linewidth=0.8)
        else:
            plt.plot(tt, tdata + dist, 'k', linewidth=0.8)
        plt.title('%s %s filtered @%4.1f-%4.1f Hz' % (sta, ccomp, freqmin, freqmax))
        plt.xlabel('time (s)')
        plt.ylabel('offset (km)')
        plt.text(disp_lag * 0.9, dist + 0.5, receiver, fontsize=6)

        # ----use to plot o times------
        if mdist < dist:
            mdist = dist
    plt.plot([0, 0], [0, mdist], 'r--', linewidth=1)

    # save figure or show
    if savefig:
        outfname = sdir + '/moveout_' + sta + '_1D_' + str(stack_method) + '.pdf'
        plt.savefig(outfname, format='png', dpi=300)
        plt.close()
    else:
        plt.show()  # GS


def plot_all_moveout_1D_9comp(sfiles, sta, dtype, freqmin, freqmax, disp_lag=None, savefig=False, sdir=None, mdist=25,
                              figsize=(14, 11)):
    '''
    display the moveout waveforms of the cross-correlation functions stacked for all time chuncks.

    PARAMETERS:
    ---------------------
    sfile: cross-correlation functions outputed by S2
    sta: source station name
    dtype: datatype either 'Allstack0pws' or 'Allstack0linear'
    freqmin: min frequency to be filtered
    freqmax: max frequency to be filtered
    disp_lag: lag times for displaying
    savefig: set True to save the figures (in pdf format)
    sdir: diresied directory to save the figure (if not provided, save to default dir)

    USAGE: 
    ----------------------
    plot_substack_moveout('temp.h5','Allstack0pws',0.1,0.2,'ZZ',200,True,'./temp')
    '''
    # open data for read
    if savefig:
        if sdir == None: print('no path selected! save figures in the default path')

    receiver = sta + '.h5'
    stack_method = dtype.split('_')[-1]
    ccomp = ['RR', 'RZ', 'RT', 'ZR', 'ZZ', 'ZT', 'TR', 'TZ', 'TT']
    # ccomp = ['ZR','ZT','ZZ','RR','RT','RZ','TR','TT','TZ']

    # extract common variables
    try:
        ds = pyasdf.ASDFDataSet(sfiles[0], mode='r')
        dt = ds.auxiliary_data[dtype][ccomp[0]].parameters['dt']
        maxlag = ds.auxiliary_data[dtype][ccomp[0]].parameters['maxlag']
    except Exception:
        print("exit! cannot open %s to read" % sfiles[0]);
        sys.exit()

    # lags for display   
    if not disp_lag: disp_lag = maxlag
    if disp_lag > maxlag: raise ValueError('lag excceds maxlag!')
    tt = np.arange(-int(disp_lag), int(disp_lag) + dt, dt)
    indx1 = int((maxlag - disp_lag) / dt)
    indx2 = indx1 + 2 * int(disp_lag / dt) + 1

    # load cc and parameter matrix
    if not figsize:
        plt.figure()
    else:
        plt.figure(figsize=figsize)
    for ic in range(len(ccomp)):
        comp = ccomp[ic]
        tmp = '33' + str(ic + 1)
        plt.subplot(tmp)

        for ii in range(len(sfiles)):
            sfile = sfiles[ii]
            iflip = 0
            treceiver = sfile.split('_')[-1]
            if treceiver == receiver:
                iflip = 1

            ds = pyasdf.ASDFDataSet(sfile, mode='r')
            try:
                # load data to variables
                dist = ds.auxiliary_data[dtype][comp].parameters['dist']
                ngood = ds.auxiliary_data[dtype][comp].parameters['ngood']
                tdata = ds.auxiliary_data[dtype][comp].data[indx1:indx2]

            except Exception:
                print("continue! cannot read %s " % sfile);
                continue

            if dist > mdist: continue
            tdata = bandpass(tdata, freqmin, freqmax, int(1 / dt), corners=4, zerophase=True)
            tdata /= np.max(tdata, axis=0)

            if iflip:
                plt.plot(tt, np.flip(tdata, axis=0) + dist, 'k', linewidth=0.8)
            else:
                plt.plot(tt, tdata + dist, 'k', linewidth=0.8)
            if ic == 1:
                plt.title('%s filtered @%4.1f-%4.1f Hz' % (sta, freqmin, freqmax))
            plt.xlabel('time (s)')
            plt.ylabel('offset (km)')
            if ic == 0:
                plt.plot([0, 2 * mdist], [0, mdist], 'r--', linewidth=0.2)
                plt.plot([0, mdist], [0, mdist], 'g--', linewidth=0.2)
            plt.text(disp_lag * 1.1, dist + 0.5, treceiver, fontsize=6)

        plt.plot([0, 0], [0, mdist], 'b--', linewidth=1)
        font = {'family': 'serif', 'color': 'red', 'weight': 'bold', 'size': 16}
        plt.text(disp_lag * 0.65, 0.9 * mdist, comp, fontdict=font)
    plt.tight_layout()

    # save figure or show
    if savefig:
        outfname = sdir + '/moveout_' + sta + '_1D_' + str(stack_method) + '.pdf'
        plt.savefig(outfname, format='png', dpi=300)
        plt.close()
    else:
        plt.show()  # GS
