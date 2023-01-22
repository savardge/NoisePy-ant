import os
import glob
import copy
import obspy
import scipy
import time
import pycwt
import pyasdf
import datetime
import numpy as np
import pandas as pd
from numba import jit
from scipy.signal import hilbert
from obspy.signal.util import _npts2nfft
from obspy.signal.invsim import cosine_taper
from scipy.fftpack import fft, ifft, next_fast_len
from obspy.signal.filter import bandpass, lowpass
from obspy.signal.regression import linear_regression
from obspy.core.util.base import _get_function_from_entry_point
from obspy.core.inventory import Inventory, Network, Station, Channel, Site

import logging
Logger = logging.getLogger(__name__)


def get_event_list(str1, str2, inc_hours):
    '''
    this function calculates the event list between time1 and time2 by increment of inc_hours
    in the formate of %Y_%m_%d_%H_%M_%S' (used in S0A & S0B)
    PARAMETERS:
    ----------------
    str1: string of the starting time -> 2010_01_01_0_0
    str2: string of the ending time -> 2010_10_11_0_0
    inc_hours: integer of incremental hours
    RETURNS:
    ----------------
    event: a numpy character list
    '''
    date1 = str1.split('_')
    date2 = str2.split('_')
    y1 = int(date1[0])
    m1 = int(date1[1])
    d1 = int(date1[2])
    h1 = int(date1[3])
    mm1 = int(date1[4])
    mn1 = int(date1[5])
    y2 = int(date2[0])
    m2 = int(date2[1])
    d2 = int(date2[2])
    h2 = int(date2[3])
    mm2 = int(date2[4])
    mn2 = int(date2[5])

    d1 = datetime.datetime(y1, m1, d1, h1, mm1, mn1)
    d2 = datetime.datetime(y2, m2, d2, h2, mm2, mn2)
    dt = datetime.timedelta(hours=inc_hours)

    event = []
    while (d1 < d2):
        event.append(d1.strftime('%Y_%m_%d_%H_%M_%S'))
        d1 += dt
    event.append(d2.strftime('%Y_%m_%d_%H_%M_%S'))

    return event


def make_timestamps(prepro_para):
    '''
    this function prepares the timestamps of both the starting and ending time of each mseed/sac file that
    is stored on local machine. this time info is used to search all stations in specific time chunck
    when preparing noise data in ASDF format. it creates a csv file containing all timestamp info if the
    file does not exist (used in S0B)f
    PARAMETERS:
    -----------------------
    prepro_para: a dic containing all pre-processing parameters used in S0B
    RETURNS:
    -----------------------
    all_stimes: numpy float array containing startting and ending time for all SAC/mseed files
    '''
    # load parameters from para dic
    wiki_file = prepro_para['wiki_file']
    messydata = prepro_para['messydata']
    RAWDATA = prepro_para['RAWDATA']
    allfiles_path = prepro_para['allfiles_path']

    if os.path.isfile(wiki_file):
        tmp = pd.read_csv(wiki_file)
        allfiles = tmp['names']
        all_stimes = np.zeros(shape=(len(allfiles), 2), dtype=np.float)
        all_stimes[:, 0] = tmp['starttime']
        all_stimes[:, 1] = tmp['endtime']

    # have to read each sac/mseed data one by one
    else:
        allfiles = glob.glob(allfiles_path)
        nfiles = len(allfiles)
        if not nfiles: raise ValueError('Abort! no data found in subdirectory of %s' % RAWDATA)
        all_stimes = np.zeros(shape=(nfiles, 2), dtype=np.float)

        if messydata:
            # get VERY precise trace-time from the header
            for ii in range(nfiles):
                try:
                    tr = obspy.read(allfiles[ii])
                    all_stimes[ii, 0] = tr[0].stats.starttime - obspy.UTCDateTime(1970, 1, 1)
                    all_stimes[ii, 1] = tr[0].stats.endtime - obspy.UTCDateTime(1970, 1, 1)
                except Exception as e:
                    #print(e);
                    Logger.warning(e)
                    continue
        else:
            # get rough estimates of the time based on the folder: need modified to accommodate your data
            for ii in range(nfiles):
                year = int(allfiles[ii].split('/')[-2].split('_')[1])
                # julia = int(allfiles[ii].split('/')[-2].split('_')[2])
                # all_stimes[ii,0] = obspy.UTCDateTime(year=year,julday=julia)-obspy.UTCDateTime(year=1970,month=1,day=1)
                month = int(allfiles[ii].split('/')[-2].split('_')[2])
                day = int(allfiles[ii].split('/')[-2].split('_')[3])
                all_stimes[ii, 0] = obspy.UTCDateTime(year=year, month=month, day=day) - obspy.UTCDateTime(year=1970,
                                                                                                           month=1,
                                                                                                           day=1)
                all_stimes[ii, 1] = all_stimes[ii, 0] + 86400

        # save name and time info for later use if the file not exist
        if not os.path.isfile(wiki_file):
            wiki_info = {'names': allfiles, 'starttime': all_stimes[:, 0], 'endtime': all_stimes[:, 1]}
            df = pd.DataFrame(wiki_info, columns=['names', 'starttime', 'endtime'])
            df.to_csv(wiki_file)
    return all_stimes, allfiles


def preprocess_raw(st, inv, prepro_para, date_info):
    '''
    this function pre-processes the raw data stream by:
        1) check samping rate and gaps in the data;
        2) remove sigularity, trend and mean of each trace
        3) filter and correct the time if integer time are between sampling points
        4) remove instrument responses with selected methods including:
            "inv"   -> using inventory information to remove_response;
            "spectrum"   -> use the inverse of response spectrum. (a script is provided in additional_module to estimate response spectrum from RESP files)
            "RESP_files" -> use the raw download RESP files
            "polezeros"  -> use pole/zero info for a crude correction of response
        5) trim data to a day-long sequence and interpolate it to ensure starting at 00:00:00.000
    (used in S0A & S0B)
    PARAMETERS:
    -----------------------
    st:  obspy stream object, containing noise data to be processed
    inv: obspy inventory object, containing stations info
    prepro_para: dict containing fft parameters, such as frequency bands and selection for instrument response removal etc.
    date_info:   dict of start and end time of the stream data
    RETURNS:
    -----------------------
    ntr: obspy stream object of cleaned, merged and filtered noise data
    '''
    # load paramters from fft dict
    rm_resp = prepro_para['rm_resp']
    if 'rm_resp_out' in prepro_para.keys():
        rm_resp_out = prepro_para['rm_resp_out']
    else:
        rm_resp_out = 'VEL'
    respdir = prepro_para['respdir']
    freqmin = prepro_para['freqmin']
    freqmax = prepro_para['freqmax']
    samp_freq = prepro_para['samp_freq']

    # parameters for butterworth filter
    f1 = 0.9 * freqmin
    f2 = freqmin
    if 1.1 * freqmax > 0.45 * samp_freq:
        f3 = 0.4 * samp_freq
        f4 = 0.45 * samp_freq
    else:
        f3 = freqmax
        f4 = 1.1 * freqmax
    pre_filt = [f1, f2, f3, f4]

    # check sampling rate and trace length
    st1 = st.copy()
    st = check_sample_gaps(st, date_info)
    if len(st) == 0:
        msg = f"No traces in Stream {st1[0].id}-{st1[0].stats.starttime} after check_sample_gaps: Continue!"
        Logger.warning(msg)
        return st
    sps = int(st[0].stats.sampling_rate)
    station = st[0].stats.station

    # remove nan/inf, mean and trend of each trace before merging
    for ii in range(len(st)):

        # -----set nan/inf values to zeros (it does happens!)-----
        tttindx = np.where(np.isnan(st[ii].data))
        if len(tttindx) > 0: st[ii].data[tttindx] = 0
        tttindx = np.where(np.isinf(st[ii].data))
        if len(tttindx) > 0: st[ii].data[tttindx] = 0

        st[ii].data = np.float32(st[ii].data)
        st[ii].data = scipy.signal.detrend(st[ii].data, type='constant')
        st[ii].data = scipy.signal.detrend(st[ii].data, type='linear')

    # merge, taper and filter the data
    if len(st) > 1: st.merge(method=1, fill_value=0)
    st[0].taper(max_percentage=0.05, max_length=50)  # taper window
    st[0].data = np.float32(bandpass(st[0].data, pre_filt[0], pre_filt[-1], df=sps, corners=4, zerophase=True))

    # make downsampling if needed
    if abs(samp_freq - sps) > 1E-4:
        # downsampling here
        st.interpolate(samp_freq, method='weighted_average_slopes')
        delta = st[0].stats.delta

        # when starttimes are between sampling points
        fric = st[0].stats.starttime.microsecond % (delta * 1E6)
        if fric > 1E-4:
            st[0].data = segment_interpolate(np.float32(st[0].data), float(fric / (delta * 1E6)))
            # --reset the time to remove the discrepancy---
            st[0].stats.starttime -= (fric * 1E-6)

    # remove traces of too small length

    # options to remove instrument response
    if rm_resp != 'no':
        if rm_resp != 'inv':
            if (respdir is None) or (not os.path.isdir(respdir)):
                raise ValueError('response file folder not found! abort!')

        if rm_resp == 'inv':
            #----check whether inventory is attached----
            if not inv[0][0][0].response:
                raise ValueError('no response found in the inventory! abort!')
            #elif inv[0][0][0].response == obspy.core.inventory.response.Response():
            #    raise ValueError('The response found in the inventory is empty (no stages)! abort!')
            else:
                try:
                    Logger.info('removing response for %s using inv'%st[0])
                    st[0].attach_response(inv)
                    st[0].remove_response(output=rm_resp_out,pre_filt=pre_filt,water_level=60)
                except Exception:
                    Logger.warning('Failed to remove response from %s. Returning empty stream.' % st[0])
                    st = []
                    return st

        elif rm_resp == 'spectrum':
            Logger.info('remove response using spectrum')
            specfile = glob.glob(os.path.join(respdir, '*' + station + '*'))
            if len(specfile) == 0:
                raise ValueError('no response sepctrum found for %s' % station)
            st = resp_spectrum(st, specfile[0], samp_freq, pre_filt)

        elif rm_resp == 'RESP':
            Logger.info('remove response using RESP files')
            resp = glob.glob(os.path.join(respdir, 'RESP.' + station + '*'))
            if len(resp) == 0:
                raise ValueError('no RESP files found for %s' % station)
            seedresp = {'filename': resp[0], 'date': date_info['starttime'], 'units': 'DIS'}
            st.simulate(paz_remove=None, pre_filt=pre_filt, seedresp=seedresp[0])

        elif rm_resp == 'polozeros':
            Logger.info('remove response using polos and zeros')
            paz_sts = glob.glob(os.path.join(respdir, '*' + station + '*'))
            if len(paz_sts) == 0:
                raise ValueError('no polozeros found for %s' % station)
            st.simulate(paz_remove=paz_sts[0], pre_filt=pre_filt)

        else:
            raise ValueError('no such option for rm_resp! please double check!')

    ntr = obspy.Stream()
    # trim a continous segment into user-defined sequences
    st[0].trim(starttime=date_info['starttime'], endtime=date_info['endtime'], pad=True, fill_value=0)
    ntr.append(st[0])

    return ntr


def stats2inv(stats, prepro_para, locs=None):
    '''
    this function creates inventory given the stats parameters in an obspy stream or a station list.
    (used in S0B)
    PARAMETERS:
    ------------------------
    stats: obspy trace stats object containing all station header info
    prepro_para: dict containing fft parameters, such as frequency bands and selection for instrument response removal etc.
    locs:  panda data frame of the station list. it is needed for convering miniseed files into ASDF
    RETURNS:
    ------------------------
    inv: obspy inventory object of all station info to be used later
    '''
    staxml = prepro_para['stationxml']
    respdir = prepro_para['respdir']
    input_fmt = prepro_para['input_fmt']

    if staxml:
        if not respdir:
            raise ValueError('Abort! staxml is selected but no directory is given to access the files')
        else:
            invfilelist = glob.glob(os.path.join(respdir, '*' + stats.station + '*'))
            if len(invfilelist) > 0:
                invfile = invfilelist[0]
                if len(invfilelist) > 1:
                    Logger.warning(
                        'More than one StationXML file was found for station %s. Keeping the first file in list.' % stats.station)
                if os.path.isfile(str(invfile)):
                    inv = obspy.read_inventory(invfile)
                    return inv
            else:
                raise ValueError('Could not find a StationXML file for station: %s.' % stats.station)

    inv = Inventory(networks=[], source="homegrown")

    if input_fmt == 'sac':
        net = Network(
            # This is the network code according to the SEED standard.
            code=stats.network,
            stations=[],
            description="created from SAC and resp files",
            start_date=stats.starttime)

        sta = Station(
            # This is the station code according to the SEED standard.
            code=stats.station,
            latitude=stats.sac["stla"],
            longitude=stats.sac["stlo"],
            elevation=stats.sac["stel"],
            creation_date=stats.starttime,
            site=Site(name="First station"))

        cha = Channel(
            # This is the channel code according to the SEED standard.
            code=stats.channel,
            # This is the location code according to the SEED standard.
            location_code=stats.location,
            # Note that these coordinates can differ from the station coordinates.
            latitude=stats.sac["stla"],
            longitude=stats.sac["stlo"],
            elevation=stats.sac["stel"],
            depth=-stats.sac["stel"],
            azimuth=stats.sac["cmpaz"],
            dip=stats.sac["cmpinc"],
            sample_rate=stats.sampling_rate)

    elif input_fmt == 'mseed':
        ista = locs[locs['station'] == stats.station].index.values.astype('int64')[0]

        net = Network(
            # This is the network code according to the SEED standard.
            code=locs.iloc[ista]["network"],
            stations=[],
            description="created from SAC and resp files",
            start_date=stats.starttime)

        sta = Station(
            # This is the station code according to the SEED standard.
            code=locs.iloc[ista]["station"],
            latitude=locs.iloc[ista]["latitude"],
            longitude=locs.iloc[ista]["longitude"],
            elevation=locs.iloc[ista]["elevation"],
            creation_date=stats.starttime,
            site=Site(name="First station"))

        cha = Channel(
            code=stats.channel,
            location_code=stats.location,
            latitude=locs.iloc[ista]["latitude"],
            longitude=locs.iloc[ista]["longitude"],
            elevation=locs.iloc[ista]["elevation"],
            depth=-locs.iloc[ista]["elevation"],
            azimuth=0,
            dip=0,
            sample_rate=stats.sampling_rate)

    response = obspy.core.inventory.response.Response()

    # Now tie it all together.
    cha.response = response
    sta.channels.append(cha)
    net.stations.append(sta)
    inv.networks.append(net)

    return inv


# Utility functions
def check_sample_gaps(stream, date_info):
    """
    this function checks sampling rate and find gaps of all traces in stream.
    PARAMETERS:
    -----------------
    stream: obspy stream object.
    date_info: dict of starting and ending time of the stream

    RETURENS:
    -----------------
    stream: List of good traces in the stream
    """
    # remove empty/big traces
    if len(stream) == 0 or len(stream) > 100:
        stream = []
        return stream

    # remove traces with big gaps
    if portion_gaps(stream, date_info) > 0.3:
        msg = f"Proportion of gaps is more than 30%. Skipping trace {stream[0].id}, {stream[0].stats.starttime}"
        Logger.warning(msg)
        stream = []
        return stream

    freqs = []
    for tr in stream:
        freqs.append(int(tr.stats.sampling_rate))
    freq = max(freqs)
    for tr in stream:
        if int(tr.stats.sampling_rate) != freq:
            msg = f"Skipping trace with mismatched sampling rate: {tr.id}, {tr.stats.starttime}, {tr.stats.sampling_rate}"
            Logger.warning(msg)
            stream.remove(tr)
        if tr.stats.npts < 10:
            msg = f"Skipping trace with less than 10 points: {tr.id}, {tr.stats.starttime}"
            Logger.warning(msg)
            stream.remove(tr)

    return stream


@jit('float32[:](float32[:],float32)')
def segment_interpolate(sig1, nfric):
    '''
    this function interpolates the data to ensure all points located on interger times of the
    sampling rate (e.g., starttime = 00:00:00.015, delta = 0.05.)
    PARAMETERS:
    ----------------------
    sig1:  seismic recordings in a 1D array
    nfric: the amount of time difference between the point and the adjacent assumed samples
    RETURNS:
    ----------------------
    sig2:  interpolated seismic recordings on the sampling points
    '''
    npts = len(sig1)
    sig2 = np.zeros(npts, dtype=np.float32)

    # ----instead of shifting, do a interpolation------
    for ii in range(npts):

        # ----deal with edges-----
        if ii == 0 or ii == npts - 1:
            sig2[ii] = sig1[ii]
        else:
            # ------interpolate using a hat function------
            sig2[ii] = (1 - nfric) * sig1[ii + 1] + nfric * sig1[ii]

    return sig2


def portion_gaps(stream, date_info):
    '''
    this function tracks the gaps (npts) from the accumulated difference between starttime and endtime
    of each stream trace. it removes trace with gap length > 30% of trace size.
    PARAMETERS:
    -------------------
    stream: obspy stream object
    date_info: dict of starting and ending time of the stream

    RETURNS:
    -----------------
    pgaps: proportion of gaps/all_pts in stream
    '''
    # ideal duration of data
    starttime = date_info['starttime']
    endtime = date_info['endtime']
    npts = (endtime - starttime) * stream[0].stats.sampling_rate

    pgaps = 0
    # loop through all trace to accumulate gaps
    for ii in range(len(stream) - 1):
        pgaps += (stream[ii + 1].stats.starttime - stream[ii].stats.endtime) * stream[ii].stats.sampling_rate
    if npts != 0: pgaps = pgaps / npts
    if npts == 0: pgaps = 1
    return pgaps


def resp_spectrum(source, resp_file, downsamp_freq, pre_filt=None):
    '''
    this function removes the instrument response using response spectrum from evalresp.
    the response spectrum is evaluated based on RESP/PZ files before inverted using the obspy
    function of invert_spectrum. a module of create_resp.py is provided in directory of 'additional_modules'
    to create the response spectrum
    PARAMETERS:
    ----------------------
    source: obspy stream object of targeted noise data
    resp_file: numpy data file of response spectrum
    downsamp_freq: sampling rate of the source data
    pre_filt: pre-defined filter parameters
    RETURNS:
    ----------------------
    source: obspy stream object of noise data with instrument response removed
    '''
    # --------resp_file is the inverted spectrum response---------
    respz = np.load(resp_file)
    nrespz = respz[1][:]
    spec_freq = max(respz[0])

    # -------on current trace----------
    nfft = _npts2nfft(source[0].stats.npts)
    sps = int(source[0].stats.sampling_rate)

    # ---------do the interpolation if needed--------
    if spec_freq < 0.5 * sps:
        raise ValueError('spectrum file has peak freq smaller than the data, abort!')
    else:
        indx = np.where(respz[0] <= 0.5 * sps)
        nfreq = np.linspace(0, 0.5 * sps, nfft // 2 + 1)
        nrespz = np.interp(nfreq, np.real(respz[0][indx]), respz[1][indx])

    # ----do interpolation if necessary-----
    source_spect = np.fft.rfft(source[0].data, n=nfft)

    # -----nrespz is inversed (water-leveled) spectrum-----
    source_spect *= nrespz
    source[0].data = np.fft.irfft(source_spect)[0:source[0].stats.npts]

    if pre_filt is not None:
        source[0].data = np.float32(
            bandpass(source[0].data, pre_filt[0], pre_filt[-1], df=sps, corners=4, zerophase=True))

    return source

