import obspy
import scipy
import numpy as np
from numba import jit
from scipy.fftpack import fft, ifft, next_fast_len
from obspy.core.util.base import _get_function_from_entry_point
from noisepy.stacking import robust_stack, selective_stack, pws
import logging

Logger = logging.getLogger(__name__)


# prepare traces *******************************************
def sta_info_from_inv(inv):
    '''
    this function outputs station info from the obspy inventory object
    (used in S0B)
    PARAMETERS:
    ----------------------
    inv: obspy inventory object
    RETURNS:
    ----------------------
    sta: station name
    net: netowrk name
    lon: longitude of the station
    lat: latitude of the station
    elv: elevation of the station
    location: location code of the station
    '''
    # load from station inventory
    sta = inv[0][0].code
    net = inv[0].code
    lon = inv[0][0].longitude
    lat = inv[0][0].latitude
    if inv[0][0].elevation:
        elv = inv[0][0].elevation
    else:
        elv = 0.

    if inv[0][0][0].location_code:
        location = inv[0][0][0].location_code
    else:
        location = '00'

    return sta, net, lon, lat, elv, location


def cut_trace_make_stat(fc_para, source):
    '''
    this function cuts continous noise data into user-defined segments, estimate the statistics of
    each segment and keep timestamp of each segment for later use. (used in S1)
    PARAMETERS:
    ----------------------
    fft_para: A dictionary containing all fft and cc parameters.
    source: obspy stream object
    RETURNS:
    ----------------------
    trace_stdS: standard deviation of the noise amplitude of each segment
    dataS_t:    timestamps of each segment
    dataS:      2D matrix of the segmented data
    '''
    # define return variables first
    source_params = []
    dataS_t = []
    dataS = []

    # load parameter from dic
    inc_hours = fc_para['inc_hours']
    cc_len = fc_para['cc_len']
    step = fc_para['step']

    # useful parameters for trace sliding
    nseg = int(np.floor((inc_hours / 24 * 86400 - cc_len) / step))
    sps = int(source[0].stats.sampling_rate)
    starttime = source[0].stats.starttime - obspy.UTCDateTime(1970, 1, 1)
    # copy data into array
    data = source[0].data

    # if the data is shorter than the time chunck, return zero values
    if data.size < sps * inc_hours * 3600:
        Logger.warning(f"The data is shorter than the time chunk! returning empty arrays {source.id}")
        return source_params, dataS_t, dataS

    # statistic to detect segments that may be associated with earthquakes
    all_madS = mad(data)  # median absolute deviation over all noise window
    all_stdS = np.std(data)  # standard deviation over all noise window
    if all_madS == 0 or all_stdS == 0 or np.isnan(all_madS) or np.isnan(all_stdS):
        Logger.info("continue! madS or stdS equals to 0 for %s" % source)
        return source_params, dataS_t, dataS

    # initialize variables
    npts = cc_len * sps
    # trace_madS = np.zeros(nseg,dtype=np.float32)
    trace_stdS = np.zeros(nseg, dtype=np.float32)
    dataS = np.zeros(shape=(nseg, npts), dtype=np.float32)
    dataS_t = np.zeros(nseg, dtype=np.float)

    indx1 = 0
    for iseg in range(nseg):
        indx2 = indx1 + npts
        dataS[iseg] = data[indx1:indx2]
        # trace_madS[iseg] = (np.max(np.abs(dataS[iseg]))/all_madS)
        trace_stdS[iseg] = (np.max(np.abs(dataS[iseg])) / all_stdS)
        dataS_t[iseg] = starttime + step * iseg
        indx1 = indx1 + step * sps

    # 2D array processing
    dataS = demean(dataS)
    dataS = detrend(dataS)
    dataS = taper(dataS)

    return trace_stdS, dataS_t, dataS


# Normalization *******************************************
def noise_processing(fft_para, dataS):
    '''
    this function performs time domain and frequency domain normalization if needed. in real case, we prefer use include
    the normalization in the cross-correlation steps by selecting coherency or decon (Prieto et al, 2008, 2009; Denolle et al, 2013)
    PARAMETERS:
    ------------------------
    fft_para: dictionary containing all useful variables used for fft and cc
    dataS: 2D matrix of all segmented noise data
    # OUTPUT VARIABLES:
    source_white: 2D matrix of data spectra

    coherency: time_norm = 'no' and freq_norm = 'rma'

    TODO: smooth_N used for both time normalization and frequency normalization, but should be allowed to be different values.
    '''
    # load parameters first
    time_norm = fft_para['time_norm']
    freq_norm = fft_para['freq_norm']
    smooth_N = fft_para['smooth_N']
    N = dataS.shape[0]

    # ------to normalize in time or not------
    if time_norm != 'no':

        if time_norm == 'one_bit':  # sign normalization
            white = np.sign(dataS)
        elif time_norm == 'rma':  # running mean: normalization over smoothed absolute average
            white = np.zeros(shape=dataS.shape, dtype=dataS.dtype)
            for kkk in range(N):
                white[kkk, :] = dataS[kkk, :] / moving_ave(np.abs(dataS[kkk, :]), smooth_N)

    else:  # don't normalize
        white = dataS

    # -----to whiten or not------
    if freq_norm != 'no':
        source_white = whiten(white, fft_para)  # whiten and return FFT
    else:
        Nfft = int(next_fast_len(int(dataS.shape[1])))
        source_white = scipy.fftpack.fft(white, Nfft, axis=1)  # return FFT

    return source_white


def noise_processing_2comps(fft_para, dataS1, dataS2):
    '''
    this function performs time domain and frequency domain normalization if needed. in real case, we prefer use include
    the normalization in the cross-correlation steps by selecting coherency or decon (Prieto et al, 2008, 2009; Denolle et al, 2013)
    This function is modified to normalize the 2 horizontal components taking into account both of them,
    so that rotation from NE -> RT is commutative with normalization

    PARAMETERS:
    ------------------------
    fft_para: dictionary containing all useful variables used for fft and cc
    dataS1: 2D matrix of all segmented noise data for horizontal component 1
    dataS2: 2D matrix of all segmented noise data for horizontal component 2
    # OUTPUT VARIABLES:
    source_white1: 2D matrix of data spectra for component 1
    source_white2: 2D matrix of data spectra for component 2

    coherency: time_norm = 'no' and freq_norm = 'rma'

    TODO: smooth_N used for both time normalization and frequency normalization, but should be allowed to be different values.
    '''
    # load parameters first
    time_norm = fft_para['time_norm']
    freq_norm = fft_para['freq_norm']
    smooth_N = fft_para['smooth_N']
    if dataS1.shape[0] != dataS2.shape[0]: raise ValueError("Size of dataS1 and dataS2 different. Abort!")
    N = dataS1.shape[0]

    # ------to normalize in time or not------
    if time_norm != 'no':

        if time_norm == 'one_bit':  # sign normalization
            raise ValueError("Can't do 1-bit normalization that is commutative with NE->RT rotation")
        elif time_norm == 'rma':  # running mean: normalization over smoothed absolute average
            white1 = np.zeros(shape=dataS1.shape, dtype=dataS1.dtype)
            white2 = np.zeros(shape=dataS2.shape, dtype=dataS2.dtype)
            for kkk in range(N):
                move_age_2c = np.vstack(
                    (moving_ave(np.abs(dataS1[kkk, :]), smooth_N), moving_ave(np.abs(dataS2[kkk, :]), smooth_N)))
                move_age_max = np.max(move_age_2c, axis=0)
                white1[kkk, :] = dataS1[kkk, :] / move_age_max
                white2[kkk, :] = dataS2[kkk, :] / move_age_max

    else:  # don't normalize
        white1 = dataS1
        white2 = dataS2

    # -----to whiten or not------
    if freq_norm != 'no':
        source_white1, source_white2 = whiten_2comps(white1, white2, fft_para)  # whiten and return FFT
    else:
        Nfft = int(next_fast_len(int(dataS1.shape[1])))
        source_white1 = scipy.fftpack.fft(white1, Nfft, axis=1)  # return FFT
        source_white2 = scipy.fftpack.fft(white2, Nfft, axis=1)  # return FFT

    return source_white1, source_white2


def noise_processing_3comps(fft_para, dataSN, dataSE, dataSZ):
    '''
    this function performs time domain and frequency domain normalization if needed. in real case, we prefer use include
    the normalization in the cross-correlation steps by selecting coherency or decon (Prieto et al, 2008, 2009; Denolle et al, 2013)
    This function is modified to normalize the 3 components by the vertical component to keep relative amplitudes accurate for e.g. ellipticity calculations.

    PARAMETERS:
    ------------------------
    fft_para: dictionary containing all useful variables used for fft and cc
    dataSN: 2D matrix of all segmented noise data for horizontal component N
    dataSE: 2D matrix of all segmented noise data for horizontal component E
    dataSZ: 2D matrix of all segmented noise data for horizontal component Z
    # OUTPUT VARIABLES:
    source_whiteN: 2D matrix of data spectra for component N
    source_whiteE: 2D matrix of data spectra for component E
    source_whiteZ: 2D matrix of data spectra for component Z

    coherency: time_norm = 'no' and freq_norm = 'rma'

    TODO: smooth_N used for both time normalization and frequency normalization, but should be allowed to be different values.
    '''
    # load parameters first
    time_norm = fft_para['time_norm']
    freq_norm = fft_para['freq_norm']
    smooth_N = fft_para['smooth_N']
    if dataSN.shape[0] != dataSE.shape[0] or dataSZ.shape[0] != dataSN.shape[0]:
        msg = f"Size of dataSN, dataSE and dataSZ are different ({dataSN.shape[0]},{dataSE.shape[0]},{dataSZ.shape[0]}). Abort!"
        raise ValueError(msg)
    N = dataSN.shape[0]

    # ------to normalize in time or not------
    if time_norm != 'no':

        if time_norm == 'one_bit':  # sign normalization
            raise ValueError("Can't do 1-bit normalization that is commutative with NE-> RT rotation")
        elif time_norm == 'rma':  # running mean: normalization over smoothed absolute average
            whiteN = np.zeros(shape=dataSN.shape, dtype=dataSN.dtype)
            whiteE = np.zeros(shape=dataSE.shape, dtype=dataSE.dtype)
            whiteZ = np.zeros(shape=dataSZ.shape, dtype=dataSZ.dtype)
            for kkk in range(N):
                move_age_z = moving_ave(np.abs(dataSZ[kkk, :]), smooth_N)
                whiteN[kkk, :] = dataSN[kkk, :] / move_age_z
                whiteE[kkk, :] = dataSE[kkk, :] / move_age_z
                whiteZ[kkk, :] = dataSZ[kkk, :] / move_age_z

    else:  # don't normalize
        whiteN = dataSN
        whiteE = dataSE
        whiteZ = dataSZ

    # -----to whiten or not------
    if freq_norm != 'no':
        source_whiteN, source_whiteE, source_whiteZ = whiten_3comps(whiteN, whiteE, whiteZ,
                                                                    fft_para)  # whiten and return FFT
    else:
        Nfft = int(next_fast_len(int(dataSN.shape[1])))
        source_whiteN = scipy.fftpack.fft(whiteN, Nfft, axis=1)  # return FFT
        source_whiteE = scipy.fftpack.fft(whiteE, Nfft, axis=1)  # return FFT
        source_whiteZ = scipy.fftpack.fft(whiteZ, Nfft, axis=1)  # return FFT

    return source_whiteN, source_whiteE, source_whiteZ


def smooth_source_spect(cc_para, fft1):
    '''
    this function smoothes amplitude spectrum of the 2D spectral matrix. (used in S1)
    PARAMETERS:
    ---------------------
    cc_para: dictionary containing useful cc parameters
    fft1:    source spectrum matrix

    RETURNS:
    ---------------------
    sfft1: complex numpy array with normalized spectrum
    '''
    cc_method = cc_para['cc_method']
    smoothspect_N = cc_para['smoothspect_N']

    if cc_method == 'deconv':

        # -----normalize single-station cc to z component-----
        temp = moving_ave(np.abs(fft1), smoothspect_N)
        try:
            sfft1 = np.conj(fft1) / temp ** 2
        except Exception:
            raise ValueError('smoothed spectrum has zero values')

    elif cc_method == 'coherency':
        temp = moving_ave(np.abs(fft1), smoothspect_N)
        try:
            sfft1 = np.conj(fft1) / temp
        except Exception:
            raise ValueError('smoothed spectrum has zero values')

    elif cc_method == 'xcorr':
        sfft1 = np.conj(fft1)

    else:
        raise ValueError('no correction correlation method is selected at L59')

    return sfft1


def whiten(data, fft_para):
    '''
    This function takes 1-dimensional timeseries array, transforms to frequency domain using fft,
    whitens the amplitude of the spectrum in frequency domain between *freqmin* and *freqmax*
    and returns the whitened fft.
    PARAMETERS:
    ----------------------
    data: numpy.ndarray contains the 1D time series to whiten
    fft_para: dict containing all fft_cc parameters such as
        dt: The sampling space of the `data`
        freqmin: The lower frequency bound
        freqmax: The upper frequency bound
        smooth_N: integer, it defines the half window length to smooth
        freq_norm: whitening method between 'one-bit' and 'RMA'
    RETURNS:
    ----------------------
    FFTRawSign: numpy.ndarray contains the FFT of the whitened input trace between the frequency bounds
    '''

    # load parameters
    delta = fft_para['dt']
    freqmin = fft_para['freqmin']
    freqmax = fft_para['freqmax']
    smooth_N = fft_para['smooth_N']
    freq_norm = fft_para['freq_norm']

    # Speed up FFT by padding to optimal size for FFTPACK
    if data.ndim == 1:
        axis = 0
    elif data.ndim == 2:
        axis = 1

    Nfft = int(next_fast_len(int(data.shape[axis])))
    Napod = 100
    freqVec = scipy.fftpack.fftfreq(Nfft, d=delta)[:Nfft // 2]
    J = np.where((freqVec >= freqmin) & (freqVec <= freqmax))[0]
    low = J[0] - Napod
    if low <= 0:
        low = 1

    left = J[0]
    right = J[-1]
    high = J[-1] + Napod
    if high > Nfft / 2:
        high = int(Nfft // 2)
    # low - high are indices for freqmin and freqmax for negative frequencies
    # left - right are indices for freqmin and freqmax for positive frequencies

    FFTRawSign = scipy.fftpack.fft(data, Nfft, axis=axis)

    if axis == 1:
        # Left tapering:
        FFTRawSign[:, 0:low] *= 0
        FFTRawSign[:, low:left] = np.cos(
            np.linspace(np.pi / 2., np.pi, left - low)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign[:, low:left]))
        # Pass band:
        if freq_norm == 'phase_only':
            FFTRawSign[:, left:right] = np.exp(1j * np.angle(FFTRawSign[:, left:right]))
        elif freq_norm == 'rma':
            for ii in range(data.shape[0]):
                tave = moving_ave(np.abs(FFTRawSign[ii, left:right]), smooth_N)
                FFTRawSign[ii, left:right] = FFTRawSign[ii, left:right] / tave
        # Right tapering:
        FFTRawSign[:, right:high] = np.cos(
            np.linspace(0., np.pi / 2., high - right)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign[:, right:high]))
        FFTRawSign[:, high:Nfft // 2] *= 0

        # Hermitian symmetry (because the input is real)
        FFTRawSign[:, -(Nfft // 2) + 1:] = np.flip(np.conj(FFTRawSign[:, 1:(Nfft // 2)]), axis=axis)
    else:
        FFTRawSign[0:low] *= 0
        FFTRawSign[low:left] = np.cos(
            np.linspace(np.pi / 2., np.pi, left - low)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign[low:left]))
        # Pass band:
        if freq_norm == 'phase_only':
            FFTRawSign[left:right] = np.exp(1j * np.angle(FFTRawSign[left:right]))
        elif freq_norm == 'rma':
            tave = moving_ave(np.abs(FFTRawSign[left:right]), smooth_N)
            FFTRawSign[left:right] = FFTRawSign[left:right] / tave
        # Right tapering:
        FFTRawSign[right:high] = np.cos(
            np.linspace(0., np.pi / 2., high - right)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign[right:high]))
        FFTRawSign[high:Nfft // 2] *= 0

        # Hermitian symmetry (because the input is real)
        FFTRawSign[-(Nfft // 2) + 1:] = FFTRawSign[1:(Nfft // 2)].conjugate()[::-1]

    return FFTRawSign


def whiten_2comps(data1, data2, fft_para):
    '''
    This function takes 1-dimensional timeseries array, transforms to frequency domain using fft,
    whitens the amplitude of the spectrum in frequency domain between *freqmin* and *freqmax*
    and returns the whitened fft.
    Modified to take into account 2 components. Normalization made with both components so that it is commutative
    with rotation of horizontal components from NE -> RT
    PARAMETERS:
    ----------------------
    data1: numpy.ndarray contains the 1D time series to whiten for component 1
    data2: numpy.ndarray contains the 1D time series to whiten for component 2
    fft_para: dict containing all fft_cc parameters such as
        dt: The sampling space of the `data`
        freqmin: The lower frequency bound
        freqmax: The upper frequency bound
        smooth_N: integer, it defines the half window length to smooth
        freq_norm: whitening method between 'one-bit' and 'RMA'
    RETURNS:
    ----------------------
    FFTRawSign1: numpy.ndarray contains the FFT of the whitened input trace between the frequency bounds for component 1
    FFTRawSign2: numpy.ndarray contains the FFT of the whitened input trace between the frequency bounds for component 2
    '''

    # load parameters
    delta = fft_para['dt']
    freqmin = fft_para['freqmin']
    freqmax = fft_para['freqmax']
    smooth_N = fft_para['smooth_N']
    freq_norm = fft_para['freq_norm']

    # Speed up FFT by padding to optimal size for FFTPACK
    if data1.ndim == 1:
        axis = 0
    elif data1.ndim == 2:
        axis = 1

    # Determine indices for tapering and bandpass
    # low - high are indices for freqmin and freqmax for negative frequencies
    # left - right are indices for freqmin and freqmax for positive frequencies
    Nfft = int(next_fast_len(int(data1.shape[axis])))
    Napod = 100
    freqVec = scipy.fftpack.fftfreq(Nfft, d=delta)[:Nfft // 2]
    J = np.where((freqVec >= freqmin) & (freqVec <= freqmax))[0]
    low = J[0] - Napod
    if low <= 0:
        low = 1
    left = J[0]
    right = J[-1]
    high = J[-1] + Napod
    if high > Nfft / 2:
        high = int(Nfft // 2)

    # Do fft
    FFTRawSign1 = scipy.fftpack.fft(data1, Nfft, axis=axis)
    FFTRawSign2 = scipy.fftpack.fft(data2, Nfft, axis=axis)

    # Normalize
    if axis == 1:
        # Left tapering:
        FFTRawSign1[:, 0:low] *= 0
        FFTRawSign2[:, 0:low] *= 0
        FFTRawSign1[:, low:left] = np.cos(
            np.linspace(np.pi / 2., np.pi, left - low)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign1[:, low:left]))
        FFTRawSign2[:, low:left] = np.cos(
            np.linspace(np.pi / 2., np.pi, left - low)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign2[:, low:left]))
        # Pass band:
        if freq_norm == 'phase_only':
            raise ValueError("This function whiten_2comps does not apply for the phase_only option")
        elif freq_norm == 'rma':
            for ii in range(data1.shape[0]):
                tave1 = moving_ave(np.abs(FFTRawSign1[ii, left:right]), smooth_N)
                tave2 = moving_ave(np.abs(FFTRawSign2[ii, left:right]), smooth_N)
                tave = np.mean(np.vstack((tave1, tave2)), axis=0)  # Take the mean of the two smoothed spectra
                FFTRawSign1[ii, left:right] = FFTRawSign1[ii, left:right] / tave
                FFTRawSign2[ii, left:right] = FFTRawSign2[ii, left:right] / tave
        # Right tapering:
        FFTRawSign1[:, right:high] = np.cos(
            np.linspace(0., np.pi / 2., high - right)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign1[:, right:high]))
        FFTRawSign2[:, right:high] = np.cos(
            np.linspace(0., np.pi / 2., high - right)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign2[:, right:high]))
        FFTRawSign1[:, high:Nfft // 2] *= 0
        FFTRawSign2[:, high:Nfft // 2] *= 0

        # Hermitian symmetry (because the input is real)
        FFTRawSign1[:, -(Nfft // 2) + 1:] = np.flip(np.conj(FFTRawSign1[:, 1:(Nfft // 2)]), axis=axis)
        FFTRawSign2[:, -(Nfft // 2) + 1:] = np.flip(np.conj(FFTRawSign2[:, 1:(Nfft // 2)]), axis=axis)
    else:
        FFTRawSign1[0:low] *= 0
        FFTRawSign2[0:low] *= 0
        FFTRawSign1[low:left] = np.cos(
            np.linspace(np.pi / 2., np.pi, left - low)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign1[low:left]))
        FFTRawSign2[low:left] = np.cos(
            np.linspace(np.pi / 2., np.pi, left - low)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign2[low:left]))
        # Pass band:
        if freq_norm == 'phase_only':
            raise ValueError("This function whiten_2comps does not apply for the phase_only option")
        elif freq_norm == 'rma':
            tave1 = moving_ave(np.abs(FFTRawSign1[left:right]), smooth_N)
            tave2 = moving_ave(np.abs(FFTRawSign2[left:right]), smooth_N)
            tave = np.mean(np.vstack((tave1, tave2)), axis=0)
            FFTRawSign1[left:right] = FFTRawSign1[left:right] / tave
            FFTRawSign2[left:right] = FFTRawSign2[left:right] / tave
        # Right tapering:
        FFTRawSign1[right:high] = np.cos(
            np.linspace(0., np.pi / 2., high - right)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign1[right:high]))
        FFTRawSign2[right:high] = np.cos(
            np.linspace(0., np.pi / 2., high - right)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign2[right:high]))
        FFTRawSign1[high:Nfft // 2] *= 0
        FFTRawSign2[high:Nfft // 2] *= 0

        # Hermitian symmetry (because the input is real)
        FFTRawSign1[-(Nfft // 2) + 1:] = FFTRawSign1[1:(Nfft // 2)].conjugate()[::-1]
        FFTRawSign2[-(Nfft // 2) + 1:] = FFTRawSign2[1:(Nfft // 2)].conjugate()[::-1]

    return FFTRawSign1, FFTRawSign2


def whiten_3comps(dataN, dataE, dataZ, fft_para):
    '''
    This function takes 1-dimensional timeseries array, transforms to frequency domain using fft,
    whitens the amplitude of the spectrum in frequency domain between *freqmin* and *freqmax*
    and returns the whitened fft.
    Modified to take into account 3 components. Normalization made with Z component so that relative amplitudes are preserved.
    PARAMETERS:
    ----------------------
    dataN: numpy.ndarray contains the 1D time series to whiten for component N
    dataE: numpy.ndarray contains the 1D time series to whiten for component E
    dataZ: numpy.ndarray contains the 1D time series to whiten for component Z
    fft_para: dict containing all fft_cc parameters such as
        dt: The sampling space of the `data`
        freqmin: The lower frequency bound
        freqmax: The upper frequency bound
        smooth_N: integer, it defines the half window length to smooth
        freq_norm: whitening method between 'one-bit' and 'RMA'
    RETURNS:
    ----------------------
    FFTRawSignN: numpy.ndarray contains the FFT of the whitened input trace between the frequency bounds for component N
    FFTRawSignE: numpy.ndarray contains the FFT of the whitened input trace between the frequency bounds for component E
    FFTRawSignZ: numpy.ndarray contains the FFT of the whitened input trace between the frequency bounds for component Z
    '''

    # load parameters
    delta = fft_para['dt']
    freqmin = fft_para['freqmin']
    freqmax = fft_para['freqmax']
    smooth_N = fft_para['smooth_N']
    freq_norm = fft_para['freq_norm']

    # Speed up FFT by padding to optimal size for FFTPACK
    if dataN.ndim == 1:
        axis = 0
    elif dataN.ndim == 2:
        axis = 1

    # Determine indices for tapering and bandpass
    # low - high are indices for freqmin and freqmax for negative frequencies
    # left - right are indices for freqmin and freqmax for positive frequencies
    Nfft = int(next_fast_len(int(dataN.shape[axis])))
    Napod = 100
    freqVec = scipy.fftpack.fftfreq(Nfft, d=delta)[:Nfft // 2]
    J = np.where((freqVec >= freqmin) & (freqVec <= freqmax))[0]
    low = J[0] - Napod
    if low <= 0:
        low = 1
    left = J[0]
    right = J[-1]
    high = J[-1] + Napod
    if high > Nfft / 2:
        high = int(Nfft // 2)

    # Do fft
    FFTRawSignN = scipy.fftpack.fft(dataN, Nfft, axis=axis)
    FFTRawSignE = scipy.fftpack.fft(dataE, Nfft, axis=axis)
    FFTRawSignZ = scipy.fftpack.fft(dataZ, Nfft, axis=axis)

    # Normalize
    if axis == 1:
        # Left tapering:
        FFTRawSignN[:, 0:low] *= 0
        FFTRawSignE[:, 0:low] *= 0
        FFTRawSignZ[:, 0:low] *= 0

        FFTRawSignN[:, low:left] = np.cos(
            np.linspace(np.pi / 2., np.pi, left - low)) ** 2 * np.exp(
            1j * np.angle(FFTRawSignN[:, low:left]))
        FFTRawSignE[:, low:left] = np.cos(
            np.linspace(np.pi / 2., np.pi, left - low)) ** 2 * np.exp(
            1j * np.angle(FFTRawSignE[:, low:left]))
        FFTRawSignZ[:, low:left] = np.cos(
            np.linspace(np.pi / 2., np.pi, left - low)) ** 2 * np.exp(
            1j * np.angle(FFTRawSignZ[:, low:left]))

        # Pass band:
        if freq_norm == 'phase_only':
            raise ValueError("This function whiten_3comps does not apply for the phase_only option")
        elif freq_norm == 'rma':
            for ii in range(dataN.shape[0]):
                #                 taven = moving_ave(np.abs(FFTRawSignZ[ii, left:right]), smooth_N)
                #                 tavee = moving_ave(np.abs(FFTRawSignE[ii, left:right]), smooth_N)
                tave = moving_ave(np.abs(FFTRawSignZ[ii, left:right]), smooth_N)
                #                 tave = np.mean(np.vstack((taven, tavee, tavez)), axis=0)  # Take the mean of the 3 smoothed spectra
                FFTRawSignN[ii, left:right] = FFTRawSignN[ii, left:right] / tave
                FFTRawSignE[ii, left:right] = FFTRawSignE[ii, left:right] / tave
                FFTRawSignZ[ii, left:right] = FFTRawSignZ[ii, left:right] / tave

        # Right tapering:
        FFTRawSignN[:, right:high] = np.cos(
            np.linspace(0., np.pi / 2., high - right)) ** 2 * np.exp(
            1j * np.angle(FFTRawSignN[:, right:high]))
        FFTRawSignE[:, right:high] = np.cos(
            np.linspace(0., np.pi / 2., high - right)) ** 2 * np.exp(
            1j * np.angle(FFTRawSignE[:, right:high]))
        FFTRawSignZ[:, right:high] = np.cos(
            np.linspace(0., np.pi / 2., high - right)) ** 2 * np.exp(
            1j * np.angle(FFTRawSignZ[:, right:high]))

        FFTRawSignN[:, high:Nfft // 2] *= 0
        FFTRawSignE[:, high:Nfft // 2] *= 0
        FFTRawSignZ[:, high:Nfft // 2] *= 0

        # Hermitian symmetry (because the input is real)
        FFTRawSignN[:, -(Nfft // 2) + 1:] = np.flip(np.conj(FFTRawSignN[:, 1:(Nfft // 2)]), axis=axis)
        FFTRawSignE[:, -(Nfft // 2) + 1:] = np.flip(np.conj(FFTRawSignE[:, 1:(Nfft // 2)]), axis=axis)
        FFTRawSignZ[:, -(Nfft // 2) + 1:] = np.flip(np.conj(FFTRawSignZ[:, 1:(Nfft // 2)]), axis=axis)
    else:
        FFTRawSignN[0:low] *= 0
        FFTRawSignE[0:low] *= 0
        FFTRawSignZ[0:low] *= 0

        FFTRawSignN[low:left] = np.cos(
            np.linspace(np.pi / 2., np.pi, left - low)) ** 2 * np.exp(
            1j * np.angle(FFTRawSignN[low:left]))
        FFTRawSignE[low:left] = np.cos(
            np.linspace(np.pi / 2., np.pi, left - low)) ** 2 * np.exp(
            1j * np.angle(FFTRawSignE[low:left]))
        FFTRawSignZ[low:left] = np.cos(
            np.linspace(np.pi / 2., np.pi, left - low)) ** 2 * np.exp(
            1j * np.angle(FFTRawSignZ[low:left]))

        # Pass band:
        if freq_norm == 'phase_only':
            raise ValueError("This function whiten_3comps does not apply for the phase_only option")
        elif freq_norm == 'rma':
            #             taven = moving_ave(np.abs(FFTRawSignN[left:right]), smooth_N)
            #             tavee = moving_ave(np.abs(FFTRawSignE[left:right]), smooth_N)
            tave = moving_ave(np.abs(FFTRawSignZ[left:right]), smooth_N)
            #             tave = np.mean(np.vstack((taven, tavee, tavez)), axis=0)
            FFTRawSignN[left:right] = FFTRawSignN[left:right] / tave
            FFTRawSignE[left:right] = FFTRawSignE[left:right] / tave
            FFTRawSignZ[left:right] = FFTRawSignZ[left:right] / tave

        # Right tapering:
        FFTRawSignN[right:high] = np.cos(
            np.linspace(0., np.pi / 2., high - right)) ** 2 * np.exp(
            1j * np.angle(FFTRawSignN[right:high]))
        FFTRawSignE[right:high] = np.cos(
            np.linspace(0., np.pi / 2., high - right)) ** 2 * np.exp(
            1j * np.angle(FFTRawSignE[right:high]))
        FFTRawSignZ[right:high] = np.cos(
            np.linspace(0., np.pi / 2., high - right)) ** 2 * np.exp(
            1j * np.angle(FFTRawSignZ[right:high]))

        FFTRawSignN[high:Nfft // 2] *= 0
        FFTRawSignE[high:Nfft // 2] *= 0
        FFTRawSignZ[high:Nfft // 2] *= 0

        # Hermitian symmetry (because the input is real)
        FFTRawSignN[-(Nfft // 2) + 1:] = FFTRawSignN[1:(Nfft // 2)].conjugate()[::-1]
        FFTRawSignE[-(Nfft // 2) + 1:] = FFTRawSignE[1:(Nfft // 2)].conjugate()[::-1]
        FFTRawSignZ[-(Nfft // 2) + 1:] = FFTRawSignZ[1:(Nfft // 2)].conjugate()[::-1]

    return FFTRawSignN, FFTRawSignE, FFTRawSignZ


# Correlate *******************************************
def correlate(fft1_smoothed_abs, fft2, D, Nfft, dataS_t):
    '''
    this function does the cross-correlation in freq domain and has the option to keep sub-stacks of
    the cross-correlation if needed. it takes advantage of the linear relationship of ifft, so that
    stacking is performed in spectrum domain first to reduce the total number of ifft. (used in S1)
    PARAMETERS:
    ---------------------
    fft1_smoothed_abs: smoothed power spectral density of the FFT for the source station
    fft2: raw FFT spectrum of the receiver station
    D: dictionary containing following parameters:
        maxlag:  maximum lags to keep in the cross correlation
        dt:      sampling rate (in s)
        nwin:    number of segments in the 2D matrix
        method:  cross-correlation methods selected by the user
        freqmin: minimum frequency (Hz)
        freqmax: maximum frequency (Hz)
    Nfft:    number of frequency points for ifft
    dataS_t: matrix of datetime object.

    RETURNS:
    ---------------------
    s_corr: 1D or 2D matrix of the averaged or sub-stacks of cross-correlation functions in time domain
    t_corr: timestamp for each sub-stack or averaged function
    n_corr: number of included segments for each sub-stack or averaged function

    MODIFICATIONS:
    ---------------------
    output the linear stack of each time chunk even when substack is selected (by Chengxin @Aug2020)
    '''
    # ----load paramters----
    dt = D['dt']
    maxlag = D['maxlag']
    method = D['cc_method']
    cc_len = D['cc_len']
    substack = D['substack']
    substack_len = D['substack_len']
    smoothspect_N = D['smoothspect_N']

    nwin = fft1_smoothed_abs.shape[0]
    Nfft2 = fft1_smoothed_abs.shape[1]

    # ------convert all 2D arrays into 1D to speed up--------
    corr = np.zeros(nwin * Nfft2, dtype=np.complex64)
    corr = fft1_smoothed_abs.reshape(fft1_smoothed_abs.size, ) * fft2.reshape(fft2.size, )

    if method == "coherency":
        temp = moving_ave(np.abs(fft2.reshape(fft2.size, )), smoothspect_N)
        corr /= temp
    corr = corr.reshape(nwin, Nfft2)

    if substack:
        if substack_len == cc_len:
            # choose to keep all fft data for a day
            s_corr = np.zeros(shape=(nwin, Nfft), dtype=np.float32)  # stacked correlation
            ampmax = np.zeros(nwin, dtype=np.float32)
            n_corr = np.zeros(nwin, dtype=np.int16)  # number of correlations for each substack
            t_corr = dataS_t  # timestamp
            crap = np.zeros(Nfft, dtype=np.complex64)
            for i in range(nwin):
                n_corr[i] = 1
                crap[:Nfft2] = corr[i, :]
                crap[:Nfft2] = crap[:Nfft2] - np.mean(crap[:Nfft2])  # remove the mean in freq domain (spike at t=0)
                crap[-(Nfft2) + 1:] = np.flip(np.conj(crap[1:(Nfft2)]), axis=0)
                crap[0] = complex(0, 0)
                s_corr[i, :] = np.real(np.fft.ifftshift(scipy.fftpack.ifft(crap, Nfft, axis=0)))

            # remove abnormal data
            ampmax = np.max(s_corr, axis=1)
            tindx = np.where((ampmax < 20 * np.median(ampmax)) & (ampmax > 0))[0]
            s_corr = s_corr[tindx, :]
            t_corr = t_corr[tindx]
            n_corr = n_corr[tindx]

        else:
            # get time information
            Ttotal = dataS_t[-1] - dataS_t[0]  # total duration of what we have now
            tstart = dataS_t[0]

            nstack = int(np.round(Ttotal / substack_len))
            ampmax = np.zeros(nstack, dtype=np.float32)
            s_corr = np.zeros(shape=(nstack, Nfft), dtype=np.float32)
            n_corr = np.zeros(nstack, dtype=np.int)
            t_corr = np.zeros(nstack, dtype=np.float)
            crap = np.zeros(Nfft, dtype=np.complex64)

            for istack in range(nstack):
                # find the indexes of all of the windows that start or end within
                itime = np.where((dataS_t >= tstart) & (dataS_t < tstart + substack_len))[0]
                if len(itime) == 0: tstart += substack_len;continue

                crap[:Nfft2] = np.mean(corr[itime, :], axis=0)  # linear average of the correlation
                crap[:Nfft2] = crap[:Nfft2] - np.mean(crap[:Nfft2])  # remove the mean in freq domain (spike at t=0)
                crap[-(Nfft2) + 1:] = np.flip(np.conj(crap[1:(Nfft2)]), axis=0)
                crap[0] = complex(0, 0)
                s_corr[istack, :] = np.real(np.fft.ifftshift(scipy.fftpack.ifft(crap, Nfft, axis=0)))
                n_corr[istack] = len(itime)  # number of windows stacks
                t_corr[istack] = tstart  # save the time stamps
                tstart += substack_len
                # Logger.info('correlation done and stacked at time %s' % str(t_corr[istack]))

            # remove abnormal data
            ampmax = np.max(s_corr, axis=1)
            tindx = np.where((ampmax < 20 * np.median(ampmax)) & (ampmax > 0))[0]
            s_corr = s_corr[tindx, :]
            t_corr = t_corr[tindx]
            n_corr = n_corr[tindx]

    else:
        # average daily cross correlation functions
        ampmax = np.max(corr, axis=1)
        tindx = np.where((ampmax < 20 * np.median(ampmax)) & (ampmax > 0))[0]
        n_corr = nwin
        s_corr = np.zeros(Nfft, dtype=np.float32)
        t_corr = dataS_t[0]
        crap = np.zeros(Nfft, dtype=np.complex64)
        crap[:Nfft2] = np.mean(corr[tindx], axis=0)
        crap[:Nfft2] = crap[:Nfft2] - np.mean(crap[:Nfft2], axis=0)
        crap[-(Nfft2) + 1:] = np.flip(np.conj(crap[1:(Nfft2)]), axis=0)
        s_corr = np.real(np.fft.ifftshift(scipy.fftpack.ifft(crap, Nfft, axis=0)))

    # trim the CCFs in [-maxlag maxlag]
    t = np.arange(-Nfft2 + 1, Nfft2) * dt
    ind = np.where(np.abs(t) <= maxlag)[0]
    if s_corr.ndim == 1:
        s_corr = s_corr[ind]
    elif s_corr.ndim == 2:
        s_corr = s_corr[:, ind]
    return s_corr, t_corr, n_corr


def correlate_nonlinear_stack(fft1_smoothed_abs, fft2, D, Nfft, dataS_t):
    '''
    this function does the cross-correlation in freq domain and has the option to keep sub-stacks of
    the cross-correlation if needed. it takes advantage of the linear relationship of ifft, so that
    stacking is performed in spectrum domain first to reduce the total number of ifft. (used in S1)
    PARAMETERS:
    ---------------------
    fft1_smoothed_abs: smoothed power spectral density of the FFT for the source station
    fft2: raw FFT spectrum of the receiver station
    D: dictionary containing following parameters:
        maxlag:  maximum lags to keep in the cross correlation
        dt:      sampling rate (in s)
        nwin:    number of segments in the 2D matrix
        method:  cross-correlation methods selected by the user
        freqmin: minimum frequency (Hz)
        freqmax: maximum frequency (Hz)
    Nfft:    number of frequency points for ifft
    dataS_t: matrix of datetime object.
    RETURNS:
    ---------------------
    s_corr: 1D or 2D matrix of the averaged or sub-stacks of cross-correlation functions in time domain
    t_corr: timestamp for each sub-stack or averaged function
    n_corr: number of included segments for each sub-stack or averaged function
    '''
    # ----load paramters----
    dt = D['dt']
    maxlag = D['maxlag']
    method = D['cc_method']
    cc_len = D['cc_len']
    substack = D['substack']
    stack_method = D['stack_method']
    substack_len = D['substack_len']
    smoothspect_N = D['smoothspect_N']

    nwin = fft1_smoothed_abs.shape[0]
    Nfft2 = fft1_smoothed_abs.shape[1]

    # ------convert all 2D arrays into 1D to speed up--------
    corr = np.zeros(nwin * Nfft2, dtype=np.complex64)
    corr = fft1_smoothed_abs.reshape(fft1_smoothed_abs.size, ) * fft2.reshape(fft2.size, )

    # normalize by receiver spectral for coherency
    if method == "coherency":
        temp = moving_ave(np.abs(fft2.reshape(fft2.size, )), smoothspect_N)
        corr /= temp
    corr = corr.reshape(nwin, Nfft2)

    # transform back to time domain waveforms
    s_corr = np.zeros(shape=(nwin, Nfft), dtype=np.float32)  # stacked correlation
    ampmax = np.zeros(nwin, dtype=np.float32)
    n_corr = np.zeros(nwin, dtype=np.int16)  # number of correlations for each substack
    t_corr = dataS_t  # timestamp
    crap = np.zeros(Nfft, dtype=np.complex64)
    for i in range(nwin):
        n_corr[i] = 1
        crap[:Nfft2] = corr[i, :]
        crap[:Nfft2] = crap[:Nfft2] - np.mean(crap[:Nfft2])  # remove the mean in freq domain (spike at t=0)
        crap[-(Nfft2) + 1:] = np.flip(np.conj(crap[1:(Nfft2)]), axis=0)
        crap[0] = complex(0, 0)
        s_corr[i, :] = np.real(np.fft.ifftshift(scipy.fftpack.ifft(crap, Nfft, axis=0)))

    ns_corr = s_corr
    for iii in range(ns_corr.shape[0]):
        ns_corr[iii] /= np.max(np.abs(ns_corr[iii]))

    if substack:
        if substack_len == cc_len:

            # remove abnormal data
            ampmax = np.max(s_corr, axis=1)
            tindx = np.where((ampmax < 20 * np.median(ampmax)) & (ampmax > 0))[0]
            s_corr = s_corr[tindx, :]
            t_corr = t_corr[tindx]
            n_corr = n_corr[tindx]

        else:
            # get time information
            Ttotal = dataS_t[-1] - dataS_t[0]  # total duration of what we have now
            tstart = dataS_t[0]

            nstack = int(np.round(Ttotal / substack_len))
            ampmax = np.zeros(nstack, dtype=np.float32)
            s_corr = np.zeros(shape=(nstack, Nfft), dtype=np.float32)
            n_corr = np.zeros(nstack, dtype=np.int)
            t_corr = np.zeros(nstack, dtype=np.float)
            crap = np.zeros(Nfft, dtype=np.complex64)

            for istack in range(nstack):
                # find the indexes of all of the windows that start or end within
                itime = np.where((dataS_t >= tstart) & (dataS_t < tstart + substack_len))[0]
                if len(itime) == 0: tstart += substack_len;continue

                crap[:Nfft2] = np.mean(corr[itime, :], axis=0)  # linear average of the correlation
                crap[:Nfft2] = crap[:Nfft2] - np.mean(crap[:Nfft2])  # remove the mean in freq domain (spike at t=0)
                crap[-(Nfft2) + 1:] = np.flip(np.conj(crap[1:(Nfft2)]), axis=0)
                crap[0] = complex(0, 0)
                s_corr[istack, :] = np.real(np.fft.ifftshift(scipy.fftpack.ifft(crap, Nfft, axis=0)))
                n_corr[istack] = len(itime)  # number of windows stacks
                t_corr[istack] = tstart  # save the time stamps
                tstart += substack_len
                # Logger.info('correlation done and stacked at time %s' % str(t_corr[istack]))

            # remove abnormal data
            ampmax = np.max(s_corr, axis=1)
            tindx = np.where((ampmax < 20 * np.median(ampmax)) & (ampmax > 0))[0]
            s_corr = s_corr[tindx, :]
            t_corr = t_corr[tindx]
            n_corr = n_corr[tindx]

    else:
        # average daily cross correlation functions
        if stack_method == 'linear':
            ampmax = np.max(s_corr, axis=1)
            tindx = np.where((ampmax < 20 * np.median(ampmax)) & (ampmax > 0))[0]
            s_corr = np.mean(s_corr[tindx], axis=0)
            t_corr = dataS_t[0]
            n_corr = len(tindx)
        elif stack_method == 'robust':
            Logger.info('do robust substacking')
            s_corr = robust_stack(s_corr, 0.001)
            t_corr = dataS_t[0]
            n_corr = nwin
    #  elif stack_method == 'selective':
    #      Logger.info('do selective substacking')
    #      s_corr = selective_stack(s_corr,0.001)
    #      t_corr = dataS_t[0]
    #      n_corr = nwin

    # trim the CCFs in [-maxlag maxlag]
    t = np.arange(-Nfft2 + 1, Nfft2) * dt
    ind = np.where(np.abs(t) <= maxlag)[0]
    if s_corr.ndim == 1:
        s_corr = s_corr[ind]
    elif s_corr.ndim == 2:
        s_corr = s_corr[:, ind]
    return s_corr, t_corr, n_corr, ns_corr[:, ind]


# Output **********************************************
def cc_parameters(cc_para, coor, tcorr, ncorr, comp):
    '''
    this function assembles the parameters for the cc function, which is used
    when writing them into ASDF files
    PARAMETERS:
    ---------------------
    cc_para: dict containing parameters used in the fft_cc step
    coor:    dict containing coordinates info of the source and receiver stations
    tcorr:   timestamp matrix
    ncorr:   matrix of number of good segments for each sub-stack/final stack
    comp:    2 character strings for the cross correlation component
    RETURNS:
    ------------------
    parameters: dict containing above info used for later stacking/plotting
    '''
    latS = coor['latS']
    lonS = coor['lonS']
    latR = coor['latR']
    lonR = coor['lonR']
    dt = cc_para['dt']
    maxlag = cc_para['maxlag']
    substack = cc_para['substack']
    cc_method = cc_para['cc_method']

    dist, azi, baz = obspy.geodetics.base.gps2dist_azimuth(latS, lonS, latR, lonR)
    parameters = {'dt': dt,
                  'maxlag': int(maxlag),
                  'dist': np.float32(dist / 1000),
                  'azi': np.float32(azi),
                  'baz': np.float32(baz),
                  'lonS': np.float32(lonS),
                  'latS': np.float32(latS),
                  'lonR': np.float32(lonR),
                  'latR': np.float32(latR),
                  'ngood': ncorr,
                  'cc_method': cc_method,
                  'time': tcorr,
                  'substack': substack,
                  'comp': comp}
    return parameters


# Utility functions *******************************************
@jit(nopython=True)
def moving_ave(A, N):  ## change the moving average calculation to take as input N the full window length to smooth
    '''
    Alternative function for moving average for an array.
    PARAMETERS:
    ---------------------
    A: 1-D array of data to be smoothed
    N: integer, it defines the full!! window length to smooth
    RETURNS:
    ---------------------
    B: 1-D array with smoothed data
    '''
    # defines an array with N extra samples at either side
    temp = np.zeros(len(A) + 2 * N)
    # set the central portion of the array to A
    temp[N: -N] = A
    # leading samples: equal to first sample of actual array
    temp[0: N] = temp[N]
    # trailing samples: Equal to last sample of actual array
    temp[-N:] = temp[-N - 1]
    # convolve with a boxcar and normalize, and use only central portion of the result
    # with length equal to the original array, discarding the added leading and trailing samples
    B = np.convolve(temp, np.ones(N) / N, mode='same')[N: -N]
    return (B)


# @jit(nopython=True)
# def moving_ave(A, N):
#     '''
#     this Numba compiled function does running smooth average for an array.
#     PARAMETERS:
#     ---------------------
#     A: 1-D array of data to be smoothed
#     N: integer, it defines the half window length to smooth
#
#     RETURNS:
#     ---------------------
#     B: 1-D array with smoothed data
#     '''
#     A = np.concatenate((A[:N], A, A[-N:]), axis=0)
#     B = np.zeros(A.shape, A.dtype)
#
#     tmp = 0.
#     for pos in range(N, A.size - N):
#         # do summing only once
#         if pos == N:
#             for i in range(-N, N + 1):
#                 tmp += A[pos + i]
#         else:
#             tmp = tmp - A[pos - N - 1] + A[pos + N]
#         B[pos] = tmp / (2 * N + 1)
#         if B[pos] == 0:
#             B[pos] = 1
#     return B[N:-N]

def mad(arr):
    """
    Median Absolute Deviation: MAD = median(|Xi- median(X)|)
    PARAMETERS:
    -------------------
    arr: numpy.ndarray, seismic trace data array
    RETURNS:
    data: Median Absolute Deviation of data
    """
    if not np.ma.is_masked(arr):
        med = np.median(arr)
        data = np.median(np.abs(arr - med))
    else:
        med = np.ma.median(arr)
        data = np.ma.median(np.ma.abs(arr - med))
    return data


def detrend(data):
    '''
    this function removes the signal trend based on QR decomposion
    NOTE: QR is a lot faster than the least square inversion used by
    scipy (also in obspy).
    PARAMETERS:
    ---------------------
    data: input data matrix
    RETURNS:
    ---------------------
    data: data matrix with trend removed
    '''
    # ndata = np.zeros(shape=data.shape,dtype=data.dtype)
    if data.ndim == 1:
        npts = data.shape[0]
        X = np.ones((npts, 2))
        X[:, 0] = np.arange(0, npts) / npts
        Q, R = np.linalg.qr(X)
        rq = np.dot(np.linalg.inv(R), Q.transpose())
        coeff = np.dot(rq, data)
        data = data - np.dot(X, coeff)
    elif data.ndim == 2:
        npts = data.shape[1]
        X = np.ones((npts, 2))
        X[:, 0] = np.arange(0, npts) / npts
        Q, R = np.linalg.qr(X)
        rq = np.dot(np.linalg.inv(R), Q.transpose())
        for ii in range(data.shape[0]):
            coeff = np.dot(rq, data[ii])
            data[ii] = data[ii] - np.dot(X, coeff)
    return data


def demean(data):
    '''
    this function remove the mean of the signal
    PARAMETERS:
    ---------------------
    data: input data matrix
    RETURNS:
    ---------------------
    data: data matrix with mean removed
    '''
    # ndata = np.zeros(shape=data.shape,dtype=data.dtype)
    if data.ndim == 1:
        data = data - np.mean(data)
    elif data.ndim == 2:
        for ii in range(data.shape[0]):
            data[ii] = data[ii] - np.mean(data[ii])
    return data


def taper(data):
    '''
    this function applies a cosine taper using obspy functions
    PARAMETERS:
    ---------------------
    data: input data matrix
    RETURNS:
    ---------------------
    data: data matrix with taper applied
    '''
    # ndata = np.zeros(shape=data.shape,dtype=data.dtype)
    if data.ndim == 1:
        npts = data.shape[0]
        # window length
        if npts * 0.05 > 20:
            wlen = 20
        else:
            wlen = npts * 0.05
        # taper values
        func = _get_function_from_entry_point('taper', 'hann')
        if 2 * wlen == npts:
            taper_sides = func(2 * wlen)
        else:
            taper_sides = func(2 * wlen + 1)
        # taper window
        win = np.hstack((taper_sides[:wlen], np.ones(npts - 2 * wlen), taper_sides[len(taper_sides) - wlen:]))
        data *= win
    elif data.ndim == 2:
        npts = data.shape[1]
        # window length
        if npts * 0.05 > 20:
            wlen = 20
        else:
            wlen = npts * 0.05
        # taper values
        func = _get_function_from_entry_point('taper', 'hann')
        if 2 * wlen == npts:
            taper_sides = func(2 * wlen)
        else:
            taper_sides = func(2 * wlen + 1)
        # taper window
        win = np.hstack((taper_sides[:wlen], np.ones(npts - 2 * wlen), taper_sides[len(taper_sides) - wlen:]))
        for ii in range(data.shape[0]):
            data[ii] *= win
    return data
