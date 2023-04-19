""" Stacking functions """


import obspy
import scipy
import numpy as np
from scipy.signal import hilbert
from scipy.fftpack import fft, ifft, next_fast_len
import logging

Logger = logging.getLogger(__name__)


# MAIN STACKING
def stacking(cc_array, cc_time, cc_ngood, stack_para):
    '''
    this function stacks the cross correlation data according to the user-defined substack_len parameter

    PARAMETERS:
    ----------------------
    cc_array: 2D numpy float32 matrix containing all segmented cross-correlation data
    cc_time:  1D numpy array of timestamps for each segment of cc_array
    cc_ngood: 1D numpy int16 matrix showing the number of segments for each sub-stack and/or full stack
    stack_para: a dict containing all stacking parameters

    RETURNS:
    ----------------------
    cc_array, cc_ngood, cc_time: same to the input parameters but with abnormal cross-correaltions removed
    allstacks1: 1D matrix of stacked cross-correlation functions over all the segments
    nstacks:    number of overall segments for the final stacks
    '''
    # load useful parameters from dict
    samp_freq = stack_para['samp_freq']
    smethod = stack_para['stack_method']
    start_date = stack_para['start_date']
    end_date = stack_para['end_date']
    npts = cc_array.shape[1]

    # remove abnormal data
    ampmax = np.max(cc_array, axis=1)
    tindx = np.where((ampmax < 20 * np.median(ampmax)) & (ampmax > 0))[0]
    if not len(tindx):
        allstacks1 = []
        allstacks2 = []
        allstacks3 = []
        allstacks4 = []
        allstacks5 = []
        nstacks = 0
        cc_array = []
        cc_ngood = []
        cc_time = []
        #         return cc_array,cc_ngood,cc_time,allstacks1,allstacks2,allstacks3,nstacks
        Logger.warning("Abnormal data. No stacking.")
        return cc_array, cc_ngood, cc_time, allstacks1, allstacks2, allstacks3, allstacks4, allstacks5, nstacks  # GS
    else:

        # remove ones with bad amplitude
        cc_array = cc_array[tindx, :]
        cc_time = cc_time[tindx]
        cc_ngood = cc_ngood[tindx]

        # do stacking
        allstacks1 = np.zeros(npts, dtype=np.float32)
        allstacks2 = np.zeros(npts, dtype=np.float32)
        allstacks3 = np.zeros(npts, dtype=np.float32)
        allstacks4 = np.zeros(npts, dtype=np.float32)
        allstacks5 = np.zeros(npts, dtype=np.float32)

        if smethod == 'linear':
            allstacks1 = np.mean(cc_array, axis=0)
        elif smethod == 'pws':
            allstacks1 = pws(cc_array, samp_freq)
        elif smethod == 'robust':
            allstacks1, w, nstep = robust_stack(cc_array, 0.001)
        elif smethod == 'auto_covariance':
            allstacks1 = adaptive_filter(cc_array, 1)
        elif smethod == 'nroot':
            allstacks1 = nroot_stack(cc_array, 2)
        elif smethod == 'all':
            allstacks1 = np.mean(cc_array, axis=0)
            allstacks2 = pws(cc_array, samp_freq)
            allstacks3, w, nstep = robust_stack(cc_array, 0.001)
            # allstacks4 = adaptive_filter(cc_array, 1)  # GS
            allstacks5 = nroot_stack(cc_array, 2)  # GS
        nstacks = np.sum(cc_ngood)

    # good to return
    return cc_array, cc_ngood, cc_time, allstacks1, allstacks2, allstacks3, allstacks4, allstacks5, nstacks  # GS


def stacking_rma(cc_array, cc_time, cc_ngood, stack_para):
    '''
    this function stacks the cross correlation data according to the user-defined substack_len parameter

    makes new substacks according to rma_substack and rma_step parameters

    PARAMETERS:
    ----------------------
    cc_array: 2D numpy float32 matrix containing all segmented cross-correlation data
    cc_time:  1D numpy array of timestamps for each segment of cc_array
    cc_ngood: 1D numpy int16 matrix showing the number of segments for each sub-stack and/or full stack
    stack_para: a dict containing all stacking parameters
    RETURNS:
    ----------------------
    cc_array, cc_ngood, cc_time: same to the input parameters but with abnormal cross-correaltions removed
    allstacks1: 1D matrix of stacked cross-correlation functions over all the segments
    nstacks:    number of overall segments for the final stacks
    '''
    # load useful parameters from dict
    samp_freq = stack_para['samp_freq']
    smethod = stack_para['stack_method']
    rma_substack = stack_para['rma_substack']
    rma_step = stack_para['rma_step']
    start_date = stack_para['start_date']
    end_date = stack_para['end_date']
    npts = cc_array.shape[1]

    # remove abnormal data
    ampmax = np.max(cc_array, axis=1)
    tindx = np.where((ampmax < 20 * np.median(ampmax)) & (ampmax > 0))[0]
    if not len(tindx):
        allstacks1 = []
        allstacks2 = []
        nstacks = 0
        cc_array = []
        cc_ngood = []
        cc_time = []
        return cc_array, cc_ngood, cc_time, allstacks1, allstacks2, nstacks
    else:

        # remove ones with bad amplitude
        cc_array = cc_array[tindx, :]
        cc_time = cc_time[tindx]
        cc_ngood = cc_ngood[tindx]

        # do substacks
        if rma_substack:
            tstart = obspy.UTCDateTime(start_date) - obspy.UTCDateTime(1970, 1, 1)
            tend = obspy.UTCDateTime(end_date) - obspy.UTCDateTime(1970, 1, 1)
            ttime = tstart
            nstack = int(np.round((tend - tstart) / (rma_step * 3600)))
            ncc_array = np.zeros(shape=(nstack, npts), dtype=np.float32)
            ncc_time = np.zeros(nstack, dtype=np.float)
            ncc_ngood = np.zeros(nstack, dtype=np.int)

            # loop through each time
            for ii in range(nstack):
                sindx = np.where((cc_time >= ttime) & (cc_time < ttime + rma_substack * 3600))[0]

                # when there are data in the time window
                if len(sindx):
                    ncc_array[ii] = np.mean(cc_array[sindx], axis=0)
                    ncc_time[ii] = ttime
                    ncc_ngood[ii] = np.sum(cc_ngood[sindx], axis=0)
                ttime += rma_step * 3600

            # remove bad ones
            tindx = np.where(ncc_ngood > 0)[0]
            ncc_array = ncc_array[tindx]
            ncc_time = ncc_time[tindx]
            ncc_ngood = ncc_ngood[tindx]

        # do stacking
        allstacks1 = np.zeros(npts, dtype=np.float32)
        allstacks2 = np.zeros(npts, dtype=np.float32)
        allstacks3 = np.zeros(npts, dtype=np.float32)
        allstacks4 = np.zeros(npts, dtype=np.float32)

        if smethod == 'linear':
            allstacks1 = np.mean(cc_array, axis=0)
        elif smethod == 'pws':
            allstacks1 = pws(cc_array, samp_freq)
        elif smethod == 'robust':
            allstacks1, w, = robust_stack(cc_array, 0.001)
        elif smethod == 'selective':
            allstacks1 = selective_stack(cc_array, epsilon=0.001, cc_th=0.7)
        elif smethod == 'all':
            allstacks1 = np.mean(cc_array, axis=0)
            allstacks2 = pws(cc_array, samp_freq)
            allstacks3 = robust_stack(cc_array, 0.001)
            allstacks4 = selective_stack(cc_array, epsilon=0.001, cc_th=0.7)
        nstacks = np.sum(cc_ngood)

    # replace the array for substacks
    if rma_substack:
        cc_array = ncc_array
        cc_time = ncc_time
        cc_ngood = ncc_ngood

    # good to return
    return cc_array, cc_ngood, cc_time, allstacks1, allstacks2, allstacks3, allstacks4, nstacks


# Stacking methods
def pws(arr, sampling_rate, power=2, pws_timegate=5.):
    '''
    Performs phase-weighted stack on array of time series. Modified on the noise function by Tim Climents.
    Follows methods of Schimmel and Paulssen, 1997.
    If s(t) is time series data (seismogram, or cross-correlation),
    S(t) = s(t) + i*H(s(t)), where H(s(t)) is Hilbert transform of s(t)
    S(t) = s(t) + i*H(s(t)) = A(t)*exp(i*phi(t)), where
    A(t) is envelope of s(t) and phi(t) is phase of s(t)
    Phase-weighted stack, g(t), is then:
    g(t) = 1/N sum j = 1:N s_j(t) * | 1/N sum k = 1:N exp[i * phi_k(t)]|^v
    where N is number of traces used, v is sharpness of phase-weighted stack

    PARAMETERS:
    ---------------------
    arr: N length array of time series data (numpy.ndarray)
    sampling_rate: sampling rate of time series arr (int)
    power: exponent for phase stack (int)
    pws_timegate: number of seconds to smooth phase stack (float)

    RETURNS:
    ---------------------
    weighted: Phase weighted stack of time series data (numpy.ndarray)
    '''

    if arr.ndim == 1:
        print('2D matrix is needed for pws')
        return arr
    N, M = arr.shape

    # construct analytical signal
    analytic = hilbert(arr, axis=1, N=next_fast_len(M))[:, :M]
    phase = np.angle(analytic)
    phase_stack = np.mean(np.exp(1j * phase), axis=0)
    phase_stack = np.abs(phase_stack) ** (power)

    # smoothing
    # timegate_samples = int(pws_timegate * sampling_rate)
    # phase_stack = moving_ave(phase_stack,timegate_samples)

    # weighted is the final waveforms
    weighted = np.multiply(arr, phase_stack)
    return np.mean(weighted, axis=0)


def robust_stack(cc_array, epsilon):
    """
    this is a robust stacking algorithm described in Palvis and Vernon 2010

    PARAMETERS:
    ----------------------
    cc_array: numpy.ndarray contains the 2D cross correlation matrix
    epsilon: residual threhold to quit the iteration
    RETURNS:
    ----------------------
    newstack: numpy vector contains the stacked cross correlation

    Written by Marine Denolle
    """
    res = 9E9  # residuals
    w = np.ones(cc_array.shape[0])
    nstep = 0
    newstack = np.median(cc_array, axis=0)
    while res > epsilon:
        stack = newstack
        for i in range(cc_array.shape[0]):
            crap = np.multiply(stack, cc_array[i, :].T)
            crap_dot = np.sum(crap)
            di_norm = np.linalg.norm(cc_array[i, :])
            ri = cc_array[i, :] - crap_dot * stack
            ri_norm = np.linalg.norm(ri)
            w[i] = np.abs(crap_dot) / di_norm / ri_norm  # /len(cc_array[:,1])
        # print(w)
        w = w / np.sum(w)
        newstack = np.sum((w * cc_array.T).T, axis=0)  # /len(cc_array[:,1])
        res = np.linalg.norm(newstack - stack, ord=1) / np.linalg.norm(newstack) / len(cc_array[:, 1])
        nstep += 1
        if nstep > 10:
            return newstack, w, nstep
    return newstack, w, nstep


def adaptive_filter(arr, g):
    '''
    the adaptive covariance filter to enhance coherent signals. Fellows the method of
    Nakata et al., 2015 (Appendix B)

    the filtered signal [x1] is given by x1 = ifft(P*x1(w)) where x1 is the ffted spectra
    and P is the filter. P is constructed by using the temporal covariance matrix.

    PARAMETERS:
    ----------------------
    arr: numpy.ndarray contains the 2D traces of daily/hourly cross-correlation functions
    g: a positive number to adjust the filter harshness
    RETURNS:
    ----------------------
    narr: numpy vector contains the stacked cross correlation function
    '''
    if arr.ndim == 1:
        print('2D matrix is needed for adaptive filtering')
        return arr
    N, M = arr.shape
    Nfft = next_fast_len(M)

    # fft the 2D array
    spec = scipy.fftpack.fft(arr, axis=1, n=Nfft)[:, :M]

    # make cross-spectrm matrix
    cspec = np.zeros(shape=(N * N, M), dtype=np.complex64)
    for ii in range(N):
        for jj in range(N):
            kk = ii * N + jj
            cspec[kk] = spec[ii] * np.conjugate(spec[jj])

    S1 = np.zeros(M, dtype=np.complex64)
    S2 = np.zeros(M, dtype=np.complex64)
    # construct the filter P
    for ii in range(N):
        mm = ii * N + ii
        S2 += cspec[mm]
        for jj in range(N):
            kk = ii * N + jj
            S1 += cspec[kk]

    p = np.power((S1 - S2) / (S2 * (N - 1)), g)

    # make ifft
    narr = np.real(scipy.fftpack.ifft(np.multiply(p, spec), Nfft, axis=1)[:, :M])
    return np.mean(narr, axis=0)


def nroot_stack(cc_array, power):
    '''
    this is nth-root stacking algorithm translated based on the matlab function
    from https://github.com/xtyangpsp/SeisStack (by Xiaotao Yang; follows the
    reference of Millet, F et al., 2019 JGR)

    Parameters:
    ------------
    cc_array: numpy.ndarray contains the 2D cross correlation matrix
    power: np.int, nth root for the stacking

    Returns:
    ------------
    nstack: np.ndarray, final stacked waveforms

    Written by Chengxin Jiang @ANU (May2020)
    '''
    if cc_array.ndim == 1:
        Logger.error('2D matrix is needed for nroot_stack')
        return cc_array
    N, M = cc_array.shape
    dout = np.zeros(M, dtype=np.float32)

    # construct y
    for ii in range(N):
        dat = cc_array[ii, :]
        dout += np.sign(dat) * np.abs(dat) ** (1 / power)
    dout /= N

    # the final stacked waveform
    # nstack = np.sign(dout) * np.abs(dout) ** (power - 1)
    nstack = dout * np.abs(dout) ** (power - 1)

    return nstack


def selective_stack(cc_array, epsilon, cc_th=0.8):
    '''
    this is a selective stacking algorithm developed by Jared Bryan/Kurama Okubo.

    PARAMETERS:
    ----------------------
    cc_array: numpy.ndarray contains the 2D cross correlation matrix
    epsilon: residual threhold to quit the iteration
    cc_th: numpy.float, threshold of correlation coefficient to be selected

    RETURNS:
    ----------------------
    newstack: numpy vector contains the stacked cross correlation
    nstep: np.int, total number of iterations for the stacking

    Originally written by Marine Denolle
    Modified by Chengxin Jiang @Harvard (Oct2020)
    '''
    if cc_array.ndim == 1:
        Logger.error('2D matrix is needed for nroot_stack')
        return cc_array
    N, M = cc_array.shape

    res = 9E9  # residuals
    cof = np.zeros(N, dtype=np.float32)
    newstack = np.mean(cc_array, axis=0)

    nstep = 0
    # start iteration
    while res > epsilon:
        for ii in range(N):
            cof[ii] = np.corrcoef(newstack, cc_array[ii, :])[0, 1]

        # find good waveforms
        indx = np.where(cof >= cc_th)[0]
        if not len(indx): raise ValueError('cannot find good waveforms inside selective stacking')
        oldstack = newstack
        newstack = np.mean(cc_array[indx], axis=0)
        res = np.linalg.norm(newstack - oldstack) / (np.linalg.norm(newstack) * M)
        nstep += 1

    return newstack, nstep


# Rotation
def rotation(bigstack, parameters, locs):
    '''
    this function transfers the Green's tensor from a E-N-Z system into a R-T-Z one

    PARAMETERS:
    -------------------
    bigstack:   9 component Green's tensor in E-N-Z system.
                order: ['EE', 'EN', 'EZ', 'NE', 'NN', 'NZ', 'ZE', 'ZN', 'ZZ']
    parameters: dict containing all parameters saved in ASDF file
    locs:       dict containing station angle info for correction purpose
    RETURNS:
    -------------------
    tcorr: 9 component Green's tensor in R-T-Z system
    '''
    # load parameter dic
    pi = np.pi
    azi = parameters['azi']
    baz = parameters['baz']
    ncomp, npts = bigstack.shape
    if ncomp < 9:
        Logger.warning('crap did not get enough components')
        tcorr = []
        return tcorr

    if len(locs):
        staS = parameters['station_source']
        staR = parameters['station_receiver']
        sta_list = list(locs['station'])
        angles = list(locs['angle'])
        # get station info from the name of ASDF file
        ind = sta_list.index(staS)
        acorr = angles[ind]
        ind = sta_list.index(staR)
        bcorr = angles[ind]

    # ---angles to be corrected----
    if len(locs):
        cosa = np.cos((azi + acorr) * pi / 180)
        sina = np.sin((azi + acorr) * pi / 180)
        cosb = np.cos((baz + bcorr) * pi / 180)
        sinb = np.sin((baz + bcorr) * pi / 180)
    else:
        cosa = np.cos(azi * pi / 180)
        sina = np.sin(azi * pi / 180)
        cosb = np.cos(baz * pi / 180)
        sinb = np.sin(baz * pi / 180)

    # rtz_components = ['ZR','ZT','ZZ','RR','RT','RZ','TR','TT','TZ']
    tcorr = np.zeros(shape=(9, npts), dtype=np.float32)
    tcorr[0] = -cosb * bigstack[7] - sinb * bigstack[6]
    tcorr[1] = sinb * bigstack[7] - cosb * bigstack[6]
    tcorr[2] = bigstack[8]
    tcorr[3] = -cosa * cosb * bigstack[4] - cosa * sinb * bigstack[3] - sina * cosb * bigstack[1] - sina * sinb * \
               bigstack[0]
    tcorr[4] = cosa * sinb * bigstack[4] - cosa * cosb * bigstack[3] + sina * sinb * bigstack[1] - sina * cosb * \
               bigstack[0]
    tcorr[5] = cosa * bigstack[5] + sina * bigstack[2]
    tcorr[6] = sina * cosb * bigstack[4] + sina * sinb * bigstack[3] - cosa * cosb * bigstack[1] - cosa * sinb * \
               bigstack[0]
    tcorr[7] = -sina * sinb * bigstack[4] + sina * cosb * bigstack[3] + cosa * sinb * bigstack[1] - cosa * cosb * \
               bigstack[0]
    tcorr[8] = -sina * bigstack[5] + cosa * bigstack[2]

    return tcorr
