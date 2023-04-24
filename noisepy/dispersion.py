""" DISPERSION PICKING FUNCTIONS"""
import numpy as np
import pycwt
from findpeaks import findpeaks  # https://github.com/erdogant/findpeaks
import scipy
import matplotlib.pyplot as plt
from scipy import fft
from scipy import interpolate
from scipy.signal import hilbert
import logging
Logger = logging.getLogger(__name__)

def get_disp_image(ccf, dist, dt, Tmin=0.4, dT=0.02, vmin=0.1, vmax=4.5, dvel=0.02, plot=True, figsize=(14,6)):
    """ Get dispersion image wtih CWT """

    # basic parameters for wavelet transform
    dj = 1 / 12  # Spacing between discrete scales. Default is Twelve sub-octaves per octaves.
    # Smaller values will result in better scale resolution, but slower calculation and plot.
    s0 = -1  # Smallest scale of the wavelet. Default value [-1] is 2*dt.
    J = -1   # Number of scales less one.
    # Scales range from s0 up to s0 * 2**(J * dj), which gives a total of (J + 1) scales.
    # Default [-1] is J = (log2(N * dt / so)) / dj.
    wvn = 'morlet'  # type of wavelet to use

    # Get period and velocity ranges
    Tmax = dist / 3.0
    fmin = 1 / Tmax
    fmax = 1 / Tmin
    per = np.arange(Tmin, Tmax, dT)
    vel = np.arange(vmin, vmax, dvel)

    # trim the data according to velocity window vmin-vmax
    npts = ccf.shape[0]
    pt1 = int(dist / vmax / dt)
    pt2 = int(dist / vmin / dt)
    if pt1 == 0:
        pt1 = 10
    if pt2 > (npts // 2):
        pt2 = npts // 2
    indx = np.arange(pt1, pt2)
    tvec = indx * dt
    ccf = ccf[indx]

    # wavelet transformation
    cwt, sj, freq, coi, _, _ = pycwt.cwt(ccf, dt, dj, s0, J, wvn)

    # do filtering here
    if (fmax > np.max(freq)) | (fmax <= fmin):
        raise ValueError('Abort: frequency out of limits!')
    freq_ind = np.where((freq >= fmin) & (freq <= fmax))[0]
    cwt = cwt[freq_ind]
    freq = freq[freq_ind]

    # use amplitude of the cwt
    period = 1 / freq
    rcwt, pcwt = np.abs(cwt) ** 2, np.angle(cwt)

    # interpolation to grids of freq-vel
    fc = scipy.interpolate.interp2d(dist / tvec, period, rcwt)
    rcwt_new = fc(vel, per)

    # do normalization for each frequency
    for ii in range(len(per)):
        rcwt_new[ii] /= np.max(rcwt_new[ii])

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(np.transpose(rcwt_new),
                  cmap='jet',
                  extent=[per[0], per[-1], vel[0], vel[-1]],
                  aspect='auto',
                  origin='lower')
        ax.set(xlabel='Period [s]', ylabel='Vg [km/s]', xlim=(Tmin, Tmax), ylim=(vmin, vmax))
        ax.set_title('Inter-station distance: %5.2f km' % dist)
        plt.tight_layout()
        plt.show()
        plt.close()

    return rcwt_new, per, vel


# function to extract the dispersion from the image
def extract_dispersion(amp, per, vel, dist, vmax=5., maxgap=3, minlambda=1.5):
    '''
    this function takes the dispersion image from CWT as input, tracks the global maxinum on
    the wavelet spectrum amplitude and extract the sections with continous and high quality data

    PARAMETERS:
    ----------------
    amp: 2D amplitude matrix of the wavelet spectrum
    phase: 2D phase matrix of the wavelet spectrum
    per:  period vector for the 2D matrix
    vel:  vel vector of the 2D matrix
    maxgap: default 5
    minlambda: minimum multiple of wavelength
    RETURNS:
    ----------------
    per:  central frequency of each wavelet scale with good data
    gv:   group velocity vector at each frequency
    ampsnr: max over median amplitude of dispersion diagram at pick time
    '''
    nper = amp.shape[0]
    gv = np.zeros(nper, dtype=np.float32)
    ampsnr = np.zeros(nper, dtype=np.float32)
    dvel = vel[1] - vel[0]

    # find global maximum
    for ii in range(nper):
        if per[ii] == 0: continue
        maxvalue = np.max(amp[ii], axis=0)
        indx = list(amp[ii]).index(maxvalue)
        gv[ii] = vel[indx]
        ampsnr[ii] = maxvalue / np.median(amp[ii], axis=0)
        # QC:
        if np.abs(gv[ii] - vmax) < 3 * dvel:  # remove points close to vg limits
            gv[ii] = 0
        elif dist / (per[ii] * gv[ii]) < minlambda:
            gv[ii] = 0

    # check the continuous of the dispersion
    for ii in range(1, nper - 15):
        # 15 is the minimum length needed for output
        if gv[ii] == 0: continue
        for jj in range(15):
            if np.abs(gv[ii + jj] - gv[ii + 1 + jj]) > maxgap * dvel:
                gv[ii] = 0
                break

    # remove the bad ones
    indx = np.where(gv > 0)[0]

    pick_per = per[indx]
    pick_gv = gv[indx]
    pick_ampsnr = ampsnr[indx]

    # Check if there are outliers (points alone with big gaps with gv before and after
    igood = list(range(len(pick_per)))
    for ii in range(2, len(pick_per) - 1):
        if pick_gv[ii] > pick_gv[ii - 1] + (pick_gv[ii + 1] - pick_gv[ii - 1]):
            igood.remove(ii)
    pick_per = pick_per[igood]
    pick_gv = pick_gv[igood]
    pick_ampsnr = pick_ampsnr[igood]

    #     return per[indx],gv[indx],ampsnr[indx]
    return pick_per, pick_gv, pick_ampsnr


def extract_dispersion_simple(amp, per, vel):
    '''
    this function takes the dispersion image from CWT as input, tracks the global maxinum on
    the wavelet spectrum amplitude and extract the sections with continous and high quality data

    PARAMETERS:
    ----------------
    amp: 2D amplitude matrix of the wavelet spectrum
    phase: 2D phase matrix of the wavelet spectrum
    per:  period vector for the 2D matrix
    vel:  vel vector of the 2D matrix
    RETURNS:
    ----------------
    per:  central frequency of each wavelet scale with good data
    gv:   group velocity vector at each frequency
    '''
    maxgap = 5
    nper = amp.shape[0]
    gv = np.zeros(nper, dtype=np.float32)
    dvel = vel[1] - vel[0]

    # find global maximum
    for ii in range(nper):
        maxvalue = np.max(amp[ii], axis=0)
        indx = list(amp[ii]).index(maxvalue)
        gv[ii] = vel[indx]

    # check the continuous of the dispersion
    for ii in range(1, nper - 15):
        # 15 is the minumum length needed for output
        for jj in range(15):
            if np.abs(gv[ii + jj] - gv[ii + 1 + jj]) > maxgap * dvel:
                gv[ii] = 0
                break

    # remove the bad ones
    indx = np.where(gv > 0)[0]

    return per[indx], gv[indx]


def extract_curves_topology(amp, per, vel, limit=0.1):
    # Get peak for each period
    fp = findpeaks(method='topology', verbose=0, limit=limit)
    peaks = []
    for iT in range(amp.shape[0]):
        X = amp[iT, :]
        results = fp.fit(X)
        imax = results["persistence"]["y"]
        scores = results["persistence"]["score"]
        for p, score in zip(imax, scores):
            if p == 0 or p == amp.shape[1] - 1:  # Skip pick at edge of image
                continue
            peaks.append((per[iT], vel[p], score))

    pick_vel = [tup[1] for tup in peaks]
    pick_per = [tup[0] for tup in peaks]
    pick_sco = [tup[2] for tup in peaks]

    return pick_per, pick_vel, pick_sco


def nb_filt_gauss(ccf, dt, fn_array, dist, alpha=5, vmin=0.5, vmax=4.5):
    """
    Narrowband Gaussian filtering to get SNR at each frequency
    Args:
        ccf: Cross-correlation function
        dt: sampling interval [s]
        fn_array: Numpy array of frequencies
        dist: distance between stations [km]
        alpha: Gaussian window parameter
        vmin: Minimum group velocity to determine signal window
        vmax: Maximum group velocity to determine signal window

    Returns:

    """
    # Define signal and noise windows
    signal_win = np.arange(int(dist / vmax / dt), int(dist / vmin / dt))
    noise_istart = len(ccf) - 2 * len(signal_win)
    noise_win = np.arange(noise_istart, noise_istart + len(signal_win))
    noise_rms = np.sqrt(np.sum(ccf[noise_win] ** 2) / len(noise_win))
    snr_bb = np.max(np.abs(ccf[signal_win])) / noise_rms  # broadband snr

    # Narrowband filtering with Gaussian
    omgn_array = 2 * np.pi * fn_array

    # Transform ccf to frequency domain
    Nfft = fft.next_fast_len(len(ccf))
    ccf_freq = fft.fft(ccf, n=Nfft)
    freq_samp = 2 * np.pi * abs(fft.fftfreq(Nfft, dt))

    # Narrowband filtering
    # ccf_time_nbG = np.zeros(shape=(len(omgn_array), len(ccf)), dtype=np.float32)
    # ccf_time_nbG_env = np.zeros(shape=(len(omgn_array), len(ccf)), dtype=np.float32)
    snr_nbG = np.zeros(shape=(len(omgn_array),), dtype=np.float32)
    for iomgn, omgn in enumerate(omgn_array):
        # Gaussian kernel
        GaussFilt = np.exp(-alpha * ((freq_samp - omgn) / omgn) ** 2)

        # Apply filter
        ccf_freq_nbG = ccf_freq * GaussFilt
        tmp = fft.ifft(ccf_freq_nbG, n=Nfft).real

        # Transform to the time domain
        ccftnbg = tmp[:len(ccf)]
        # ccf_time_nbG[iomgn, :] = ccftnbg

        # Get envelope
        analytic_signal = hilbert(ccftnbg)
        amplitude_envelope = np.abs(analytic_signal)
        # ccf_time_nbG_env[iomgn, :] = amplitude_envelope

        # SNR
        # check if max is at edge of lag time limits
        isnr = np.argmax(amplitude_envelope)
        if isnr == 0 or isnr == len(amplitude_envelope) - 1:
            snr_nbG[iomgn] = 0
        else:
            noise_rms = np.sqrt(np.sum(ccftnbg[noise_win] ** 2) / len(noise_win))
            snr_nbG[iomgn] = np.max(ccftnbg[signal_win]) / noise_rms

    return snr_nbG, snr_bb  # ccf_time_nbG , ccf_time_nbG_env, snr_nbG
