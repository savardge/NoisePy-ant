"""
DISPERSION PICKING FUNCTIONS
Code modified from the older version of the NoisePy repository https://github.com/noisepy/NoisePy
"""
import numpy as np
import pycwt
from findpeaks import findpeaks  # https://github.com/erdogant/findpeaks
import scipy
import matplotlib.pyplot as plt
from scipy import fft
from scipy import interpolate
from scipy.signal import hilbert
import logging
from matplotlib.ticker import AutoMinorLocator
from obspy.imaging.cm import pqlx
from scipy.signal.windows import tukey as tukey_window


Logger = logging.getLogger(__name__)


def get_disp_image(ccf, dist, dt, Tmin=0.4, dT=0.02, vmin=0.1, vmax=4.5, dvel=0.02, vave=3., plot=True, figsize=(14, 6)):
    '''
    Get the group dispersion image with the Continuous Wavelet Transform
    Args:
        ccf: Cross-correlation function (symmetric)
        dist: inter-station distance for the given ccf in km
        dt: Sampling interval in second
        Tmin: Minimum period
        dT: Spacing of the period axis on the dispersion image in second
        vmin: Minimum group velocity
        vmax: Maximum group velocity
        dvel: Spacing of the group velocity axis in km/s
        vave: average velocity in km/s used to calculate Tmax = dist/ vave
        plot: Whether to plot or not (bool)
        figsize: Figure size if plotting

    Returns:
        rcwt_new: Group dispersion image (2D numpy array)
        per, vel: corresponding period and group velocity vectors
    '''

    # Basic parameters for wavelet transform
    dj = 1 / 12  # Spacing between discrete scales. Default is Twelve sub-octaves per octaves.
    # Smaller values will result in better scale resolution, but slower calculation and plot.
    s0 = -1  # Smallest scale of the wavelet. Default value [-1] is 2*dt.
    J = -1  # Number of scales less one.
    # Scales range from s0 up to s0 * 2**(J * dj), which gives a total of (J + 1) scales.
    # Default [-1] is J = (log2(N * dt / so)) / dj.
    wvn = 'morlet'  # type of wavelet to use

    # Get period and velocity ranges
    Tmax = dist / vave # Max period assumes a velocity of 3 km/s
    fmin = 1 / Tmax
    fmax = 1 / Tmin
    per = np.arange(Tmin, Tmax, dT)  # Periods
    vel = np.arange(vmin, vmax, dvel)  # Group velocities

    # Trim the CCF according to velocity window vmin-vmax
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

    # Filter the image within the requested frequency band
    if (fmax > np.max(freq)) | (fmax <= fmin):
        raise ValueError('Abort: frequency out of limits!')
    freq_ind = np.where((freq >= fmin) & (freq <= fmax))[0]
    cwt = cwt[freq_ind]
    freq = freq[freq_ind]

    # Calculate the amplitude and phase of the cwt
    period = 1 / freq
    rcwt = np.abs(cwt) ** 2  # Amplitude
    pcwt = np.angle(cwt)  # Phase

    # Interpolation of the image to the requested intervals in period and velocity
    fc = scipy.interpolate.interp2d(dist / tvec, period, rcwt)
    rcwt_new = fc(vel, per)

    # Normalization amplitude at each frequency
    for ii in range(len(per)):
        rcwt_new[ii] /= np.max(rcwt_new[ii])

    # Plot
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


def get_disp_image_taper(ccf, dist, dt, Tmin=0.4, dT=0.02, vmin=0.1, vmax=4.5, dvel=0.02, vave=3., plot=True,
                         figsize=(14, 6)):
    '''
    Get the group dispersion image with the Continuous Wavelet Transform
    Args:
        ccf: Cross-correlation function (symmetric)
        dist: inter-station distance for the given ccf in km
        dt: Sampling interval in second
        Tmin: Minimum period
        dT: Spacing of the period axis on the dispersion image in second
        vmin: Minimum group velocity
        vmax: Maximum group velocity
        dvel: Spacing of the group velocity axis in km/s
        vave: average velocity in km/s used to calculate Tmax = dist/ vave
        plot: Whether to plot or not (bool)
        figsize: Figure size if plotting

    Returns:
        rcwt_new: Group dispersion image (2D numpy array)
        per, vel: corresponding period and group velocity vectors
    '''

    # Basic parameters for wavelet transform
    dj = 1 / 12  # Spacing between discrete scales. Default is Twelve sub-octaves per octaves.
    # Smaller values will result in better scale resolution, but slower calculation and plot.
    s0 = -1  # Smallest scale of the wavelet. Default value [-1] is 2*dt.
    J = -1  # Number of scales less one.
    # Scales range from s0 up to s0 * 2**(J * dj), which gives a total of (J + 1) scales.
    # Default [-1] is J = (log2(N * dt / so)) / dj.
    wvn = 'morlet'  # type of wavelet to use

    # Get period and velocity ranges
    Tmax = dist / vave  # Max period assumes a velocity of 3 km/s
    fmin = 1 / Tmax
    fmax = 1 / Tmin
    per = np.arange(Tmin, Tmax, dT)  # Periods
    vel = np.arange(vmin, vmax, dvel)  # Group velocities

    # Trim the CCF according to velocity window vmin-vmax
    npts = ccf.shape[0]
    tvec = np.arange(0, npts) * dt
    pt1 = int(dist / vmax / dt)
    pt2 = int(dist / vmin / dt)
    if pt1 == 0:
        pt1 = 10
    if pt2 > (npts // 2):
        pt2 = npts // 2
    indx = np.arange(pt1, pt2)

    # Taper the window
    taper = np.zeros(shape=npts)
    window = tukey_window(len(indx), alpha=0.05, sym=True)
    taper[indx] = window
    ccf *= taper
    # Cut the part after the taper (to speed up calculation)
    ccf = ccf[:pt2]
    tvec = tvec[:pt2]

    # wavelet transformation
    cwt, sj, freq, coi, _, _ = pycwt.cwt(ccf, dt, dj, s0, J, wvn)

    # Filter the image within the requested frequency band
    if (fmax > np.max(freq)) | (fmax <= fmin):
        raise ValueError('Abort: frequency out of limits!')
    freq_ind = np.where((freq >= fmin) & (freq <= fmax))[0]
    cwt = cwt[freq_ind]
    freq = freq[freq_ind]

    # Calculate the amplitude and phase of the cwt
    period = 1 / freq
    rcwt = np.abs(cwt) ** 2  # Amplitude
    # pcwt = np.angle(cwt)  # Phase

    # Remove t=0 sample
    tvec = tvec[1:]
    rcwt = rcwt[:, 1:]
    coi = coi[1:]

    # Interpolation of the image to the requested intervals in period and velocity
    velocity = dist / tvec
    fc = scipy.interpolate.interp2d(velocity, period, rcwt)
    rcwt_new = fc(vel, per)

    # Interpolation of coi
    ff = scipy.interpolate.interp1d(velocity, coi, fill_value='extrapolate', assume_sorted=False)
    coi_new = ff(vel)

    # Normalization amplitude at each frequency
    rcwt_new /= np.max(rcwt_new, axis=1)[:, np.newaxis]

    # Plot
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(np.transpose(rcwt_new),
                  cmap='jet',
                  extent=[per[0], per[-1], vel[0], vel[-1]],
                  aspect='auto',
                  origin='lower')
        ax.scatter(coi_new, vel, c="k", s=5)
        ax.set(xlabel='Period [s]', ylabel='Vg [km/s]', xlim=(Tmin, Tmax), ylim=(vmin, vmax))
        ax.set_title('Inter-station distance: %5.2f km' % dist)
        plt.tight_layout()
        plt.show()
        plt.close()

    return rcwt_new, per, vel, coi_new

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
    maxgap: Maximum gap in group velocity between successive picks in samples
    minlambda: minimum multiple of wavelength (default 1.5 wavelength required)

    RETURNS:
    ----------------
    per:  central frequency of each wavelet scale with good data
    gv:   group velocity vector at each frequency
    ampsnr: max over median amplitude of dispersion diagram at pick time
    '''
    nper = amp.shape[0]  # Number of period samples
    gv = np.zeros(nper, dtype=np.float32)  # Group velocity
    ampsnr = np.zeros(nper, dtype=np.float32)  # SNR
    dvel = vel[1] - vel[0]  # Group velocity spacing
    minimum_output_length = 15  # The minimum length of the curve needed for output in samples

    # Find global maximum at each period
    for ii in range(nper):
        if per[ii] == 0:
            continue
        maxvalue = np.max(amp[ii], axis=0)
        indx = list(amp[ii]).index(maxvalue)
        gv[ii] = vel[indx]
        ampsnr[ii] = maxvalue / np.median(amp[ii], axis=0)

        # QC:
        if np.abs(gv[ii] - vmax) < 3 * dvel:  # remove points close to vg limits (within 3 samples of the max)
            gv[ii] = 0
        elif dist / (per[ii] * gv[ii]) < minlambda:  # remove if number of wavelengths is less than threshold
            gv[ii] = 0

    # Check the continuity of the dispersion curve
    for ii in range(1, nper - minimum_output_length):
        if gv[ii] == 0:
            continue
        for jj in range(minimum_output_length):
            if np.abs(gv[ii + jj] - gv[ii + 1 + jj]) > maxgap * dvel:  # If there is a large velocity gap, set to 0
                gv[ii] = 0
                break

    # Remove discarded picks set to 0
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
    minimum_output_length = 15  # The minimum length of the curve needed for output
    maxgap = 5  # Maximum gap in group velocity between successive picks in samples
    nper = amp.shape[0]
    gv = np.zeros(nper, dtype=np.float32)
    dvel = vel[1] - vel[0]

    # Find global maximum at each period
    for ii in range(nper):
        maxvalue = np.max(amp[ii], axis=0)
        indx = list(amp[ii]).index(maxvalue)
        gv[ii] = vel[indx]

    # Check continuity of the dispersion curve
    for ii in range(1, nper - minimum_output_length):
        for jj in range(minimum_output_length):
            if np.abs(gv[ii + jj] - gv[ii + 1 + jj]) > maxgap * dvel:
                gv[ii] = 0
                break

    # remove the bad ones
    indx = np.where(gv > 0)[0]

    return per[indx], gv[indx]


def extract_curves_topology(amp, per, vel, limit=0.1):
    """
    Pick dispersion curves using the topology method (c.f. https://github.com/erdogant/findpeaks)
    Args:
        amp: FTAN image
        per: periods
        vel: velocities
        limit: Minimum score

    Returns:
        pick_per, pick_vel: 1D arrays with the picked period and group velocity
        pick_sco: Corresponding peak score, between 0 (not a peak) and 1 (most significant peak)

    """
    # Get peak for each period
    fp = findpeaks(method='topology', verbose=0, limit=limit)
    peaks = []
    for iT in range(amp.shape[0]):
        X = amp[iT, :]
        try:
            results = fp.fit(X)
            imax = results["persistence"]["y"]
            scores = results["persistence"]["score"]
            for p, score in zip(imax, scores):
                if p == 0 or p == amp.shape[1] - 1:  # Skip pick at edge of image
                    continue
                peaks.append((per[iT], vel[p], score))
        except:
            pass

    pick_vel = [tup[1] for tup in peaks]
    pick_per = [tup[0] for tup in peaks]
    pick_sco = [tup[2] for tup in peaks]

    return pick_per, pick_vel, pick_sco


def remove_picks_coi(pick_per, pick_vel, pick_sco, vel, coi):
    """
    Remove picks inside cone of influence (coi) of CWT image
    Args:
        pick_per: period vector of picks
        pick_vel: velocity vector of picks
        pick_sco: persistence score of picks
        vel: velocity vector corresponding to coi vector
        coi: cone of influence vector

    Returns:
        Cleaned up picks
        periods, velocities, score
    """
    ibad = []
    for ii in range(len(pick_vel)):
        ix = np.argwhere(vel == pick_vel[ii])[0][0]
        if pick_per[ii] > coi[ix]:
            ibad.append(ii)
    pick_per_f = np.delete(pick_per, ibad)
    pick_vel_f = np.delete(pick_vel, ibad)
    pick_sco_f = np.delete(pick_sco, ibad)
    if ibad:
        maxT_coi = min([p for ii, p in enumerate(pick_per) if ii in ibad])
        ibad2 = np.argwhere(pick_per_f > maxT_coi).flatten()
        pick_per_f = np.delete(pick_per_f, ibad2)
        pick_vel_f = np.delete(pick_vel_f, ibad2)
        pick_sco_f = np.delete(pick_sco_f, ibad2)
    return pick_per_f, pick_vel_f, pick_sco_f

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
        snr_nbG: SNR array for the CCF filtered at each frequency of fn_array
        snr_bb: SNR value calculated without filtering (broadband CCF)
        ccf_time_nbG: 2D array of CCF narrowband-filtered at each frequency of fn_array
        ccf_time_nbG_env: 2D array of the envelope of the CCF narrowband-filtered at each frequency of fn_array
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
    ccf_time_nbG = np.zeros(shape=(len(omgn_array), len(ccf)), dtype=np.float32)
    ccf_time_nbG_env = np.zeros(shape=(len(omgn_array), len(ccf)), dtype=np.float32)
    snr_nbG = np.zeros(shape=(len(omgn_array),), dtype=np.float32)
    for iomgn, omgn in enumerate(omgn_array):
        # Gaussian kernel
        GaussFilt = np.exp(-alpha * ((freq_samp - omgn) / omgn) ** 2)

        # Apply filter
        ccf_freq_nbG = ccf_freq * GaussFilt
        tmp = fft.ifft(ccf_freq_nbG, n=Nfft).real

        # Transform to the time domain
        ccftnbg = tmp[:len(ccf)]
        ccf_time_nbG[iomgn, :] = ccftnbg

        # Get envelope
        analytic_signal = hilbert(ccftnbg)
        amplitude_envelope = np.abs(analytic_signal)
        ccf_time_nbG_env[iomgn, :] = amplitude_envelope

        # SNR
        # check if max is at edge of lag time limits
        # isnr = np.argmax(amplitude_envelope)
        # if isnr == 0 or isnr == len(amplitude_envelope) - 1:
        #    snr_nbG[iomgn] = 0
        # else:
        #    noise_rms = np.sqrt(np.sum(ccftnbg[noise_win] ** 2) / len(noise_win))
        #    snr_nbG[iomgn] = np.max(ccftnbg[signal_win]) / noise_rms
        noise_rms = np.sqrt(np.sum(amplitude_envelope[noise_win] ** 2) / len(noise_win))
        snr_nbG[iomgn] = np.max(amplitude_envelope[signal_win]) / noise_rms

    return snr_nbG, snr_bb, ccf_time_nbG, ccf_time_nbG_env


def get_mean(inst_periods, group_velocity):
    """
    Get mean and standard deviation of group velocity picks along periods
    Args:
        inst_periods: Numpy array of instantaneous periods
        group_velocity: Numpy array of group velocity

    Returns:
        Mean dispersion curve:
        period, mean group velocity, standard deviation

    """
    inst_periods_uniq = np.unique(inst_periods)
    gv_moy = np.zeros(shape=(len(inst_periods_uniq),))
    gv_std = np.zeros(shape=(len(inst_periods_uniq),))
    for iper, per in enumerate(inst_periods_uniq):
        gv_moy[iper] = np.mean(group_velocity[inst_periods == per])
        gv_std[iper] = np.std(group_velocity[inst_periods == per])
    return inst_periods_uniq, gv_moy, gv_std


def plot_picks(picks, ax=None, dmax=None, bins=100, title="Pick density", cmap=pqlx, std_multiple=2):
    """
    Plot an histogram of dispersion picks with the mean and 2*sigma bounds
    Args:
        picks: Pandas.DataFrame of picks with columns inst_period, group_velocity
        ax: pyplot axes to plot in
        dmax: Maximum group velocity
        bins: Matrix of period bins and group velocity bins ([period bins; velocity bins])
        title: plot title
        cmap: colormap for histogram

    Returns:
        heatmap, xedges, yedges, image handle
    """
    picks2 = picks.copy()

    inst_periods_uniq, gv_moy, gv_std = get_mean(picks2.inst_period.values, picks2.group_velocity.values)

    heatmap, xedges, yedges = np.histogram2d(picks2.inst_period, picks2.group_velocity, bins=bins)

    # Cut first column and first row
    heatmap = heatmap[1:, 1:]
    xedges = xedges[1:]
    yedges = yedges[1:]

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    print(np.min(heatmap), np.max(heatmap))

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        if dmax:
            im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap=cmap, vmin=0, vmax=dmax)
        else:
            im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap=cmap)
        ax.set_title(title)
        plt.tight_layout()
        cb = plt.colorbar(im, shrink=0.5, label="# picks")
        plt.show()
        plt.close()
    else:
        if dmax:
            im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap=cmap, vmin=0, vmax=dmax)
        else:
            im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap=cmap)
        ax.plot(inst_periods_uniq, gv_moy, c="w", ls="--", lw=2)
        ax.plot(inst_periods_uniq, gv_moy + std_multiple * gv_std, c="w", ls=":", lw=1)
        ax.plot(inst_periods_uniq, gv_moy - std_multiple * gv_std, c="w", ls=":", lw=1)
        ax.set_title(title)
        ax.set(xlim=(xedges[0], xedges[-1]), ylim=(yedges[0], yedges[-1]))
        ax.xaxis.set_minor_locator(AutoMinorLocator(10))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.tick_params(which='both', width=1)
        ax.tick_params(which='major', length=9)
        ax.tick_params(which='minor', length=5, color='k')
    return heatmap, xedges, yedges, im
