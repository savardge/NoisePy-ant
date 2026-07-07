"""
DISPERSION PICKING FUNCTIONS
Code modified from the older version of the NoisePy repository https://github.com/noisepy/NoisePy
"""
import numpy as np
import pycwt
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


def get_disp_image(ccf, dist, dt, Tmin=0.4, dT=0.02, vmin=0.1, vmax=4.5, dvel=0.02, vave=3., plot=True,
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
    rcwt_new = _interp_image(dist / tvec, period, rcwt, vel, per)

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


def _interp_image(x, y, z, xnew, ynew):
    '''
    Drop-in replacement for the (removed) scipy.interpolate.interp2d usage in this module.

    Interpolates z, defined on grid (y, x) with shape (len(y), len(x)), onto the
    grid (ynew, xnew), returning an array of shape (len(ynew), len(xnew)). Axes are
    sorted internally (RegularGridInterpolator requires strictly increasing axes) and
    values are linearly extrapolated outside the input range (matching the previous
    behaviour closely enough for the dispersion image).
    '''
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xi = np.argsort(x)
    yi = np.argsort(y)
    rgi = scipy.interpolate.RegularGridInterpolator(
        (y[yi], x[xi]), np.asarray(z)[yi][:, xi],
        method='linear', bounds_error=False, fill_value=None)
    YY, XX = np.meshgrid(np.asarray(ynew, dtype=float), np.asarray(xnew, dtype=float), indexing='ij')
    return rgi((YY, XX))


def compute_cwt(ccf, dist, dt, Tmin=0.4, vmin=0.1, vmax=4.5, vave=3., taper=True):
    '''
    Compute the (complex) Continuous Wavelet Transform of a cross-correlation lag and
    return the raw coefficients plus everything needed for downstream measurements
    (group-velocity image, Shapiro/Levshin period corrections, phase velocity).

    This factors out the wavelet machinery shared by get_disp_image_taper so the complex
    coefficients (and the signal spectrum) survive past image building and are computed
    only once per (component, lag).

    Args:
        ccf: Cross-correlation function for one lag (causal side, increasing lag time)
        dist: inter-station distance [km]
        dt: sampling interval [s]
        Tmin: minimum period [s] (sets the upper frequency band edge)
        vmin, vmax: group-velocity window used to trim/taper the lag
        vave: average velocity used for Tmax = dist/vave (lower frequency band edge)
        taper: apply a Tukey taper over the velocity window before the transform

    Returns:
        dict cwt_data with keys:
            cwt: complex coefficients, shape (n_band_scales, n_time)
            freq: Fourier frequency per scale [Hz], shape (n_band_scales,)
            sj: wavelet scales for those frequencies, shape (n_band_scales,)
            tvec: lag-time vector [s], shape (n_time,)
            velocity: dist/tvec [km/s], shape (n_time,)
            coi: cone-of-influence period vs time, shape (n_time,)
            ccf_fft: FFT of the (trimmed, tapered) ccf used in the transform
            omega: angular frequencies of ccf_fft [rad/s]
            dt, dist: passed through
            f0: Morlet central (non-dimensional) frequency (6)
    '''
    ccf = ccf.copy()

    # Basic parameters for wavelet transform (kept identical to get_disp_image_taper)
    dj = 1 / 12  # Twelve sub-octaves per octave
    s0 = -1      # Smallest scale -> 2*dt
    J = -1       # Number of scales determined automatically
    wvn = 'morlet'
    f0 = 6.0     # Morlet central frequency used by pycwt for 'morlet'

    Tmax = dist / vave
    fmin = 1 / Tmax
    fmax = 1 / Tmin

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
    if taper:
        taper_arr = np.zeros(shape=npts)
        taper_arr[indx] = tukey_window(len(indx), alpha=0.05, sym=True)
        ccf *= taper_arr
    # Cut the part after the taper (to speed up calculation)
    ccf = ccf[:pt2]
    tvec = tvec[:pt2]

    # wavelet transformation
    cwt, sj, freq, coi, _, _ = pycwt.cwt(ccf, dt, dj, s0, J, wvn)

    # Filter to the requested frequency band
    if (fmax > np.max(freq)) | (fmax <= fmin):
        raise ValueError('Abort: frequency out of limits!')
    freq_ind = np.where((freq >= fmin) & (freq <= fmax))[0]
    cwt = cwt[freq_ind]
    freq = freq[freq_ind]
    sj = sj[freq_ind]

    # Remove t=0 sample
    tvec = tvec[1:]
    cwt = cwt[:, 1:]
    coi = coi[1:]

    # Signal spectrum (for the Shapiro centroid correction)
    ccf_fft = fft.fft(ccf)
    omega = 2 * np.pi * fft.fftfreq(len(ccf), dt)

    return {
        'cwt': cwt, 'freq': freq, 'sj': sj, 'tvec': tvec, 'velocity': dist / tvec,
        'coi': coi, 'ccf_fft': ccf_fft, 'omega': omega, 'dt': dt, 'dist': dist, 'f0': f0,
    }


def disp_image_from_cwt(cwt_data, dist, Tmin=0.4, dT=0.02, vmin=0.1, vmax=4.5, dvel=0.02, vave=3.,
                        normalize='per_period'):
    '''
    Build the (period, velocity) group-dispersion image from pre-computed CWT coefficients.

    normalize:
        'per_period' (default) -- divide each period row by its own max (every period equally
            bright; best for ridge tracking, but hides where the signal is genuinely strong);
        'global' -- divide the whole image by its single maximum ("normalized by the maximum
            energy of the signal", Esteve et al. 2025 / Shirzad et al. 2025): preserves the
            relative amplitude across periods so genuinely high-energy regions stand out and
            weak periods fade;
        'none' -- raw |CWT|^2.

    Returns (rcwt_new, per, vel, coi_new).
    '''
    freq = cwt_data['freq']
    velocity = cwt_data['velocity']
    period = 1 / freq
    rcwt = np.abs(cwt_data['cwt']) ** 2

    Tmax = dist / vave
    per = np.arange(Tmin, Tmax, dT)
    vel = np.arange(vmin, vmax, dvel)

    rcwt_new = _interp_image(velocity, period, rcwt, vel, per)

    ff = scipy.interpolate.interp1d(velocity, cwt_data['coi'], fill_value='extrapolate', assume_sorted=False)
    coi_new = ff(vel)

    if normalize == 'per_period':
        rcwt_new /= np.max(rcwt_new, axis=1)[:, np.newaxis]
    elif normalize == 'global':
        rcwt_new /= np.max(rcwt_new)
    elif normalize != 'none':
        raise ValueError(f"unknown normalize={normalize!r}")
    return rcwt_new, per, vel, coi_new


def synthesize_rayleigh_modes(zz, rr, rz, zr, receiver_side_flip=False):
    '''
    Nayak & Thurber (2020) multi-component Rayleigh-wave mode separation (their eqs 3 and 4).

    Phase-correct the four radial-vertical cross-correlation components and stack them into two
    traces: G_LR0, which constructively stacks the retrograde fundamental mode (and suppresses the
    prograde 1st higher mode), and G_LR1, which does the reverse to enhance the 1st higher mode.
    Feeding G_LR0 / G_LR1 to compute_cwt + disp_image_from_cwt gives FTAN images in which a single
    mode dominates, so the ridge picker (extract_dispersion / extract_curves_topology) follows one
    mode at a time instead of jumping between modes.

        G_LR0 = ZZ + RR + IFFT( RZ e^{+i pi/2} + ZR e^{-i pi/2} )      (eq. 3)
        G_LR1 = ZZ + RR + IFFT(-RZ e^{+i pi/2} - ZR e^{-i pi/2} )      (eq. 4)

    Args:
        zz, rr, rz, zr: one-sided (symmetric-fold, increasing-lag) cross-correlation traces of the
            ZZ, RR, RZ and ZR components, all the same length. Convention (matches the NoisePy-ant
            E-N-Z -> R-T-Z rotation): the first letter is the source component, the second the
            receiver component, so RZ = R-at-source / Z-at-receiver = paper G_RZ, and ZR = paper
            G_ZR. The four components must share a common amplitude normalization (the normZ
            processing does this), since this is a waveform stack.
        receiver_side_flip: which eigenfunction-sign assumption to use for G_LR1.
            False (default) -> eq (4): the 1st-higher-mode vertical eigenfunction is flipped at
            BOTH stations (same-medium assumption; components with a single vertical instance,
            RZ and ZR, get the minus sign). Right choice when both stations sit in the same
            geological setting (e.g. both in the basin).
            True -> the paper's real-data variant (their section 3.1): the flip is assumed at the
            RECEIVER station only (components with Z at the receiver, ZZ and RZ, are multiplied
            by -1 before applying eq. 3). They used this for source stations on Coast Range hard
            rock and receiver stations on Great Valley sediments; use it for pairs straddling a
            strong structural boundary.

    Returns:
        (g0, g1): the fundamental (G_LR0) and 1st-higher-mode (G_LR1) Rayleigh traces, same length
        as the inputs.

    Note:
        The +-pi/2 sign is the paper convention and has been validated empirically on these data
        (G_LR0 tracks the ZZ fundamental). If a dataset's component convention differs, the sign of
        the pi/2 shifts (equivalently swapping rz and zr) would need to flip.
    '''
    c0, c1 = phase_corrected_components(zz, rr, rz, zr, receiver_side_flip=receiver_side_flip)
    return np.sum(c0, axis=0), np.sum(c1, axis=0)


def phase_corrected_components(zz, rr, rz, zr, receiver_side_flip=False):
    '''
    The four phase-corrected R-Z components entering the Nayak & Thurber stacks, BEFORE summation.

    Returns (comps0, comps1): two lists of four equal-length traces whose plain sums are G_LR0 and
    G_LR1 (see synthesize_rayleigh_modes for the conventions and the receiver_side_flip variants).
    Exposed separately so the stacking operator is a free choice: linear sum (paper eqs 3/4) or a
    phase-weighted stack of the four traces (paper section 3.1 uses a t-f domain PWS on real data;
    see tf_pws).
    '''
    zz = np.asarray(zz, dtype=float)
    rr = np.asarray(rr, dtype=float)
    n = len(zz)
    rz90 = np.fft.irfft(np.fft.rfft(np.asarray(rz, dtype=float)) * np.exp(1j * np.pi / 2), n=n)
    zr90 = np.fft.irfft(np.fft.rfft(np.asarray(zr, dtype=float)) * np.exp(-1j * np.pi / 2), n=n)
    comps0 = [zz, rr, rz90, zr90]
    if receiver_side_flip:
        # section 3.1 variant: Z-at-receiver components (ZZ, RZ) flipped, then eq (3)
        comps1 = [-zz, rr, -rz90, zr90]
    else:
        # eq (4): flip the single-vertical-instance components (RZ, ZR)
        comps1 = [zz, rr, -rz90, -zr90]
    return comps0, comps1


def tf_pws(traces, dt, wu=2.0, unbiased=True, dj=1 / 12):
    '''
    Time-frequency phase-weighted stack (Schimmel & Gallart 2007) in the wavelet domain,
    following the ts-PWS formulation of Ventosa et al. (GJI 2017; reference C implementation in
    ~/Codes/ts-PWS): CWT each trace, weight the linear stack of coefficients by the phase
    coherence of the ensemble, and invert back to the time domain.

        W_lin(s,t)  = (1/K) sum_k W_k(s,t)
        c(s,t)      = | (1/K) sum_k W_k/|W_k| |          (phase coherence, 0..1)
        W_pws(s,t)  = W_lin * w(s,t)

    with w = c^wu, or for wu=2 the unbiased estimator of Ventosa et al. (2017)
    w = (K c^2 - 1)/(K - 1) clipped at 0, which removes the 1/K random-phase bias -- important
    for small ensembles like the K=4 phase-corrected components of Nayak & Thurber (2020), whose
    real-data processing uses exactly this kind of t-f PWS to suppress wave packets that are not
    in phase across the four [R/Z] components.

    Args:
        traces: sequence of K equal-length 1-D arrays (e.g. from phase_corrected_components)
        dt: sampling interval [s]
        wu: phase-weight power (2 = standard)
        unbiased: use the unbiased coherence weight (only defined for wu=2)
        dj: wavelet scale resolution (same default as compute_cwt)

    Returns:
        1-D real array, same length as the inputs (amplitude scale is that of the mean stack).
    '''
    X = np.asarray(traces, dtype=float)
    K, n = X.shape
    Ws = None
    for k in range(K):
        W, sj, freq, coi, _, _ = pycwt.cwt(X[k], dt, dj, -1, -1, 'morlet')
        if Ws is None:
            Ws = np.empty((K,) + W.shape, dtype=complex)
        Ws[k] = W
    lin = Ws.mean(axis=0)
    mag = np.abs(Ws)
    with np.errstate(divide='ignore', invalid='ignore'):
        unit = np.where(mag > 0, Ws / mag, 0)
    coh = np.abs(unit.mean(axis=0))                     # phase coherence in [0, 1]
    if unbiased and wu == 2:
        w = np.clip((K * coh ** 2 - 1.0) / (K - 1.0), 0.0, None)
    else:
        w = coh ** wu
    rec = pycwt.icwt(lin * w, sj, dt, dj, 'morlet')
    return np.real(rec)[:n]


def phase_image_from_cwt(cwt_data, dist, Tmin=0.4, dT=0.02, vmin=0.1, vmax=4.5, dvel=0.02, vave=3.,
                         normalize=True):
    '''
    Build a phase-velocity image from pre-computed CWT coefficients.

    The value plotted is the (per-scale normalized) real part of the Morlet-filtered analytic
    signal, mapped onto the (period, velocity) grid where velocity = dist/t is the apparent
    *phase* velocity. The bright/dark fringes are constant-phase contours: the fringe matching
    the reference phase-velocity curve is the dispersion branch, and the adjacent fringes are
    the 2*pi*N ambiguity branches. This is the CWT-consistent analogue of Douglas' phase
    spectrogram (TimeShift_PhaseSpectrograms), without the ad-hoc CenterT/8 time shift.

    Returns (phase_img, per, vel) with phase_img shape (len(per), len(vel)).
    '''
    freq = cwt_data['freq']
    period = 1 / freq
    velocity = cwt_data['velocity']
    cwt = cwt_data['cwt']
    if normalize:
        norm = np.max(np.abs(cwt), axis=1, keepdims=True)
        field = np.real(cwt) / np.where(norm > 0, norm, 1.0)
    else:
        field = np.real(cwt)

    Tmax = dist / vave
    per = np.arange(Tmin, Tmax, dT)
    vel = np.arange(vmin, vmax, dvel)
    phase_img = _interp_image(velocity, period, field, vel, per)
    return phase_img, per, vel


def compute_narrowband(ccf, dist, dt, Tmin=0.4, vmin=0.1, vmax=4.5, vave=3., alpha=18.0,
                       n_center=100, taper=True):
    '''
    Classic multiple-filter (narrowband Gaussian) FTAN, Levshin et al. (1989) / Bensen et al.
    (2007) eqs. 3-6. The analytic signal Sa(w)=S(w)(1+sgn w) is multiplied by a narrowband
    Gaussian G(w-w0)=exp(-alpha*((w-w0)/w0)^2) at a set of centre frequencies w0 and inverse-
    transformed to the analytic narrowband field A(t,w0) e^{i phi(t,w0)}.

    Returns a dict with the SAME keys as compute_cwt (cwt=complex field n_center x n_time, freq,
    tvec, velocity, coi, dt, dist, ...), so disp_image_from_cwt / phase_velocity_image /
    measure_point / extract_dispersion_viterbi all work on it unchanged -- enabling a direct
    comparison with the CWT (Morlet) approach.

    `alpha` is the Bensen narrowband width (distance-dependent in classic FTAN). alpha=18 makes
    the Gaussian's relative bandwidth ~ a Morlet wavelet with w0=6 (alpha_equiv = w0^2/2), so the
    two methods are compared at matched resolution; lower alpha = broader filter.
    '''
    ccf = ccf.copy()
    Tmax = dist / vave
    fmin, fmax = 1.0 / Tmax, 1.0 / Tmin
    npts = len(ccf)
    tvec = np.arange(npts) * dt
    pt1 = max(int(dist / vmax / dt), 10)
    pt2 = min(int(dist / vmin / dt), npts)
    indx = np.arange(pt1, pt2)
    if taper:
        w = np.zeros(npts)
        w[indx] = tukey_window(len(indx), 0.05, sym=True)
        ccf *= w
    ccf = ccf[:pt2]
    tvec = tvec[:pt2]
    N = len(ccf)

    Sa = fft.fft(ccf)
    freqs = fft.fftfreq(N, dt)
    omega = 2.0 * np.pi * freqs
    analytic = np.zeros(N)                       # Sa(w) = S(w)(1+sgn w): double +, zero -
    analytic[freqs > 0] = 2.0
    analytic[freqs == 0] = 1.0
    Sa = Sa * analytic

    fcs = np.geomspace(fmin, fmax, n_center)      # centre frequencies [Hz]
    field = np.zeros((n_center, N), dtype=complex)
    for j, fc in enumerate(fcs):
        w0 = 2.0 * np.pi * fc
        G = np.exp(-alpha * ((omega - w0) / w0) ** 2)
        field[j, :] = fft.ifft(Sa * G)            # analytic narrowband A(t) e^{i phi}

    tvec = tvec[1:]
    field = field[:, 1:]
    return {
        'cwt': field, 'freq': fcs, 'sj': np.zeros(n_center), 'tvec': tvec,
        'velocity': dist / tvec, 'coi': np.full(len(tvec), Tmax),     # no COI for narrowband
        'ccf_fft': Sa, 'omega': omega, 'dt': dt, 'dist': dist, 'f0': 6.0, 'alpha': alpha,
    }


def phase_velocity_image(cwt_data, dist, Tmin=0.4, dT=0.02, vmin=0.1, vmax=4.5, dvel=0.02, vave=3.,
                         phase_shift=np.pi / 4.0, phase_offset=0.0,
                         group_per=None, group_vel=None):
    '''
    Proper phase-velocity image whose POSITIVE crests are the phase-velocity branches.

        img(T, c) = cos( w*dist*(1/c - 1/U(T)) - phi(t_u) - phase_shift - phase_offset )

    For the picks from measure_corrections_and_phase() to land EXACTLY on the crests, the image
    must use the SAME group arrival as the picks: pass the picked group-velocity curve via
    (group_per, group_vel) and, at each scale, U(T) and phi(t_u) are read with measure_point at
    that ridge velocity (identical to the picks). If no curve is given it falls back to the
    per-scale global envelope peak (whose branches will NOT match picks taken off a different
    ridge -- always pass the ridge).

    Returns (img, per, vel) with img shape (len(per), len(vel)); NaN outside the band.
    '''
    freq = cwt_data['freq']
    tvec = cwt_data['tvec']
    W = cwt_data['cwt']
    Tmax = dist / vave
    per = np.arange(Tmin, Tmax, dT)
    vel = np.arange(vmin, vmax, dvel)
    s_c = 1.0 / vel
    period_native = 1.0 / freq
    img_native = np.full((len(freq), len(vel)), np.nan)
    have_ridge = (group_per is not None and group_vel is not None
                  and len(np.atleast_1d(group_per)) >= 2)
    for j in range(len(freq)):
        Tj = period_native[j]
        if have_ridge:
            U_guess = float(np.interp(Tj, group_per, group_vel))   # ridge velocity at this period
            m = measure_point(cwt_data, Tj, U_guess, dist)          # SAME reading the picks use
            U, phi = m['U'], m['phase']
            if not np.isfinite(phi):
                continue
        else:
            row = W[j, :]
            it = int(np.argmax(np.abs(row)))
            if it <= 0 or it >= len(row) - 1:
                continue
            U = dist / tvec[it]
            phi = float(np.angle(row[it]))
        w = 2.0 * np.pi * freq[j]
        img_native[j, :] = np.cos(w * dist * (s_c - 1.0 / U) - phi - phase_shift - phase_offset)
    o = np.argsort(period_native)
    img = scipy.interpolate.interp1d(period_native[o], img_native[o, :], axis=0,
                                     bounds_error=False, fill_value=np.nan)(per)
    return img, per, vel


def coi_mask(coi_new, per, vel):
    '''
    Boolean mask (len(per) x len(vel)) that is True where a (period, velocity) cell is
    OUTSIDE the cone of influence (i.e. reliable: period <= coi at that velocity).
    Matches the orientation of the group/phase images returned by this module.
    '''
    mask = np.zeros((len(per), len(vel)), dtype=np.uint8)
    for jv in range(len(vel)):
        mask[:, jv] = (per <= coi_new[jv]).astype(np.uint8)
    return mask


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
        coi_new: cone of influence interpolated on the velocity axis
    '''
    cwt_data = compute_cwt(ccf, dist, dt, Tmin=Tmin, vmin=vmin, vmax=vmax, vave=vave, taper=True)
    rcwt_new, per, vel, coi_new = disp_image_from_cwt(
        cwt_data, dist, Tmin=Tmin, dT=dT, vmin=vmin, vmax=vmax, dvel=dvel, vave=vave)

    # Plot
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(np.transpose(rcwt_new),
                  cmap='jet',
                  extent=[per[0], per[-1], vel[0], vel[-1]],
                  aspect='auto',
                  origin='lower')
        ax.scatter(coi_new, vel, c="k", s=5)
        ax.set(xlabel='Period [s]', ylabel='Vg [km/s]', xlim=(Tmin, dist / vave), ylim=(vmin, vmax))
        ax.set_title('Inter-station distance: %5.2f km' % dist)
        plt.tight_layout()
        plt.show()
        plt.close()

    return rcwt_new, per, vel, coi_new


# function to extract the dispersion from the image
def extract_dispersion(amp, per, vel, dist, vmax=5., maxgap=3, minlambda=1.5,
                       segments=False, min_seg=5):
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
    segments: continuity policy. False (default) = original behaviour: a pick is kept only if
        the following 15 samples contain no jump > maxgap, which discards everything up to
        1.5 s of period BEFORE a genuine steep step of the dispersion curve (e.g. the
        near-vertical fundamental-mode transition across a strong basin/bedrock contrast).
        True = split the argmax curve at jumps > maxgap and keep EVERY continuous segment of
        at least min_seg samples: steep but real branch steps are preserved, while short
        unstable runs (spikes, body-wave blobs a few samples wide) are dropped.
    min_seg: minimum segment length in samples (segments=True only).

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
        indx = int(np.argmax(amp[ii]))
        maxvalue = amp[ii][indx]
        gv[ii] = vel[indx]
        ampsnr[ii] = maxvalue / np.median(amp[ii], axis=0)

        # QC:
        if np.abs(gv[ii] - vmax) < 3 * dvel:  # remove points close to vg limits (within 3 samples of the max)
            gv[ii] = 0
        elif dist / (per[ii] * gv[ii]) < minlambda:  # remove if number of wavelengths is less than threshold
            gv[ii] = 0

    if segments:
        # Split the picked curve at velocity jumps and keep every continuous segment that is at
        # least min_seg samples long. A steep (real) step in the curve then costs nothing but the
        # single across-step pair, instead of the 15 preceding samples.
        keep = np.zeros(nper, dtype=bool)
        ii = 0
        while ii < nper:
            if gv[ii] == 0:
                ii += 1
                continue
            jj = ii
            while (jj + 1 < nper and gv[jj + 1] > 0
                   and np.abs(gv[jj + 1] - gv[jj]) <= maxgap * dvel):
                jj += 1
            if (jj - ii + 1) >= min_seg:
                keep[ii:jj + 1] = True
            ii = jj + 1
        gv = np.where(keep, gv, 0)
    else:
        # Check the continuity of the dispersion curve (original 15-sample look-ahead)
        for ii in range(1, nper - minimum_output_length):
            if gv[ii] == 0:
                continue
            for jj in range(minimum_output_length):
                if np.abs(gv[ii + jj] - gv[ii + 1 + jj]) > maxgap * dvel:  # large velocity gap
                    gv[ii] = 0
                    break

    # Remove discarded picks set to 0
    indx = np.where(gv > 0)[0]
    pick_per = per[indx]
    pick_gv = gv[indx]
    pick_ampsnr = ampsnr[indx]

    # Check if there are outliers: points that deviate from the linear interpolation of
    # their two neighbours by more than maxgap velocity samples (spike removal).
    # NOTE: the previous test `pick_gv[ii] > pick_gv[ii-1] + (pick_gv[ii+1] - pick_gv[ii-1])`
    # algebraically reduced to `pick_gv[ii] > pick_gv[ii+1]` and dropped any locally
    # decreasing point; this is the intended midpoint-deviation test.
    igood = list(range(len(pick_per)))
    for ii in range(2, len(pick_per) - 1):
        if segments and (indx[ii + 1] - indx[ii - 1]) != 2:
            continue    # neighbours span a segment boundary -- midpoint test not meaningful
        midpoint = 0.5 * (pick_gv[ii - 1] + pick_gv[ii + 1])
        if np.abs(pick_gv[ii] - midpoint) > maxgap * dvel:
            igood.remove(ii)
    pick_per = pick_per[igood]
    pick_gv = pick_gv[igood]
    pick_ampsnr = pick_ampsnr[igood]

    #     return per[indx],gv[indx],ampsnr[indx]
    return pick_per, pick_gv, pick_ampsnr


def extract_dispersion_viterbi(amp, per, vel, smooth_weight=2.0, max_step=0.2,
                               short_priority=1.0, coi=None, coi_penalty=10.0):
    """
    Track the group-velocity ridge through the FTAN amplitude image by Viterbi DP.

    States = velocity bins; emission favours high (per-period normalised) amplitude;
    transition penalises the velocity jump |Δv| [km/s] between adjacent periods AND HARD-FORBIDS
    any jump larger than `max_step` per period. The hard cap is essential: with only a linear
    penalty a single big jump can be cheaper than staying dim across many periods, so the ridge
    would leap between energy bands ("huge gaps"). The cap guarantees a continuous curve while
    still allowing steep but smooth gradients (|Δv| up to max_step per period step).

    Args:
        amp: FTAN image (nper x nvel), normalised to ~[0,1] per period
        per, vel: period and velocity axes
        smooth_weight: cost per km/s of inter-period velocity jump (within the cap)
        max_step: maximum allowed |Δv| [km/s] between adjacent periods (hard continuity cap)
        short_priority: power >=0 weighting the amplitude reward by (Tmin/T)**short_priority, so
             the ridge is anchored on the strong, reliable SHORT-period peaks and the long-period
             tail follows by continuity (rather than the path trading short-period accuracy away).
        coi: optional cone-of-influence period vs velocity (len nvel); cells with per>coi are
             penalised so the ridge avoids the COI
        coi_penalty: emission penalty added inside the COI

    Returns:
        (per, ridge_vel): one velocity per period along the optimal continuous ridge.
    """
    amp = np.asarray(amp, dtype=float)
    nper, nvel = amp.shape
    V = np.asarray(vel, dtype=float)
    pw = (per[0] / np.asarray(per, dtype=float)) ** short_priority   # short-period emphasis
    emis = -amp * pw[:, None]                              # prefer bright cells, more so at short T
    if coi is not None:
        inside = per[:, None] > np.asarray(coi)[None, :]   # (nper x nvel)
        emis = np.where(inside, emis + coi_penalty, emis)
    dV = np.abs(V[:, None] - V[None, :])                   # (prev x cur), km/s
    trans = smooth_weight * dV
    trans[dV > max_step] = 1e9                             # hard cap: forbid huge per-step gaps

    cost = emis[0].copy()
    back = np.zeros((nper, nvel), dtype=int)
    for i in range(1, nper):
        total = cost[:, None] + trans + emis[i][None, :]
        back[i] = np.argmin(total, axis=0)
        cost = np.min(total, axis=0)
    k = int(np.argmin(cost))
    ridge = np.zeros(nper)
    for i in range(nper - 1, -1, -1):
        ridge[i] = V[k]
        k = back[i, k]
    return per.copy(), ridge


def viterbi_select_candidates(cand_per, cand_vel, cand_w, smooth_weight=2.0):
    """
    Select one candidate per period from a scattered set of picks (e.g. topology peaks or a
    pooled argmax+topology set) so the chosen sequence is a smooth, high-weight ridge.

    Same idea as extract_dispersion_viterbi but the states at each period are only the supplied
    candidate velocities there (not the full velocity grid). Returns a boolean mask (aligned
    with the inputs) flagging which candidates were selected, plus the selected (per, vel).

    Args:
        cand_per, cand_vel: 1D arrays of candidate periods and velocities
        cand_w: 1D weight per candidate (amplitude or persistence score; higher = preferred)
        smooth_weight: cost per km/s of velocity jump between adjacent selected periods
    """
    cand_per = np.asarray(cand_per, dtype=float)
    cand_vel = np.asarray(cand_vel, dtype=float)
    cand_w = np.asarray(cand_w, dtype=float)
    n = len(cand_per)
    selected = np.zeros(n, dtype=bool)
    if n == 0:
        return selected, np.array([]), np.array([])

    uper = np.unique(cand_per)
    groups = [np.where(cand_per == p)[0] for p in uper]    # candidate indices per period
    # Viterbi over the variable-size state sets
    prev_cost = -cand_w[groups[0]]
    prev_idx = groups[0]
    back = [np.full(len(prev_idx), -1)]
    states = [prev_idx]
    for g in range(1, len(uper)):
        cur = groups[g]
        vcur = cand_vel[cur]
        vprev = cand_vel[prev_idx]
        trans = smooth_weight * np.abs(vcur[None, :] - vprev[:, None])   # (prev x cur)
        total = prev_cost[:, None] + trans + (-cand_w[cur])[None, :]
        bp = np.argmin(total, axis=0)
        prev_cost = np.min(total, axis=0)
        back.append(bp)
        states.append(cur)
        prev_idx = cur
    k = int(np.argmin(prev_cost))
    sel_per, sel_vel = [], []
    for g in range(len(uper) - 1, -1, -1):
        idx = states[g][k]
        selected[idx] = True
        sel_per.append(cand_per[idx]); sel_vel.append(cand_vel[idx])
        k = back[g][k]
    return selected, np.array(sel_per[::-1]), np.array(sel_vel[::-1])


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
    from findpeaks import findpeaks  # https://github.com/erdogant/findpeaks (optional dependency)
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
        # ix = np.argwhere(vel == pick_vel[ii])[0][0]  # doesn't work if rounding precision error
        ix = np.argmin(np.abs(vel - pick_vel[ii]))
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


def _scale_index_for_period(cwt_data, period):
    """Index of the CWT scale whose Fourier frequency is closest to 1/period."""
    return int(np.argmin(np.abs(cwt_data['freq'] - 1.0 / period)))


def measure_point(cwt_data, period, U, dist, refine=True, half=8):
    """
    Measure the group arrival, analytic-signal phase and instantaneous frequency for one
    pick, on the CWT row whose scale matches `period`.

    Phase velocity is extremely sensitive to the exact group-arrival time (a one-sample
    error gives a phase error of ~ w*dt), so this:
      * relocates to the local envelope maximum near t_u = dist/U (within +/- `half`
        samples) and refines it to sub-sample precision by parabolic interpolation;
      * reads the phase by linear interpolation of the *unwrapped* phase at that time;
      * returns the self-consistent group velocity U = dist/t_peak so the s_u term in the
        phase-velocity formula cancels correctly.

    Returns dict: {t_peak, U, phase, omega_inst, T_inst} (values may be nan at edges).
    """
    j = _scale_index_for_period(cwt_data, period)
    row = cwt_data['cwt'][j, :]
    tvec = cwt_data['tvec']
    dt = cwt_data['dt']
    ntime = row.shape[0]
    env = np.abs(row)
    phase_unw = np.unwrap(np.angle(row))

    it0 = int(np.argmin(np.abs(tvec - dist / U)))
    if refine:
        lo = max(1, it0 - half)
        hi = min(ntime - 1, it0 + half + 1)
        itp = lo + int(np.argmax(env[lo:hi]))
    else:
        itp = it0

    if itp <= 0 or itp >= ntime - 1:
        return {'t_peak': np.nan, 'U': np.nan, 'phase': np.nan,
                'omega_inst': np.nan, 'T_inst': np.nan}

    # Sub-sample envelope peak (parabolic)
    y0, y1, y2 = env[itp - 1], env[itp], env[itp + 1]
    denom = (y0 - 2 * y1 + y2)
    frac = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    frac = float(np.clip(frac, -0.5, 0.5))
    t_peak = tvec[itp] + frac * dt

    # Sub-sample phase via the unwrapped phase, then return the principal value in
    # [-pi, pi] (the integer cycles are carried by the 2*pi*N ambiguity term, not here).
    phase_unwrapped_at_peak = np.interp(t_peak, tvec, phase_unw)
    phase = float((phase_unwrapped_at_peak + np.pi) % (2.0 * np.pi) - np.pi)
    omega_inst = (phase_unw[itp + 1] - phase_unw[itp - 1]) / (2.0 * dt)
    T_inst = 2.0 * np.pi / omega_inst if omega_inst > 0 else np.nan
    return {'t_peak': float(t_peak), 'U': float(dist / t_peak), 'phase': phase,
            'omega_inst': float(omega_inst), 'T_inst': float(T_inst)}


def instantaneous_period_at(cwt_data, period, U, dist):
    """
    Instantaneous period at the group arrival time (Levshin 1989 eq. 5.92; Bensen 2007).

    The CWT row at a fixed scale is the Gaussian/Morlet-filtered analytic signal S(w,t).
    Its instantaneous angular frequency Omega = d/dt arg S, evaluated at the envelope peak
    (the group arrival time), is the unbiased frequency to which the group-velocity
    measurement should be assigned, instead of the nominal scale period.

    Returns the instantaneous period [s], or np.nan if it cannot be evaluated.
    """
    return measure_point(cwt_data, period, U, dist)['T_inst']


def centroid_period(cwt_data, period):
    """
    Centroid-frequency period correction (Shapiro & Singh 1999 eq. 6).

    The systematic FTAN error comes from the spectral amplitude varying across the
    (Morlet) filter band, which shifts the filtered-spectrum centroid away from the
    nominal centre frequency. We compute the energy centroid of the filtered spectrum
    |psi(s*w)|^2 |X(w)|^2 and express the correction *relative* to the flat-spectrum
    centroid, so that T_centroid == period when the spectrum is flat (this removes the
    Morlet flambda-vs-f0 convention offset and isolates the spectral-falloff shift).

    Args:
        cwt_data: dict returned by compute_cwt
        period: nominal (picked) period [s]

    Returns:
        Corrected (centroid) period [s], or np.nan if it cannot be evaluated.
    """
    j = _scale_index_for_period(cwt_data, period)
    sj = cwt_data['sj'][j]
    f0 = cwt_data['f0']
    omega = cwt_data['omega']
    pos = omega > 0
    w = omega[pos]
    gauss = np.exp(-(sj * w - f0) ** 2)          # |Morlet(s*w)|^2 shape (positive freqs)
    spec = np.abs(cwt_data['ccf_fft'][pos]) ** 2  # |X(w)|^2
    denom_real = np.sum(gauss * spec)
    denom_flat = np.sum(gauss)
    if denom_real <= 0 or denom_flat <= 0:
        return np.nan
    omega_c = np.sum(w * gauss * spec) / denom_real   # centroid with the real spectrum
    omega_c0 = np.sum(w * gauss) / denom_flat          # centroid for a flat spectrum
    if omega_c <= 0:
        return np.nan
    # Relative shift applied to the nominal period (convention-free).
    return period * (omega_c0 / omega_c)


def phase_at(cwt_data, period, U, dist):
    """
    Phase of the analytic (Morlet-filtered) signal at the group arrival time t_u = dist/U.

    This is phi(t_u, w0) in Bensen et al. (2007) eq. 9, used for phase-velocity extraction.
    NOTE: the Morlet wavelet carries its own phase convention relative to a zero-phase
    Gaussian filter; calibrate any constant offset on a synthetic of known phase velocity
    (pass it via `phase_offset` of phase_velocity) before trusting absolute phase speeds.
    """
    return measure_point(cwt_data, period, U, dist)['phase']


def phase_velocity(phi_tu, U, dist, period, c_ref, phase_shift=np.pi / 4.0,
                   phase_offset=0.0, n_search=6):
    """
    Phase velocity from the analytic-signal phase, with the 2*pi*N ambiguity resolved by a
    reference curve (Bensen et al. 2007 eqs. 10-11).

        s_c(N) = 1/U + (w*dist)^-1 * (-phi(t_u) + phase_offset + 2*pi*N + phase_shift)
        c(N)   = 1 / s_c(N)

    The measured phase enters NEGATED: the Morlet-filtered analytic signal of an arrival with
    far-field spectrum e^{-i(kr - pi/4)} is W(t) ~ e^{i(w t - k r + pi/4)}, so
    k r = w t_u - phi(t_u) + pi/4 + 2*pi*N, i.e. phase_shift = +pi/4 for ZZ-convention
    components. Validated on a synthetic of exactly known c(T) (ladder contains the true value
    to ~0.003 km/s); the previous +phi / -pi/4 form tilted the fringe ladder the wrong way.
    The integer N is chosen so c(N) is closest to the reference phase velocity c_ref(period).
    Source phase phi_s = 0 (ambient noise).

    Args:
        phi_tu: analytic-signal phase at the group arrival time [rad] (from phase_at)
        U: group velocity [km/s]
        dist: inter-station distance [km]
        period: period [s] (nominal or corrected; sets w = 2*pi/period)
        c_ref: callable c_ref(period) -> reference phase velocity [km/s] (may return nan)
        phase_shift: stationary-phase term. -pi/4 for vertical Rayleigh (ZZ),
            +pi/4 for radial; pass the appropriate value for Love (see Snieder 2004 / Lin 2007).
        phase_offset: calibration constant for the Morlet phase convention [rad].
        n_search: search N in [-n_search, n_search].

    Returns:
        (c, N): phase velocity [km/s] and chosen integer N, or (np.nan, 0) if unresolved.
    """
    cref = float(c_ref(period))
    if not np.isfinite(cref):
        return np.nan, 0
    omega = 2.0 * np.pi / period
    s_u = 1.0 / U
    base = -phi_tu + phase_offset + phase_shift
    Ns = np.arange(-n_search, n_search + 1)
    s_c = s_u + (base + 2.0 * np.pi * Ns) / (omega * dist)
    with np.errstate(divide='ignore', invalid='ignore'):
        c_cand = np.where(s_c > 0, 1.0 / s_c, np.nan)
    err = np.abs(c_cand - cref)
    if np.all(np.isnan(err)):
        return np.nan, 0
    k = int(np.nanargmin(err))
    return float(c_cand[k]), int(Ns[k])


def load_reference_curve(source, key=None):
    """
    Build a reference phase-velocity function c_ref(period) for resolving the 2*pi*N
    ambiguity, supporting per-pair / per-region curves.

    Args:
        source: one of
            - dict mapping a key (station pair or region) to (periods, velocities) or to a
              file path; a 'default' entry is used as fallback when `key` is not found;
            - a path (str) to a 2-column text file (period[s]  phase_velocity[km/s]);
            - a (periods, velocities) tuple/array pair.
        key: pair/region key to select within a dict source.

    Returns:
        Callable c_ref(period) returning the interpolated phase velocity, or np.nan
        outside the tabulated period range (so out-of-range picks get no phase velocity).
    """
    def _from_table(periods, velocities):
        periods = np.asarray(periods, dtype=float)
        velocities = np.asarray(velocities, dtype=float)
        order = np.argsort(periods)
        return scipy.interpolate.interp1d(
            periods[order], velocities[order], kind='linear',
            bounds_error=False, fill_value=np.nan)

    def _load_entry(entry):
        if isinstance(entry, str):
            arr = np.loadtxt(entry)
            return _from_table(arr[:, 0], arr[:, 1])
        periods, velocities = entry
        return _from_table(periods, velocities)

    if isinstance(source, dict):
        if key is not None and key in source:
            return _load_entry(source[key])
        if 'default' in source:
            return _load_entry(source['default'])
        raise KeyError(f"No reference curve for key={key!r} and no 'default' entry")
    return _load_entry(source)


def resolve_phase_curve(periods, phases, gv, dist, c_ref, phase_shift=np.pi / 4.0,
                        phase_offset=0.0, n_search=8, smooth_weight=3.0, max_step=None):
    """
    Resolve the 2*pi*N phase-velocity ambiguity jointly over a whole dispersion curve.

    For each pick i there is a closed-form ladder of candidate phase velocities (one per
    integer N, the fringes of the phase image):

        c_i(N) = 1 / ( 1/U_i + (-phi_i + phase_offset + phase_shift + 2*pi*N) / (w_i*dist) )

    (measured phase negated -- see phase_velocity for the convention derivation/validation).

    A single per-period argmin against the reference can hop to a neighbouring branch wherever
    the reference is locally off by more than half the branch spacing. Instead we pick the
    N-sequence that minimises, by Viterbi dynamic programming,

        sum_i |c_i(N_i) - c_ref(T_i)|/c_ref(T_i)            (stay near the reference)
      + smooth_weight * sum_i |c_i(N_i) - c_{i-1}(N_{i-1})|/c_i(N_i)   (keep c(T) smooth)

    Note N is NOT forced constant: a physical branch lets N step by +/-1 across periods; the
    smoothness is imposed on the velocity curve, which is what actually tracks one ridge.

    Args:
        periods, phases, gv: 1D arrays (same length) of pick period [s], analytic-signal phase
            phi(t_u) [rad] and group velocity [km/s]. Order is arbitrary (sorted internally).
        dist: inter-station distance [km]
        c_ref: callable c_ref(period) -> reference phase velocity [km/s] (may return nan)
        phase_shift, phase_offset: as in phase_velocity
        n_search: search N in [-n_search, n_search]
        smooth_weight: relative weight of the curve-continuity term vs the reference term

    Returns:
        (c_phase, N): arrays aligned with the input order. NaN / 0 where unresolved.
    """
    periods = np.asarray(periods, dtype=float)
    phases = np.asarray(phases, dtype=float)
    gv = np.asarray(gv, dtype=float)
    m = len(periods)
    out_c = np.full(m, np.nan)
    out_N = np.zeros(m, dtype=int)
    if m == 0:
        return out_c, out_N

    order = np.argsort(periods)
    T, phi, U = periods[order], phases[order], gv[order]
    cref = np.array([float(c_ref(t)) for t in T])
    if not np.any(np.isfinite(cref)):
        return out_c, out_N  # no anchor -> cannot resolve absolute branch

    Ns = np.arange(-n_search, n_search + 1)
    omega = 2.0 * np.pi / T
    s = 1.0 / U[:, None] + (-phi[:, None] + phase_offset + phase_shift
                            + 2.0 * np.pi * Ns[None, :]) / (omega[:, None] * dist)
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.where(s > 0, 1.0 / s, np.nan)               # candidate velocities (m x K)

    INF = 1e9
    # Emission cost: distance to reference (relative). Where the reference is nan, no anchor.
    with np.errstate(invalid='ignore'):
        emis = np.abs(c - cref[:, None]) / np.where(np.isfinite(cref[:, None]) & (cref[:, None] > 0),
                                                    cref[:, None], 1.0)
    emis = np.where(np.isfinite(cref[:, None]), emis, 0.0)
    emis = np.where(np.isfinite(c), emis, INF)

    # Viterbi
    cost = emis[0].copy()
    back = np.zeros((m, len(Ns)), dtype=int)
    for i in range(1, m):
        dc = np.abs(c[i][None, :] - c[i - 1][:, None])
        with np.errstate(invalid='ignore'):
            trans = smooth_weight * dc / \
                np.where(np.isfinite(c[i][None, :]) & (c[i][None, :] > 0), c[i][None, :], 1.0)
        trans = np.where(np.isfinite(trans), trans, INF)        # (K_prev x K_cur)
        if max_step is not None:                                # hard cap: forbid branch hops
            trans = np.where(dc > max_step, INF, trans)
        total = cost[:, None] + trans + emis[i][None, :]
        back[i] = np.argmin(total, axis=0)
        cost = np.min(total, axis=0)

    k = int(np.argmin(cost))
    for i in range(m - 1, -1, -1):
        if np.isfinite(c[i, k]):
            out_c[order[i]] = c[i, k]
            out_N[order[i]] = Ns[k]
        k = back[i, k]
    return out_c, out_N


def resolve_phase_curve_unwrap(periods, phases, gv, dist, c_ref, phase_shift=np.pi / 4.0,
                               phase_offset=0.0, n_search=8, seg_dU=0.25, seg_gap=2.5):
    '''
    Single-N phase-velocity resolution by continuous frequency-unwrapping (Bensen et al. 2007:
    "the 2*pi ambiguity inherent to any phase spectrum", their eq. 11 -- N is ONE integer for
    the unwrapped spectrum, chosen against a reference, NOT a free integer per period).

    Steps:
      1. kr_raw_i = w_i * dist/U_i + (-phi_i + phase_offset + phase_shift)   (principal value)
      2. unwrap along ascending frequency: the measured group slowness is the derivative of the
         phase spectrum (dk/dw = 1/U), so the predicted increment between adjacent picks is
         d(kr) ~ dist * dw * mean(1/U); each pick's integer is set to bring kr_raw within pi of
         that prediction -- this enforces the continuity the physics requires and eliminates
         ridge-hopping.
      3. one GLOBAL integer M for the whole curve: choose M minimising the median relative
         distance between c_i(M) = w_i*dist/(kr_i + 2*pi*M) and the reference curve.

    Segmentation: the one-N continuity argument holds within ONE wave packet's phase spectrum.
    Where the picked group curve jumps (mixed-geology paths: the argmax switches between two
    arrivals) or has a period gap, the phases on either side belong to different packets and
    must NOT be glued into a single continuum -- the unwrap breaks there and each segment gets
    its own global integer (seg_dU, seg_gap control the break criteria).

    Returns (c_phase, N_amb) aligned with the inputs; NaN/0 where undefined. N_amb is the total
    per-pick integer (unwrap steps + M) for bookkeeping -- its steps across period compensate
    principal-value wraps only; the physical ambiguity resolved is one M per segment.
    '''
    periods = np.asarray(periods, dtype=float)
    phases = np.asarray(phases, dtype=float)
    gv = np.asarray(gv, dtype=float)
    m = len(periods)
    out_c = np.full(m, np.nan)
    out_N = np.zeros(m, dtype=int)
    ok = np.isfinite(periods) & np.isfinite(phases) & np.isfinite(gv) & (periods > 0) & (gv > 0)
    if ok.sum() < 2:
        return out_c, out_N
    idx = np.where(ok)[0]
    w = 2.0 * np.pi / periods[idx]
    order = np.argsort(w)
    idx = idx[order]
    w = w[order]
    U = gv[idx]
    phi = phases[idx]

    kr_raw = w * dist / U + (-phi + phase_offset + phase_shift)
    T = 2.0 * np.pi / w
    cref = np.array([float(c_ref(t)) for t in T]) if c_ref is not None else \
        np.full(len(w), np.nan)

    # segment boundaries: group-velocity jump (packet switch) or period gap
    dTmed = np.median(np.abs(np.diff(T))) if len(T) > 1 else 0.0
    brk = np.zeros(len(idx), dtype=bool)
    for i in range(1, len(idx)):
        if np.abs(U[i] - U[i - 1]) > seg_dU or \
           (dTmed > 0 and np.abs(T[i] - T[i - 1]) > seg_gap * dTmed):
            brk[i] = True
    seg_id = np.cumsum(brk)

    steps = np.zeros(len(idx), dtype=int)
    kr = kr_raw.copy()
    c_out = np.full(len(idx), np.nan)
    N_out = np.zeros(len(idx), dtype=int)
    for s in range(seg_id.max() + 1):
        ii = np.where(seg_id == s)[0]
        # unwrap within the segment (group slowness = dk/dw as predictor)
        for j in ii[1:]:
            pred = kr[j - 1] + dist * (w[j] - w[j - 1]) * 0.5 * (1.0 / U[j] + 1.0 / U[j - 1])
            steps[j] = int(np.round((pred - kr_raw[j]) / (2.0 * np.pi)))
            kr[j] = kr_raw[j] + 2.0 * np.pi * steps[j]
        # one global integer per segment, median-anchored to the reference
        best_M, best_cost = None, np.inf
        for M in range(-n_search, n_search + 1):
            denom = kr[ii] + 2.0 * np.pi * M
            with np.errstate(divide='ignore', invalid='ignore'):
                c = np.where(denom > 0, w[ii] * dist / denom, np.nan)
            cost = np.nanmedian(np.abs(c - cref[ii]) / cref[ii])
            if np.isfinite(cost) and cost < best_cost:
                best_cost, best_M = cost, M
        if best_M is None:
            continue
        denom = kr[ii] + 2.0 * np.pi * best_M
        with np.errstate(divide='ignore', invalid='ignore'):
            c_out[ii] = np.where(denom > 0, w[ii] * dist / denom, np.nan)
        N_out[ii] = steps[ii] + best_M
    out_c[idx] = c_out
    out_N[idx] = N_out
    return out_c, out_N


def phase_from_group(pick_per, pick_gv, dist, phases, phase_shift=np.pi / 4.0,
                     phase_offset=0.0, cmin=0.5, cmax=5.0, n_scan=2001):
    '''
    Unique phase-velocity curve predicted from a picked group-velocity curve.

    Since U = dw/dk, the wavenumber follows by integration,

        k(w) = k_ref + int_{w_ref}^{w} dw' / U(w'),      c(w) = w / k(w),

    which determines c(T) only UP TO the integration constant k_ref (differentiation destroys
    the absolute phase). k_ref is resolved here from the waveform itself: the analytic-signal
    phases measured at the group arrivals constrain the slowness to a discrete fringe ladder
    (mod 2*pi/(w*dist), Bensen et al. 2007 eqs 10-11), and a 1-D scan picks the k_ref whose
    rigid group-integrated curve best matches the ladder in the circular (mod-rung) sense.
    Group shape + measured phase anchor -> a single curve, no external reference needed.
    IMPORTANT degeneracy: shifting k_ref by exactly 2*pi/dist moves the predicted slowness by
    one full rung at EVERY frequency simultaneously, so the circular misfit is periodic in
    k_ref and all rung-aliases fit the measured phases identically (and share identical group
    velocity -- the constant is precisely what differentiation destroys). The fit therefore
    determines k_ref only mod 2*pi/dist; the alias is selected by physics: the curve must
    satisfy cmin <= c <= cmax and c >= U (normal dispersion; small tolerance allowed), and
    among the feasible aliases the one closest above U is taken. This usually leaves exactly
    one candidate, but sanity-check against an independent reference (FK) where available --
    strongly inversely-dispersive branches could in principle defeat the c >= U selector.

    Args:
        pick_per, pick_gv: picked group curve (period [s], U [km/s]); any order, one per period
        dist: inter-station distance [km]
        phases: analytic-signal phase at the group arrival per pick [rad] (measure_point);
            nan entries are ignored in the anchor fit
        phase_shift, phase_offset: as in phase_velocity (component's initial phase term)
        cmin, cmax: physical bounds for the reference-frequency phase velocity in the scan
        n_scan: k_ref scan resolution

    Returns:
        c_pred: predicted phase velocity aligned with the input picks (nan where undefined)
    '''
    from scipy.integrate import cumulative_trapezoid

    per = np.asarray(pick_per, dtype=float)
    U = np.asarray(pick_gv, dtype=float)
    phi = np.asarray(phases, dtype=float)
    c_pred = np.full(len(per), np.nan)
    ok = np.isfinite(per) & np.isfinite(U) & (per > 0) & (U > 0)
    if ok.sum() < 3:
        return c_pred
    idx = np.where(ok)[0]
    w = 2.0 * np.pi / per[idx]
    order = np.argsort(w)                    # ascending omega
    idx = idx[order]
    w = w[order]
    Uo = U[idx]
    phio = phi[idx]

    kint = cumulative_trapezoid(1.0 / Uo, w, initial=0.0)     # k(w) - k_ref
    # fringe ladder from the measured phases: base slowness and rung spacing per pick
    # (phase negated -- see phase_velocity for the convention derivation/validation)
    s0 = 1.0 / Uo + (-phio + phase_offset + phase_shift) / (w * dist)
    rung = 2.0 * np.pi / (w * dist)
    have_phase = np.isfinite(s0)

    if not have_phase.any():
        return c_pred

    # Step 1: fine offset within ONE rung period (cost is exactly periodic in 2*pi/dist).
    alias_step = 2.0 * np.pi / dist
    k_lo = w[0] / cmax
    kref_grid = np.linspace(k_lo, k_lo + alias_step, n_scan, endpoint=False)
    costs = np.empty(n_scan)
    for i, kref in enumerate(kref_grid):
        s_pred = (kref + kint) / w
        r = np.mod(s_pred - s0 + 0.5 * rung, rung) - 0.5 * rung   # circular residual per rung
        costs[i] = np.nanmean((r[have_phase] / rung[have_phase]) ** 2)
    k_fine = kref_grid[int(np.argmin(costs))]

    # Step 2: pick the rung alias by physics. All k_fine + m*alias_step fit the phases
    # identically; require cmin <= c <= cmax and c >= U (normal dispersion, 3% tolerance),
    # then take the feasible alias closest above U.
    U_tol = 0.97
    best = None
    m = 0
    while k_fine + m * alias_step <= w[0] / cmin:
        k_m = k_fine + m * alias_step
        with np.errstate(divide='ignore', invalid='ignore'):
            c_m = w / (k_m + kint)
        m += 1
        if np.any(~np.isfinite(c_m)) or c_m.max() > cmax or c_m.min() < cmin:
            continue
        if np.any(c_m < U_tol * Uo):                      # violates c >= U
            continue
        gap = float(np.mean(c_m - Uo))                    # supersonic gap above the group curve
        if best is None or gap < best[0]:
            best = (gap, c_m)
    if best is None:
        return c_pred                                     # no physically admissible alias
    c_pred[idx] = best[1]
    return c_pred


def group_from_phase(periods, c_phase):
    """
    Predict group velocity from a phase-velocity curve via the dispersion relation
    (Bensen et al. 2007 eq. 7). Group and phase velocity are NOT independent:

        k = w / c = w * s_c ,   U = dw/dk   =>   s_u = dk/dw = s_c + w * ds_c/dw

    with s_c = 1/c (phase slowness) and s_u = 1/U (group slowness). Differentiating the
    extracted phase-velocity curve should reproduce the directly-measured group-velocity
    ridge -- a strong internal consistency check (a mismatch flags a wrong 2*pi*N branch or
    bad picks).

    Args:
        periods: 1D array of periods [s]
        c_phase: 1D array of phase velocities [km/s] (same length)

    Returns:
        U_pred: predicted group velocity [km/s], aligned with the input order (NaN where
        it cannot be evaluated, e.g. fewer than 3 finite points).
    """
    periods = np.asarray(periods, dtype=float)
    c_phase = np.asarray(c_phase, dtype=float)
    out = np.full(len(periods), np.nan)
    good = np.isfinite(periods) & np.isfinite(c_phase) & (periods > 0) & (c_phase > 0)
    if good.sum() < 3:
        return out
    idx = np.where(good)[0]
    # sort by angular frequency ascending (period descending) for the derivative
    order = idx[np.argsort(periods[idx])[::-1]]
    w = 2.0 * np.pi / periods[order]
    s_c = 1.0 / c_phase[order]
    ds_c_dw = np.gradient(s_c, w)                 # handles non-uniform omega
    s_u = s_c + w * ds_c_dw
    with np.errstate(divide='ignore', invalid='ignore'):
        U = np.where(s_u > 0, 1.0 / s_u, np.nan)
    out[order] = U
    return out


def measure_corrections_and_phase(cwt_data, pick_per, pick_gv, dist, c_ref=None,
                                  phase_shift=np.pi / 4.0, phase_offset=0.0,
                                  use_period='nominal', joint=True, smooth_weight=3.0,
                                  phase_max_step=None):
    """
    Convenience wrapper: for each (nominal period, group velocity) pick, compute the
    Shapiro centroid period, the Levshin instantaneous period, and (if c_ref is given)
    the phase velocity with its resolved integer N.

    Args:
        cwt_data: dict from compute_cwt (same component/lag the picks came from)
        pick_per, pick_gv: 1D arrays of picked nominal periods [s] and group velocities [km/s]
        dist: inter-station distance [km]
        c_ref: reference phase-velocity callable, or None to skip phase velocity
        phase_shift, phase_offset: passed to phase_velocity
        use_period: 'nominal', 'centroid' or 'inst' — which period to assign omega in the
            phase-velocity formula (and to report as the measurement period downstream)

        joint: if True, resolve the 2*pi*N ambiguity jointly across all picks (Viterbi,
            curve-continuity + reference), which tracks one branch and avoids per-period
            cycle hops. If False, resolve each pick independently (per-period argmin).
        smooth_weight: curve-continuity weight passed to resolve_phase_curve (joint mode)

    Returns:
        dict of equal-length arrays: T_centroid, T_inst, phase_velocity, N_ambiguity
    """
    n = len(pick_per)
    T_centroid = np.full(n, np.nan)
    T_inst = np.full(n, np.nan)
    c_phase = np.full(n, np.nan)
    N_amb = np.zeros(n, dtype=int)

    # Pass 1: corrections + the single phase measurement and omega-period for every pick.
    phases = np.full(n, np.nan)
    Uref = np.full(n, np.nan)
    T_omega = np.full(n, np.nan)
    for i in range(n):
        T = float(pick_per[i])
        U = float(pick_gv[i])
        if T <= 0 or U <= 0:
            continue
        T_centroid[i] = centroid_period(cwt_data, T)
        m = measure_point(cwt_data, T, U, dist)
        T_inst[i] = m['T_inst']
        phases[i] = m['phase']
        Uref[i] = m['U']        # refined group velocity (dist/t_peak), self-consistent with phase
        T_omega[i] = {'nominal': T, 'centroid': T_centroid[i], 'inst': T_inst[i]}[use_period]

    # Pass 2: resolve the 2*pi*N ambiguity.
    if c_ref is not None:
        valid = np.isfinite(phases) & np.isfinite(T_omega) & (T_omega > 0)
        idx = np.where(valid)[0]
        if len(idx):
            if joint == 'unwrap':
                # single-N resolution via continuous frequency-unwrapping (Bensen 2007 eq. 11)
                c_sub, N_sub = resolve_phase_curve_unwrap(
                    T_omega[idx], phases[idx], Uref[idx], dist, c_ref,
                    phase_shift=phase_shift, phase_offset=phase_offset)
                c_phase[idx] = c_sub
                N_amb[idx] = N_sub
            elif joint:
                c_sub, N_sub = resolve_phase_curve(
                    T_omega[idx], phases[idx], Uref[idx], dist, c_ref,
                    phase_shift=phase_shift, phase_offset=phase_offset,
                    smooth_weight=smooth_weight, max_step=phase_max_step)
                c_phase[idx] = c_sub
                N_amb[idx] = N_sub
            else:
                for i in idx:
                    c_phase[i], N_amb[i] = phase_velocity(
                        phases[i], Uref[i], dist, T_omega[i], c_ref,
                        phase_shift=phase_shift, phase_offset=phase_offset)

    # Group velocity predicted from the extracted phase curve (Bensen eq. 7) -- a consistency
    # check against the directly-measured group ridge (large misfit => wrong branch / bad pick).
    U_from_phase = group_from_phase(T_omega, c_phase)
    return {'T_centroid': T_centroid, 'T_inst': T_inst,
            'phase_velocity': c_phase, 'N_ambiguity': N_amb, 'U_from_phase': U_from_phase}


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
