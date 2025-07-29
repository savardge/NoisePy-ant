import pycwt
import numpy as np
import math
from cmath import inf
import scipy
from scipy.fftpack import fft, ifft
from scipy.interpolate import interp1d
from scipy.signal import hilbert, windows, firwin, lfilter
from scipy.signal.windows import tukey as tukey_window
import matplotlib.pyplot as plt
import obspy
import os
import glob
import pyasdf
import sys

""" 
Necessary modules and functions

Make sure to download the packages pycwt, scipy, pyasdf, obspy and matplotlib
"""


def get_disp_image_taper(ccf, dist, dt, Tmin=0.2, Tmax=7, dT=0.01, vmin=0.5, vmax=3.5, dvel=0.005, vave=3., plot=True,
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
    ccf = ccf.copy()

    # Basic parameters for wavelet transform
    dj = 1 / 120  # Spacing between discrete scales. Default is Twelve sub-octaves per octaves.
    # Smaller values will result in better scale resolution, but slower calculation and plot.
    s0 = -1  # Smallest scale of the wavelet. Default value [-1] is 2*dt.
    J = -1  # Number of scales less one.
    # Scales range from s0 up to s0 * 2**(J * dj), which gives a total of (J + 1) scales.
    # Default [-1] is J = (log2(N * dt / so)) / dj.
    wvn = 'morlet'  # type of wavelet to use

    # Get period and velocity ranges
    # Tmax = dist / vave  # Max period assumes a velocity of 3 km/s
    fmin = 1 / Tmax
    fmax = 1 / Tmin

    # Douglas - added +dT +dvel to have consistent shape
    per = np.arange(Tmin, Tmax + dT, dT)  # Periods
    vel = np.arange(vmin, vmax + dvel, dvel)  # Group velocities

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
    pcwt = np.angle(cwt)  # Phase

    # Remove t=0 sample
    tvec = tvec[1:]
    rcwt = rcwt[:, 1:]
    pcwt = pcwt[:, 1:]
    coi = coi[1:]

    # Interpolation of the image to the requested intervals in period and velocity
    velocity = dist / tvec
    fc = scipy.interpolate.interp2d(velocity, period, rcwt)
    rcwt_new = fc(vel, per)
    fp = scipy.interpolate.interp2d(velocity, period, pcwt)
    pcwt_new = fp(vel, per)

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

    return rcwt_new, pcwt_new, per, vel, coi_new


def generate_coi_mask(coi_new, vel_grid, per_grid):
    mask = np.zeros_like(vel_grid)

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if per_grid[i, j] < coi_new[j]:
                mask[i, j] = 1
            else:
                mask[i, j] = 0

    return mask


# Not needed but keep in case
# def fold_trace(tr):
#    """fold_trace() takes a trace and computes the symmetric component by 
#    summing the time-reversed acausal component with the causal coponent"""
# Pick some constants we'll need from the trace:
#    npts2 = tr.stats.npts
#    npts = int((npts2-1)/2)

# Fold the CCFs:
#    tr_folded= tr.copy()
#    causal =  tr.data[npts:-1]
#    acausal = tr.data[npts:0:-1]
#    tr_folded.data =  (causal+acausal)/2

#    return tr_folded

###########################################################################################
""" 
Import data from NoisePy stack files
"""

# Input stack file name
fname = sys.argv[1]  # First input argument specified on command line
print(f"Processing this file: {fname}")

# Specify stack directory:
export_cwt_path = "/home/users/s/stumpp4/scratch/RoccaStrada/E_FTAN_and_picking/B_automatic_picking/CWT_DisperPicker/data/TestData/group_image/"
export_coi_path = "/home/users/s/stumpp4/scratch/RoccaStrada/E_FTAN_and_picking/CWT_coi_dat/"

""" 
Select the stacking method. Options: 
- Allstack_linear (linear stacking, taking the mean)
- Allstack_pws (phase weighted stack) 
- Allstack_nroot (N root stack Millet, F et al., 2019 JGR, with power=2)
- Allstack_robust (Palvis and Vernon 2010)
- Allstack_auto_covariance (Adaptive filter of Nakata et al., 2015 appendix B: with filter harshness g=1)
"""
stack_method = "Allstack_pws"

"""
Select the component: 
Options: ['EE', 'EN', 'EZ', 'NE', 'NN', 'NZ', 'RR', 'RT', 'RZ', 'TR', 'TT', 'TZ', 'ZE', 'ZN', 'ZR', 'ZT', 'ZZ']
"""
component = "ZZ"

# Setup up period and velocity limits and spacing
vmin = 0.5
vmax = 3.5
dvel = 0.005
Tmin = 0.2
Tmax = 7
dT = 0.01

# Process the input file fname
try:
    with pyasdf.ASDFDataSet(fname, mode="r") as ds:
        # print(ds.auxiliary_data.list()) # This shows the list of stack methods available
        # print(ds.auxiliary_data[stack_method].list()) # This shows the list of components available
        ccf_full = ds.auxiliary_data[stack_method][component].data[:]
        params = ds.auxiliary_data[stack_method][component].parameters

    pair = os.path.split(fname)[1].split(".h5")[0]
    # print(f"pair: {pair}, distance: {params['dist']}")

    # Make an Obspy trace for full symmetric component
    tr_full = obspy.Trace(ccf_full, header={'delta': params['dt'], 'station': pair, 'channel': component,
                                            'distance': params['dist']})
    npts2 = tr_full.stats.npts
    npts = int((npts2 - 1) / 2)
    causal = tr_full.data[npts:-1]
    acausal = tr_full.data[npts:0:-1]

    # Get the data
    dt = tr_full.stats.delta
    dist = params['dist']

    # Maxamplitude fonction and CF to EGFs conversion through hilbert transform
    PtNum = npts
    samp_rate = tr_full.stats.sampling_rate  # Sampling frequency (SampleF)
    Time = np.arange(0, (PtNum - 1) * dt, dt)

    maxamp = max(max(causal), max(acausal))

    if maxamp > 0:
        causal_new = causal / maxamp
        acausal_new = acausal / maxamp

        Green_causal = np.imag(hilbert(causal_new))
        Green_acausal = np.imag(hilbert(acausal_new))
        stackEGF = (Green_causal + Green_acausal) / 2.0

        amp, phase, per, vel, coi = get_disp_image_taper(stackEGF,  # numpy array of the symmetric CCF
                                                         dist,  # inter-station distance in km
                                                         dt,  # sampling interval in seconds
                                                         Tmin=Tmin,
                                                         Tmax=Tmax,  # min period
                                                         dT=dT,  # period step
                                                         vmin=vmin,  # min Vg
                                                         vmax=vmax,  # max Vg
                                                         dvel=dvel,  # steps in Vg
                                                         plot=False)

        vel_grid, per_grid = np.meshgrid(vel, per)
        coi_mask = generate_coi_mask(coi, vel_grid, per_grid)

        parts = pair.split('_')

        # 1. cwt amplitude
        cwt_export = np.transpose(amp)
        cwt_name_export = parts[0].split('.')[1] + '.' + parts[1].split('.')[1] + '.dat'
        np.savetxt(os.path.join(export_cwt_path, cwt_name_export), cwt_export)

        # 2. Coi mask/line
        coi_mask_export = np.flip(np.transpose(coi_mask))
        coi_mask_name_export = 'mask_' + parts[0].split('.')[1] + '.' + parts[1].split('.')[1] + '.dat'
        np.savetxt(os.path.join(export_coi_path, coi_mask_name_export), coi_mask_export)
        # coi_line_export = coi
        # coi_line_name_export = 'scatter_' + parts[0].split('.')[1] + '.' + parts[1].split('.')[1] + '.dat'
        # np.savetxt(os.path.join(export_coi_path,coi_line_name_export),coi_line_export)
    else:
        print("maxamp is 0 for the CCF.... skipping")

except:
    print(f"Could not export CWT for file {fname}")
