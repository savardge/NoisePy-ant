
from matplotlib import mlab
from obspy.signal.util import prev_pow_2
from obspy.signal.spectral_estimation import fft_taper
import matplotlib.pyplot as plt
import numpy as np


def plot_psd(substacks, sampling_rate=25.0, flims=None, figsize=(15, 5), dB=False):
    """
    Plot the mean power spectral amplitudes of substacks and their min-max bounds
    Modified from Obspy (obspy/signal/spectral_estimation.py L954)
    Args:
        substacks: numpy 2D array of substacks. Lag time along rows, subwindow time along columns
        sampling_rate: in Hz [float, 25]
        flims: Frequency limits for plot [tuple, None]
        figsize: size of figure [tuple, (15,5)]
        dB: to plot with amplitude in dB [bool, False]

    Returns:
        frequencies, mean spectrum, minimum spectrum, max spectrum
    """
    nfft = prev_pow_2(substacks.shape[1])
    nsta = substacks.shape[0]
    specs_all = np.zeros(shape=(nsta, int(nfft / 2)))
    print(nsta, nfft)
    for k in range(substacks.shape[0]):
        data = substacks[k, :]
        spec, freq = mlab.psd(data, nfft, sampling_rate,
                              detrend=mlab.detrend_linear, window=fft_taper,
                              noverlap=int(0.75 * nfft), sides='onesided',
                              scale_by_freq=True)
        # leave out first entry (offset)
        spec = spec[1:]

        # avoid calculating log of zero
        dtiny = np.finfo(0.0).tiny
        idx = spec < dtiny
        spec[idx] = dtiny

        if dB:
            # go to dB
            spec = np.log10(spec)
            spec *= 10

        # Smooth
        kernel_size = 10
        kernel = np.ones(kernel_size) / kernel_size
        spec_smooth = np.convolve(spec, kernel, mode='same')

        # fig, ax = plt.subplots(1,1, figsize=(16,5))
        # ax.plot(freq[1:], spec, c="k")
        # ax.plot(freq[1:], spec_smooth, c="r")
        # ax.set_xlabel("Frequency (Hz)")
        # ax.set_ylabel("Amplitude")
        # plt.show()
        # plt.close()

        specs_all[k, :] = spec_smooth

    mean_spec = np.mean(specs_all, axis=0)
    min_spec = np.min(specs_all, axis=0)
    max_spec = np.max(specs_all, axis=0)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(freq[1:], min_spec, c="g", ls="--")
    ax.plot(freq[1:], max_spec, c="g", ls="--")
    ax.plot(freq[1:], mean_spec, c="k")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD amplitude (dB)")
    if flims:
        ax.set_xlim(flims)
    plt.show()
    plt.close()

    return freq[1:], mean_spec, min_spec, max_spec
