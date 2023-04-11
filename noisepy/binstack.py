import pyasdf
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, next_fast_len
import scipy
import time
from obspy.signal.filter import lowpass, bandpass
import scipy.fftpack as sfft
from numpy import matlib as mb


def get_stack_gather(sfiles, stack_method="Allstack_pws", comp="ZZ"):
    """
    This function takes in a list of the H5 stack files and outputs the CCFs and key information into numpy arrays,
    for further analysis (e.g. bin stacks, beamforming, FK analysis)
    Args:
        sfiles: List of stack files in H5 format outputted by NoisePy. One file per station pair.
        stack_method: Stack method to use as labelled in the H5 files (e.g. "Allstack_linear")
        comp: Cross-component to extract (e.g. "ZZ", "TT", ...)

    Returns:
        ncfs0: 2D array of FFT of CCFs. Dimensions: (station pairs, frequencies)
        r0: Vector of inter-station distances, in the same order as first dimension of ncfs0 [km]
        f: Vector of frequencies, in the same order as second dimension of ncfs0
        ncts0: 2D array of CCFs in time domain. Dimensions: (station pairs, time lag)
        t: Vector of time lags, corresponding to 2nd dimension of ncts0
        dt: Sampling time interval (1 / sampling rate)
        azimuth: Azimuth from source station to receiver station
        numgood: Number of individual raw CCFs used in the stack

    """
    # Get parameters
    nPairs = len(sfiles)
    with pyasdf.ASDFDataSet(sfiles[0], mode="r") as ds:
        dt = ds.auxiliary_data[stack_method][comp].parameters["dt"]  # 0.04
        n = np.array(ds.auxiliary_data[stack_method][comp].data).shape[0]  # 3001 for 60 s lag, dt = 0.04
    Nfft = int(next_fast_len(n))  # 3072

    # Necessary variables for CC-FJpy: r, f, ncfs
    ncfs = np.zeros([nPairs, Nfft], dtype=np.complex64)  # array of CCFs in spectral domain
    r = np.zeros(nPairs)  # array of distances between pairs
    t = np.arange(-((n - 1) / 2) * dt, ((n) / 2) * dt, dt)  # Array of lag time
    ncts = np.zeros([nPairs, n], dtype=np.float32)  # Array of CCFs in time domain
    azimuth = np.zeros(nPairs)
    numgood = np.zeros(nPairs)

    # Get ncfs
    t0 = time.time()  # To get runtime of code
    ibad = []  # Indices for corrupted data files
    for _i, filename in enumerate(sfiles):

        if _i % 1000 == 0: print(f"{_i + 1}/{nPairs}")

        # *** Read data from .h5
        try:
            with pyasdf.ASDFDataSet(filename, mode="r") as ds:
                dist = ds.auxiliary_data[stack_method][comp].parameters["dist"]
                ngood = ds.auxiliary_data[stack_method][comp].parameters["ngood"]
                tdata = np.array(ds.auxiliary_data[stack_method][comp].data)
                lonR = ds.auxiliary_data[stack_method][comp].parameters["lonR"]
                lonS = ds.auxiliary_data[stack_method][comp].parameters["lonS"]
                if lonS > lonR:  # Flip so we have West to East for positive lags
                    tdata = np.flip(tdata)
                    azi = ds.auxiliary_data[stack_method][comp].parameters["baz"]
                else:
                    azi = ds.auxiliary_data[stack_method][comp].parameters["azi"]
        except:
            ibad.append(_i)
            continue

        # *** fft
        data_fft = fft(tdata, Nfft, axis=0)  # [:Nfft//2]
        f = scipy.fftpack.fftfreq(Nfft, d=dt)  # Frequencies

        # *** Save distance and spectrum
        r[_i] = dist
        spec = ncfs[_i, :] + data_fft
        spec /= np.max(np.abs(spec))
        ncfs[_i, :] = spec
        numgood[_i] = ngood
        azimuth[_i] = azi

        # *** Save time domain CCF
        ncts[_i, :] = tdata

    print(f"Time elapsed to read data: {time.time() - t0:.0f}")

    # *** Remove bad indices
    ncfs = np.delete(ncfs, ibad, axis=0)
    ncts = np.delete(ncts, ibad, axis=0)
    r = np.delete(r, ibad, axis=0)

    # *** Sort by increasing distance
    indx = np.argsort(r)
    r0 = r[indx]
    ncfs0 = ncfs[indx, :]
    ncts0 = ncts[indx, :]

    return ncfs0, r0, f, ncts0, t, dt, azimuth, numgood


def symmetric_stack_time(ncts, t, r, plot=True, tmaxplot=20):
    """
    Calculate the symmetric CCFs in the time domain as a function of inter-station distance and plot.
    Args:
        ncts: 2D array of CCFs in time domain, dimensions = (station pairs, time lag)
        t: Vector of time lag
        r: Vector of inter-station distances [km]
        plot: Whether to plot or not [bool]
        tmaxplot: Maximum time lag for plotting

    Returns:
        Mp: Normalized 2D array of CCFs
        Mpsym: Normallized and symmetric 2D array of CCFs

    """
    M = ncts.copy()
    trace_num = np.arange(0, len(r))
    Mp = M / np.max(np.abs(M), axis=1, keepdims=True)  # Normalize by max
    # stack positive and negative lags
    imid = len(t) // 2
    Msym = M[:, imid:].copy()
    Msym[:, 1:] += np.flip(M[:, :imid].copy(), axis=1)
    Msym /= 2
    Msym /= np.max(np.abs(Msym), axis=1, keepdims=True)  # Normalize by max

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].pcolormesh(t, trace_num, Mp, cmap='gray', vmin=-1, vmax=1)
        ax[0].set_title("Causal and acausal")
        ax[0].set_ylabel('trace #')
        ax[0].set_xlabel('Time (s)')
        ax[0].set_xlim((-tmaxplot, tmaxplot))

        ax[1].pcolormesh(t[imid:], trace_num, Msym, cmap='gray', vmin=-1, vmax=1)
        ax[1].set_title("Symmetric CCF")
        ax[1].set_ylabel('trace #')
        ax[1].set_xlabel('Time (s)')
        ax[1].set_xlim((0, tmaxplot))

        plt.show()
        plt.close()

    return Mp, Msym


def binned_stack_time(M, Msym, dt, t, r, dr=150, plot=True, tmaxplot=20, dmaxplot=None):
    """
    Calculate the Binned Stack for a given distance interval.
    All CCFs falling into a given inter-station distance interval dr are stacked.
    Args:
        M: 2D array of normalized CCFs in time domain, dimensions = (station pairs, time lag)
        Msym: 2D array of symmetric and normalized CCFs in time domain, dimensions = (station pairs, time lag)
        dt: Sampling time interval [s]
        t: Vector of time lags [s]
        r: Vector of inter-station distances [m]
        dr: Distance interval for each bin in the stack [m]
        plot: Whether to plot the binned stack [bool]
        tmaxplot: Max time lag for plotting
        dmaxplot: Max distance for plotting

    Returns:
        ncts_binned: Binned stack in the time domain.
        ncts_sym_binned_nonan: Symmetric binned stack in the time domain with intervals with no CCFs found removed
        edges: Distance intervals for the binned stack (ncts_binned)
        time: time lag for the symmetric binned stack (ncts_sym_binned_nonan)
        distances: Distance intervals for the binned stack with no empty rows (ncts_sym_binned_nonan)
        num_per_bin0: Number of CCFs in each bin (ncts_binned)
    """
    # *** Define bins
    H, edges = np.histogram(r, bins=np.arange(0, np.max(r), dr))

    # *** Initialize
    ncts_sym_binned = np.zeros([len(edges) - 1, Msym.shape[1]], dtype=np.float32)
    ncts_binned = np.zeros([len(edges) - 1, M.shape[1]], dtype=np.float32)

    # *** Stack in bins
    ibad = []
    num_per_bin = np.zeros(edges.shape)
    for k in np.arange(1, len(edges)):
        ix = np.argwhere((r < edges[k]) & (r >= edges[k - 1]))
        num_per_bin[k] = len(ix)
        if len(ix):
            # sym
            stack = np.sum(Msym[ix, :], axis=0)
            ncts_sym_binned[k - 1, :] = stack / np.max(np.abs(stack))

            # non-sym
            stack = np.sum(M[ix, :], axis=0)
            ncts_binned[k - 1, :] = stack / np.max(np.abs(stack))

        else:
            ibad.append(k - 1)

    #     print(f"Edges: {edges[k-1]} - {edges[k]} m, num CCFs = {len(ix)}")
    imid = len(t) // 2
    time = t[imid:].copy()

    distances0 = edges[1:].astype(np.float32).copy()
    num_per_bin0 = num_per_bin[1:].copy()

    # *** Remove nan rows
    ncts_sym_binned_nonan = np.delete(ncts_sym_binned, ibad, axis=0)
    distances = np.delete(distances0, ibad, axis=0)

    # *** Bandpass
    # ncts_binned = bandpass(ncts_binned,freqmin=0.5,freqmax=12,df=int(1/dt),corners=4, zerophase=True)

    # *** normalize
    # ncts_binned /= np.max(np.abs(ncts_binned), axis=1, keepdims=True)

    # *** Plot
    if plot:
        fig, ax = plt.subplots(1, 3, figsize=(15, 6), sharex=True, sharey=True)

        ax[0].pcolormesh(time, distances, ncts_sym_binned_nonan, cmap='gray', vmin=-1, vmax=1)
        ax[0].plot(time, time * 5000, c="b", lw=2, ls=":")
        ax[0].plot(time, time * 3000, c="r", lw=2, ls=":")
        ax[0].plot(time, time * 1200, c="g", lw=2, ls=":")
        ax[0].set_title("broadband")
        ax[0].set_ylabel('Distance (m)')
        ax[0].set_xlabel('Time (s)')
        ax[0].set_xlim((0, tmaxplot))
        if dmaxplot is None:
            ax[0].set_ylim((0, max(distances)))
        else:
            ax[0].set_ylim((0, dmaxplot))
        # ax[0].set_ylim((0,16000))

        D = ncts_sym_binned_nonan
        D1 = bandpass(D.copy(), freqmin=0.2, freqmax=1, df=int(1 / dt), corners=4, zerophase=True)
        ax[1].pcolormesh(time, distances, D1 / np.max(np.abs(D1)), cmap='gray', vmin=-1, vmax=1)
        ax[1].set_title("0.2 - 1 Hz")
        D2 = bandpass(D.copy(), freqmin=.6, freqmax=3.5, df=int(1 / dt), corners=4, zerophase=True)
        ax[2].pcolormesh(time, distances, D2 / np.max(np.abs(D2)), cmap='gray', vmin=-1, vmax=1)
        ax[2].set_title("0.6 - 3.5 Hz")
        ax[2].plot(time, time * 5000, c="b", lw=2, ls=":")
        ax[2].plot(time, time * 3000, c="r", lw=2, ls=":")
        ax[2].plot(time, time * 1200, c="g", lw=2, ls=":")

        plt.show()
        plt.close()

    return ncts_binned, ncts_sym_binned_nonan, edges.astype(np.float32), time, distances, num_per_bin0


def fk_decomposition(ncts_binned, dt=0.04, dr=0.150, plot=True, title=None, kmaxplot=10, doublelength=False):
    """
    Frequency-wavenumber decomposition applied to the binned stack
    Args:
        ncts_binned: 2D array of the binned stack, dimensions = (station pairs, time lags)
        dt: Sampling time interval [s]
        dr: Bin distance interval [km]
        plot: Whether to plot or not [bool]
        title: Title for the plot
        kmaxplot: Max wavenumber for plotting
        doublelength: Whether to increase Nfft to twice the number of time samples for the FFT [bool]

    Returns:
        fk: 2D array of the FK decomposition
        omega0: Frequency [rad/s]
        k0: Wavenumber [rad/km]
    """
    if doublelength:
        Nfft = sfft.next_fast_len(max(ncts_binned.shape) * 2)
    else:
        Nfft = sfft.next_fast_len(max(ncts_binned.shape))
    D = ncts_binned.copy()
    fk = sfft.fft2(D, shape=[Nfft, Nfft])
    fk_shift = sfft.fftshift(fk)

    omega0 = np.fft.fftfreq(fk.shape[1], d=dt) * 2 * np.pi  # omega in rad/s
    omega = np.fft.fftshift(omega0)
    k0 = np.fft.fftfreq(fk.shape[0], d=dr) * 2 * np.pi  # k in rad/km
    k = np.fft.fftshift(k0)

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.pcolormesh(omega, k, np.abs(fk_shift), cmap='jet')
        ax.set_xlabel(r'$\omega$ (rad/s)')
        ax.set_ylabel('k (rad/km)')
        ax.set_ylim((-kmaxplot, kmaxplot))
        ax.set_xlim((-kmaxplot, kmaxplot))
        if title:
            ax.set_title(title)
        ax.grid(c="w")
        plt.show()
        plt.close()

    return fk, omega0, k0


def fk_filtering_then_plot(ncts_binned, edges, t, dt=0.04, dr=0.150, cmin=0.7, cmax=7.0, lambda_min=0, lambda_max=20, plot=True):
    """
    FK decomposition followed by filtering based on phase velocity and wavelength,
    then inverse transform back to time domain to get filtered time domain CCFs

    Args:
        ncts_binned: Binned stack in the time domain
        edges: Distance bins corresponding to ncts_binned
        t: Vector of time lags
        dt: Sampling time interval [s]
        dr: Distance bin interval [km]
        cmin: minimum phase velocity [km/s]
        cmax: maximum phase velocity [km/s]
        lambda_min: minimum wavelength [km]
        lambda_max: maximum wavelength [km]
        plot: Whether to plot or not [bool]

    Returns:
        D_filt: CCFs filtered in the phase velocity and wavelength domain
        fk_filt: FK plot filtered in the phase velocity and wavelength domain
    """

    # Fk image via fft2
    D = ncts_binned.copy()
    fk = sfft.fft2(D)
    f = np.fft.fftfreq(fk.shape[1], d=dt)
    k = np.fft.fftfreq(fk.shape[0], d=dr) * 2 * np.pi  # k in rad/km
    omega = f * 2 * np.pi  # omega in rad/s

    # Get phase velocity, T, lambda... for each point in fk image
    fmat = mb.repmat(f, len(k), 1)
    kmat = mb.repmat(k, len(omega), 1).T
    Omat = mb.repmat(omega, len(k), 1)
    kmat[kmat == 0] = np.nan
    Omat[Omat == 0] = np.nan
    cmat = Omat * np.reciprocal(kmat)
    # Tmat = 2 * np.pi * np.reciprocal(Omat.astype(np.float32))
    lambmat = 2 * np.pi * np.reciprocal(kmat)

    # Make a mask
    with np.errstate(invalid='ignore', divide='ignore'):
        mask = np.zeros(fk.shape)
        mask[(np.abs(cmat) > cmin) & (np.abs(cmat) < cmax) & (lambmat > lambda_min) & (lambmat < lambda_max)] = 1

    # Now filter
    fkamp = np.abs(fk)  # Amplitude
    fkpha = np.angle(fk)  # Phase
    fkamp_filt = fkamp * mask  # Apply mask to amplitude only
    fk_filt = fkamp_filt * np.exp(1j * fkpha)  # Reconstruct complex image

    # Inverse fft2 to get back to time domain
    D_filt = sfft.ifft2(fk_filt)

    # Plot
    if plot:
        distances = edges.copy() * 1e-3
        fig, ax = plt.subplots(2, 1, figsize=(12, 12), sharex=True, sharey=True)
        ax[0].pcolormesh(t, distances, ncts_binned, cmap='gray', vmin=-1, vmax=1)
        ax[0].set_title("Before f-k filtering -- Broadband")
        ax[0].set_xlim(-30, 30)
        ax[0].set_xlabel("lag time (s)")
        ax[0].set_ylabel("distance (km)")

        #     D2_filt = bandpass(D_filt.copy(),freqmin=.2,freqmax=3.5,df=int(1/dt),corners=4, zerophase=True)
        #     ax[1].pcolormesh(t,distances,np.real(D2_filt),cmap='gray', vmin=-1, vmax=1)
        #     ax[1].set_title("After f-k filtering -- Filtered between 0.2 - 3.5 Hz")
        ax[1].pcolormesh(t, distances, np.real(D_filt), cmap='gray', vmin=-1, vmax=1)
        ax[1].set_title("After f-k filtering")
        ax[1].set_xlabel("lag time (s)")
        ax[1].set_ylabel("distance (km)")
        plt.show()
        plt.close()

    return D_filt, fk_filt


def fk_decomposition_pos(ncts_binned, dt, dr, plot=False, ax=None, title=None):
    """
    Plot the FK decomposition with positive frequencies and wavenumbers (fold the quadrants with negative values)
    Args:
        ncts_binned: Binned stack
        dt: Sampling time interval [s]
        dr: Distance bin interval [km]
        plot: Whether to plot of not [bool]
        ax: Matplotlib.pyplot axes where to plot
        title: Plot title
    Returns:
        newf: Frequency array (Hz)
        newk_km: Wavenumber array (km)
        fk_pos: 2D array of FK decomposition
        fk_pos_dB: 2D array of FK decomposition with amplitude in dB

    """
    # Get FK
    fk, omega, k = fk_decomposition(ncts_binned, dt=dt, dr=dr * 1e-3, plot=False, doublelength=False)
    n1 = fk.shape[0] // 2
    n2 = fk.shape[1] // 2
    newk = k[:n1]
    newom = omega[:n2]
    newf = np.divide(newom, 2 * np.pi)
    newk_km = np.divide(newk, 2 * np.pi)

    # FFT Shift
    fk = sfft.fftshift(fk)
    omega = np.fft.fftshift(omega)
    k = np.fft.fftshift(k)

    # Fold 4 quadrants of FK plot into positive quadrant

    if fk.shape[0] % 2 != 0:  # odd
        fkp3 = np.flipud(np.fliplr(np.abs(fk[:n1, :n2])))  # quadrant 3
        fkp4 = np.flipud(np.abs(fk[:n1, n2 + 1:]))  # quadrant 4
        fkp1 = np.abs(fk[n1 + 1:, n2 + 1:])  # quadrant 1
        fkp2 = np.fliplr(np.abs(fk[n1 + 1:, :n2]))  # quadrant 2
    else:  # even
        fkp3 = np.flipud(np.fliplr(np.abs(fk[:n1, :n2])))  # Q3
        fkp4 = np.flipud(np.abs(fk[:n1, n2:]))  # Q4
        fkp1 = np.abs(fk[n1:, n2:])  # Q1
        fkp2 = np.fliplr(np.abs(fk[n1:, :n2]))  # Q2
    fk_pos = 0.25 * (fkp1 + fkp2 + fkp3 + fkp4)  # Mean of 4 quadrants
    fk_pos_dB = 20 * np.log(np.abs(fk_pos))  # convert amplitude to decibels

    # Plot
    if plot:
        # ax.pcolormesh(newom,newk,np.abs(fk_pos),cmap='jet')
        # ax.set_xlabel(r'$\omega$ (rad/s)')
        # ax.set_ylabel('k (rad/km)')
        fk_pos /= np.max(fk_pos.flatten())  # Normalize amplitudes
        ax.pcolormesh(newf, newk_km, fk_pos, cmap='jet')
        ax.set_xlabel(r'Frequency (Hz)')
        ax.set_ylabel(r'Wavenumber (1/km)')
        ax.set_title(title)
        ax.grid(c="w", ls=":", lw=.5)

    return newf, newk_km, fk_pos, fk_pos_dB
