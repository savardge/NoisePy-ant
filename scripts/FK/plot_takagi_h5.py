""" Rotate substacks from E-N-Z to R-T-Z system, reading from H5 files"""
from collections import Counter
from noisepy.stacking import rotation
from noisepy import binstack
import os
import sys
import obspy
import pyasdf
import numpy as np
import matplotlib.pyplot as plt
import time
from noisepy.filter import lowpass, bandpass
import glob

datadir = "/home/users/s/savardg/scratch/riehen/STACK_CHRI_normZ/"
sfiles = glob.glob(os.path.join(datadir, "*", "*.h5"))

window = sys.argv[1]  # "T1662840000"
output_dir = "/home/users/s/savardg/scratch/extract_ncfs/riehen/by_windows_ZR-RZ/"
output_file_time = os.path.join(output_dir, f"riehen_ncfs_wCH_{window}_ZR-RZtime.jpg")
output_file_fk = os.path.join(output_dir, f"riehen_ncfs_wCH_{window}_ZR-RZfk.jpg")


def get_substack_gather_1comp(sfiles, dtype, comp="ZZ"):
    # Get parameters common to all pairs
    nPairs = len(sfiles)
    n = 0
    for sfile in sfiles:
        try:
            with pyasdf.ASDFDataSet(sfiles[0], mode="r") as ds:
                maxlag = ds.auxiliary_data[dtype][comp].parameters["maxlag"]
                dt = ds.auxiliary_data[dtype][comp].parameters["dt"]  # 0.04
                n = np.array(ds.auxiliary_data[dtype][comp].data).shape[0]  # 3001 for 60 s lag, dt = 0.04
            break
        except:
            continue
    if n == 0: return 0

    # Initialize
    r = np.zeros(nPairs)  # array of distances between pairs
    t = np.arange(-((n - 1) / 2) * dt, ((n) / 2) * dt, dt)  # Array of lag time
    ncts = np.zeros([nPairs, n], dtype=np.float32)  # Array of CCFs in time domain
    azimuth = np.zeros(nPairs)
    backazimuth = np.zeros(nPairs)
    station_source = np.chararray(nPairs, itemsize=17)
    station_receiver = np.chararray(nPairs, itemsize=17)
    numgood = np.zeros(nPairs)

    # Get ncfs
    t0 = time.time()  # To get runtime of code
    ibad = []  # Indices for corrupted data files
    list_of_data = []
    for _i, filename in enumerate(sfiles):
        if _i % 1000 == 0: print(f"{_i + 1}/{nPairs}")
        pair = os.path.split(filename)[1].split(".h5")[0]
        net1, sta1 = pair.split("_")[0].split(".")
        net2, sta2 = pair.split("_")[1].split(".")

        # *** Read data from .h5
        try:
            with pyasdf.ASDFDataSet(filename, mode="r") as ds:
                r[_i] = ds.auxiliary_data[dtype][comp].parameters["dist"]
                numgood[_i] = ds.auxiliary_data[dtype][comp].parameters["ngood"]
                tdata = ds.auxiliary_data[dtype][comp].data[:]
                lonR = ds.auxiliary_data[dtype][comp].parameters["lonR"]
                lonS = ds.auxiliary_data[dtype][comp].parameters["lonS"]

                # Correct polarity issue
                if net1 != net2:
                    tdata *= -1

                # Flip so we have West to East for positive lags
                if lonS > lonR:
                    ncts[_i, :] = np.flip(tdata)
                    azimuth[_i] = ds.auxiliary_data[dtype][comp].parameters["baz"]
                    backazimuth[_i] = ds.auxiliary_data[dtype][comp].parameters["azi"]
                    station_source[_i] = f"{net2}.{sta2}"
                    station_receiver[_i] = f"{net1}.{sta1}"

                else:
                    ncts[_i, :] = tdata
                    azimuth[_i] = ds.auxiliary_data[dtype][comp].parameters["azi"]
                    backazimuth[_i] = ds.auxiliary_data[dtype][comp].parameters["baz"]
                    station_source[_i] = f"{net1}.{sta1}"
                    station_receiver[_i] = f"{net2}.{sta2}"

        except:
            ibad.append(_i)
            continue

    print(f"Time elapsed to read data: {time.time() - t0:.0f}")

    # *** Remove bad indices
    ncts = np.delete(ncts, ibad, axis=0)
    r = np.delete(r, ibad, axis=0)
    numgood = np.delete(numgood, ibad, axis=0)
    azimuth = np.delete(azimuth, ibad, axis=0)
    backazimuth = np.delete(backazimuth, ibad, axis=0)
    station_source = np.delete(station_source, ibad, axis=0)
    station_receiver = np.delete(station_receiver, ibad, axis=0)

    # *** Sort by increasing distance
    indx = np.argsort(r)
    r = r[indx]
    ncts = ncts[indx, :]
    numgood = numgood[indx]
    azimuth = azimuth[indx]
    backazimuth = backazimuth[indx]
    station_source = station_source[indx]
    station_receiver = station_receiver[indx]

    # *** Save
    data = {
        "r": r,
        "ncts": ncts,
        "t": t,
        "numgood": numgood,
        "azimuth": azimuth,
        "backazimuth": backazimuth,
        "station_source": station_source,
        "station_receiver": station_receiver,
        "dt": dt,
        "maxlag": maxlag
    }

    return data


# Extract substack data for a time window
list_of_data = []
t0 = time.time()
for ccomp in ['EE', 'EN', 'EZ', 'NE', 'NN', 'NZ', 'ZE', 'ZN', 'ZZ']:
    t1 = time.time()
    data = get_substack_gather_1comp(sfiles, dtype="T1662163200", comp=ccomp)
    list_of_data.append(data)
    print(f"Time to read component {ccomp}: {time.time() - t1} s")
print(f"Time to read all stacks for all components: {time.time() - t0} s")


""" Now find station pairs for which there are 9 components for this time window """
def find_common_values(list_of_arrays):
    # Convert each array to a Counter object
    counters = [Counter(arr) for arr in list_of_arrays]

    # Get the intersection of all the Counter objects
    intersection = set.intersection(*[set(c.keys()) for c in counters])

    # Return the common values
    return [value for value in intersection if all(c[value] > 0 for c in counters)]


list_of_arrays = []
for data in list_of_data:
    station_join = np.char.add(data["station_source"], data["station_receiver"])
    list_of_arrays.append(station_join)
common_pairs = np.array(find_common_values(list_of_arrays))

# Exclude CH stations
exclude = []
for ipair, pair in enumerate(common_pairs):
    if "CH." in pair.decode():
        exclude.append(ipair)
common_pairs = np.delete(common_pairs, exclude)

print(f"Number of pairs with 9-component data: {common_pairs.shape[0]}")

list_of_indices = []
list_of_ncts = []
for array, data in zip(list_of_arrays, list_of_data):
    indices = np.nonzero(np.in1d(array, common_pairs))[0]
    list_of_indices.append(indices)
    list_of_ncts.append(data["ncts"][indices, :])

""" Now calculate RTZ cross-components """

list_of_azimuth = list_of_data[0]["azimuth"][list_of_indices[0]]
list_of_backazimuth = list_of_data[0]["backazimuth"][list_of_indices[0]]
npairs = len(common_pairs)
parameters = data
list_of_rotated_stacks = []
for ipair in range(npairs):
    ccfs = [ncts[ipair, :] for ncts in list_of_ncts]
    bigstack = np.vstack(ccfs)
    params = {"azi": list_of_azimuth[ipair], "baz": list_of_backazimuth[ipair]}
    bigstack_rotated = rotation(bigstack, params, locs=[])
    # ouput order: ['ZR','ZT','ZZ','RR','RT','RZ','TR','TT','TZ']
    list_of_rotated_stacks.append(bigstack_rotated)

""" Make the binned stacks """
dr = 150  # bin interval
t = list_of_data[0]["t"]
r = list_of_data[0]["r"][list_of_indices[0]]
dt = list_of_data[0]["dt"]


def get_binstack(stack_comp):
    M, Msym = binstack.symmetric_stack_time(stack_comp, t, r, plot=False, tmaxplot=20)
    _, ncts_sym_binned_nonan, _, tsym, distsym, _ = binstack.binned_stack_time(M, Msym, dt=0.04, t=t, r=r * 1e3, dr=150,
                                                                               plot=False, tmaxplot=20, dmaxplot=None)

    # *** Remove nan rows
    # distances = edges[1:].astype(np.float32).copy()
    # np.nan_to_num(ncts_binned, copy=False)
    np.nan_to_num(ncts_sym_binned_nonan, copy=False)

    # Normalize
    D = ncts_sym_binned_nonan / np.max(np.abs(ncts_sym_binned_nonan))

    return D, tsym, distsym


print("(ZR+RZ)/2")
ZRpRZcomp = np.vstack([0.5 * (stack[0, :] + stack[5, :]) for stack in list_of_rotated_stacks])
ZRpRZ, tsym, distsym = get_binstack(ZRpRZcomp)

print("(ZR-RZ)/2")
ZRmRZcomp = np.vstack([0.5 * (stack[0, :] - stack[5, :]) for stack in list_of_rotated_stacks])
ZRmRZ, _, _ = get_binstack(ZRmRZcomp)

print("ZZ")
ZZcomp = np.vstack([stack[2, :] for stack in list_of_rotated_stacks])
ZZ, _, _ = get_binstack(ZZcomp)

print("TT")
TTcomp = np.vstack([stack[7, :] for stack in list_of_rotated_stacks])
TT, _, _ = get_binstack(TTcomp)

""" Plot the time gathers """
freqmin = 0.25
freqmax = 1.2
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
ZZf = bandpass(ZZ, freqmin=freqmin, freqmax=freqmax, df=1 / dt, corners=4, zerophase=True)
axs[0][0].pcolormesh(tsym, distsym, ZZf / np.max(ZZf), cmap='gray', shading='auto', vmin=-1, vmax=1)
axs[0][0].set_title("ZZ")
ZRpRZf = bandpass(ZRpRZ, freqmin=freqmin, freqmax=freqmax, df=1 / dt, corners=4, zerophase=True)
axs[0][1].pcolormesh(tsym, distsym, ZRpRZf / np.max(ZRpRZf), cmap='gray', shading='auto', vmin=-1, vmax=1)
axs[0][1].set_title("(ZR + RZ)/ 2")
ZRmRZf = bandpass(ZRmRZ, freqmin=freqmin, freqmax=freqmax, df=1 / dt, corners=4, zerophase=True)
axs[1][0].pcolormesh(tsym, distsym, ZRmRZf / np.max(ZRmRZf), cmap='gray', shading='auto', vmin=-1, vmax=1)
axs[1][0].set_title("(ZR - RZ)/ 2")
TTf = bandpass(TT, freqmin=freqmin, freqmax=freqmax, df=1 / dt, corners=4, zerophase=True)
axs[1][1].pcolormesh(tsym, distsym, TTf / np.max(TTf), cmap='gray', shading='auto', vmin=-1, vmax=1)
axs[1][1].set_title("TT")
for ax in axs.ravel():
    ax.set(xlabel="Lag time [s]", ylabel="Distance [m]", xlim=(0, 20), ylim=(0, np.max(distsym)))
    ax.plot(tsym, tsym * 5870, c="b", lw=2, ls=":")  # Reference line for V=5 km/s
    ax.plot(tsym, tsym * 3460, c="r", lw=2, ls=":")  # Reference line for V=3 km/s
    ax.plot(tsym, tsym * 2000, c="g", lw=2, ls=":")  # Reference line for V=1.5 km/s
    ax.plot(tsym, tsym * 1500, c="g", lw=2, ls=":")  # Reference line for V=1.5 km/s

timestr = obspy.UTCDateTime(int(window[1:])).strftime("%Y-%m-%dT%H:%M:%S")
fig.suptitle(f"{window}: {timestr}")
fig.tight_layout()
# plt.show()
plt.savefig(output_file_time, format="JPEG")
plt.close()
print(f"Figure saved to {output_file_time}")

""" F-K decomposition """
f, k, fkZZ, _ = binstack.fk_decomposition_pos(ZZ, dt=dt, dr=dr)
f, k, fkZRpRZ, _ = binstack.fk_decomposition_pos(ZRpRZ, dt=dt, dr=dr)
f, k, fkZRmRZ, _ = binstack.fk_decomposition_pos(ZRmRZ, dt=dt, dr=dr)
f, k, fkTT, _ = binstack.fk_decomposition_pos(TT, dt=dt, dr=dr)

# Plotting limits
fmax = 1  # Hz
indf = max(np.argwhere(f < fmax))[0]
kmax = .5  # 1/km
indk = max(np.argwhere(k < kmax))[0]

# Plot
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
axs[0][0].pcolormesh(f[:indf], k[:indk], fkZZ[:indk, :indf], cmap='jet', shading='auto')
axs[0][0].set_title("ZZ")
axs[0][1].pcolormesh(f[:indf], k[:indk], fkZRpRZ[:indk, :indf], cmap='jet', shading='auto')
axs[0][1].set_title("(ZR + RZ)/ 2")
axs[1][0].pcolormesh(f[:indf], k[:indk], fkZRmRZ[:indk, :indf], cmap='jet', shading='auto')
axs[1][0].set_title("(ZR - RZ)/ 2")
axs[1][1].pcolormesh(f[:indf], k[:indk], fkTT[:indk, :indf], cmap='jet', shading='auto')
axs[1][1].set_title("TT")
for ax in axs.ravel():
    ax.set_xlabel(r'Frequency (Hz)')
    ax.set_ylabel(r'Wavenumber (1/km)')
    ax.grid(c="w", ls=":", lw=.5)
# axs[0][0].set_xlim((0, 1.0))
# axs[0][0].set_ylim((0, 0.5))
timestr = obspy.UTCDateTime(int(window[1:])).strftime("%Y-%m-%dT%H:%M:%S")
fig.suptitle(f"{window}: {timestr}")
fig.tight_layout()
plt.savefig(output_file_fk, format="JPEG")
plt.close()
print(f"Figure saved to {output_file_fk}")