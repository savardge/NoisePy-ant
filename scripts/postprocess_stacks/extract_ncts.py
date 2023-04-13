import pyasdf
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, next_fast_len
import scipy
import time
from obspy.signal.filter import lowpass
import sys


def get_stack_gather(sfiles, dtype="Allstack_pws", comp="ZZ", output_fname="dum.npz"):
    # Get parameters common to all pairs
    nPairs = len(filelist)
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
    # Necessary variables for CC-FJpy: r, f, ncfs
    # Necessary variables for beamforming: azimuth, backazimuth, distance, ncfs, (stations, coordinates)
    r = np.zeros(nPairs)  # array of distances between pairs
    t = np.arange(-((n - 1) / 2) * dt, ((n) / 2) * dt, dt)  # Array of lag time
    ncts = np.zeros([nPairs, n], dtype=np.float32)  # Array of CCFs in time domain
    azimuth = np.zeros(nPairs)
    backazimuth = np.zeros(nPairs)
    station_source = np.chararray(nPairs, itemsize=17)
    longitude_source = np.zeros(nPairs)
    latitude_source = np.zeros(nPairs)
    station_receiver = np.chararray(nPairs, itemsize=17)
    longitude_receiver = np.zeros(nPairs)
    latitude_receiver = np.zeros(nPairs)
    numgood = np.zeros(nPairs)

    # Get ncfs
    t0 = time.time()  # To get runtime of code
    ibad = []  # Indices for corrupted data files
    for _i, filename in enumerate(filelist):
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
                latR = ds.auxiliary_data[dtype][comp].parameters["latR"]
                lonS = ds.auxiliary_data[dtype][comp].parameters["lonS"]
                latS = ds.auxiliary_data[dtype][comp].parameters["latS"]

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
                    longitude_source[_i] = lonR
                    longitude_receiver[_i] = lonS
                    latitude_source[_i] = latR
                    latitude_receiver[_i] = latS
                else:
                    ncts[_i, :] = tdata
                    azimuth[_i] = ds.auxiliary_data[dtype][comp].parameters["azi"]
                    backazimuth[_i] = ds.auxiliary_data[dtype][comp].parameters["baz"]
                    station_source[_i] = f"{net1}.{sta1}"
                    station_receiver[_i] = f"{net2}.{sta2}"
                    longitude_source[_i] = lonS
                    longitude_receiver[_i] = lonR
                    latitude_source[_i] = latS
                    latitude_receiver[_i] = latR


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
    longitude_source = np.delete(longitude_source, ibad, axis=0)
    latitude_source = np.delete(latitude_source, ibad, axis=0)
    latitude_receiver = np.delete(latitude_receiver, ibad, axis=0)
    longitude_receiver = np.delete(longitude_receiver, ibad, axis=0)

    # *** Sort by increasing distance
    indx = np.argsort(r)
    r = r[indx]
    ncts = ncts[indx, :]
    numgood = numgood[indx]
    azimuth = azimuth[indx]
    backazimuth = backazimuth[indx]
    station_source = station_source[indx]
    station_receiver = station_receiver[indx]
    longitude_source = longitude_source[indx]
    latitude_source = latitude_source[indx]
    latitude_receiver = latitude_receiver[indx]
    longitude_receiver = longitude_receiver[indx]

    # *** Save
    np.savez(output_fname,
             r=r,
             ncts=ncts,
             t=t,
             numgood=numgood,
             azimuth=azimuth,
             backazimuth=backazimuth,
             station_source=station_source,
             station_receiver=station_receiver,
             longitude_source=longitude_source,
             latitude_source=latitude_source,
             latitude_receiver=latitude_receiver,
             longitude_receiver=longitude_receiver,
             dt=dt,
             maxlag=maxlag
             )

    return r  # , r0, f, ncts0, t, dt, azimuth, numgood, pairname0


if __name__ == "__main__":
    # Parse arguments
    comp = sys.argv[1]  # "RR"
    dtype = sys.argv[2]  # "Allstack_pws"
    datadir = sys.argv[3]  # "/home/users/s/savardg/scratch/riehen/STACK_CHRI_norm/"
    output_fname = sys.argv[4]  # f"riehen_ncfs_wCH_{dtype}_{comp}.npz"
    print(f"comp={comp}, dtype={dtype}, datadir={datadir}, output_fname={output_fname}")
    filelist = glob.glob(os.path.join(datadir, "*", "*.h5"))
    print(f"Number of files: {len(filelist)}")
    _ = get_stack_gather(filelist, dtype=dtype, comp=comp, output_fname=output_fname)
