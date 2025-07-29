import glob
import numpy as np
import os
import pyasdf
import pycwt
import scipy
import sys
import matplotlib.pyplot as plt
import obspy
from scipy import fft
from scipy import interpolate
from scipy.signal import hilbert
import pandas as pd
from obspy.geodetics.base import gps2dist_azimuth


# FTAN
def nb_filt_gauss(ccf, dt, fn_array, dist, alpha=5):
    vmin = 1.5
    vmax = 4.5
    t = np.arange(0, len(ccf)) * dt
    # Define signal and noise windows
    signal_win = np.arange(int(dist / vmax / dt), int(dist / vmin / dt))
    noise_istart = len(ccf) - 2 * len(signal_win)
    noise_win = np.arange(noise_istart, noise_istart + len(signal_win))
    # noise_rms = np.sqrt(np.sum(ccf[noise_win]**2)/len(noise_win))
    # snr_bb = np.max(np.abs(ccf[signal_win])) / noise_rms # broadband snr

    # Narrowband filtering with Gaussian
    omgn_array = 2 * np.pi * fn_array

    # Transform ccf to frequency domain
    Nfft = fft.next_fast_len(len(ccf))
    ccf_freq = fft.fft(ccf, n=Nfft)
    freq_samp = 2 * np.pi * abs(fft.fftfreq(Nfft, dt))

    # Narrowband filtering
    # ccf_time_nbG = np.zeros(shape=(len(omgn_array), len(ccf)), dtype=np.float32)
    # ccf_time_nbG_env = np.zeros(shape=(len(omgn_array), len(ccf)), dtype=np.float32)
    tmax_nbG = np.zeros(shape=(len(omgn_array),), dtype=np.float32)
    Amax_nbG = np.zeros(shape=(len(omgn_array),), dtype=np.float32)
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
        isnr = np.argmax(amplitude_envelope)  # + signal_win[0]
        Amax_nbG[iomgn] = amplitude_envelope[isnr]
        tmax_nbG[iomgn] = t[isnr]
        if isnr == 0 or isnr == len(amplitude_envelope) - 1:
            snr_nbG[iomgn] = 0
        else:
            noise_rms = np.sqrt(np.sum(ccftnbg[noise_win] ** 2) / len(noise_win))
            snr_nbG[iomgn] = np.max(ccftnbg[signal_win]) / noise_rms

    return snr_nbG, tmax_nbG, Amax_nbG  # , ccf_time_nbG , ccf_time_nbG_env


# Get lags (symmetric, positive, negative)
def get_ccf_lag(tdata, params):
    npts = int(1 / params["dt"]) * 2 * params["maxlag"] + 1
    indx = npts // 2
    data_neg = np.flip(tdata[:indx + 1], axis=0)
    data_pos = tdata[indx:]
    data_sym = 0.5 * tdata[indx:] + 0.5 * np.flip(tdata[:indx + 1], axis=0)

    return data_sym, data_pos, data_neg


# Read stack file and get ZZ, ZR, RZ, RR
def get_zr_ccomps(sfile):
    tmp = sfile.split('/')[-1].split('_')
    spair = tmp[0] + '_' + tmp[1][:-3]
    net1 = spair.split("_")[0].split(".")[0]
    net2 = spair.split("_")[1].split(".")[0]

    st = obspy.Stream()
    for comp in ["ZZ", "ZR", "RR", "RZ"]:

        with pyasdf.ASDFDataSet(sfile, mode='r') as ds:
            dtype = 'Allstack_' + stack_method
            try:
                # print(ds.auxiliary_data[dtype].list())
                tdata = ds.auxiliary_data[dtype][comp].data[:]
                params = ds.auxiliary_data[dtype][comp].parameters
                dt = params["dt"]
            except Exception as e:
                raise ValueError(e)

        # Polarity fix if needed        
        if net1 != net2:
            tdata *= -1

        # Get lags
        data_sym, data_pos, data_neg = get_ccf_lag(tdata, params)

        # Make Obspy Traces
        tr = obspy.Trace(data=data_sym, header={"delta": dt, "station": spair, "channel": comp, "location": "sym"})
        st.append(tr)
        tr = obspy.Trace(data=data_pos, header={"delta": dt, "station": spair, "channel": comp, "location": "pos"})
        st.append(tr)
        tr = obspy.Trace(data=data_neg, header={"delta": dt, "station": spair, "channel": comp, "location": "neg"})
        st.append(tr)

    return st, params


def get_hv(sfile, Tn_array, station_receiver, lag="pos"):
    tmp = sfile.split('/')[-1].split('_')
    sta_src = tmp[0]
    sta_rcv = tmp[1][:-3]

    # Get ZZ, ZR, RZ, RR and all lags
    st, params = get_zr_ccomps(sfile)
    dist = params["dist"]
    dt = params["dt"]

    # Convert CCF to EGF (see eg Lin 2008)
    st.differentiate()
    for tr in st:
        tr.data *= -1

    st.taper(0.005)

    fn_array = np.divide(1., Tn_array)

    # Get amplitude and arrival times    
    df = pd.DataFrame({"period": Tn_array})
    # df["sta1"] = sta_src
    # df["sta2"] = sta_rcv
    df["dist_over_lambda"] = np.divide(dist, vs_ave * Tn_array)
    for icomp, comp in enumerate(["ZZ", "ZR", "RZ", "RR"]):
        ccf = st.select(location=lag, channel=comp)[0].data
        snr_nbG, tmax_nbG, Amax_nbG = nb_filt_gauss(ccf, dt, fn_array, dist, alpha=5)
        df["snr_" + comp] = snr_nbG
        df["time_" + comp] = tmax_nbG
        df["amp_" + comp] = Amax_nbG

    # Calculate H/V ratio
    if sta_rcv.split(".")[1] == station_receiver:
        # Source is 1st station: vertical source = ZZ-ZR, radial source = RR-RZ 
        df["source"] = sta_src
        df["receiver"] = sta_rcv
        hv_vertical = df["amp_ZR"].values / df["amp_ZZ"].values
        hv_radial = df["amp_RR"].values / df["amp_RZ"].values

    elif sta_src.split(".")[1] == station_receiver:
        # Source is 2nd station: vertical source = ZZ-RZ, radial source = ZR-RR
        df["source"] = sta_rcv
        df["receiver"] = sta_src
        hv_vertical = df["amp_RZ"].values / df["amp_ZZ"]
        hv_radial = df["amp_RR"].values / df["amp_ZR"]
    else:
        raise ValueError(f"station_receiver is in neither pair...{sta_rcv}, {sta_src}")
    df["HV_vertical"] = hv_vertical
    df["HV_radial"] = hv_radial
    df["percent_diff"] = np.abs(hv_radial - hv_vertical) / hv_vertical * 100
    return df


# ********************************************************************************************************************

# PARAMS
# datadir = '/home/users/s/savardg/scratch/aargau/STACK_CH-AA' 
datadir = '/home/users/s/savardg/scratch/aargau/STACK_CHAA_normZ'
# datadir = '/home/users/s/savardg/scratch/riehen/STACK_CHRI_norm' 
# datadir = '/home/users/s/savardg/scratch/riehen/STACK_CHRI_normZ'
stack_method = "pws"
station_file = "/home/users/s/savardg/aargau_ant/station_locations_CH-AA.csv"
# station_file = "/home/users/s/savardg/riehen/noisepy-nodes/stations_nodes_sed_noisepy.csv"
output_dir = "output_files_AA_normZ"
try:
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
except:
    pass

# Select station
station = str(sys.argv[1])  # "3007077"
stations = pd.read_csv(station_file)
print(f"Station is: {station}. Station file: {station_file}")
stations = stations.loc[stations["channel"].str.contains("Z")]
slat = stations.loc[stations.station == station, "latitude"].values[0]
slon = stations.loc[stations.station == station, "longitude"].values[0]

# Get distances to other stations 
dists = []
for irow, row in stations.iterrows():
    lat2 = row.latitude
    lon2 = row.longitude
    dist_m, _, _ = gps2dist_azimuth(slat, slon, lat2, lon2)
    # if dist_m == 0: continue
    dists.append(dist_m * 1e-3)
stations["distance"] = np.array(dists)

# Select stations in far field
vs_ave = 3.0
Tn_array = np.arange(0.2, 8, 0.2)
# T = 3.0
# select = stations.loc[ stations.distance > 1*T*vs_ave, :]
# select.head(30)

select = stations

# Get input file

flist = glob.glob(os.path.join(datadir, "*", f"*{station}*.h5"))
dfmaster = pd.DataFrame({'period': [],
                         'dist_over_lambda': [],
                         'snr_ZZ': [], 'time_ZZ': [], 'amp_ZZ': [],
                         'snr_ZR': [], 'time_ZR': [], 'amp_ZR': [],
                         'snr_RZ': [], 'time_RZ': [], 'amp_RZ': [],
                         'snr_RR': [], 'time_RR': [], 'amp_RR': [],
                         'source': [], 'receiver': [],
                         'HV_vertical': [], 'HV_radial': [],
                         'percent_diff': [], 'lag_type': []})
for irow, row in select.iterrows():
    sta = row.station
    dum = [sfile for sfile in flist if sta in sfile]
    if not len(dum):
        print(f"{sta} no found")
        continue
    sfile = dum[0]
    # print(sta, sfile)

    try:
        dfpos = get_hv(sfile, Tn_array=Tn_array, station_receiver=station, lag="pos")
        dfpos["lag_type"] = "pos"
        dfneg = get_hv(sfile, Tn_array=Tn_array, station_receiver=station, lag="neg")
        dfneg["lag_type"] = "neg"
        dfsym = get_hv(sfile, Tn_array=Tn_array, station_receiver=station, lag="sym")
        dfsym["lag_type"] = "sym"
        dfmaster = dfmaster.append(dfpos)
        dfmaster = dfmaster.append(dfneg)
        dfmaster = dfmaster.append(dfsym)
    except:
        pass

# Save to file
output_file = f"{output_dir}/{station}_HV_{stack_method}.csv"
dfmaster.to_csv(output_file)
print(f"DONE. Saved to {output_file}")
# df = dfmaster.loc[dfmaster["percent_diff"] < 20, :]
# df.loc[df["lag_type"] == "sym", "HV_vertical"].hist(bins=np.arange(0,5,0.1))
# df.loc[df["lag_type"] == "sym", "HV_radial"].hist(bins=np.arange(0,5,0.1))
