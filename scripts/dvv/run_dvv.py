"""
Script to run dvv functions
Takes as input a npy file containing:
CCdata: 3D array containing substacks (window time along rows, lag time along columns)
tlag: Lag time
times: timestamp of subwindows
sampling_rate: sampling rate in Hz
fmin, fmax: frequency of bandpass filter applied to CCdata
comp: component
maxlag: maximum lag time

"""
from noisepy import dvv_monitoring as dvv
import numpy as np
import os
import matplotlib.pyplot as plt
#from obspy import UTCDateTime
import datetime

input_file = "/home/genevieve/research/dvv_Francisco/AEGEAN_SEA/data/HL_LIA_HL_SMTH_stacks_NZ_0.05_1.0.npy"
AeganSeaEQ = np.datetime64("2014-05-24T09:25:02")

# Define new substack length and step
stacklen_new = np.timedelta64(5, "D")
step = np.timedelta64(5, "D")

# Define reference stack start-end time
tref_start = np.datetime64("2014-02-20")
tref_end = np.datetime64("2014-05-01")

# Define correlation threshold
cc_thresh = 0.3

# Define lag time window
vmin = 2.0  # minimum group velocity km/s
dist = 70.  # inter-station distance in km
lwin = 10  # length of window in s
twin = [int(dist/vmin), int(dist/vmin) + lwin]

# --------------------------------------------------------
# Get station names from file name
dum = os.path.split(input_file)[1].split("_stacks")[0].split("_")
sta1 = f"{dum[0]}.{dum[1]}"
sta2 = f"{dum[2]}.{dum[3]}"
print(f"Station pair: {sta1} - {sta2}")

# Read file
data = np.load("/home/genevieve/research/dvv_Francisco/AEGEAN_SEA/data/HL_LIA_HL_SMTH_stacks_NZ_0.05_1.0.npy",
               allow_pickle=True)
ndata = data["CCdata"].T
tlag_all = data["tlag"]
timestamp = data["times"].astype(np.datetime64)
sampling_rate = data["sampling_rate"]
fmin = data["fmin"]
fmax = data["fmax"]
comp = data["comp"]
maxlag = data["maxlag"]
npts_all = ndata.shape[1]
npts = npts_all//2
tlag = tlag_all[npts:]
ndata_pos = ndata[:, npts:]  # positive lag
ndata_neg = np.fliplr(ndata[:, :npts + 1])  # negative lag
ndata_sym = np.mean(np.vstack((ndata_pos[np.newaxis], ndata_neg[np.newaxis])), axis=0)
twin_indx = np.where((tlag >= np.min(twin)) & (tlag < np.max(twin)))[0]
npts_win = len(twin_indx)

# Plot positive, negative and symmetric lags for 1st row
fig, ax = plt.subplots(4, 1, figsize=(15, 10))
ix = 0
ax[0].plot(tlag_all, ndata[ix, :])
ax[0].set_title(f"substack at time {timestamp[0]}, both lags")
ax[1].plot(tlag, ndata_pos[ix, :])
ax[1].set_title(f"substack at time {timestamp[0]}, + lag")
ax[2].plot(tlag, ndata_neg[ix, :])
ax[2].set_title(f"substack at time {timestamp[0]}, - lag")
ax[3].plot(tlag, ndata_sym[ix, :])
ax[3].set_title(f"substack at time {timestamp[0]}, symmetric lag")
plt.show()

# Get reference stack
iref = np.where((timestamp > tref_start) & (timestamp < tref_end))[0]
refstack = np.mean(ndata[iref, :], axis=0)
refstack_pos = np.mean(ndata_pos[iref, :], axis=0)
refstack_neg = np.mean(ndata_neg[iref, :], axis=0)
refstack_sym = np.mean(ndata_sym[iref, :], axis=0)

# Change substack length
timestamp, ndata_sym = dvv.change_substack_length(timestamp, ndata_sym, stacklen_new, step, dt=1 / sampling_rate)
_, ndata_pos = dvv.change_substack_length(timestamp, ndata_pos, stacklen_new, step, dt=1 / sampling_rate)
_, ndata_neg = dvv.change_substack_length(timestamp, ndata_neg, stacklen_new, step, dt=1 / sampling_rate)

# Get correlation coefficient between substack and reference (before stretching)
nwin = ndata_sym.shape[0]
cc_pos = np.zeros(shape=(nwin,), dtype=np.float32)
cc_neg = np.zeros(shape=(nwin,), dtype=np.float32)
cc_sym = np.zeros(shape=(nwin,), dtype=np.float32)
for iwin in range(nwin):
    cc_pos[iwin] = np.corrcoef(refstack_pos[twin_indx], ndata_pos[iwin, twin_indx])[0, 1]
    cc_neg[iwin] = np.corrcoef(refstack_neg[twin_indx], ndata_neg[iwin, twin_indx])[0, 1]
    cc_sym[iwin] = np.corrcoef(refstack_sym[twin_indx], ndata_sym[iwin, twin_indx])[0, 1]

fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(15,10))
ax[0].plot(timestamp, cc_pos)
ax[0].axhline(cc_thresh, c="r", ls=":")
ax[0].set_title("Correlation with reference before stretching: positive lag")
ax[1].plot(timestamp, cc_neg)
ax[1].axhline(cc_thresh, c="r", ls=":")
ax[1].set_title("Correlation with reference before stretching: negative lag")
ax[2].plot(timestamp, cc_sym)
ax[2].axhline(cc_thresh, c="r", ls=":")
ax[2].set_title("Correlation with reference before stretching: symmetric lag")
for a in ax:
    a.set(ylabel="Corr. coeff.")
plt.show()

