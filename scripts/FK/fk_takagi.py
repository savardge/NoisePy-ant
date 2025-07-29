"""
Example script on how to produce a FK plot for the ZZ, (ZR-RZ)/2, (ZR+RZ)/2 and TT cross-components
from the H5 stack files or from the files produced by extract_ncts.py
"""
import time
import glob
import os
from noisepy import binstack
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 24})

# If starting from H5 stack file list
useH5 = True
stack_dir = "/media/savardg/sandisk4TB/aargau-data/STACK_CHAA_normZ/"
sfiles = glob.glob(os.path.join(stack_dir, "AA.*", "AA.*AA.*.h5"))

# Define distance interval in meters
dr = 150

# List of components to extract
components = ["ZZ", "RR", "TT", "ZR", "RZ"]  # "RT",  "TR", "TZ", "ZR", "ZT"]
Dlist = []  # Hold the binned stack for each component
Dfiltlist = []  # Binned stacks but with filtering
for i, comp in enumerate(components):
    irow = i // 3
    icol = i % 3

    # Get the stacks
    if useH5:
        # If starting from the H5 stack files (very very slow!!!):
        t_start = time.time()
        _, r, f, ncts, t, dt, _, _ = binstack.get_stack_gather(sfiles, stack_method="Allstack_pws", comp="ZZ")
        print(f"Took {time.time() - t_start:.1f} seconds to extract CCF data from stack files")
        # r = r * 1e3  # Inter-station distances in m
    else:
        # If extract_ncts was run previously, use the produced files with stacks as numpy arrays
        # print(glob.glob(os.path.join("/home/users/s/savardg/riehen/extract_ncfs",f"riehen_ncfs_Allstack_pws_{comp}.npz")))
        data = np.load(glob.glob(os.path.join("/home/users/s/savardg/scratch/extract_ncfs", "aargau",
                                              f"aargau_ncfs_wCH_Allstack_pws_{comp}.npz"))[0])
        ncts = data['ncts']
        r = data['r']  # *1e3 # Inter-station distances in m
        t = data['t']
        dt = data['dt']

    # Get symmetric binned stack
    Mp, Msym = binstack.symmetric_stack_time(ncts, t, r, plot=False)
    ncts_binned, ncts_sym_binned_nonan, edges, tsym, distsym, _ = binstack.binned_stack_time(Mp, Msym, dt, t, r, dr=dr,
                                                                                             plot=False)
    # *** Remove nan rows
    distances = edges[1:].astype(np.float32).copy()
    np.nan_to_num(ncts_binned, copy=False)
    np.nan_to_num(ncts_sym_binned_nonan, copy=False)

    # Plot
    # D = ncts_binned/np.max(np.abs(ncts_binned))
    D = ncts_sym_binned_nonan / np.max(np.abs(ncts_sym_binned_nonan))
    # D_filt = bandpass(D,freqmin=.2,freqmax=1,df=int(1/dt),corners=4, zerophase=True)
    # D_filt = bandpass(D,freqmin=.1,freqmax=1,df=int(1/dt),corners=4, zerophase=True)
    # D_filt = bandpass(D,freqmin=.5,freqmax=1.5,df=int(1/dt),corners=4, zerophase=False)

    Dlist.append(D)
    # Dfiltlist.append(D_filt)

# Get FK
ZZ = Dlist[0]
RR = Dlist[1]
TT = Dlist[2]
ZR = Dlist[3]
RZ = Dlist[4]
ZRpRZ = (ZR + RZ) * 0.5
ZRmRZ = (ZR - RZ) * 0.5
f, k, fkZZ, _ = binstack.fk_decomposition_pos(ZZ, dt=dt, dr=dr)
f, k, fkZRpRZ, _ = binstack.fk_decomposition_pos(ZRpRZ, dt=dt, dr=dr)
f, k, fkZRmRZ, _ = binstack.fk_decomposition_pos(ZRmRZ, dt=dt, dr=dr)
f, k, fkTT, _ = binstack.fk_decomposition_pos(TT, dt=dt, dr=dr)

# Plotting limits
fmax = 3  # Hz
indf = max(np.argwhere(f < fmax))[0]
kmax = 3  # 1/km
indk = max(np.argwhere(k < kmax))[0]

# Plot
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(15, 15))
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
axs[0][0].set_xlim((0, 1.0))
axs[0][0].set_ylim((0, 0.5))
fig.tight_layout()
plt.show()
plt.close()
