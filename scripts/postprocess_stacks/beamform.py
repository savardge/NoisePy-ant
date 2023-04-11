import numpy as np
import time
import sys
import os, glob
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('font', **{'size':20})
from obspy.signal.filter import bandpass
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def plot_beam(P, title="Beamform",save=0,savename='none',cmax=0):
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_axes([0.1,0.1,0.6,0.6])  #x0,y0,dx,dy
    cmap = plt.get_cmap('inferno')
    i = plt.pcolor(ux-dux/2,uy-duy/2,np.real(P.T),cmap=cmap,rasterized=True)#,vmin=-4,vmax=4)
    if(cmax==0):
        cmax=np.max(np.abs(P))
    cmin=np.min(P)
    plt.clim(cmin, cmax)
    
    plt.axis('equal')
    plt.axis('tight')
    plt.xlim(min(ux)+dux,max(ux)-dux)
    plt.ylim(min(uy)+duy,max(uy)-duy)
    plt.xlabel('Slowness East-West [s/km]')
    plt.ylabel('Slowness North-South [s/km]')
    ax.tick_params(top=True,right=True)
    plt.plot([np.min(ux), np.max(ux)],[0,0],'w')
    plt.plot([0,0],[np.min(uy), np.max(uy)],'w')
    theta = np.linspace( 0 , 2 * np.pi , 150 )
    for radius in [0.1, 0.2, 0.3, 0.4, 0.5]:
        plt.plot(radius * np.cos( theta ), radius * np.sin( theta ), "w--")    
        plt.text(radius, 0, f"{radius}", c="w")
    plt.text(0, 0.83*max(uy), "N", c="w", fontsize=30)
    plt.text(0.83*max(ux), 0, "E", c="w", fontsize=30)
    plt.text(0, 0.9*min(uy), "S", c="w", fontsize=30)
    plt.text(0.9*min(ux), 0, "W", c="w", fontsize=30)
    plt.title(title)
    colorbar_ax = fig.add_axes([0.75, 0.1, 0.03, 0.6])  #x0,y0,dx,dy
    fig.colorbar(i, cax=colorbar_ax)
    if(save==1):
        plt.savefig(savename, bbox_inches='tight', format="PNG")
    plt.close()

    
ncf_file = sys.argv[1] #"/home/users/s/savardg/extract_ncfs/aargau/aargau_ncfs_wCH_pure_Allstack_linear_ZZ.npz"
beam_file = ncf_file.replace(".npz", "_beam.npz")
fig_file = beam_file.replace(".npz", ".png")
logging.info(f"Input file: {ncf_file}")
logging.info(f"Output file: {beam_file}")
logging.info(f"Figure file: {fig_file}")

# Load input data
data = np.load(ncf_file)
azimuth = data["azimuth"]
backazimuth = data["backazimuth"]
distance = data["r"]
ccf = data["ncts"]
dt = 1/25. #data["dt"]
tt_corr = data["t"]

# Define bandpass filter and apply
vs_ave = 3.
freqmin = 1/(np.max(distance)/vs_ave)
freqmax = 1/(np.min(distance)/vs_ave)
logging.info(f"freqmin = {freqmin:.3f} Hz, freqmax = {freqmax:.3f} Hz")
ccf_filt = np.zeros(ccf.shape, dtype=np.float32)
for k in range(ccf.shape[0]):
    ccf_filt[k,:] = bandpass(ccf[k,:], freqmin=freqmin, freqmax=freqmax, df=1/dt, corners=4, zerophase=True)

# Define a grid of slownesses to test
sl = .75 # second/km
# number of pixels in x and y
nux = 121
nuy = 121
ux = np.linspace(-sl,sl,nux)
uy = np.linspace(-sl,sl,nuy)

# Initialize beam array
Paz = np.zeros([nux,nuy])
Pbaz = np.zeros([nux,nuy])

# Loop over slowness values
t0 = time.time()
counter_grid = 0
logging.info("starting gridpoints: {0}".format(nux*nuy))
for ix in range(0,nux):
    for iy in range(0,nuy):
        counter_grid += 1
        if(counter_grid % 100 == 0):
            logging.info(f"counter_grid={counter_grid}, {time.time()-t0} s elapsed")
            
        # Calculate the effective time for each pair of station given the azimuth and distance (and backazimuth and distance)
        time_effective1 = (-ux[ix]*np.cos(np.deg2rad(azimuth)) - uy[iy]*np.sin(np.deg2rad(azimuth))) * distance
        # time_effective2 = (-ux[ix]*np.cos(np.deg2rad(backazimuth)) - uy[iy]*np.sin(np.deg2rad(backazimuth))) * distance
        
        # Find the index in the time vector closest to the predicted time ("time effective)")
        index_time1 = np.abs(tt_corr[np.newaxis,:] - time_effective1[:,np.newaxis]).argmin(axis=1)
        # index_time2 = np.abs(tt_corr[np.newaxis,:] - time_effective2[:,np.newaxis]).argmin(axis=1)
        
        # Extract the values for those time index from the timeseries ndarray and sum
        Paz[ix,iy] = np.sum(ccf_filt[np.arange(time_effective1.shape[0]),index_time1])
        # Pbaz[ix,iy] = np.sum(ccf_filt[np.arange(time_effective2.shape[0]),index_time2])
            
logging.info("Done.")
np.savez(beam_file ,Paz=Paz, ux=ux, uy=uy)
# np.savez(beam_file ,Paz=Paz, Pbaz=Pbaz, ux=ux, uy=uy)
logging.info(f"Data saved to {beam_file}")

#P = Paz #+ Pbaz
#plot_beam(P, title=os.path.split(ncf_file)[1], save=True, savename=fig_file)
