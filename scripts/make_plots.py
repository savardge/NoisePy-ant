import sys
import glob
from noisepy import plotting
import time

ccomp = sys.argv[1]
sfiles = glob.glob('/home/users/s/savardg/scratch/aargau/STACK_25sps_3c_adaptive/*/*.h5')
figdir = '/home/users/s/savardg/scratch/aargau/STACK_25sps_3c_adaptive/figures'
savefig = True
dist_inc = 0.2 #km
disp_lag = 20
freqmin = 0.1
freqmax = 1.0

t1 = time.time()
print(f"cross-component {ccomp}, bandpass filter: {freqmin}-{freqmax} Hz, distance increment: {dist_inc} km")
plotting_modules.plot_all_moveout(sfiles,
                                  'Allstack_auto_covariance',
                                  freqmin=freqmin,
                                  freqmax=freqmax,
                                  ccomp=ccomp,
                                  dist_inc=dist_inc,
                                  disp_lag=disp_lag,
                                  savefig=savefig,
                                  sdir=figdir)

#plotting_modules.plot_all_moveout(sfiles,'Allstack_pws',freqmin=freqmin,freqmax=freqmax,ccomp=ccomp,dist_inc=dist_inc,disp_lag=disp_lag,savefig=savefig,sdir=figdir)
#plotting_modules.plot_all_moveout(sfiles,'Allstack_linear',freqmin=freqmin,freqmax=freqmax,ccomp=ccomp,dist_inc=dist_inc,disp_lag=disp_lag,savefig=savefig,sdir=figdir) 
#plotting_modules.plot_all_moveout(sfiles,'Allstack_robust',freqmin=freqmin,freqmax=freqmax,ccomp=ccomp,dist_inc=dist_inc,disp_lag=disp_lag,savefig=savefig,sdir=figdir)

print(f"Elapsed time: {time.time() - t1}")
