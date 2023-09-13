"""
For each given station pair, plot all the substacks (see S1_params.yaml file for length substack_len) in the time and frequency domains.
"""
import plotting_modules as pm
import glob, os

# Define bandpass filter
freqmin = 0.1
freqmax = 1

# Maximum time lag for plotting
disp_lag = 60

# Cross-component to plot
ccomp = 'ZZ' 

savefig = True

# Get list of files containing the stacked CCFs in h5 format
stackdir = "/home/genevieve.savard/CanoeReach/noisepy/stack_pure" 
sfiles = glob.glob(os.path.join(stackdir,"BG.*/*.h5")) # Get the list of h5 files
sfiles.sort()

for sfile in sfiles:

    # Create/setup output directory name
    sdir = os.path.join(stackdir,"figs/substacks")
    if not os.path.exists(sdir):
        os.makedirs(sdir)  
        
    # Skip if plot already made        
    outfname = sdir + '/{0:s}_{1:s}_{2:4.2f}_{3:4.2f}Hz_spect.png'.format(sfile.split('/')[-1], ccomp, freqmin, freqmax) # Pattern needs to match the one inside the called function, in plotting_modules.py
    if os.path.exists(outfname):
        continue    
        
    # Do plotting
    pm.plot_substack_all_spect(sfile, freqmin, freqmax, ccomp=ccomp, disp_lag=disp_lag, savefig=savefig, sdir=sdir, figsize=(14, 14))