import glob
import numpy as np
import os
import pyasdf
import pycwt
import scipy
import sys
import matplotlib.pyplot as plt
from scipy import fft
from scipy import interpolate 
from scipy.signal import hilbert
from findpeaks import findpeaks
from noisepy import dispersion

############################################
############ PARAMETER SECTION #############
############################################

# input file info
rootpath = '/home/users/s/savardg/scratch/aargau/STACK_CHAA_normZ/'  # root path for this data processing. available stack methods: auto_covariance, linear, nroot, pws, robust
#rootpath = '/home/users/s/savardg/scratch/aargau/STACK_25sps_3c_adaptive/'  # root path for this data processing
sfile = os.path.join(rootpath, sys.argv[1])  # ASDF file containing stacked data
#sfile = os.path.join(rootpath, "CH.SBAV/CH.SBAV_RI.INZ12.h5")

output_dir_root = os.path.join(rootpath, f"dispersion_V2")  # dir where to output dispersive image and extracted dispersion
try:
    if not os.path.exists(outdir): os.makedirs(outdir)
except:
    pass

print(f"Input file: {sfile}")
overwrite = True

# data type and cross-component
#stack_methods =  ["pws"] # which stacked data to measure dispersion info # auto_covariance
stack_methods =  ['pws', 'linear'] #,'robust', 'nroot', 'auto_covariance']
lag_types = ['neg','pos','sym']  # options to do measurements on the 'neg', 'pos' or 'sym' lag (average of neg and pos)

#rtz_system = ['ZZ']
#rtz_system = ['ZZ', 'RR', 'TT']
rtz_system = ['RR', 'RZ', 'TT', 'ZR', 'ZZ']
#rtz_system = ['RR', 'RT', 'RZ', 'TR', 'TT', 'TZ', 'ZR', 'ZT', 'ZZ']
    

# get station-pair name ready for output
tmp = sfile.split('/')[-1].split('_')
station1 = tmp[0]
spair = tmp[0] + '_' + tmp[1][:-3]


dcfile = os.path.join(output_dir_root , spair + '_group_all.csv')
if os.path.exists(dcfile) and overwrite:
    os.remove(dcfile)
elif os.path.exists(dcfile) and not overwrite:
    print(f"File already exists. Skipping. {dcfile}")
    #continue
    
# Open file
fphase = open(dcfile, 'w')
fphase.write('inst_period,group_velocity,score,snr_nbG,snr_bb,ratio_d_lambda,azimuth,backazimuth,distance,lag,component,stack_method,pick_method\n')

# Define period limits. Don't go above one wavelength
# targeted freq bands for dispersion analysis
Tmin = 0.2
dT = 0.1
vmin = 0.5 #0.5
vmax = 4.0 #4.5
dvel = 0.01
vel = np.arange(vmin, vmax, dvel)
maxgap = 3

##################################################
############ MEASURE GROUP VELOCITY ##############
##################################################

# Loop over stack_method
for stack_method in stack_methods:

    # load basic data information including dt, dist and maxlag
    with pyasdf.ASDFDataSet(sfile, mode='r') as ds:
        dtype = 'Allstack_' + stack_method
        try:
            maxlag = ds.auxiliary_data[dtype]['ZZ'].parameters['maxlag']
            dist = ds.auxiliary_data[dtype]['ZZ'].parameters['dist']
            dt = ds.auxiliary_data[dtype]['ZZ'].parameters['dt']
            azi = ds.auxiliary_data[dtype]['ZZ'].parameters['azi']
            baz = ds.auxiliary_data[dtype]['ZZ'].parameters['baz']
        except Exception as e:
            raise ValueError(e)

    CCFdata = {
        # "dist": dist,
        # "dt": dt,
        # "maxlag": maxlag,
        # "azi": azi,
        # "baz": baz,
        "ccf": {"neg":{}, "pos":{}, "sym":{}},
        "FTAN": {"neg":{}, "pos":{}, "sym":{}} } #,
        #"FTAN_params": {"Tmin":Tmin, "dT": dT, "vmin":vmin, "vmax":vmax, "dvel":dvel}
    #}
    
    # loop through each component
    for comp in rtz_system:

        # load cross-correlation functions
        with pyasdf.ASDFDataSet(sfile, mode='r') as ds:
            try:
                tdata = ds.auxiliary_data[dtype][comp].data[:]
            except Exception as e:
                raise ValueError(e)
                
        # Get positive / negative /symmetric sides
        npts = len(tdata)
        indx = npts // 2
        CCFdata["ccf"]["neg"][comp] = tdata[:indx + 1][::-1]
        CCFdata["ccf"]["pos"][comp] = tdata[indx:]
        CCFdata["ccf"]["sym"][comp] = 0.5 * (tdata[indx:] + np.flip(tdata[:indx + 1], axis=0))
        
       # Get FTAN images\
        for lag in ["pos", "neg", "sym"]:
            amp, per, vel = dispersion.get_disp_image(CCFdata["ccf"][lag][comp], dist, dt, Tmin=Tmin, dT=dT, vmin=vmin, vmax=vmax, dvel=dvel, plot=False)
            CCFdata["FTAN"][lag][comp] = amp

            for pick_method in ["argmax", "topology"]:
                # Get dispersion for this component
                if pick_method == "argmax":
                    nper, gv, score = dispersion.extract_dispersion(CCFdata["FTAN"][lag][comp], per, vel, dist, maxgap=maxgap)
                elif pick_method == "topology":
                    nper, gv, score = dispersion.extract_curves_topology(CCFdata["FTAN"][lag][comp],per,vel, limit=0.3)

                # Calculate SNR using narroband Gauss filters
                snr_nbG, snr_bb = dispersion.nb_filt_gauss(CCFdata["ccf"][lag][comp], dt, np.divide(1, nper), dist, alpha=10)

                # Write picks to file
                for iii in range(len(nper)):
                    if nper[iii] == 0: continue
                    slambda =  nper[iii] * gv[iii]
                    ratio_d_lambda = np.divide(dist,slambda) # Ratio d/lambda # GS
                    #fphase.write('%6.2f,%6.3f,%5.3f,%5.3f,%5.3f,%6.3f,%3d,%3d,%7.3f,%s,%s\n' % (nper[iii], gv[iii], score[iii], snr_nbG[iii], snr_bb, ratio_d_lambda, azi, baz,dist, "sym", comp)) #GS
                    line=f"{nper[iii]:4.1f},{gv[iii]:5.2f},{score[iii]:5.2f},{snr_nbG[iii]:6.2f},{snr_bb:6.2f},{ratio_d_lambda:5.1f},{azi:5.2f},{baz:5.2f},{dist:6.3f},{lag},{comp},{stack_method},{pick_method}\n"
                    fphase.write(line)

        print(f"Wrote dispersion curves for {comp} in file: {dcfile}")

    # Now get picks for combinations
    amp = (CCFdata["FTAN"]["pos"]["ZZ"] * 
           CCFdata["FTAN"]["neg"]["ZZ"] *
           CCFdata["FTAN"]["pos"]["ZR"] *
           CCFdata["FTAN"]["neg"]["ZR"] *
           CCFdata["FTAN"]["pos"]["RZ"] *
           CCFdata["FTAN"]["neg"]["RZ"] *
           CCFdata["FTAN"]["pos"]["RR"] *
           CCFdata["FTAN"]["neg"]["RR"] ) ** (1/8)
    lag = "all2"
    comp = "all4"
    for pick_method in ["argmax", "topology"]:
        # Get dispersion for this component
        if pick_method == "argmax":
            nper, gv, score = dispersion.extract_dispersion(amp, per, vel,dist, vmax=4.0,maxgap=maxgap)
        elif pick_method == "topology":
            nper, gv, score = dispersion.extract_curves_topology(amp,per,vel,limit=0.3)
        # Calculate SNR using narroband Gauss filters
        snr_nbG, snr_bb = dispersion.nb_filt_gauss(CCFdata["ccf"]["sym"]["ZZ"], dt, np.divide(1, nper), dist, alpha=10)
        # Write picks to file
        for iii in range(len(nper)):
            if nper[iii] == 0: continue
            slambda =  nper[iii] * gv[iii]
            ratio_d_lambda = np.divide(dist,slambda) # Ratio d/lambda # GS
            #fphase.write('%6.2f,%6.3f,%5.3f,%5.3f,%5.3f,%6.3f,%3d,%3d,%7.3f,%s,%s\n' % (nper[iii], gv[iii], score[iii], snr_nbG[iii], snr_bb, ratio_d_lambda, azi, baz,dist, "sym", comp)) #GS
            line=f"{nper[iii]:4.1f},{gv[iii]:5.2f},{score[iii]:5.2f},{snr_nbG[iii]:6.2f},{snr_bb:6.2f},{ratio_d_lambda:5.1f},{azi:5.2f},{baz:5.2f},{dist:6.3f},{lag},{comp},{stack_method},{pick_method}\n"
            fphase.write(line)
    print(f"Wrote dispersion curve ZZ-ZR-RR-RZ product in file: {dcfile}")
    
    
    amp = (CCFdata["FTAN"]["sym"]["ZZ"] * 
           CCFdata["FTAN"]["sym"]["ZR"] ) ** (1/2)
    lag = "sym"
    comp = "ZZ-ZR"
    # Write picks to file
    for pick_method in ["argmax", "topology"]:
        # Get dispersion for this component
        if pick_method == "argmax":
            nper, gv, score = dispersion.extract_dispersion(amp, per, vel,dist, vmax=4.0,maxgap=maxgap)
        elif pick_method == "topology":
            nper, gv, score = dispersion.extract_curves_topology(amp,per,vel,limit=0.3)
        # Calculate SNR using narroband Gauss filters
        snr_nbG, snr_bb = dispersion.nb_filt_gauss(CCFdata["ccf"]["sym"]["ZZ"], dt, np.divide(1, nper), dist, alpha=10)
        # Write picks to file
        for iii in range(len(nper)):
            if nper[iii] == 0: continue
            slambda =  nper[iii] * gv[iii]
            ratio_d_lambda = np.divide(dist,slambda) # Ratio d/lambda # GS
            #fphase.write('%6.2f,%6.3f,%5.3f,%5.3f,%5.3f,%6.3f,%3d,%3d,%7.3f,%s,%s\n' % (nper[iii], gv[iii], score[iii], snr_nbG[iii], snr_bb, ratio_d_lambda, azi, baz,dist, "sym", comp)) #GS
            line=f"{nper[iii]:4.1f},{gv[iii]:5.2f},{score[iii]:5.2f},{snr_nbG[iii]:6.2f},{snr_bb:6.2f},{ratio_d_lambda:5.1f},{azi:5.2f},{baz:5.2f},{dist:6.3f},{lag},{comp},{stack_method},{pick_method}\n"
            fphase.write(line)
    print(f"Wrote dispersion curve ZZ-ZR product in file: {dcfile}")
    
    
    amp = (CCFdata["FTAN"]["sym"]["RR"] * 
           CCFdata["FTAN"]["sym"]["RZ"] ) ** (1/2)
    lag = "sym"
    comp = "RR-RZ"
    # Write picks to file
    for pick_method in ["argmax", "topology"]:
        # Get dispersion for this component
        if pick_method == "argmax":
            nper, gv, score = dispersion.extract_dispersion(amp, per, vel,dist, vmax=4.0, maxgap=maxgap)
        elif pick_method == "topology":
            nper, gv, score = dispersion.extract_curves_topology(amp,per,vel,limit=0.3)
        # Calculate SNR using narroband Gauss filters
        snr_nbG, snr_bb = dispersion.nb_filt_gauss(CCFdata["ccf"]["sym"]["ZZ"], dt, np.divide(1, nper), dist, alpha=10)
        # Write picks to file
        for iii in range(len(nper)):
            if nper[iii] == 0: continue
            slambda =  nper[iii] * gv[iii]
            ratio_d_lambda = np.divide(dist,slambda) # Ratio d/lambda # GS
            #fphase.write('%6.2f,%6.3f,%5.3f,%5.3f,%5.3f,%6.3f,%3d,%3d,%7.3f,%s,%s\n' % (nper[iii], gv[iii], score[iii], snr_nbG[iii], snr_bb, ratio_d_lambda, azi, baz,dist, "sym", comp)) #GS
            line=f"{nper[iii]:4.1f},{gv[iii]:5.2f},{score[iii]:5.2f},{snr_nbG[iii]:6.2f},{snr_bb:6.2f},{ratio_d_lambda:5.1f},{azi:5.2f},{baz:5.2f},{dist:6.3f},{lag},{comp},{stack_method},{pick_method}\n"
            fphase.write(line)
    print(f"Wrote dispersion curve RR-RZ product in file: {dcfile}")
    
fphase.close()
