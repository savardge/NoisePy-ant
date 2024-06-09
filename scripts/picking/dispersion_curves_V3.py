import numpy as np
import os
import pyasdf
import sys
from noisepy import dispersion

# input file info
sfile = sys.argv[1]  # ASDF file containing stacked data (full path)

# get station-pair name
tmp = sfile.split('/')[-1].split('_')
station1 = tmp[0]
spair = tmp[0] + '_' + tmp[1][:-3]

# Output path
dum = os.path.split(sfile)[0].split("/")[:-1]
rootpath = "/".join(dum)
output_dir_root = os.path.join(rootpath, f"dispersion_V3", station1)  # dir where to output
print(f"Output directory: {output_dir_root}")
print(f"Input file: {sfile}")
overwrite = True

try:
    if not os.path.exists(output_dir_root):
        os.makedirs(output_dir_root)
except Exception as e:
    print(e)
    pass

# Parameters for data type and cross-component
# which stacked data to measure dispersion info # auto_covariance
stack_methods = ['pws', 'linear', 'nroot']  # ,'robust', , 'auto_covariance']
lag_types = ['neg', 'pos', 'sym']  # options to do measurements on the 'neg', 'pos' or 'sym' lag
rtz_system = ['RR', 'RZ', 'ZR', 'ZZ', 'TT']
pick_methods = ["argmax", "topology"]

# Define period limits. Don't go above one wavelength
# targeted freq bands for dispersion analysis
Tmin = 0.2
dT = 0.1
vmin = 0.5  # 0.5
vmax = 4.0  # 4.5
dvel = 0.01
vel = np.arange(vmin, vmax, dvel)
maxgap = int(0.2 / 0.01)  # 0.2 km/s max jump in vg in time interval of dT=0.1 s
min_score = 0.7  # minimum persistence score for topology method
gauss_alpha = 5.  # Gaussian filter parameter to get SNR

# Open file
dcfile = os.path.join(output_dir_root, spair + '_group_all.csv')
if os.path.exists(dcfile) and overwrite:
    os.remove(dcfile)
elif os.path.exists(dcfile) and not overwrite:
    print(f"File already exists. Skipping. {dcfile}")
    sys.exit()
fphase = open(dcfile, 'w')
fphase.write(
    'inst_period,group_velocity,score,snr_nbG,snr_bb,ratio_d_lambda,azimuth,backazimuth,distance,lag,component,stack_method,pick_method\n')

# MEASURE GROUP VELOCITY ##############
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

    # Dictionary structure to save data to
    CCFdata = {
        "ccf": {"neg": {}, "pos": {}, "sym": {}},
        "FTAN": {"neg": {}, "pos": {}, "sym": {}},
        "COI": {"neg": {}, "pos": {}, "sym": {}}
    }

    # loop through each component
    for comp in rtz_system:

        # load cross-correlation functions
        with pyasdf.ASDFDataSet(sfile, mode='r') as ds:
            try:
                tdata = ds.auxiliary_data[dtype][comp].data[:]
            except Exception as e:
                print(f"Could not read data for comp {comp}")
                continue

        # Get positive / negative /symmetric sides
        npts = len(tdata)
        indx = npts // 2
        CCFdata["ccf"]["neg"][comp] = tdata[:indx + 1][::-1]
        CCFdata["ccf"]["pos"][comp] = tdata[indx:]
        CCFdata["ccf"]["sym"][comp] = 0.5 * (tdata[indx:] + np.flip(tdata[:indx + 1], axis=0))

        # Get FTAN images
        for lag in lag_types:
            amp, per, vel, coi = dispersion.get_disp_image_taper(CCFdata["ccf"][lag][comp], dist, dt, Tmin=Tmin, dT=dT,
                                                                 vmin=vmin,
                                                                 vmax=vmax, dvel=dvel, plot=False)
            CCFdata["FTAN"][lag][comp] = amp
            CCFdata["COI"][lag][comp] = coi

            for pick_method in pick_methods:

                try:
                    # Get dispersion for this component
                    if pick_method == "argmax":
                        nper, gv, score = dispersion.extract_dispersion(CCFdata["FTAN"][lag][comp], per, vel, dist,
                                                                        maxgap=maxgap)
                    elif pick_method == "topology":
                        nper, gv, score = dispersion.extract_curves_topology(CCFdata["FTAN"][lag][comp], per, vel,
                                                                             limit=min_score)
                    else:
                        continue

                    # Remove picks inside cone of influence
                    nper, gv, score = dispersion.remove_picks_coi(nper, gv, score, vel, CCFdata["COI"][lag][comp])

                    # Calculate SNR using narrowband Gauss filters
                    fn_array = np.divide(1, nper)
                    snr_nbG, snr_bb, _, _ = dispersion.nb_filt_gauss(CCFdata["ccf"][lag][comp], dt, fn_array, dist,
                                                                     alpha=gauss_alpha, vmin=vmin, vmax=vmax)

                    # Write picks to file
                    for iii in range(len(nper)):
                        if nper[iii] == 0:
                            continue
                        slambda = nper[iii] * gv[iii]
                        ratio_d_lambda = np.divide(dist, slambda)  # Ratio d/lambda # GS
                        line = f"{nper[iii]:4.1f},{gv[iii]:5.2f},{score[iii]:5.2f},{snr_nbG[iii]:6.2f},{snr_bb:6.2f},{ratio_d_lambda:5.1f},{azi:5.2f},{baz:5.2f},{dist:6.3f},{lag},{comp},{stack_method},{pick_method}\n"
                        fphase.write(line)

                except Exception as e:
                    print(
                        f"Error occured when picking dispersion for comp {comp}, lag {lag} and pick_method {pick_method}")

        print(f"Wrote dispersion curves for {comp}, stack {stack_method}")

    # Now get picks for combinations
    amp_list = [
        ((CCFdata["FTAN"]["pos"]["ZZ"] *
          CCFdata["FTAN"]["neg"]["ZZ"] *
          CCFdata["FTAN"]["pos"]["ZR"] *
          CCFdata["FTAN"]["neg"]["ZR"] *
          CCFdata["FTAN"]["pos"]["RZ"] *
          CCFdata["FTAN"]["neg"]["RZ"] *
          CCFdata["FTAN"]["pos"]["RR"] *
          CCFdata["FTAN"]["neg"]["RR"]) ** (1 / 8)),
        ((CCFdata["FTAN"]["sym"]["ZZ"] *
          CCFdata["FTAN"]["sym"]["ZR"] *
          CCFdata["FTAN"]["sym"]["RZ"] *
          CCFdata["FTAN"]["sym"]["RR"]) ** (1 / 4)),
        (CCFdata["FTAN"]["sym"]["ZZ"] *
         CCFdata["FTAN"]["sym"]["ZR"]) ** (1 / 2),
        (CCFdata["FTAN"]["sym"]["RR"] *
         CCFdata["FTAN"]["sym"]["RZ"]) ** (1 / 2)]
    lag_list = ["all2", "sym", "sym", "sym"]
    comp_list = ["all4", "all4", "ZZ-ZR", "RR-RZ"]

    for amp, lag, comp in zip(amp_list, lag_list, comp_list):
        for pick_method in pick_methods:

            try:
                # Get dispersion for this component
                if pick_method == "argmax":
                    nper, gv, score = dispersion.extract_dispersion(amp, per, vel, dist, vmax=vmax, maxgap=maxgap)
                elif pick_method == "topology":
                    nper, gv, score = dispersion.extract_curves_topology(amp, per, vel, limit=min_score)
                else:
                    continue

                # Calculate SNR using narrowband Gauss filters
                fn_array = np.divide(1, nper)
                if comp == "RR-RZ":
                    snr_nbG, snr_bb, _, _ = dispersion.nb_filt_gauss(CCFdata["ccf"]["sym"]["RZ"], dt, fn_array,
                                                                     dist, alpha=gauss_alpha, vmin=vmin, vmax=vmax)
                else:
                    snr_nbG, snr_bb, _, _ = dispersion.nb_filt_gauss(CCFdata["ccf"]["sym"]["ZZ"], dt, fn_array,
                                                                     dist, alpha=gauss_alpha, vmin=vmin, vmax=vmax)
                # Write picks to file
                for iii in range(len(nper)):
                    if nper[iii] == 0:
                        continue
                    slambda = nper[iii] * gv[iii]
                    ratio_d_lambda = np.divide(dist, slambda)  # Ratio d/lambda # GS
                    line = f"{nper[iii]:4.1f},{gv[iii]:5.2f},{score[iii]:5.2f},{snr_nbG[iii]:6.2f},{snr_bb:6.2f},{ratio_d_lambda:5.1f},{azi:5.2f},{baz:5.2f},{dist:6.3f},{lag},{comp},{stack_method},{pick_method}\n"
                    fphase.write(line)
            except Exception as e:
                print(f"Error occured when picking dispersion for comp {comp}, lag {lag} and pick_method {pick_method}")

        print(f"Wrote dispersion curves from product of FTAN images for {comp}-{lag}  stack {stack_method}")

fphase.close()
print(dcfile)
