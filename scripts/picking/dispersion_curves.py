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
from noisepy.dispersion import *

############################################
############ PARAMETER SECTION #############
############################################

# input file info
sfile = sys.argv[1]  # ASDF file containing stacked data (full path)
dum = os.path.split(sfile)[0].split("/")[:-1]
rootpath = "/".join(dum)

pick_method = "topology"
output_dir_root = os.path.join(rootpath,
                               'dispersion_' + pick_method)  # dir where to output dispersive image and extracted dispersion
try:
    if not os.path.exists(output_dir_root):
        os.makedirs(output_dir_root)
except:
    pass

print(f"Input file: {sfile}")
dofigure = False
overwrite = True

# data type and cross-component
# stack_methods =  ["pws"] # which stacked data to measure dispersion info # auto_covariance
stack_methods = ['pws']  # ,'robust', 'nroot', 'auto_covariance']
lag_types = ['neg', 'pos',
             'sym']  # options to do measurements on the 'neg', 'pos' or 'sym' lag (average of neg and pos)
ncomp = 3

if ncomp == 1:
    rtz_system = ['ZZ']
elif ncomp == 3:
    rtz_system = ['ZZ', 'RR', 'TT']
    # index for plotting the figures
    post1 = [0, 1, 2]
    post2 = [0, 1, 2]
else:
    #     rtz_system = ['ZR', 'ZT', 'ZZ', 'RR', 'RT', 'RZ', 'TR', 'TT', 'TZ']
    rtz_system = ['RR', 'RT', 'RZ', 'TR', 'TT', 'TZ', 'ZR', 'ZT', 'ZZ']

    # index for plotting the figures
    post1 = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    post2 = [0, 1, 2, 0, 1, 2, 0, 1, 2]

# set time window for dispersion analysis
vmin = 1.0  # 0.5
vmax = 4.5  # 4.5
vel = np.arange(vmin, vmax, 0.02)

# Maximum gap in vg (multiple of dvg)
maxgap = 3

# basic parameters for wavelet transform
dj = 1 / 12
s0 = -1
J = -1
wvn = 'morlet'

# get station-pair name ready for output
tmp = sfile.split('/')[-1].split('_')
station1 = tmp[0]
spair = tmp[0] + '_' + tmp[1][:-3]


################################################################
################ DISPERSION EXTRACTION FUNCTIONS ###############
################################################################

# SNR
def nb_filt_gauss(ccf, dt, fn_array, dist, alpha=5):
    # Define signal and noise windows
    signal_win = np.arange(int(dist / vmax / dt), int(dist / vmin / dt))
    noise_istart = len(ccf) - 2 * len(signal_win)
    noise_win = np.arange(noise_istart, noise_istart + len(signal_win))
    noise_rms = np.sqrt(np.sum(ccf[noise_win] ** 2) / len(noise_win))
    snr_bb = np.max(np.abs(ccf[signal_win])) / noise_rms  # broadband snr

    # Narrowband filtering with Gaussian
    omgn_array = 2 * np.pi * fn_array

    # Transform ccf to frequency domain
    Nfft = fft.next_fast_len(len(ccf))
    ccf_freq = fft.fft(ccf, n=Nfft)
    freq_samp = 2 * np.pi * abs(fft.fftfreq(Nfft, dt))

    # Narrowband filtering
    # ccf_time_nbG = np.zeros(shape=(len(omgn_array), len(ccf)), dtype=np.float32)
    # ccf_time_nbG_env = np.zeros(shape=(len(omgn_array), len(ccf)), dtype=np.float32)
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
        isnr = np.argmax(amplitude_envelope)
        if isnr == 0 or isnr == len(amplitude_envelope) - 1:
            snr_nbG[iomgn] = 0
        else:
            noise_rms = np.sqrt(np.sum(ccftnbg[noise_win] ** 2) / len(noise_win))
            snr_nbG[iomgn] = np.max(ccftnbg[signal_win]) / noise_rms

    return snr_nbG, snr_bb  # ccf_time_nbG , ccf_time_nbG_env, snr_nbG


# function to extract the dispersion from the image
def extract_dispersion(amp, per, vel, maxgap=5, minlambda=1.5):
    '''
    this function takes the dispersion image from CWT as input, tracks the global maxinum on
    the wavelet spectrum amplitude and extract the sections with continous and high quality data

    PARAMETERS:
    ----------------
    amp: 2D amplitude matrix of the wavelet spectrum
    phase: 2D phase matrix of the wavelet spectrum
    per:  period vector for the 2D matrix
    vel:  vel vector of the 2D matrix
    maxgap: default 5
    minlambda: minimum multiple of wavelength 
    RETURNS:
    ----------------
    per:  central frequency of each wavelet scale with good data
    gv:   group velocity vector at each frequency
    ampsnr: max over median amplitude of dispersion diagram at pick time
    '''
    nper = amp.shape[0]
    gv = np.zeros(nper, dtype=np.float32)
    ampsnr = np.zeros(nper, dtype=np.float32)
    dvel = vel[1] - vel[0]

    # find global maximum
    for ii in range(nper):
        if per[ii] == 0: continue
        maxvalue = np.max(amp[ii], axis=0)
        indx = list(amp[ii]).index(maxvalue)
        gv[ii] = vel[indx]
        ampsnr[ii] = maxvalue / np.median(amp[ii], axis=0)
        # QC:
        if np.abs(gv[ii] - vmax) < 3 * dvel:  # remove points close to vg limits
            gv[ii] = 0
        elif dist / (per[ii] * gv[ii]) < minlambda:
            gv[ii] = 0

    # check the continuous of the dispersion
    for ii in range(1, nper - 15):
        # 15 is the minimum length needed for output
        if gv[ii] == 0: continue
        for jj in range(15):
            if np.abs(gv[ii + jj] - gv[ii + 1 + jj]) > maxgap * dvel:
                gv[ii] = 0
                break

    # remove the bad ones
    indx = np.where(gv > 0)[0]

    pick_per = per[indx]
    pick_gv = gv[indx]
    pick_ampsnr = ampsnr[indx]

    # Check if there are outliers (points alone with big gaps with gv before and after
    igood = list(range(len(pick_per)))
    for ii in range(2, len(pick_per) - 1):
        if pick_gv[ii] > pick_gv[ii - 1] + (pick_gv[ii + 1] - pick_gv[ii - 1]):
            igood.remove(ii)
    pick_per = pick_per[igood]
    pick_gv = pick_gv[igood]
    pick_ampsnr = pick_ampsnr[igood]

    #     return per[indx],gv[indx],ampsnr[indx]
    return pick_per, pick_gv, pick_ampsnr


def extract_curves_topology(amp, per, vel, limit=0.1):
    # Get peak for each period
    fp = findpeaks(method='topology', verbose=0, limit=limit)
    peaks = []
    for iT in range(amp.shape[0]):
        X = amp[iT, :]
        results = fp.fit(X)
        imax = results["persistence"]["y"]
        scores = results["persistence"]["score"]
        for p, score in zip(imax, scores):
            if p == 0 or p == amp.shape[1] - 1:  # Skip pick at edge of image
                continue
            peaks.append((per[iT], vel[p], score))

    pick_vel = [tup[1] for tup in peaks]
    pick_per = [tup[0] for tup in peaks]
    pick_sco = [tup[2] for tup in peaks]

    return pick_per, pick_vel, pick_sco


##################################################
############ MEASURE GROUP VELOCITY ##############
##################################################

# Loop over stack_method
for stack_method in stack_methods:

    outdir = os.path.join(output_dir_root,
                          stack_method)  # dir where to output dispersive image and extracted dispersion
    try:
        if not os.path.exists(outdir): os.makedirs(outdir)
    except:
        pass

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

    # Define period limits. Don't go above one wavelength
    # targeted freq bands for dispersion analysis
    Tmin = 0.2
    Tmax = dist / 2.0  # use 2 km/s as average
    period_lims = (Tmin, Tmax)
    fmin = 1 / Tmax
    fmax = 1 / Tmin
    # print(f"Tmin = {Tmin}, Tmax = {Tmax}")
    per = np.arange(Tmin, Tmax, 0.02)

    # Loop over lag type
    for lag_type in lag_types:
        if dofigure:
            # initialize the plotting procedure
            if ncomp == 3:
                fig, ax = plt.subplots(1, 3, figsize=(12, 3), sharex=True)
            elif ncomp == 1:
                plt.figure(figsize=(4, 3))
            else:
                fig, ax = plt.subplots(3, 3, figsize=(12, 9), sharex=True)

        # loop through each component
        for comp in rtz_system:

            # Check if DC pick file already exists
            outdir2 = os.path.join(outdir, 'vg_' + comp, station1)
            try:
                if not os.path.exists(outdir2):
                    os.makedirs(outdir2)
            except:
                print("Error while checking if output directory exists and creating it. Skipping.")

            dcfile = os.path.join(outdir2, spair + '_group_' + comp + '_lag' + lag_type + '.csv')
            if os.path.exists(dcfile) and overwrite:
                os.remove(dcfile)
            elif os.path.exists(dcfile):
                print(f"File already exists. Skipping. {dcfile}")
                continue

            # For plotting axes indices
            cindx = rtz_system.index(comp)
            pos1 = post1[cindx]
            pos2 = post2[cindx]

            # load cross-correlation functions
            with pyasdf.ASDFDataSet(sfile, mode='r') as ds:
                try:
                    tdata = ds.auxiliary_data[dtype][comp].data[:]
                except Exception as e:
                    raise ValueError(e)

            # stack positive and negative lags
            npts = int(1 / dt) * 2 * maxlag + 1
            indx = npts // 2

            if lag_type == 'neg':
                data = tdata[:indx + 1]
                data = data[::-1]  # flip
            elif lag_type == 'pos':
                data = tdata[indx:]
            elif lag_type == 'sym':
                data = 0.5 * tdata[indx:] + 0.5 * np.flip(tdata[:indx + 1], axis=0)
            else:
                raise ValueError('parameter of lag_type (L35) is not right! please double check')
            data0 = data.copy()

            # trim the data according to vel window
            pt1 = int(dist / vmax / dt)
            pt2 = int(dist / vmin / dt)
            if pt1 == 0:
                pt1 = 10
            if pt2 > (npts // 2):
                pt2 = npts // 2
            indx = np.arange(pt1, pt2)
            tvec = indx * dt
            data = data[indx]

            # wavelet transformation
            cwt, sj, freq, coi, _, _ = pycwt.cwt(data, dt, dj, s0, J, wvn)

            # do filtering here
            if (fmax > np.max(freq)) | (fmax <= fmin):
                raise ValueError('Abort: frequency out of limits!')
            freq_ind = np.where((freq >= fmin) & (freq <= fmax))[0]
            cwt = cwt[freq_ind]
            freq = freq[freq_ind]

            # use amplitude of the cwt
            period = 1 / freq
            rcwt, pcwt = np.abs(cwt) ** 2, np.angle(cwt)

            # interpolation to grids of freq-vel
            fc = scipy.interpolate.interp2d(dist / tvec, period, rcwt)
            rcwt_new = fc(vel, per)

            # do normalization for each frequency
            for ii in range(len(per)):
                rcwt_new[ii] /= np.max(rcwt_new[ii])

            # extract dispersion curves for ZZ, RR and TT
            if comp == 'ZZ' or comp == 'RR' or comp == 'TT':

                if pick_method == "argmax":
                    nper, gv, score = extract_dispersion(rcwt_new, per, vel, maxgap=maxgap)
                elif pick_method == "topology":
                    nper, gv, score = extract_curves_topology(rcwt_new, per, vel, limit=0.1)

                # Calculate SNR using narroband Gauss filters
                snr_nbG, snr_bb = nb_filt_gauss(data0, dt, np.divide(1, nper), dist, alpha=10)

                # Write picks to file
                fphase = open(dcfile, 'w')
                fphase.write(
                    'inst_period,group_velocity,score,snr_nbG,snr_bb,ratio_d_lambda,azimuth,backazimuth,distance\n')
                for iii in range(len(nper)):
                    if nper[iii] == 0: continue
                    slambda = nper[iii] * gv[iii]
                    ratio_d_lambda = np.divide(dist, slambda)  # Ratio d/lambda # GS
                    fphase.write('%6.2f,%6.3f,%5.3f,%5.3f,%5.3f,%6.3f,%3d,%3d,%7.3f\n' % (
                    nper[iii], gv[iii], score[iii], snr_nbG[iii], snr_bb, ratio_d_lambda, azi, baz, dist))  # GS
                    # if comp == 'ZZ':
                    #     print('%6.2f,%6.3f,%5.3f,%5.3f,%5.3f,%6.3f,%3d,%3d,%7.3f' % (nper[iii], gv[iii], score[iii], snr_nbG[iii], snr_bb, ratio_d_lambda, azi, baz,dist)) #GS
                fphase.close()
                print(f"Wrote dispersion curve {comp} in file: {dcfile}")

            if dofigure:
                # plot wavelet spectrum
                if ncomp == 1:
                    plt.imshow(np.transpose(rcwt_new), cmap='jet', extent=[per[0], per[-1], vel[0], vel[-1]],
                               aspect='auto',
                               origin='lower')
                    # extracted disperison curves
                    plt.plot(nper, gv, 'w--', marker="o", ms=3, color="k")
                    plt.xlabel('Period [s]')
                    plt.ylabel('U [km/s]')
                    plt.xlim(period_lims)
                    plt.title('%s %5.2fkm linear' % (spair, dist))
                    font = {'family': 'serif', 'color': 'green', 'weight': 'bold', 'size': 16}
                    plt.text(int(per[-1] * 0.85), vel[-1] - 0.5, comp, fontdict=font)
                    plt.tight_layout()
                elif ncomp == 3:
                    # dispersive image 
                    im = ax[pos1].imshow(np.transpose(rcwt_new), cmap='jet', extent=[per[0], per[-1], vel[0], vel[-1]],
                                         aspect='auto', origin='lower')
                    # extracted dispersion curves
                    if comp == 'ZZ' or comp == 'RR' or comp == 'TT':
                        ax[pos1].plot(nper, gv, 'w:', marker="o", ms=3, color="k")
                    ax[pos1].set_xlabel('Period [s]')
                    ax[pos1].set_ylabel('U [km/s]')
                    ax[pos1].set_xlim(period_lims)
                    if cindx == 1:
                        ax[pos1].set_title('%s %5.2fkm linear' % (spair, dist))
                    ax[pos1].xaxis.set_ticks_position('bottom')
                    cbar = fig.colorbar(im, ax=ax[pos1])
                    font = {'family': 'serif', 'color': 'green', 'weight': 'bold', 'size': 16}
                    ax[pos1].text(int(per[-1] * 0.85), vel[-1] - 0.5, comp, fontdict=font)

                else:
                    # dispersive image 
                    im = ax[pos1, pos2].imshow(np.transpose(rcwt_new), cmap='jet',
                                               extent=[per[0], per[-1], vel[0], vel[-1]],
                                               aspect='auto', origin='lower')
                    # extracted dispersion curves
                    if comp == 'ZZ' or comp == 'RR' or comp == 'TT':
                        ax[pos1, pos2].plot(nper, gv, 'w--', marker="o", ms=3, color="k")
                    ax[pos1, pos2].set_xlabel('Period [s]')
                    ax[pos1, pos2].set_ylabel('U [km/s]')
                    ax[pos1, pos2].set_xlim(period_lims)
                    if cindx == 1:
                        ax[pos1, pos2].set_title('%s %5.2fkm linear' % (spair, dist))
                    ax[pos1, pos2].xaxis.set_ticks_position('bottom')
                    cbar = fig.colorbar(im, ax=ax[pos1, pos2])
                    font = {'family': 'serif', 'color': 'green', 'weight': 'bold', 'size': 16}
                    ax[pos1, pos2].text(int(per[-1] * 0.85), vel[-1] - 0.5, comp, fontdict=font)
            #         ax[pos1, pos2].text(int(per[-1] * 0.85), vel[-1] - 0.5, comp, fontdict=font)

        if dofigure:
            # save figures
            outdir2 = os.path.join(outdir, 'images', station1)
            if not os.path.exists(outdir2): os.makedirs(outdir2)
            outfname = os.path.join(outdir2, '{0:s}_{1:s}.png'.format(spair, lag_type))
            if ncomp == 3:
                fig.tight_layout()
                fig.savefig(outfname, format='png', dpi=150)
            else:
                plt.savefig(outfname, format='png', dpi=150)
            # plt.show()
            plt.close()
            print(f"Figure save in file: {outfname}")
