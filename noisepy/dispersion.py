""" DISPERSION FUNCTIONS"""

import numpy as np
import logging
from findpeaks import findpeaks  # https://github.com/erdogant/findpeaks
Logger = logging.getLogger(__name__)


# function to extract the dispersion from the image
def extract_dispersion(amp, per, vel, dist, vmax=5., maxgap=5, minlambda=1.5):
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


def extract_dispersion_simple(amp, per, vel):
    '''
    this function takes the dispersion image from CWT as input, tracks the global maxinum on
    the wavelet spectrum amplitude and extract the sections with continous and high quality data

    PARAMETERS:
    ----------------
    amp: 2D amplitude matrix of the wavelet spectrum
    phase: 2D phase matrix of the wavelet spectrum
    per:  period vector for the 2D matrix
    vel:  vel vector of the 2D matrix
    RETURNS:
    ----------------
    per:  central frequency of each wavelet scale with good data
    gv:   group velocity vector at each frequency
    '''
    maxgap = 5
    nper = amp.shape[0]
    gv = np.zeros(nper, dtype=np.float32)
    dvel = vel[1] - vel[0]

    # find global maximum
    for ii in range(nper):
        maxvalue = np.max(amp[ii], axis=0)
        indx = list(amp[ii]).index(maxvalue)
        gv[ii] = vel[indx]

    # check the continuous of the dispersion
    for ii in range(1, nper - 15):
        # 15 is the minumum length needed for output
        for jj in range(15):
            if np.abs(gv[ii + jj] - gv[ii + 1 + jj]) > maxgap * dvel:
                gv[ii] = 0
                break

    # remove the bad ones
    indx = np.where(gv > 0)[0]

    return per[indx], gv[indx]


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
