import numpy as np
import os
import pyasdf
import obspy
from scipy.io import savemat
import sys


# Get lags (symmetric, positive, negative)
def get_ccf_lag(tdata, params):
    npts = int(1 / params["dt"]) * 2 * params["maxlag"] + 1
    indx = npts // 2
    if params["lonS"] < params["lonR"]:  # W to E
        data_neg = np.flip(tdata[:indx + 1], axis=0)
        data_pos = tdata[indx:]
    else:
        data_pos = np.flip(tdata[:indx + 1], axis=0)
        data_neg = tdata[indx:]
    data_sym = 0.5 * tdata[indx:] + 0.5 * np.flip(tdata[:indx + 1], axis=0)

    return data_sym, data_pos, data_neg


# Read stack file and get ZZ, ZR, RZ, RR
def get_all_ccomps(sfile):
    tmp = sfile.split('/')[-1].split('_')
    spair = tmp[0] + '_' + tmp[1][:-3]
    net1 = spair.split("_")[0].split(".")[0]
    net2 = spair.split("_")[1].split(".")[0]

    st = obspy.Stream()
    for comp in ['RR', 'RT', 'RZ', 'TR', 'TT', 'TZ', 'ZR', 'ZT', 'ZZ']:

        with pyasdf.ASDFDataSet(sfile, mode='r') as ds:
            dtype = 'Allstack_linear'
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


if __name__ == "__main__":
    # for f in `ls /media/genevieve/sandisk4TB/aargau-data/STACK_CHAA_normZ/AA.*/AA.*_AA.*.h5`; do python export_ccf_2matlab.py $f; done

    # sfile = "/media/savardg/sandisk4TB/aargau-data/STACK_CHAA_normZ/CH.EMMET/CH.EMMET_CH.FLACH.h5"
    #sfile = "/media/savardg/sandisk4TB/aargau-data/STACK_CHAA_normZ/AA.3006391/AA.3006391_AA.3007204.h5"
    sfile = sys.argv[1]
    output_dir = "/media/genevieve/sandisk4TB/SP-TFF/Aargau/"

    tmp = sfile.split('/')[-1].split('_')
    sta_src = tmp[0]
    sta_rcv = tmp[1][:-3]

    # Get ZZ, ZR, RZ, RR and all lags
    stream, params = get_all_ccomps(sfile)
    dist = params["dist"]
    dt = params["dt"]

    # Convert CCF to EGF (see eg Lin 2008)
    stream.differentiate()
    for tr in stream:
        tr.data *= -1

    stream.taper(0.05)
    stream.trim(starttime=stream[0].stats.starttime, endtime=stream[0].stats.starttime+30)

    # Save to Matlab
    times = np.arange(0, stream[0].stats.npts*dt, dt)
    fname = os.path.join(output_dir, os.path.split(sfile)[1].replace(".h5", ".mat"))
    mdict = {
        "sta_src": sta_src,
        "sta_rcv": sta_rcv,
        "dt": dt,
        "times": times,
        "params": params
    }
    for ccomp in ['RR', 'RT', 'RZ', 'TR', 'TT', 'TZ', 'ZR', 'ZT', 'ZZ']:
        mdict[f"wave_{ccomp}"] = stream.select(location="sym", channel=ccomp)[0].data
    savemat(fname, mdict=mdict)
    print(f"CCF data saved to {fname}.")
