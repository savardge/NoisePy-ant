import numpy as np
import math
from cmath import inf
import scipy
from scipy.signal import hilbert, windows, firwin, lfilter
from scipy.fftpack import fft, ifft
from scipy.interpolate import interp1d
from scipy.signal.windows import tukey as tukey_window
import matplotlib.pyplot as plt
import obspy
import os
import glob
import pyasdf
import sys
import yaml
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
logger = logging.getLogger(__name__)


def generate_phase(stackEGF, fs, vmin, vmax, dvel, Tmin, Tmax, dT, dist, npts, Time, dt):
    
    # find relevant ccf window index based on interstation distance and velocity range
    g_WinMaxPtNum = round(fs*dist//vmin)
    g_WinMinPtNum = round(fs*dist//vmax) 
        
    if g_WinMaxPtNum >= npts:
        g_WinMaxPtNum = npts-1
        vmin          = np.ceil(10*dist/Time[-1])/10
        logger.warning(f"Min velocity reset to: {vmin}")
        
    # number of time and velocity points
    NumCtrT = round((Tmax - Tmin)/dT)+1
    #Tpoint  = np.linspace(Tmin, Tmax, NumCtrT)
    NumCtrV = round((vmax - vmin)/dvel)+1
    Vpoint  = np.linspace(vmax, vmin, NumCtrV)
                
    # calculate the window function
    Window, TaperLen = generate_signal_window(g_WinMinPtNum, g_WinMaxPtNum, npts, winalpha=0.05)
    # Window the stacked EGF (causal + acausal)
    WinWave = stackEGF * Window
    
    # Extract Noise window after the windowed surface wave 
    WinWaveClip, WaveClipPt  = extract_noise_window(dt, TaperLen, g_WinMaxPtNum, npts, stackEGF, WinWave, NoiseTime=150)
    
    PhaseVIm = time_shift(WinWaveClip, WaveClipPt, fs, Vpoint, Tmin, dT, dist, dt, NumCtrT, g_WinMinPtNum, g_WinMaxPtNum, FilterKaiserPara = 6, MaxFilterLengthLog = 13)

    return PhaseVIm

def generate_signal_window(g_WinMinPtNum, g_WinMaxPtNum, npts, winalpha):
    # window length
    win_len = int((g_WinMaxPtNum - g_WinMinPtNum)/(1-winalpha))+1
    # generate window function
    Window = windows.tukey(win_len, winalpha)
    TaperLen = round(win_len * winalpha / 2)
    
    # crop or add the left side
    pad_left_len = int(g_WinMinPtNum) - TaperLen
    if pad_left_len > 0:
        Window = np.pad(Window, (pad_left_len, 0), 'constant')
    else:
        Window = Window[-pad_left_len:]
    # crop or add the right side
    if Window.shape[0] < npts:
        Window = np.pad(Window, (0, npts - Window.shape[0]), 'constant')
    else:
        Window = Window[:npts]
    
    return Window, TaperLen

def extract_noise_window(dt, TaperLen, g_WinMaxPtNum, npts, stackEGF, WinWave, NoiseTime=150):
    # Extract Noise window after the windowed surface wave       
    NoisePt   = round(NoiseTime / dt)
    NoiseStartIndex = int(g_WinMaxPtNum) + TaperLen

    if (NoiseStartIndex + NoisePt) < npts:
        NoiseWinWave = stackEGF[NoiseStartIndex:NoiseStartIndex + NoisePt]
    else:
        NoiseWinWave = stackEGF[NoiseStartIndex:]
        NoisePt      = len(NoiseWinWave)
        logger.warning(f"Noise Window Length of {NoiseWinWave.shape[0]} is not long enough")
    
    WaveClipPt  = min((int(g_WinMaxPtNum) + TaperLen), npts)
    WinWaveClip = WinWave[:WaveClipPt]
    
    return WinWaveClip, WaveClipPt

def time_shift(WinWaveClip, WaveClipPt, fs, Vpoint, Tmin, dT, dist, dt, NumCtrT, g_WinMinPtNum, g_WinMaxPtNum, FilterKaiserPara=6, MaxFilterLengthLog=13):
    # Apply time shift method in time domain

    BandWidth = dT
    
    exponential   = min(math.ceil(np.log2(1024*fs)), MaxFilterLengthLog)
    filter_length = int(2 ** exponential)

    HalfFilterNum = int(filter_length / 2)
    
    WinWave2 = np.concatenate((np.copy(WinWaveClip), np.zeros(HalfFilterNum)))
      
    PhaseIm = []
    for numt in range(NumCtrT):
        CtrT  = Tmin + numt * dT
        #CtrF  = (2 / fs) / CtrT
        LowF  = (2 / fs) / (CtrT + 0.5 * BandWidth)
        HighF = (2 / fs) / (CtrT - 0.5 * BandWidth)
    
        filter_data = firwin(filter_length + 1, [LowF, HighF], pass_zero=False, window=('kaiser',FilterKaiserPara))
    
        # We do not have information of reference group velocity window for using 
        # frequency-time filtering analysis on variable MFT(?) to window the original waveform
    
        # Therefore, rather use uniform group velocity window and apply phase shift technique in time domain (?)
        # i.e. two-pass filtering (time and time-reverse) in order to remove phase shift
    
        # filtering
        FilteredWave = lfilter(filter_data, 1, WinWave2)
        # inverse order
        FilteredWave = FilteredWave[::-1]
        # filtering
        FilteredWave = lfilter(filter_data, 1, FilteredWave)
        # inverse order
        FilteredWave = FilteredWave[::-1]
        # clip
        FilteredWave = (FilteredWave[:WaveClipPt])
        # normalization
        FilteredWave = FilteredWave / np.max(np.abs(FilteredWave))
        PhaseIm.append(FilteredWave)
   

    timeptnum  = np.array(range(int(g_WinMinPtNum),int(g_WinMaxPtNum)))
    phase_time = timeptnum * dt
    PhaseVIm   = []

    for i in range(NumCtrT):
               
        CenterT = Tmin + i * dT
        TravPtV = dist/(phase_time - CenterT/8)
    
        # time - CenterT/8 may be zero
        TravPtV[TravPtV == inf] = 100
    
        PhaseVIm.append(interp1d(TravPtV[::-1], (PhaseIm[i][int(g_WinMinPtNum):int(g_WinMaxPtNum)])[::-1], kind='cubic', bounds_error=False,fill_value=0)(Vpoint))

    PhaseVIm = np.transpose(np.array(PhaseVIm))
    # reverse
    PhaseVIm = PhaseVIm[::-1]
    
    return PhaseVIm

# Not needed but keep in case
#def fold_trace(tr):
#    """fold_trace() takes a trace and computes the symmetric component by 
#    summing the time-reversed acausal component with the causal coponent"""
    # Pick some constants we'll need from the trace:
#    npts2 = tr.stats.npts
#    npts = int((npts2-1)/2)

    # Fold the CCFs:
#    tr_folded= tr.copy()
#    causal =  tr.data[npts:-1]
#    acausal = tr.data[npts:0:-1]
#    tr_folded.data =  (causal+acausal)/2
    
#    return tr_folded

################################################################
# Specify stack directory:
stack_dir = "/home/itopie/Bureau/RN.N001"
export_path = "/home/itopie/Bureau/N001_phase_image/"

#filelist = glob.glob(os.path.join(stack_dir,"*", "*.h5"))
filelist = glob.glob(os.path.join(stack_dir, "*.h5"))
filelist.sort()

""" 
Select the stacking method. Options: 
- Allstack_linear (linear stacking, taking the mean)
- Allstack_pws (phase weighted stack) 
- Allstack_nroot (N root stack Millet, F et al., 2019 JGR, with power=2)
- Allstack_robust (Palvis and Vernon 2010)
- Allstack_auto_covariance (Adaptive filter of Nakata et al., 2015 appendix B: with filter harshness g=1)
"""
stack_method = "Allstack_pws"  

"""
Select the component: 
Options: ['EE', 'EN', 'EZ', 'NE', 'NN', 'NZ', 'RR', 'RT', 'RZ', 'TR', 'TT', 'TZ', 'ZE', 'ZN', 'ZR', 'ZT', 'ZZ']
"""
component = "ZZ"

# Setup up period and velocity limits and spacing
vmin = 0.5
vmax = 3.5
dvel = 0.005
Tmin = 0.2 # Tmin should not be at the limit given by 1/(2*fs) 
Tmax = 7
dT = 0.01


# Loop over files to check them out
for fname in filelist:

    print(f"Processing this file: {fname}")
    try:
        with pyasdf.ASDFDataSet(fname, mode="r") as ds:
                #print(ds.auxiliary_data.list()) # This shows the list of stack methods available
                #print(ds.auxiliary_data[stack_method].list()) # This shows the list of components available
                ccf_full = ds.auxiliary_data[stack_method][component].data[:]
                params = ds.auxiliary_data[stack_method][component].parameters

        pair = os.path.split(fname)[1].split(".h5")[0]
        #print(f"pair: {pair}, distance: {params['dist']}")

        # Make an Obspy trace for full symmetric component
        tr_full = obspy.Trace(ccf_full, header={'delta': params['dt'], 'station': pair, 'channel': component, 'distance': params['dist']})
        npts2   = tr_full.stats.npts
        npts    = int((npts2-1)/2)
        causal  = tr_full.data[npts:-1]
        acausal = tr_full.data[npts:0:-1]
    
        # Get the data
        dt = tr_full.stats.delta
        dist = params['dist']
      
        # Maxamplitude fonction and CF to EGFs conversion through hilbert transform
        PtNum     = npts
        fs        = tr_full.stats.sampling_rate  # Sampling frequency (SampleF)
        Time      = np.arange(0,(PtNum-1)*dt,dt)

        maxamp    = max(max(causal), max(acausal))

        if maxamp > 0:
                causal_new  = causal/maxamp
                acausal_new = acausal/maxamp

        Green_causal  = np.imag(hilbert(causal_new))
        Green_acausal = np.imag(hilbert(acausal_new))
        stackEGF = (Green_causal + Green_acausal) / 2.0

        PhaseVIm = generate_phase(stackEGF, fs, vmin, vmax, dvel, Tmin, Tmax, dT, dist, npts, Time, dt)

        parts    = pair.split('_')
        
        phase_name_export = parts[0].split('.')[1] + '.' + parts[1].split('.')[1] + '.dat'  
        np.savetxt(os.path.join(export_path,phase_name_export), PhaseVIm)

    except:
        print(f"Could not export Phase spectrogram for file {fname}")





























































