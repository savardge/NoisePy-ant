import sys
import obspy
import glob
import os
from obspy.signal import PPSD
from obspy.imaging.cm import pqlx

# ***
from obspy.imaging.util import _set_xaxis_obspy_dates
from obspy.imaging.cm import obspy_sequential
import numpy as np
def plot_spectrogram(self, cmap=obspy_sequential, clim=None, xlims=None, ylims=None, grid=True,
                         filename=None, show=True):
    """
    Plot the temporal evolution of the PSD in a spectrogram-like plot.

    .. note::
        For example plots see the :ref:`Obspy Gallery <gallery>`.

    :type cmap: :class:`matplotlib.colors.Colormap`
    :param cmap: Specify a custom colormap instance. If not specified, then
        the default ObsPy sequential colormap is used.
    :type clim: list
    :param clim: Minimum/maximum dB values for lower/upper end of colormap.
        Specified as type ``float`` or ``None`` for no clipping on one end
        of the scale (e.g. ``clim=[-150, None]`` for a lower limit of
        ``-150`` dB and no clipping on upper end).
    :type grid: bool
    :param grid: Enable/disable grid in histogram plot.
    :type filename: str
    :param filename: Name of output file
    :type show: bool
    :param show: Enable/disable immediately showing the plot.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    quadmeshes = []
    yedges = self.period_xedges

    for times, psds in self._get_gapless_psd():
        xedges = [t.matplotlib_date for t in times] + \
            [(times[-1] + self.step).matplotlib_date]
        meshgrid_x, meshgrid_y = np.meshgrid(xedges, yedges)
        data = np.array(psds).T

        quadmesh = ax.pcolormesh(meshgrid_x, meshgrid_y, data, cmap=cmap,
                                 zorder=-1)
        quadmeshes.append(quadmesh)

    if clim is None:
        cmin = min(qm.get_clim()[0] for qm in quadmeshes)
        cmax = max(qm.get_clim()[1] for qm in quadmeshes)
        clim = (cmin, cmax)

    for quadmesh in quadmeshes:
        quadmesh.set_clim(*clim)

    cb = plt.colorbar(quadmesh, ax=ax)

    if grid:
        ax.grid()

    if self.special_handling is None:
        cb.ax.set_ylabel('Amplitude [$m^2/s^4/Hz$] [dB]')
    elif self.special_handling == "infrasound":
        ax.set_ylabel('Amplitude [$Pa^2/Hz$] [dB]')
    else:
        cb.ax.set_ylabel('Amplitude [dB]')
    ax.set_ylabel('Period [s]')

    fig.autofmt_xdate()
    _set_xaxis_obspy_dates(ax)

    ax.set_yscale("log")
    if xlims:
        ax.set_xlim(xlims)
    else:
        ax.set_xlim(self.times_processed[0].matplotlib_date,
                    (self.times_processed[-1] + self.step).matplotlib_date)
    if ylims:
        ax.set_ylim(ylims)
    else: 
        ax.set_ylim(yedges[0], yedges[-1])
    try:
        ax.set_facecolor('0.8')
    # mpl <2 has different API for setting Axes background color
    except AttributeError:
        ax.set_axis_bgcolor('0.8')

    fig.tight_layout()

    if filename is not None:
        plt.savefig(filename)
        plt.close()
    elif show:
        plt.draw()
        plt.show()
    else:
        plt.draw()
    return fig
# ***

# Input 
station = sys.argv[1] # "3006977"
channel = sys.argv[2] # DPZ

# Paths
datadir = "/home/share/cdff/riehen/raw_data/"
figdir = "/home/users/s/savardg/scratch/riehen/ppsd/figures"
npzdir = "/home/users/s/savardg/scratch/riehen/ppsd/ppsd_npz"
inv = obspy.read_inventory("/home/users/s/savardg/riehen/ppsd/riehen_stations.xml")
smartsolo = True # Whether we are doing PPSD for smartSolo geophones or not

# Output files
npz_filename =  os.path.join(npzdir, f"{station}_{channel}_ppsd.npz")
outfile1 = os.path.join(figdir, f"{station}_{channel}_ppsd.png")
outfile2 = os.path.join(figdir, f"{station}_{channel}_temporal.png")
outfile3 = os.path.join(figdir, f"{station}_{channel}_spectrogram.png")

def fix_trace(tr):
    ''' Fix trace amplitude for SmartSolo geophone'''
    tr.stats.sampling_rate = 250.0 # Force sampling rate to be exactly 250
    tr.data /= 1000 # Convert mV to V
    return tr

if not os.path.exists(npz_filename):
    # File list
    sfiles = glob.glob(os.path.join(datadir, station, f"*{channel}*"))

    # Initialize ppsd object
    trace = obspy.read(os.path.join(datadir, station, f"{station}*.1.*{channel}.mseed"), headonly=True)[0]
    if smartsolo: trace = fix_trace(trace)
    ppsd = PPSD(trace.stats, metadata=inv)

    # Add other files
    for sfile in sfiles:
        try:
            trace = obspy.read(sfile)[0]
        except:
            print(f"ERROR while reading file {sfile}. Skipping")
            continue
        if smartsolo: trace = fix_trace(trace)
        ppsd.add(trace)

    print("number of psd segments:", len(ppsd.times_processed))

    # SAVE OBJECT
    print(f"Saving PPSD object to pickle: {npz_filename}")
    ppsd.save_npz(npz_filename)
    
else:
    print(f"Reading PPSD object from pickle: {npz_filename}")
    ppsd = PPSD.load_npz(npz_filename)
    
makefig = False
if makefig:
# MAKE FIGS
    import matplotlib.pyplot as plt
    plt.rcParams["figure.figsize"] = (12,8)
    minT = 0.02
    maxT = 10.0
    fig = ppsd.plot(show=False, show_mean=True, cmap=pqlx)
    ax = fig.axes[0]
    ax.set_xlim((minT, maxT))
    coverage = fig.axes[1]
    fig.delaxes(coverage)
    print(f"Saving figure to: {outfile1}")
    plt.savefig(outfile1, format="PNG")
    plt.close()

    fig = ppsd.plot_temporal([0.1, 0.2, 1], color=None, marker=".", show=False)
    fig.suptitle(f'Evolution of PPSD at period bins of 0.1, 0.2 and 1 s, station {station}')
    print(f"Saving figure to: {outfile2}")
    plt.savefig(outfile2, format="PNG")
    plt.close()

    xlims = (obspy.UTCDateTime(2020,12,5,0,0,0), obspy.UTCDateTime(2021,1,5,0,0,0))
    #fig = plot_spectrogram(ppsd, cmap=pqlx, clim=(-160, -100), xlims=xlims, ylims=(minT, maxT), grid=False, show=False)
    fig = plot_spectrogram(ppsd, cmap=pqlx, xlims=xlims, ylims=(minT, maxT), grid=False, show=False)
    fig.set_size_inches(16, 7)
    print(f"Saving figure to: {outfile3}")
    plt.savefig(outfile3, format="PNG")
    plt.close()
