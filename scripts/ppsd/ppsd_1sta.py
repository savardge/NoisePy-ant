"""
PPSD Generator Script for a Single Station

This script reads MiniSEED files, computes the Probabilistic Power Spectral Density (PPSD)
using ObsPy, and optionally creates plots. Configuration is provided through a YAML file.

Usage:
    python ppsd_station.py [station] [channel] [config_file]

Author: Refactored by ChatGPT (Python GPT)
"""

import os
import sys
import glob
import logging
import yaml
import argparse
from typing import List, Optional

import obspy
from obspy import read_inventory, read, UTCDateTime
from obspy.signal import PPSD
from obspy.imaging.cm import pqlx, obspy_sequential
from obspy.imaging.util import _set_xaxis_obspy_dates
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Matplotlib configuration for consistent styling
plt.rcParams["figure.figsize"] = (18, 12)
font = {'weight': 'normal', 'size': 22}
matplotlib.rc('font', **font)

# Ensure output directories exist before saving figures
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Created directory: {path}")

def load_config(path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    logging.info(f"Loaded config: {path}")
    return config


def find_mseed_files(datadir: str, station: str, channel: str, pattern_template: Optional[str] = None) -> List[str]:
    """Find MiniSEED files using a pattern template from config or a default fallback."""
    if pattern_template:
        pattern = pattern_template.format(datadir=datadir, station=station, channel=channel)
    else:
        pattern = os.path.join(datadir, "*", station, channel, "*")
    files = glob.glob(pattern)
    if not files:
        logging.warning(f"No MiniSEED files found with pattern: {pattern}")
    return files


def create_ppsd(files: List[str], inv_path: str, npz_path: str, overwrite: bool = False) -> PPSD:
    """
    Create or load PPSD object.

    Parameters:
        files: List of MiniSEED files
        inv_path: Path to StationXML file
        npz_path: Output .npz file to store/load PPSD
        overwrite: Whether to overwrite existing PPSD .npz

    Returns:
        PPSD object
    """
    if os.path.exists(npz_path) and not overwrite:
        logging.info(f"Loading PPSD from cache: {npz_path}")
        return PPSD.load_npz(npz_path)

    inv = read_inventory(inv_path)
    trace = read(files[0], headonly=True)[0]
    ppsd = PPSD(trace.stats, metadata=inv)

    for f in files:
        try:
            trace = read(f)[0]
            ppsd.add(trace)
        except Exception as e:
            logging.warning(f"Failed to read file {f}: {e}")

    logging.info(f"Processed {len(ppsd.times_processed)} PSD segments.")
    logging.info(f"Saving PPSD to: {npz_path}")
    ppsd.save_npz(npz_path)
    return ppsd


def plot_spectrogram(ppsd: PPSD, filename: Optional[str] = None, cmap=obspy_sequential,
                     clim=None, xlims=None, ylims=None, grid=True, show=True):
    """
    Plot a spectrogram showing temporal evolution of PSD values.

    Parameters:
        ppsd: PPSD object
        filename: If provided, saves figure to this path
        cmap: Colormap to use
        clim: Color limits (min, max) in dB
        xlims: Tuple of (start, end) time limits
        ylims: Tuple of (min, max) period limits
        grid: Show grid or not
        show: Show figure interactively

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots()
    yedges = ppsd.period_xedges
    quadmeshes = []

    for times, psds in ppsd._get_gapless_psd():
        xedges = [t.matplotlib_date for t in times] + [(times[-1] + ppsd.step).matplotlib_date]
        meshgrid_x, meshgrid_y = np.meshgrid(xedges, yedges)
        data = np.array(psds).T
        qm = ax.pcolormesh(meshgrid_x, meshgrid_y, data, cmap=cmap, zorder=-1)
        quadmeshes.append(qm)

    if clim is None:
        cmin = min(qm.get_clim()[0] for qm in quadmeshes)
        cmax = max(qm.get_clim()[1] for qm in quadmeshes)
        clim = (cmin, cmax)

    for qm in quadmeshes:
        qm.set_clim(*clim)

    cb = plt.colorbar(qm, ax=ax)
    cb.ax.set_ylabel('Amplitude [dB]')
    ax.set_ylabel('Period [s]')
    ax.set_yscale("log")
    if grid:
        ax.grid()
    _set_xaxis_obspy_dates(ax)
    if xlims:
        ax.set_xlim(xlims)
    if ylims:
        ax.set_ylim(ylims)
    ax.set_facecolor('0.8')
    fig.tight_layout()

    if filename:
        ensure_dir(os.path.dirname(filename))
        fig.set_size_inches(16, 7)
        fig.savefig(filename, format="PNG")
        plt.close(fig)
        logging.info(f"Saved spectrogram to: {filename}")
    elif show:
        plt.show()
    return fig


def generate_figures(ppsd: PPSD, config: dict, station: str, channel: str):
    """Generate and save PPSD, temporal evolution, and spectrogram plots."""
    figdir = config["figdir"]
    ensure_dir(figdir)
    outfile1 = os.path.join(figdir, f"{station}_{channel}_ppsd.png")
    outfile2 = os.path.join(figdir, f"{station}_{channel}_temporal.png")
    outfile3 = os.path.join(figdir, f"{station}_{channel}_spectrogram.png")

    minT = config["minT"]
    maxT = config["maxT"]
    period_bins = config["period_bins"]
    starttime = UTCDateTime(config["starttime"])._get_datetime()
    endtime = UTCDateTime(config["endtime"])._get_datetime()

    # Basic PPSD plot
    fig = ppsd.plot(show=False, show_mean=True, cmap=pqlx)
    fig.axes[0].set_xlim((minT, maxT))
    fig.delaxes(fig.axes[1])
    fig.set_size_inches(15, 6)
    fig.savefig(outfile1, format="PNG")
    plt.close(fig)
    logging.info(f"Saved PPSD figure to: {outfile1}")

    # Temporal evolution plot
    fig = ppsd.plot_temporal(period_bins, color=None, marker=".", show=False)
    fig.set_size_inches(14, 6)
    fig.savefig(outfile2, format="PNG")
    plt.close(fig)
    logging.info(f"Saved temporal plot to: {outfile2}")

    # Spectrogram
    plot_spectrogram(ppsd, filename=outfile3, cmap=pqlx, xlims=(starttime, endtime),
                     ylims=(minT, maxT), grid=False, show=False)


def main():
    """Main function to parse arguments and run the workflow."""
    parser = argparse.ArgumentParser(description="Compute and optionally plot PPSD from MiniSEED files.")
    parser.add_argument("station", type=str, help="Station code")
    parser.add_argument("channel", type=str, help="Channel code")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    station, channel = args.station, args.channel

    pattern_template = config.get("file_pattern")
    files = find_mseed_files(config["datadir"], station, channel, pattern_template)
    if not files:
        logging.error("No MiniSEED files found. Exiting.")
        sys.exit(1)

    npz_path = os.path.join(config["npzdir"], f"{station}_{channel}_ppsd.npz")
    ppsd = create_ppsd(files, config["stationxml_file"], npz_path, config.get("overwrite", False))

    if config.get("makefig", False):
        generate_figures(ppsd, config, station, channel)


if __name__ == "__main__":
    main()
