#!/usr/bin/env python3
"""
plot_ppsd_all_stations.py

Module for summarizing Probabilistic Power Spectral Densities (PPSDs)
across multiple stations and generating summary statistics and plots.

Usage:
    python plot_ppsd_all_stations.py \
        --input-dir /path/to/npz \
        --pattern "*DPZ_ppsd.npz" \
        --output-file summary.png \
        --metric mean \
        [--colormap pqlx] \
        [--figsize 15 10] \
        [--font-size 20]
"""
import argparse
import glob
import logging
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from obspy.signal import PPSD
from obspy.signal.spectral_estimation import get_nhnm, get_nlnm
from obspy.imaging.cm import pqlx
from matplotlib.ticker import FormatStrFormatter


def configure_logging(level: str = "INFO") -> None:
    """
    Configure the root logger.

    Parameters
    ----------
    level : str
        Logging level (e.g., 'DEBUG', 'INFO', 'WARNING').
    """
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s", level=getattr(logging, level)
    )


def find_ppsd_files(input_dir: str, pattern: str) -> List[str]:
    """
    Find all PPSD NPZ files matching the given pattern.

    Parameters
    ----------
    input_dir : str
        Directory to search for files.
    pattern : str
        Glob pattern to match filenames (e.g., '*DPZ_ppsd.npz').

    Returns
    -------
    List[str]
        Sorted list of matching file paths.
    """
    search_path = os.path.join(input_dir, pattern)
    files = sorted(glob.glob(search_path))
    logging.info("Found %d PPSD files", len(files))
    return files


def load_ppsd_metrics(
    files: List[str], metric: str = "mean"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load PPSD files and compute the chosen metric for each station.

    Parameters
    ----------
    files : List[str]
        List of NPZ file paths.
    metric : str
        'mean' or 'mode' to compute station-wise summary.

    Returns
    -------
    periods : np.ndarray
        Array of periods from the last file loaded.
    all_values : np.ndarray
        2D array of shape (nperiods, nstations) with the metric values.
    db_edges : np.ndarray
        Array of dB bin edges for histogram plotting.
    """
    nsta = len(files)
    ppsd0 = PPSD.load_npz(files[0])
    nperiods = ppsd0.current_histogram.shape[0]
    all_values = np.zeros((nperiods, nsta))

    for i, fp in enumerate(files):
        ppsd = PPSD.load_npz(fp)
        periods, values = getattr(ppsd, f"get_{metric}")()
        all_values[:, i] = values
        logging.debug("Loaded %s for station %d", metric, i)

    return periods, all_values, ppsd0.db_bin_edges, ppsd0.period_xedges


def compute_stats(all_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and standard deviation across stations.

    Parameters
    ----------
    all_values : np.ndarray
        Array of shape (nperiods, nstations).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        mean and std arrays of length nperiods.
    """
    mean_vals = np.mean(all_values, axis=1)
    std_vals = np.std(all_values, axis=1)
    return mean_vals, std_vals


def compute_histogram_percent(
    all_values: np.ndarray, db_edges: np.ndarray
) -> np.ndarray:
    """
    Compute histogram of values for each period and normalize to percentage.

    Parameters
    ----------
    all_values : np.ndarray
        2D array (nperiods, nstations).
    db_edges : np.ndarray
        dB bin edges used for histogramming.

    Returns
    -------
    np.ndarray
        Histogram percentages of shape (nperiods, nbins).
    """
    nperiods = all_values.shape[0]
    nbins = len(db_edges) - 1
    hist = np.zeros((nperiods, nbins))
    for idx in range(nperiods):
        hist[idx, :] = np.histogram(all_values[idx, :], bins=db_edges)[0]
    percent = hist * 100.0 / all_values.shape[1]
    return percent


def plot_ppsd_summary(
    periods: np.ndarray,
    db_edges: np.ndarray,
    percent: np.ndarray,
    output_file: str,
    period_edges: np.ndarray,
    cmap_name: str = "pqlx",
    figsize: Tuple[int, int] = (15, 10),
    font_size: int = 20,
) -> None:
    """
    Plot the PPSD summary heatmap with noise models and save to file.

    Parameters
    ----------
    periods : np.ndarray
        1D array of period centers (log-spaced).
    db_edges : np.ndarray
        dB bin edges.
    percent : np.ndarray
        Histogram percentage data (nperiods x nbins).
    output_file : str
        Path to save the figure.
    period_edges : np.ndarray
        Edges of the period bins.
    cmap_name : str
        Name of the colormap to use.
    figsize : Tuple[int,int]
        Figure size in inches (width, height).
    font_size : int
        Global font size for plotting.
    """
    plt.rcParams.update({"font.size": font_size})
    cmap = getattr(pqlx, cmap_name) if hasattr(pqlx, cmap_name) else pqlx

    fig, ax = plt.subplots(figsize=figsize)
    mesh_x, mesh_y = np.meshgrid(period_edges, db_edges)
    h = ax.pcolormesh(
        mesh_x, mesh_y, percent.T, cmap=cmap, zorder=-1
    )
    cb = fig.colorbar(h, ax=ax)
    cb.set_label("Percentage [%]")

    ax.set_xscale('log')
    ax.set_xlim(period_edges[0], period_edges[-1])
    ax.set_ylim(db_edges[0], db_edges[-1])
    ax.set_xlabel('Period [s]')
    ax.set_ylabel('Amplitude [dB]')
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))

    # Add noise models
    nhnm_periods, nhnm = get_nhnm()
    nlnm_periods, nlnm = get_nlnm()
    ax.plot(nhnm_periods, nhnm, 'r', linewidth=2, label='NHNM', zorder=10)
    ax.plot(nlnm_periods, nlnm, 'g', linewidth=2, label='NLNM', zorder=10)
    ax.legend(loc='upper right')

    ax.grid(which='major', linestyle='--', linewidth=0.5)
    ax.grid(which='minor', linestyle=':', linewidth=0.5)

    fig.savefig(output_file, bbox_inches='tight')
    logging.info("Saved summary plot to %s", output_file)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize PPSD datasets and plot summary heatmap."
    )
    parser.add_argument(
        '--input-dir', required=True, help='Directory of PPSD NPZ files'
    )
    parser.add_argument(
        '--pattern', default='*DPZ_ppsd.npz', help='Glob pattern for NPZ files'
    )
    parser.add_argument(
        '--metric', choices=['mean', 'mode'], default='mean',
        help="Metric to summarize (mean/mode)"
    )
    parser.add_argument(
        '--output-file', default='ppsd_summary.png',
        help='Output figure filename'
    )
    parser.add_argument(
        '--colormap', default='pqlx', help='Colormap for heatmap'
    )
    parser.add_argument(
        '--figsize', nargs=2, type=int, default=[15, 10],
        help='Figure size as two integers: width height'
    )
    parser.add_argument(
        '--font-size', type=int, default=20,
        help='Font size for plot labels'
    )
    parser.add_argument(
        '--log-level', default='INFO', help='Logging level'
    )
    args = parser.parse_args()

    configure_logging(args.log_level)
    files = find_ppsd_files(args.input_dir, args.pattern)
    if not files:
        logging.error("No files found. Exiting.")
        return

    periods, all_vals, db_edges, period_edges = load_ppsd_metrics(
        files, metric=args.metric
    )
    percent = compute_histogram_percent(all_vals, db_edges)
    plot_ppsd_summary(
        periods, db_edges, percent, args.output_file,
        period_edges, cmap_name=args.colormap,
        figsize=tuple(args.figsize), font_size=args.font_size
    )


if __name__ == '__main__':
    main()
