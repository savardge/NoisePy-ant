#!/usr/bin/env python3
"""
Preprocess and clean SAC/MSEED files for ASDF conversion with NoisePy, now with MPI scatter/gather and robust error reporting.

Usage:
  mpirun -n <nprocs> python preprocess_asdf_mpi.py config.yaml

Features:
  - Argparse CLI with YAML config
  - MPI scatter to distribute time-chunk pairs evenly
  - Gather and report errors back to rank 0 for centralized handling
  - Modular structure and clear logging

Original Authors: by Chengxin Jiang, Marine Denolle (Jul.30.2019)
Modified by: Genevieve Savard
Refactored by: ChatGPT
"""
import argparse
import logging
import sys
import time
import warnings
import os

from pathlib import Path
import re
import numpy as np
import pandas as pd
import yaml
from mpi4py import MPI
import obspy
import pyasdf
from noisepy import preprocess_h5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean SAC/MSEED files and assemble ASDF dataset for NoisePy.")
    parser.add_argument(
        "config", type=Path,
        help="Path to YAML config file with preprocessing parameters.")
    return parser.parse_args()


def setup_logging(level: str = 'INFO'):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def validate_paths(cfg: dict):
    root = Path(cfg['rootpath'])
    rawdir = Path(cfg['RAWDATA'])
    outdir = Path(cfg['DATADIR'])
    locfile = Path(cfg['locations'])

    rawdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    if not locfile.is_file():
        raise FileNotFoundError(f"Station list not found: {locfile}")
    return rawdir, outdir, locfile


def estimate_memory(nsta, inc_hours, samp_freq, cc_len, step):
    """
    Estimate required memory for a time-chunk using SLURM_MEM_PER_CPU.
    Raises MemoryError if required memory exceeds allocation.
    """
    # Compute required memory (GB)
    nsec_chunk = inc_hours * 3600
    nseg = int(np.floor((nsec_chunk - cc_len) / step)) + 1
    npts = int(nseg * cc_len * samp_freq)
    required_gb = nsta * npts * 4 / 1024**3

    # Read memory limit from Slurm env var
    mem_env = os.environ.get('SLURM_MEM_PER_CPU')
    if not mem_env:
        raise EnvironmentError("Environment variable SLURM_MEM_PER_CPU not set.")
    m = re.match(r"(\d+(?:\.\d+)?)([KMG]?)", mem_env.upper())
    if not m:
        raise ValueError(f"Cannot parse SLURM_MEM_PER_CPU='{mem_env}'")
    val, unit = float(m.group(1)), m.group(2)
    if unit == 'G':
        max_mem_gb = val
    elif unit == 'M' or unit == '':
        max_mem_gb = val / 1024
    elif unit == 'K':
        max_mem_gb = val / (1024**2)
    else:
        raise ValueError(f"Unsupported unit '{unit}' in SLURM_MEM_PER_CPU")

    logging.info(f"Required memory: {required_gb:.2f} GB; Allocated per-CPU: {max_mem_gb:.2f} GB")
    if required_gb > max_mem_gb:
        raise MemoryError(
            f"Chunk requires {required_gb:.2f} GB, exceeds allocated {max_mem_gb:.2f} GB from SLURM_MEM_PER_CPU.")
    if required_gb > 0.8 * max_mem_gb:
        logging.warning(
            f"High memory usage: {required_gb/max_mem_gb:.0%} of per-CPU allocation.")

def broadcast(obj, comm, root=0):
    return comm.bcast(obj, root=root)


def process_station(station_info, tfiles, cfg, date_info):
    station = station_info['station']
    comp = station_info['channel']
    files = [f for f in tfiles if station in f and comp in f]
    if not files:
        logging.debug(f"No files for {station}.{comp} in window.")
        return

    stream = obspy.Stream()
    for fpath in files:
        try:
            st = obspy.read(fpath)
            stream += st
        except Exception as e:
            logging.warning(f"Failed to read {fpath}: {e}")
    if not stream:
        logging.warning(f"No valid traces for {station}.{comp}, skipping.")
        return

    try:
        inv = preprocess_h5.stats2inv(stream[0].stats, cfg, locs=None)
        trimmed = preprocess_h5.preprocess_raw(stream, inv, cfg, date_info)
        if not trimmed or np.all(trimmed[0].data == 0):
            logging.warning(f"Preprocessing zero-length for {station}.{comp}.")
            return
    except Exception as e:
        logging.error(f"Error preprocessing {station}.{comp}: {e}")
        raise

    ofile = Path(cfg['DATADIR']) / f"{date_info['starttime'].strftime('%Y%m%dT%H%M%S')}_{date_info['endtime'].strftime('%Y%m%dT%H%M%S')}.h5"
    try:
        with pyasdf.ASDFDataSet(str(ofile), mode='a', mpi=False, compression='gzip-3') as ds:
            ds.add_stationxml(inv)
            tag = f"{comp.lower()}_00"
            ds.add_waveforms(trimmed, tag=tag)
    except Exception as e:
        logging.error(f"Failed to write ASDF for {station}.{comp}: {e}")
        raise


def main():
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.get('log_level', 'INFO'))

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Rank 0 prepares data
    if rank == 0:
        rawdir, outdir, locfile = validate_paths(cfg)
        locs = pd.read_csv(locfile, dtype={'station': str})
        all_stimes, allfiles = preprocess_h5.make_timestamps(cfg)
        chunks = preprocess_h5.get_event_list(
            cfg['start_date'], cfg['end_date'], cfg['inc_hours'])
        if len(chunks) < 2:
            raise ValueError("No time chunks generated.")
        estimate_memory(
            len(locs), cfg['inc_hours'], cfg['samp_freq'],
            cfg['cc_len'], cfg['step'], cfg['MAX_MEM']
        )
        # prepare paired chunks and scatter
        chunk_pairs = list(zip(chunks[:-1], chunks[1:]))
        chunk_sublists = [chunk_pairs[i::size] for i in range(size)]
    else:
        locs = all_stimes = allfiles = None
        chunk_sublists = None

    # Broadcast locs and file/time indices
    locs      = broadcast(locs,      comm)
    all_stimes= broadcast(all_stimes, comm)
    allfiles  = broadcast(allfiles,   comm)

    # Scatter chunk assignments
    assigned = comm.scatter(chunk_sublists, root=0)
    local_errors = []

    for start, end in assigned:
        date_info = {
            'starttime': obspy.UTCDateTime(start),
            'endtime':   obspy.UTCDateTime(end)
        }
        try:
            # identify file indices for this chunk
            t1 = date_info['starttime'] - obspy.UTCDateTime(1970,1,1)
            t2 = date_info['endtime']   - obspy.UTCDateTime(1970,1,1)
            idx = np.where(
                ((all_stimes[:,0] <= t1) & (t1 < all_stimes[:,1])) |
                ((all_stimes[:,0] <  t2) & (t2 <=all_stimes[:,1])) |
                ((all_stimes[:,0] >= t1) & (all_stimes[:,1] <=t2))
            )[0]
            if not len(idx):
                logging.info(f"No data for chunk {start} to {end}.")
                continue
            tfiles = [allfiles[j] for j in idx]

            # process each station
            for _, row in locs.iterrows():
                process_station(row, tfiles, cfg, date_info)

            logging.info(f"Rank {rank}: Chunk {start}-{end} completed.")
        except Exception as e:
            logging.error(f"Rank {rank}: Failure on chunk {start}-{end}: {e}")
            local_errors.append({'rank': rank, 'chunk': (start, end), 'error': str(e)})

    # synchronize and gather errors
    comm.Barrier()
    all_errors = comm.gather(local_errors, root=0)
    if rank == 0 and any(all_errors):
        flat = [e for sub in all_errors for e in sub]
        logging.error("Errors encountered during processing:")
        for e in flat:
            s, e_str = e['chunk'], e['error']
            logging.error(f" Rank {e['rank']} chunk {s[0]}-{s[1]}: {e_str}")
        sys.exit(1)

if __name__ == '__main__':
    main()
