#!/usr/bin/env python3
"""
Compute FFT-based cross-correlations in parallel via MPI for ASDF noise chunks.

Usage:
  mpirun -n <nprocs> python refactored_fft_cc_mpi.py params.yaml

Features:
  - Argparse CLI reading YAML config
  - Pathlib for robust file handling
  - Scatter/gather MPI pattern for chunk-level parallelism
  - Temporary stamp files to mark progress and resume
  - Modular functions for loading, FFT, normalization, and correlation
  - SLURM_MEM_PER_CPU-based memory check
  - Detailed logging and exception catching
"""
import argparse
import logging
import sys
import os
import re
import gc
import time

from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import yaml
from mpi4py import MPI
import pyasdf
from scipy.fft import fft
from scipy.fftpack.helper import next_fast_len
from noisepy import cross_correlation


def parse_args():
    parser = argparse.ArgumentParser(
        description="FFT + cross-correlation of noise data with MPI parallelism.")
    parser.add_argument(
        "params", type=Path,
        help="YAML parameters file for FFT/CC processing.")
    return parser.parse_args()


def setup_logging(level="INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Params file not found: {path}")
    fc_para = yaml.safe_load(path.read_text())
    # load useful download info if start from ASDF
    dfile = os.path.join(fc_para['DATADIR'], 'download_info.yaml')
    with open(dfile, "r") as file:
        down_info = yaml.safe_load(file)
    samp_freq = down_info['samp_freq']

    # Add down_info parameters to fc_para
    fc_para.update(down_info)

    dt = 1 / samp_freq
    fc_para['dt'] = dt
    return fc_para


def validate_paths(cfg: dict):
    ccf_dir = Path(cfg["CCFDIR"])
    data_dir = Path(cfg["DATADIR"])
    ccf_dir.mkdir(parents=True, exist_ok=True)
    if not data_dir.exists():
        raise FileNotFoundError(f"Noise data dir not found: {data_dir}")
    return ccf_dir, data_dir


def parse_memory_limit():
    """Parse SLURM_MEM_PER_CPU into GB."""
    mem_env = os.environ.get("SLURM_MEM_PER_CPU")
    if not mem_env:
        raise EnvironmentError("SLURM_MEM_PER_CPU not set.")
    m = re.match(r"(\d+(?:\.\d+)?)([KMG]?)", mem_env.upper())
    if not m:
        raise ValueError(f"Cannot parse SLURM_MEM_PER_CPU={mem_env}")
    val, unit = float(m.group(1)), m.group(2)
    if unit == "G": return val
    if unit in ("M", ""): return val / 1024
    if unit == "K": return val / (1024**2)
    raise ValueError(f"Unsupported unit {unit} in memory limit")


def list_chunks(data_dir: Path) -> List[Path]:
    return sorted(data_dir.glob("*.h5"))


def chunk_progress_file(ccf_dir: Path, chunk: Path) -> Path:
    name = chunk.stem + ".tmp"
    return ccf_dir / name


def estimate_memory(nsta: int, cc_len: int, step: int, samp_freq: float,
                    inc_hours: float, mem_limit: float):
    """Check required memory vs SLURM_MEM_PER_CPU."""
    nsec = inc_hours * 3600
    nseg = int(np.floor((nsec - cc_len) / step))
    npts = nseg * cc_len * samp_freq
    req_gb = nsta * npts * 4 / 1024**3
    logging.info(f"Memory req: {req_gb:.2f} GB, limit {mem_limit:.2f} GB")
    if req_gb > mem_limit:
        raise MemoryError(f"Require {req_gb:.2f} GB > {mem_limit:.2f} GB")
    if req_gb > 0.8 * mem_limit:
        logging.warning(f"High memory use: {req_gb/mem_limit:.0%}")


def load_waveform_list(chunk: Path) -> List[str]:
    with pyasdf.ASDFDataSet(str(chunk), mode='r', mpi=False) as ds:
        return ds.waveforms.list()


def build_fft_arrays(nsta: int, nseg: int, nnfft2: int):
    """Initialize arrays for FFT data and metadata."""
    fft_data = np.zeros((nsta, nseg * nnfft2), dtype=np.complex64)
    fft_std = np.zeros((nsta, nseg), dtype=np.float32)
    fft_flag = np.zeros(nsta, dtype=bool)
    fft_time = np.zeros((nsta, nseg), dtype=np.float64)
    return fft_data, fft_std, fft_flag, fft_time


def process_chunk(chunk: Path, ccf_dir: Path, cfg: dict, mem_limit: float):
    logger = logging.getLogger(f"chunk:{chunk.stem}")
    progress = chunk_progress_file(ccf_dir, chunk)
    if progress.exists() and progress.read_text().endswith("done"):
        logger.info("Already done")
        return
    elif progress.exists():
        logger.info("Incomplete, reprocessing")
        progress.unlink()
    progress.write_text("start\n")

    samp_freq = cfg['samp_freq']
    cc_len = cfg['cc_len']
    step = cfg['step']
    inc_hours = cfg['inc_hours']
    ncomp = cfg['ncomp']

    ds = pyasdf.ASDFDataSet(str(chunk), mode='r', mpi=False)
    stations = load_waveform_list(chunk)
    nsta = len(stations) * ncomp
    if not stations:
        logger.warning("No stations in chunk")
        return

    estimate_memory(nsta, cc_len, step, samp_freq, inc_hours, mem_limit)

    nseg = int(np.floor((inc_hours*3600 - cc_len) / step))
    nnfft = next_fast_len(int(cc_len * samp_freq))
    nnfft2 = nnfft // 2

    fft_data, fft_std, fft_flag, fft_time = build_fft_arrays(nsta, nseg, nnfft2)
    meta = []

    idx = 0
    for sta in stations:
        try:
            inv = ds.waveforms[sta]['StationXML']
        except KeyError:
            logger.warning(f"Missing inventory for {sta}")
            continue
        comps = ds.waveforms[sta].get_waveform_tags()
        tags = {c:None for c in ('Z','N','E')}
        for i, tag in enumerate(comps): tags[tag[2].upper()] = tag
        if any(v is None for v in tags.values()):
            logger.info(f"Incomplete comps for {sta}")
            continue

        stats_seg, time_seg, data_seg = {}, {}, {}
        for comp, tag in tags.items():
            stream = ds.waveforms[sta][tag]
            stds, times, data = cross_correlation.cut_trace_make_stat(cfg, stream)
            if data.shape[0] != nseg:
                raise ValueError(f"Segment count mismatch {sta}.{comp}")
            stats_seg[comp], time_seg[comp], data_seg[comp] = stds, times, data

        white = cross_correlation.noise_processing_3comps(
            cfg,
            data_seg['N'], data_seg['E'], data_seg['Z']
        )
        for comp in ('Z','N','E'):
            arr = fft(white[comp], nnfft)[:, :nnfft2]
            fft_data[idx] = arr.ravel()
            fft_std[idx] = stats_seg[comp]
            fft_flag[idx] = True
            fft_time[idx] = time_seg[comp]
            meta.append((sta, comp))
            idx += 1

    del ds

    cc_file = ccf_dir / f"{chunk.stem}.h5"
    with pyasdf.ASDFDataSet(str(cc_file), mode='a', mpi=False) as ccf:
        for i in range(idx):
            for j in range(i, idx):
                if cfg.get('acorr_only') and meta[i][0] != meta[j][0]:
                    continue
                good = (fft_std[i] < cfg['max_over_std']) & (fft_std[i] > 0)
                if not good.any(): continue
                s1 = fft_data[i].reshape(nseg, nnfft2)[good]
                s2 = fft_data[j].reshape(nseg, nnfft2)[good]
                corr, times_corr, nc = cross_correlation.correlate(
                    s1, s2, cfg, nnfft, fft_time[j][good]
                )
                params = cross_correlation.cc_parameters(
                    cfg, {}, times_corr, nc, meta[i][1]+meta[j][1]
                )
                path = f"{meta[i][1]}_{meta[j][1]}"
                dtype = f"{meta[i][0]}_{meta[j][0]}"
                ccf.add_auxiliary_data(
                    data=corr, data_type=dtype, path=path, parameters=params
                )

    progress.write_text(progress.read_text() + "done")
    gc.collect()
    logger.info(f"Completed chunk {chunk.stem}")


def main():
    args = parse_args()
    cfg = load_config(args.params)
    setup_logging(cfg.get('flag') and 'DEBUG' or 'INFO')

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    ccf_dir, data_dir = validate_paths(cfg)
    mem_limit = parse_memory_limit()
    chunks = list_chunks(data_dir)
    if not chunks:
        raise IOError("No ASDF chunks found in DATADIR")

    sublists = [chunks[i::size] for i in range(size)]
    assigned = comm.scatter(sublists, root=0)

    for chunk in assigned:
        try:
            process_chunk(chunk, ccf_dir, cfg, mem_limit)
        except Exception as e:
            logging.error(f"Rank {rank} error on {chunk.stem}: {e}")

    comm.Barrier()
    if rank == 0:
        logging.info("All chunks processed.")
        sys.exit(0)

if __name__ == '__main__':
    main()
