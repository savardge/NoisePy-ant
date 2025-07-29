#!/usr/bin/env python3
"""
Stack cross-correlation functions in parallel via MPI.

Usage:
  mpirun -n <nprocs> python refactored_stacking.py S2_params.yaml

Features:
  - Argparse CLI reading YAML config
  - Pathlib for robust file handling
  - Scatter/gather MPI pattern over station-pairs
  - SLURM_MEM_PER_CPU-based memory check
  - Modular functions for data loading, stacking, and rotation
  - Temporary stamp files for resume logic
  - Detailed logging and error handling
"""
import argparse
import logging
import sys
import os
import re
import gc
import time

from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from mpi4py import MPI
import pyasdf
from noisepy import stacking

# Define component orders
ENZ_ORDER = ['EE','EN','EZ','NE','NN','NZ','ZE','ZN','ZZ']
RTZ_COMPONENTS = ['ZR','ZT','ZZ','RR','RT','RZ','TR','TT','TZ']


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stack cross-correlations with MPI parallelism.")
    parser.add_argument(
        "params", type=Path,
        help="YAML parameters file for stacking.")
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
    return yaml.safe_load(path.read_text())


def validate_paths(cfg: dict):
    ccf_dir = Path(cfg["CCFDIR"])
    stack_dir = Path(cfg["STACKDIR"])
    loc_file = Path(cfg["locations"])
    ccf_dir.mkdir(parents=True, exist_ok=True)
    stack_dir.mkdir(parents=True, exist_ok=True)
    if not loc_file.is_file():
        raise FileNotFoundError(f"Locations file not found: {loc_file}")
    return ccf_dir, stack_dir, loc_file


def parse_memory_limit():
    mem_env = os.environ.get("SLURM_MEM_PER_CPU")
    if not mem_env:
        raise EnvironmentError("SLURM_MEM_PER_CPU not set.")
    m = re.match(r"(\d+(?:\.\d+)?)([KMG]?)", mem_env.upper())
    if not m:
        raise ValueError(f"Cannot parse SLURM_MEM_PER_CPU={mem_env}")
    val, unit = float(m.group(1)), m.group(2)
    if unit == "G": return val
    if unit in ("M", ""): return val/1024
    if unit == "K": return val/(1024**2)
    raise ValueError(f"Unsupported unit {unit}")


def list_ccf_files(ccf_dir: Path) -> list[Path]:
    return sorted(ccf_dir.glob("*.h5"))


def get_station_pairs(loc_file: Path) -> list[str]:
    df = pd.read_csv(loc_file, dtype={'station': str})
    ids = df['network'] + "." + df['station']
    uniq = sorted(ids.unique())
    return [f"{uniq[i]}_{uniq[j]}" for i in range(len(uniq)) for j in range(i, len(uniq))]


def pair_progress_file(stack_dir: Path, pair: str) -> Path:
    return stack_dir / f"{pair}.tmp"


def estimate_memory(num_ccf: int, num_segs: int, seg_len: int, mem_limit: float):
    req_gb = num_ccf * num_segs * seg_len * 4 / 1024**3
    logging.info(f"Memory required: {req_gb:.2f} GB, limit: {mem_limit:.2f} GB")
    if req_gb > mem_limit:
        raise MemoryError(f"Require {req_gb:.2f} GB > {mem_limit:.2f} GB")
    if req_gb > 0.8 * mem_limit:
        logging.warning(f"High memory use: {req_gb/mem_limit:.0%}")


def process_pair(pair: str, ccf_files: list[Path], stack_dir: Path, cfg: dict, mem_limit: float):
    logger = logging.getLogger(f"pair:{pair}")
    stamp = pair_progress_file(stack_dir, pair)
    if stamp.exists() and stamp.read_text().strip() == "done":
        logger.info("Already done")
        return
    if stamp.exists():
        logger.info("Incomplete, reprocessing")
        stamp.unlink()
    stamp.write_text("start\n")

    # parse station IDs
    src, rec = pair.split('_')
    snet, ssta = src.split('.')
    rnet, rsta = rec.split('.')

    # load location metadata if rotation
    if cfg.get('rotation'):
        locs_df = pd.read_csv(cfg['locations'], dtype={'station': str})

    # parameters
    ncomp = cfg['ncomp']
    maxlag = cfg['maxlag']
    samp_freq = cfg['samp_freq']
    inc_hours = cfg['inc_hours']
    cc_len = cfg['cc_len']
    step = cfg['step']
    substack = cfg['substack']
    substack_len = cfg['substack_len']

    # memory estimate
    num_ccf = len(ccf_files) * ncomp * ncomp
    num_segs = 1 if not substack else int(inc_hours*3600/substack_len)
    seg_len = int(2*maxlag*samp_freq) + 1
    estimate_memory(num_ccf, num_segs, seg_len, mem_limit)

    # allocate storage for components
    cc_data = np.zeros((num_ccf * num_segs, seg_len), dtype=np.float32)
    cc_time = np.zeros(num_ccf * num_segs)
    cc_ngood = np.zeros(num_ccf * num_segs, dtype=int)
    cc_comp = np.chararray(num_ccf * num_segs, itemsize=2)

    idx = 0
    for file in ccf_files:
        ds = pyasdf.ASDFDataSet(str(file), mode='r', mpi=False)
        try:
            paths = ds.auxiliary_data[pair].list()
        except KeyError:
            ds.close()
            continue
        for p in paths:
            data = ds.auxiliary_data[pair][p].data
            params = ds.auxiliary_data[pair][p].parameters.copy()
            times = params['time']
            ngood = params['ngood']
            comp = p.split('_')[0][-1] + p.split('_')[1][-1]
            if substack:
                for s in range(data.shape[0]):
                    cc_data[idx] = data[s]
                    cc_time[idx] = times[s]
                    cc_ngood[idx] = ngood[s]
                    cc_comp[idx] = comp
                    idx += 1
            else:
                cc_data[idx] = data
                cc_time[idx] = times
                cc_ngood[idx] = ngood
                cc_comp[idx] = comp
                idx += 1
        ds.close()

    if idx == 0:
        logger.warning(f"No data for pair {pair}, skipping.")
        return

    # stacking
    outfile = stack_dir / f"{pair}.h5"
    # collect stacks by method and comp for rotation
    stacks_by_method = {m: {} for m in stacking.methods(cfg)}

    for comp in np.unique(cc_comp[:idx]):
        mask = cc_comp[:idx] == comp
        data = cc_data[:idx][mask]
        times = cc_time[:idx][mask]
        ng = cc_ngood[:idx][mask]
        (cc_final, ng_final, stamps, *allstacks, nstacks) = stacking.stacking(
            data, times, ng, cfg)
        methods = stacking.methods(cfg)
        with pyasdf.ASDFDataSet(str(outfile), mode='a', mpi=False) as ds2:
            for method, arr in zip(methods, allstacks):
                params = {'time': stamps[0], 'ngood': nstacks, 'comp': comp}
                ds2.add_auxiliary_data(
                    data=arr, data_type=f"Allstack_{method}", path=comp, parameters=params
                )
                stacks_by_method[method][comp] = arr

    # rotation
    if cfg.get('rotation'):
        with pyasdf.ASDFDataSet(str(outfile), mode='a', mpi=False) as ds2:
            for method, comp_dict in stacks_by_method.items():
                # build 9xT array in ENZ order
                bigstack = np.zeros((len(ENZ_ORDER), seg_len), dtype=np.float32)
                for i, c in enumerate(ENZ_ORDER):
                    bigstack[i] = comp_dict.get(c, np.zeros(seg_len))
                # rotate into RTZ
                rot = stacking.rotation(
                    bigstack,
                    {'station_source': ssta, 'station_receiver': rsta},
                    locs_df
                )
                # write rotated
                for i, rcomp in enumerate(RTZ_COMPONENTS):
                    params = {'time': stamps[0], 'ngood': nstacks, 'comp': rcomp}
                    ds2.add_auxiliary_data(
                        data=rot[i],
                        data_type=f"Allstack_{method}",
                        path=rcomp,
                        parameters=params
                    )
    stamp.write_text(stamp.read_text() + "done")
    logger.info(f"Completed pair {pair}")


def main():
    args = parse_args()
    cfg = load_config(args.params)
    setup_logging(cfg.get('flag') and 'DEBUG' or 'INFO')

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    ccf_dir, stack_dir, loc_file = validate_paths(cfg)
    mem_limit = parse_memory_limit()
    ccf_files = list_ccf_files(ccf_dir)
    if not ccf_files:
        raise IOError("No CCF files found")
    pairs = get_station_pairs(loc_file)

    sublists = [pairs[i::size] for i in range(size)]
    assigned = comm.scatter(sublists, root=0)

    for pair in assigned:
        try:
            process_pair(pair, ccf_files, stack_dir, cfg, mem_limit)
        except Exception as e:
            logging.error(f"Rank {rank} error on {pair}: {e}")

    comm.Barrier()
    if rank == 0:
        logging.info("All pairs done")
        sys.exit(0)

if __name__ == '__main__':
    main()
