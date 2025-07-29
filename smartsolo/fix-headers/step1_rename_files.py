#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Script: rename_mseed_by_station.py
#
# Author: Geneviève Savard
# Date: 2025-04-19
#
# Description:
#   This script renames SmartSolo-exported miniseed files based on a station
#   mapping provided in a CSV file. It parses metadata from filenames, removes
#   unwanted elements, and restructures the output into per-station subfolders
#   organized by component and data type. Failures are logged, and a CSV of
#   unprocessed serial numbers is created for review.
#
# Usage:
#   - Configure the paths in the CONFIGURATION section.
#   - Run with Python 3.x.
#
# Outputs:
#   - Renamed and moved miniseed files in a structured directory tree
#   - Log file with detailed info and warnings
#   - CSV report of failed renames
#
# Dependencies:
#   - pandas, logging, shutil, glob, os
#
# -----------------------------------------------------------------------------

import os
import glob
import shutil
import pandas as pd
import logging
from datetime import datetime
import csv

# --- TRACKING FAILURES ---
failed_entries = []

# --- CONFIGURATION ---
stainfo_path = "/home/share/cdff/thurgau/fix-headers/coordinates_from_log_mode_wName.csv"  # CSV mapping serial numbers to station names
datadir = "/srv/beegfs/scratch/shares/cdff/thurgau/DATA_MSEED/TG_mseed"  # Root directory containing original miniseed files
network = "SS"
location = "SW"
datatype = "D"
dry_run = False  # If True, only simulate renaming without moving files

# --- LOGGING SETUP ---
log_dir = os.path.join(os.getcwd(), "rename_logs")
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"rename_log_{datetime.now():%Y%m%d_%H%M%S}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

# --- LOAD STATION INFO ---
stainfo = pd.read_csv(stainfo_path, dtype=str).fillna("")
serial_to_station = dict(zip(stainfo["serial_number"], stainfo["station"]))

# --- PROCESS FILES ---
for count, (sn, station) in enumerate(serial_to_station.items(), 1):
    logging.info(f"{count}: Processing SN={sn} -> Station={station}")
    olddir = os.path.join(datadir, sn)
    newdir = os.path.join(datadir, station)

    if not os.path.isdir(olddir):
        logging.warning(f"  [Skip] Original directory {olddir} does not exist.")
        row = stainfo[stainfo["serial_number"] == sn].copy()
        row["reason"] = "Missing directory"
        failed_entries.append(row)
        continue

    os.makedirs(newdir, exist_ok=True)
    flist = glob.glob(os.path.join(olddir, f"{sn}.*"))
    renamed_any = False

    for oldfile_path in flist:
        oldfile = os.path.basename(oldfile_path)

        # Remove the second element from filename (e.g., remove "0002" from "SN.0002.YYYY.MM.DD...")
        parts = oldfile.split(".")
        if len(parts) >= 10:
            parts.pop(1)
            cleaned_oldfile = ".".join(parts)
        else:
            logging.warning(f"  [Skip] Unexpected filename format: {oldfile}")
            continue

        # Determine component code
        if ".Z." in oldfile:
            comp = "DPZ"
        elif ".N." in oldfile:
            comp = "DPN"
        elif ".E." in oldfile:
            comp = "DPE"
        else:
            logging.warning(f"  [Skip] Unknown component in file {oldfile}")
            continue

        # Create subfolder for this component (e.g., TG01/DPZ.D/)
        subdir = os.path.join(newdir, f"{comp}.{datatype}")
        os.makedirs(subdir, exist_ok=True)

        # Construct the new filename using network, station, location, component, and SN
        newfile = cleaned_oldfile.replace(sn, f"{network}.{station}.{location}.{comp}.{datatype}")
        newfile = newfile.replace(f".{comp[-1]}.miniseed", f".{sn}.miniseed")
        newfile_path = os.path.join(subdir, newfile)

        if os.path.exists(newfile_path):
            logging.warning(f"  [Skip] Target file {newfile} already exists.")
            continue

        try:
            if dry_run:
                logging.info(f"  [Dry Run] Would rename: {oldfile_path} -> {newfile_path}")
            else:
                shutil.move(oldfile_path, newfile_path)
                logging.info(f"  [OK] Renamed: {oldfile_path} -> {newfile_path}")
                renamed_any = True
        except Exception as e:
            logging.error(f"  [Error] Failed to rename {oldfile}: {e}")
            row = stainfo[stainfo["serial_number"] == sn].copy()
            row["reason"] = f"Rename error: {e}"
            failed_entries.append(row)

    if not renamed_any:
        row = stainfo[stainfo["serial_number"] == sn].copy()
        row["reason"] = "No files renamed"
        failed_entries.append(row)

    # Optionally remove old directory if it's empty
    try:
        if not os.listdir(olddir):
            if not dry_run:
                os.rmdir(olddir)
            logging.info(f"  [Clean] Removed empty directory {olddir}")
    except Exception as e:
        logging.warning(f"  [Warning] Could not remove {olddir}: {e}")

# --- EXPORT FAILURES ---
logging.info(f"\nFinished. Full log saved to {log_filename}")

if failed_entries:
    failed_df = pd.concat(failed_entries, ignore_index=True).drop_duplicates()
    failed_filename = os.path.join(log_dir, f"failed_renames_{datetime.now():%Y%m%d_%H%M%S}.csv")
    failed_df.to_csv(failed_filename, index=False)
    logging.warning(f"\nSome serials could not be renamed. See: {failed_filename}")
else:
    logging.info("\nAll serials processed successfully.")

