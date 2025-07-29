"""
SmartSolo LOG Metadata Extractor

This script parses SmartSolo DigiSolo.LOG files to extract:
- Temperature data over time
- GPS synchronization data, including:
    - eCompass North direction
    - Tilted, Roll, and Pitch angles
    - Latitude, Longitude, Altitude

Outputs:
- CSV files for temperature and GPS metadata
- A diagnostic multi-panel plot showing temporal evolution of all values

Usage:
    python extract_QC_stats.py DCCDATA_DIR OUTPUT_DIR START_TIME END_TIME

Arguments:
    DCCDATA_DIR   Path to root directory containing SmartSolo LOG files
    OUTPUT_DIR    Directory to save CSVs and plots
    START_TIME    ISO date string in UTC (e.g. "2024/03/01,00:00:00")
    END_TIME      ISO date string in UTC (e.g. "2024/04/25,00:00:00")

Author:
    Geneviève Savard — genevieve.savard@unige.ch
    Refactored by: ChatGPT
"""

import os
import re
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

LOCAL_TIME_ZONE = 'CET'


def parse_log_file(lines):
    """Extract temperature and GPS info from a SmartSolo LOG file (lines)."""
    temp_data = []
    gps_data = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Temperature block
        if line.startswith("[Temperature"):
            timestamp_line = lines[i+1].strip()
            temp_line = lines[i+2].strip()
            try:
                ts = pd.to_datetime(timestamp_line.split()[-1].replace('"', ''), utc=True, format="%Y/%m/%d,%H:%M:%S")
                temp = float(temp_line.split()[-1])
                temp_data.append((ts, temp))
            except Exception as e:
                print(f"Warning: failed to parse temperature block at line {i}: {e}")
            i += 3

        # GPS Sync block
        elif line.startswith("[GPS"):
            try:
                status = lines[i+1].split('=')[-1].strip()
                if status == "GPS Synchronization":
                    ts = pd.to_datetime(lines[i+2].split()[-1].strip('"'), utc=True, format="%Y/%m/%d,%H:%M:%S")
                    gps_entry = {'time_UTC': ts, 'compass': np.nan, 'tilt': np.nan, 'roll': np.nan, 'pitch': np.nan,
                                 'longitude': np.nan, 'latitude': np.nan, 'altitude': np.nan}
                    j = i + 4  # skip to potential data lines
                    while j < len(lines) and not lines[j].startswith("["):
                        field = lines[j].strip()
                        if "eCompass" in field:
                            gps_entry['compass'] = float(field.split()[-1])
                        elif "Tilted Angle" in field:
                            gps_entry['tilt'] = float(field.split()[-1])
                        elif "Roll Angle" in field:
                            gps_entry['roll'] = float(field.split()[-1])
                        elif "Pitch Angle" in field:
                            gps_entry['pitch'] = float(field.split()[-1])
                        elif "Longitude" in field:
                            gps_entry['longitude'] = float(field.split()[-1])
                        elif "Latitude" in field:
                            gps_entry['latitude'] = float(field.split()[-1])
                        elif "Altitude" in field:
                            try:
                                gps_entry['altitude'] = float(field.split()[-1])
                            except:
                                gps_entry['altitude'] = np.nan
                        j += 1
                    gps_data.append(gps_entry)
            except Exception as e:
                print(f"Warning: failed to parse GPS block at line {i}: {e}")
            i += 1
        else:
            i += 1

    temp_df = pd.DataFrame(temp_data, columns=['time_UTC', 'temperature'])
    temp_df['time_local'] = temp_df['time_UTC'].dt.tz_convert(LOCAL_TIME_ZONE)

    gps_df = pd.DataFrame(gps_data)
    if not gps_df.empty:
        gps_df['time_local'] = gps_df['time_UTC'].dt.tz_convert(LOCAL_TIME_ZONE)

    return temp_df, gps_df


def plot_qc_data(temp, gps, serial_number, outdir):
    """Generate and save QC plot from temperature and GPS DataFrames."""
    figname = os.path.join(outdir, f"{serial_number}_stats.jpg")
    fig, axs = plt.subplots(6, 1, figsize=(16, 12), sharex=True)

    axs[0].plot(temp.time_local, temp.temperature, c="b")
    axs[0].set_title(f"Temperature [C] std = {temp.temperature.std():.3f}")
    axs[0].set_ylabel("Temperature")

    axs[1].plot(gps.time_local, gps.latitude, label='Lat', c="b")
    ax2 = axs[1].twinx()
    ax2.plot(gps.time_local, gps.longitude, label='Lon', c="g")
    axs[1].set_title(f"Lat/Lon std = {gps.latitude.std():.6f}, {gps.longitude.std():.6f}")
    axs[1].set_ylabel("Latitude")
    ax2.set_ylabel("Longitude")

    axs[2].plot(gps.time_local, gps.altitude, c="b")
    axs[2].set_title(f"Altitude std = {gps.altitude.std():.3f}")
    axs[2].set_ylabel("Altitude")

    axs[3].plot(gps.time_local, gps.compass, c="b")
    axs[3].set_title(f"Compass std = {gps.compass.std():.3f}")
    axs[3].set_ylabel("eCompass")

    axs[4].plot(gps.time_local, gps.tilt, c="b")
    axs[4].set_title(f"Tilt std = {gps.tilt.std():.3f}")
    axs[4].set_ylabel("Tilt")

    axs[5].plot(gps.time_local, gps.roll, c="b", label="Roll")
    ax2 = axs[5].twinx()
    ax2.plot(gps.time_local, gps.pitch, c="g", label="Pitch")
    axs[5].set_title(f"Roll/Pitch std = {gps.roll.std():.3f}, {gps.pitch.std():.3f}")
    axs[5].set_ylabel("Roll")
    ax2.set_ylabel("Pitch")

    plt.suptitle(f"SmartSolo QC for Serial: {serial_number}")
    plt.tight_layout()
    plt.savefig(figname)
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dccdir", help="Parent DCC directory containing log files")
    parser.add_argument("outdir", help="Output directory for CSVs and plots")
    parser.add_argument("start", help="Start UTC time (e.g. 2024/03/01,00:00:00")
    parser.add_argument("end", help="End UTC time (e.g. 2024/04/25,00:00:00")
    args = parser.parse_args()

    start = pd.to_datetime(args.start, utc=True)
    end = pd.to_datetime(args.end, utc=True)

    os.makedirs(args.outdir, exist_ok=True)
    logfiles = sorted(glob.glob(os.path.join(args.dccdir, "*", "*", "DigiSolo.LOG")))
    print(f"Found {len(logfiles)} LOG files.")

    for path in logfiles:
        with open(path) as f:
            lines = f.readlines()

        temp, gps = parse_log_file(lines)
        temp = temp[(temp.time_UTC >= start) & (temp.time_UTC <= end)]
        gps = gps[(gps.time_UTC >= start) & (gps.time_UTC <= end)]

        serial = os.path.basename(os.path.dirname(os.path.dirname(path)))
        temp.to_csv(os.path.join(args.outdir, f"{serial}_stats_temperature.csv"), index=False)
        gps.to_csv(os.path.join(args.outdir, f"{serial}_stats_GPSinfo.csv"), index=False)

        if not temp.empty and not gps.empty:
            plot_qc_data(temp, gps, serial, args.outdir)


if __name__ == "__main__":
    main()
