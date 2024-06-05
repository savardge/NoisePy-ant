"""
Extract metadata from the raw log files created by SmartSolo. Makes a plot showing:
- Temperature
- Latitude & Longitude
- Altitude
- eCompass north (careful! absolute value is wrong if node not calibrated...)
- Tilted angle
- Roll and Pitch angles

Input arguments:
> python extract_QC_stats.py DCCDATA_DIR OUTPUT_DIR
where 
DCCDATA_DIR is parent folder containing the subfolders for each serial number (453...)
OUTPUT_DIR is where to save figure (JPG format) and data tables (CSV)

Author: genevieve.savard@unige.ch
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

LOCAL_TIME_ZONE = 'CET'


def get_temperature(fname):
    temps = []
    times = []
    with open(fname, "r") as f:
        for line in f:
            if line.startswith("[Temperature"):
                dum = next(f).split()[-1].replace("\"", "")
                t = pd.to_datetime(dum, utc=True, format="%Y/%m/%d,%H:%M:%S")
                temp = np.float32(next(f).split()[-1])
                times.append(t)
                temps.append(temp)

    df = pd.DataFrame({"time_UTC": times})
    df['time_local'] = df['time_UTC'].dt.tz_convert(LOCAL_TIME_ZONE)
    df['temperature'] = temps
    return df


def get_gps_info(fname):
    times = []
    compass = []
    tilt = []
    roll = []
    pitch = []
    longitude = []
    latitude = []
    altitude = []
    with open(fname, "r") as f:
        for line in f:
            if line.startswith("[GPS"):
                status = next(f).split("=")[-1].strip()  # GPS Status
                if status == "GPS Synchronization":
                    # print(line)
                    dum = next(f).split()[-1].strip("\"")
                    times.append(pd.to_datetime(dum, utc=True, format="%Y/%m/%d,%H:%M:%S"))
                    dum = next(f)  # Lead Second
                    for k in range(15):
                        if dum.startswith("eCompass"):
                            compass.append(np.float32(dum.split()[-1]))
                        elif dum.startswith("Tilted Angle"):
                            tilt.append(np.float32(dum.split()[-1]))
                        elif dum.startswith("Roll Angle"):
                            roll.append(np.float32(dum.split()[-1]))
                        elif dum.startswith("Pitch Angle"):
                            pitch.append(np.float32(dum.split()[-1]))
                        elif dum.startswith("Longitude"):
                            longitude.append(np.float32(dum.split()[-1]))
                        elif dum.startswith("Latitude"):
                            latitude.append(np.float32(dum.split()[-1]))
                        elif dum.startswith("Altitude"):
                            try:
                                altitude.append(np.float32(dum.split()[-1]))
                            except:
                                altitude.append(np.nan)
                        if dum == "\n":
                            for val in [compass, tilt, roll, pitch, longitude, latitude, altitude]:
                                if len(val) < len(times):
                                    val.append(np.nan)
                            break
                        else:
                            try:
                                dum = next(f)
                            except:
                                for val in [compass, tilt, roll, pitch, longitude, latitude, altitude]:
                                    if len(val) < len(times):
                                        val.append(np.nan)
                                pass

    df = pd.DataFrame({"time_UTC": times})
    df['time_local'] = df['time_UTC'].dt.tz_convert(LOCAL_TIME_ZONE)
    df['compass'] = compass
    df["tilt"] = tilt
    df["roll"] = roll
    df["pitch"] = pitch
    df["longitude"] = longitude
    df["latitude"] = latitude
    df["altitude"] = altitude
    return df


if __name__ == "__main__":

    DCCDATA_DIR = sys.argv[1]  # DCCDATA folder path
    OUTPUT_DIR = sys.argv[2]  # Output directory for csv and figure files
    start_date_str = sys.argv[3]  # e.g. "2024/03/01,00:00:00"
    end_date_str = sys.argv[4]  # e.g. "2024/04/25,00:00:00"
    start_date = pd.to_datetime(start_date_str, utc=True) #, format="%Y/%m/%d,%H:%M:%S")
    end_date = pd.to_datetime(end_date_str, utc=True) #, format="%Y/%m/%d,%H:%M:%S")
    print(f"Keeping data between {start_date_str} and {end_date_str}")

    filelist = glob.glob(os.path.join(DCCDATA_DIR, "*", "*", "DigiSolo.LOG"))
    print(f"Reading data from directory: {DCCDATA_DIR}: Number of DigiSolo.LOG files found: {len(filelist)}")
    filelist.sort()
    for fname in filelist:
        print(fname)

        # Extract metadata
        temp = get_temperature(fname)
        gps = get_gps_info(fname)

        # Keep only dates after start_date
        temp = temp.loc[(temp.time_UTC >= start_date) & (temp.time_UTC <= end_date), :]
        gps = gps.loc[(gps.time_UTC >= start_date) & (gps.time_UTC <= end_date), :]

        # Get serial number
        n = len(DCCDATA_DIR.rstrip("/").split("/"))
        serial_number = fname.split("/")[n]

        # Save data to CSV tables
        fname_table1 = os.path.join(OUTPUT_DIR, f"{serial_number}_stats_temperature.csv")
        temp.to_csv(fname_table1, index=False)
        fname_table2 = os.path.join(OUTPUT_DIR, f"{serial_number}_stats_GPSinfo.csv")
        gps.to_csv(fname_table2, index=False)

        # Plot 
        figname = os.path.join(OUTPUT_DIR, f"{serial_number}_stats.jpg")
        fig, axs = plt.subplots(6, 1, figsize=(16, 6 * 2), sharex=True)
        # Temperature
        axs[0].plot(temp.time_local.values, temp.temperature.values, c="b")
        axs[0].set_title(f"Temperature [C] with std deviation = {temp.temperature.std():.3f}")
        axs[0].set_ylabel("Temperature [C]")
        # Latitude
        axs[1].plot(gps.time_local.values, gps.latitude.values, c="b")
        axs[1].set_title(
            f"Latitude and Longitude with std deviations = {gps.latitude.std():.9f}, {gps.longitude.std():.9f}")
        axs[1].set_ylabel("Latitude")
        ax = axs[1].twinx()
        ax.plot(gps.time_local.values, gps.longitude.values, c="g")
        ax.set_ylabel("Longitude")
        # Elevation
        axs[2].plot(gps.time_local.values, gps.altitude.values, c="b")
        axs[2].set_title(f"Altitude [m] with std deviation = {gps.altitude.std():.3f}")
        axs[2].set_ylabel("Altitude [m]")
        # Compass north
        axs[3].plot(gps.time_local.values, gps.compass.values, c="b")
        axs[3].set_title(f"eCompass North with std deviation = {gps.compass.std():.3f}")
        axs[3].set_ylabel("eCompass North direction")
        # Tilt
        axs[4].plot(gps.time_local.values, gps.tilt.values, c="b")
        axs[4].set_title(f"Tilt angle with std deviation = {gps.tilt.std():.3f}")
        axs[4].set_ylabel("Tilt angle")
        # Roll and pitch
        axs[5].plot(gps.time_local.values, gps.roll.values, c="b")
        axs[5].set_title(f"Roll and pitch angles with std deviations = {gps.roll.std():.3f}, {gps.pitch.std():.3f}")
        axs[5].set_ylabel("Roll angle")
        ax = axs[5].twinx()
        ax.plot(gps.time_local.values, gps.pitch.values, c="g")
        ax.set_ylabel("Pitch angle")
        plt.suptitle(f"Serial number: {serial_number}")
        plt.tight_layout()
        plt.savefig(figname, format="JPG")
        plt.close()
