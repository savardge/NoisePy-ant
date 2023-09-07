"""
Output the station serial number for which the standard deviation of the positional angles logged is above a given threshold.
"""

import pandas as pd
import os
import glob

fdir = "./csvfiles"  # Directory where csv files created by "extract_QC_stats.py" are
flist = glob.glob(os.path.join(fdir, "*_stats_GPSinfo.csv"))  # Get list of files
thresh_angle = 5  # Threshold for standard deviation of each angle, in degrees

for f in flist:
    gps = pd.read_csv(f)

    # Get standard deviation for the position angles
    north_std = gps.compass.std()
    tilt_std = gps.tilt.std()
    roll_std = gps.roll.std()
    pitch_std = gps.pitch.std()

    if north_std > thresh_angle or tilt_std > thresh_angle or roll_std > thresh_angle or pitch_std > thresh_angle:
        print(f)
        print(
            f"File: {f}\n\t--> North: {north_std:.3f}\tTilt: {tilt_std:.3f}\tRoll: {roll_std:.3f}\tPitch: {pitch_std:.3f}")