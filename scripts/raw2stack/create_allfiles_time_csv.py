# Create allfiles_time.csv for NoisePy S0B

import glob
import obspy
import pandas as pd
import os

RAWDATA   = '/home/users/s/savardg/scratch/aargau_broadband/RAW_DATA_MSEED'

tmpfname = "allfiles_time_temp_SED.csv"
fname = "allfiles_time_SED.csv"

# STEP 1 IF INPUT IN SAC:
# SAC command to get start time info:
# echo "filepath,starttime,length_seconds" > allfiles_time_temp.csv
# saclst kzdate kztime e f /home/users/s/savardg/aargau_data/453*/*.sac | awk '{print $1","$2"T"$3","$4}' >> allfiles_time_temp.txt

# STEP 1 IF INPUT IN MSEED:
flist = glob.glob(os.path.join(RAWDATA, "*", "*.mseed"))
with open(tmpfname,"w") as of:
    of.write("filepath,starttime,length_seconds\n")
    for ind, f in enumerate(flist):
        if ind % 100 == 0: print(f"{ind+1}/{len(flist)}")
        dum = obspy.read(f, headonly=True)
        tstart = min([tr.stats.starttime for tr in dum])
        tend = max([tr.stats.endtime for tr in dum])
        #print(f,tstart,tend)
        length_seconds = tend - tstart
        starttime = tstart.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        of.write(f"{f},{starttime},{length_seconds}\n")
print("Finished writing temporary file")

# STEP 2:
df = pd.read_csv(tmpfname)
# create test data
dates = pd.to_datetime(df["starttime"])

# calculate unix datetime
start_sec = (dates - pd.Timestamp("1970-01-01")) / pd.Timedelta('1s')
end_sec = start_sec.astype(float) + df["length_seconds"].astype(float)

newdf = pd.DataFrame({"names": df["filepath"], "starttime": start_sec, "endtime": end_sec})


# Check by converting back to date time format
newdf["starttime_"] = pd.to_datetime(newdf["starttime"], unit='s', origin='unix') 
newdf["endtime_"] = pd.to_datetime(newdf["endtime"], unit='s', origin='unix') 
print(newdf["starttime_"].min())
print(newdf["starttime_"].max())

newdf.head()

#print(newdf.iloc[0,0])
#print(newdf.iloc[0,:])

for irow, row in newdf.iterrows():
    t1 = row["starttime"]
    t2 = row["endtime"]
    print(row)
    print(f"Starttime: {obspy.UTCDateTime(1970,1,1)+t1}, Endtime: {obspy.UTCDateTime(1970,1,1)+t2}")
    break
# Save
newdf.to_csv(fname, index=False)
