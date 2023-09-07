import os
import glob
import pandas as pd

stainfo = pd.read_csv("/srv/beegfs/scratch/shares/cdff/MIGRATE/Tuscany_July2023/fix-headers/stations_RoccaNodes.csv")
datadir = "/srv/beegfs/scratch/shares/cdff/MIGRATE/Tuscany_July2023/miniseed"
network = "RN"

for SN, station in zip(stainfo.serial_number.astype(str), stainfo.station.astype(str)):
    print(station,SN)
    olddir = os.path.join(datadir, SN)    
    if os.path.exists(olddir):
        newdir = os.path.join(datadir, station)        
        
        # Change file names 
        # Assumes SmartSolo export file pattern like 453007143.11.2023.06.26.00.00.00.000.E.miniseed (starts with serial number)
        flist = glob.glob(os.path.join(olddir, f"{SN}.*"))
        for oldfile in flist:
            if ".Z." in oldfile:
                comp = "DPZ"
                newfile = oldfile.replace(SN,f"{network}.{station}..{comp}.{SN}")
                os.rename(oldfile,newfile)
            elif ".N." in oldfile:
                comp = "DPN"
                newfile = oldfile.replace(SN,f"{network}.{station}..{comp}.{SN}")
                os.rename(oldfile,newfile)
            elif ".E." in oldfile:
                comp = "DPE"
                newfile = oldfile.replace(SN,f"{network}.{station}..{comp}.{SN}")
                os.rename(oldfile,newfile)            
            
        # Change directory name
        os.rename(olddir,newdir)
