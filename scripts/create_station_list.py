# Create station list for NoisePy from StationXML files
# format csv: network,station,channel,latitude,longitude,elevation

import obspy
import os, glob
from obspy import read_inventory

respdir = "/home/share/cdff/aegansea/stationxml"
flist = glob.glob(os.path.join(respdir, "*.xml"))
stalst = [os.path.split(f)[1].split(".")[1] for f in flist]
print(stalst)
#stalst = ["IFIL","ILOS","IST3","ISTR","IVCR","IVGP","IVLT","IVPL","IVUG","LIBRI","MCPD","MCSR","ME12","ME15","MILZ","MPNC","MUCR","NOV","STRG"]

stationfile = "stations.csv"

inv_total = obspy.Inventory(networks=[],source="")

with open(stationfile, "w") as of:
    of.write("network,station,channel,latitude,longitude,elevation\n")
    for ista, sta in enumerate(stalst):
        respfile = glob.glob(os.path.join(respdir, "*.%s.xml" % sta))[0]
        if os.path.exists(respfile):
            inv = read_inventory(respfile)
            inv_total += inv
            net = inv[0].code
            staname = inv[0][0].code
            for channel in inv[0][0]:
                #if channel.code[0] == "B" or channel.code[1] == "N": continue
                print(f"{net},{staname}, {channel.code},{channel.latitude},{channel.longitude},{channel.elevation}\n")
                of.write(f"{net},{staname},{channel.code},{channel.latitude},{channel.longitude},{channel.elevation}\n")                
                
inv_total.write("stations.xml", format="StationXML")
