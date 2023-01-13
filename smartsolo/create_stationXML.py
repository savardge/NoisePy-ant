from obspy.core.inventory import Inventory, Network, Station, Channel, Site
from obspy import read_inventory, UTCDateTime
from obspy.core.inventory.util import Equipment
import pandas as pd
import os

##### INPUT ########
source = "UNIGE CDFF"

# Paths
outdir = "/home/users/s/savardg/scratch/aargau/resp" #"/home/users/s/savardg/scratch/riehen/resp" #
output_file = "aargau_stations_nodes.xml" #"riehen_stations_nodes.xml" # 
# resp_file = "/home/users/s/savardg/smartsolo/sensor_response_info/RESP.XX.NS680..SPZ.DTSOLO.5.1850.43000.76_6_toV" # Input RESP file
# stainfo = pd.read_csv("/home/users/s/savardg/riehen/station_locations_riehen.csv") # Needs columns station,latitude,longitude,(elevation)
resp_file = "/home/users/s/savardg/smartsolo/sensor_response_info/RESP.XX.NS680..SPZ.DTSOLO.5.1850.43000.76_6_toV_36dB" # Input RESP file
stainfo = pd.read_csv("/home/users/s/savardg/aargau_ant/text_files/station_locations_noisepy.csv")
stainfo.station = stainfo.station.astype(str)
stations = list(stainfo['station'])

# Network parameters
network_code = "AA" #"RI" #
network_desc = "Aargau nodal ANT: Dec 2020" #"Riehen nodal ANT: Sep 2022" #
start_date = UTCDateTime(2020, 12, 4) #UTCDateTime(2022, 9, 3) #
end_date = UTCDateTime(2021, 1, 6) #UTCDateTime(2022, 9, 24) #
sampling_rate = 250
channel_prefix = "DP"  # First 2 letters of the SEED channel code. See https://ds.iris.edu/ds/nodes/dmc/data/formats/seed-channel-naming/
zero_elevation = False # To fix elevation at zero. Otherwise need "elevation" column in stainfo

######################

# Get response
dum = read_inventory(resp_file, format='RESP')
response = dum.networks[0].stations[0].channels[0].response

# We'll first create all the various objects. These strongly follow the
# hierarchy of StationXML files.
inv = Inventory(
    # We'll add networks later.
    networks=[],
    # The source should be the id whoever create the file.
    source=source)

net = Network(
    # This is the network code according to the SEED standard.
    code=network_code,
    # A list of stations. We'll add one later.
    stations=[],
    description=network_desc,
    #description="March-May deployment",
    # Start-and end dates are optional.
    start_date=start_date,
    end_date=end_date
)

for station in stations:    
    if station in list(set([sta.code for sta in net.stations])): continue # skip duplicate        
    latitude = stainfo.loc[stainfo['station'] == station, 'latitude'].values[0]
    longitude = stainfo.loc[stainfo['station'] == station, 'longitude'].values[0]
    serial_number = stainfo.loc[stainfo['station'] == station, 'serial_number'].values[0]
    if zero_elevation:
        elevation = 0
    else:
        elevation = stainfo.loc[stainfo['station'] == station, 'elevation'].values[0]
    # create sensor object
    sensor = Equipment(type='SmartSolo 3C',
                   description='3C 10 Hz',
                   manufacturer='Dynamic Technology (DTCC)',
                   model='IGU-16',
                   serial_number=serial_number,
                   installation_date=start_date,
                   removal_date=end_date
                   )
    sta = Station(
        # This is the station code according to the SEED standard.
        code=station,
        latitude=latitude,
        longitude=longitude,
        elevation=elevation,
        creation_date=start_date,
        site=Site(name="")
    )

    chaN = Channel(
        # This is the channel code according to the SEED standard.
        code=channel_prefix +"N",
        # This is the location code according to the SEED standard.
        location_code="",
        # Note that these coordinates can differ from the station coordinates.
        latitude=latitude,
        longitude=longitude,
        elevation=elevation,
        depth=0,
        azimuth=0,
        dip=0.0,
        sample_rate=sampling_rate,
        start_date=start_date,
        end_date=end_date
    )
    chaE = Channel(
        # This is the channel code according to the SEED standard.
        code=channel_prefix +"E",
        # This is the location code according to the SEED standard.
        location_code="",
        # Note that these coordinates can differ from the station coordinates.
        latitude=latitude,
        longitude=longitude,
        elevation=elevation,
        depth=0,
        azimuth=90,
        dip=0.0,
        sample_rate=sampling_rate,
        start_date=start_date,
        end_date=end_date
    )
    chaZ = Channel(
        # This is the channel code according to the SEED standard.
        code=channel_prefix +"Z",
        # This is the location code according to the SEED standard.
        location_code="",
        # Note that these coordinates can differ from the station coordinates.
        latitude=latitude,
        longitude=longitude,
        elevation=elevation,
        depth=0,
        azimuth=0.0,
        dip=-90.0,
        sample_rate=sampling_rate,
        start_date=start_date,
        end_date=end_date
    )


    # Now tie it all together.
    chaN.response = response
    chaE.response = response
    chaZ.response = response
    chaN.sensor = sensor
    chaE.sensor = sensor
    chaZ.sensor = sensor
    sta.channels.append(chaN)
    sta.channels.append(chaE)
    sta.channels.append(chaZ)
    net.stations.append(sta)

inv.networks.append(net)

# And finally write it to a StationXML file. We also force a validation against
# the StationXML schema to ensure it produces a valid StationXML file.
#
# Note that it is also possible to serialize to any of the other inventory
# output formats ObsPy supports.
inv.write(output_file, format="stationxml", validate=True)

# Write 1 file per station
if not os.path.exists(outdir):
    os.mkdir(outdir)
for sta in inv[0]:
    fname = os.path.join(outdir, f"{network_code}.{sta.code}.xml")
    inv.select(station=sta.code).write(fname, format="StationXML")
    print(fname)
    