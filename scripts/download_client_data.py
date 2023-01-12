import obspy
from obspy.clients.fdsn.mass_downloader import CircularDomain, \
    Restrictions, MassDownloader
import os

# Circular domain around volcano for which we want data. This will download all data between
# 70 and 90 degrees distance from the epicenter. This module also offers
# rectangular and global domains. More complex domains can be defined by
# inheriting from the Domain class.
domain = CircularDomain(latitude=40.29, longitude=25.40, minradius=0.0, maxradius=1.2)  #(0.1 degree is ~ 11 km)

# Define function for naming convention of mseed
ROOT = "/home/share/cdff/aegansea"
def get_mseed_storage(network, station, location, channel, starttime,
                      endtime):
    filepath = os.path.join(ROOT, network, station, channel, "%s.%s.%s.%s.%s.mseed" % (network, station,
                                                     location, channel, starttime.strftime("%Y.%j.%H.%M.%S.%f")[:-3]))
    # Returning True means that neither the data nor the StationXML file
    # will be downloaded.
    if os.path.exists(filepath):
        return True
    # If a string is returned the file will be saved in that location.
    return filepath

restrictions = Restrictions(
    # Get data for specific period.
    starttime=obspy.UTCDateTime(2012, 1, 1, 0, 0, 0),
    endtime=obspy.UTCDateTime(2015, 5, 1, 0, 0, 0),
    # Chunk it to have one file per day.
    chunklength_in_sec=86400,
    # Considering the enormous amount of data associated with continuous
    # requests, you might want to limit the data based on SEED identifiers.
    # If the location code is specified, the location priority list is not
    # used; the same is true for the channel argument and priority list.
    network="*", station="KVLA", location="*", channel="HN*",
    channel_priorities=('BH[ZNE]','HH[ZNE]'),
    location_priorities=('','20','00'),
    # The typical use case for such a data set are noise correlations where
    # gaps are dealt with at a later stage.
    reject_channels_with_gaps=False,
    # Same is true with the minimum length. All data might be useful.
    minimum_length=0.0,
    # Guard against the same station having different names.
    minimum_interstation_distance_in_m=100.0)

mdl = MassDownloader(providers=["KOERI","RESIF","NOA","ORFEUS","IRIS"])
mdl.download(domain, restrictions, threads_per_client=8, mseed_storage=get_mseed_storage, stationxml_storage="/home/share/cdff/aegansea/stationxml")
#mdl.download(domain, restrictions, chunk_size_in_mb=2000, threads_per_client=3, mseed_storage=get_mseed_storage, stationxml_storage="stations")

