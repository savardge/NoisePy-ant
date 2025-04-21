import pandas as pd
from obspy import read_inventory, UTCDateTime
from obspy.core.inventory import Inventory, Network, Station, Channel, Site
from obspy.core.inventory.util import Equipment
import os


def read_station_data(station_location_file):
    """Read and process station location data from a CSV."""
    stainfo = pd.read_csv(station_location_file)
    stainfo.station = stainfo.station.astype(str)  # Ensure station names are strings
    return stainfo


def build_sensor(serial_number, start_date, end_date):
    """Return an Equipment object representing the sensor."""
    return Equipment(
        type="SmartSolo 3C",
        description="3C 10 Hz",
        manufacturer="Dynamic Technology (DTCC)",
        model="IGU-16",
        serial_number=serial_number,
        installation_date=start_date,
        removal_date=end_date,
    )


def build_channel(code, latitude, longitude, elevation, sampling_rate, start_date, end_date, azimuth, dip,
                  location_code):
    """Create a Channel with the specified parameters."""
    return Channel(
        code=code,
        location_code=location_code,
        latitude=latitude,
        longitude=longitude,
        elevation=elevation,
        depth=0,
        azimuth=azimuth,
        dip=dip,
        sample_rate=sampling_rate,
        start_date=start_date,
        end_date=end_date,
    )


def build_station(
        station_code,
        latitude,
        longitude,
        elevation,
        start_date,
        sensor,
        channel_prefix,
        sampling_rate,
        response,
        end_date,
        location_code,
):
    """Build a complete Station object with channels and response."""
    station = Station(
        code=station_code,
        latitude=latitude,
        longitude=longitude,
        elevation=elevation,
        creation_date=start_date,
        site=Site(name=""),
    )

    # Add channels with response and sensor
    for suffix, azimuth, dip in [("N", 0, 0.0), ("E", 90, 0.0), ("Z", 0, -90.0)]:
        channel_code = f"{channel_prefix}{suffix}"
        channel = build_channel(
            code=channel_code,
            latitude=latitude,
            longitude=longitude,
            elevation=elevation,
            sampling_rate=sampling_rate,
            start_date=start_date,
            end_date=end_date,
            azimuth=azimuth,
            dip=dip,
            location_code=location_code,
        )
        channel.response = response
        channel.sensor = sensor
        station.channels.append(channel)

    return station


def build_inventory(stainfo, response, config):
    """Build the full inventory with networks, stations, and channels."""
    inventory = Inventory(networks=[], source=config["source"])

    network = Network(
        code=config["network_code"],
        stations=[],
        description=config["network_desc"],
        start_date=config["start_date"],
        end_date=config["end_date"],
    )

    for _, row in stainfo.iterrows():
        station_code = row.station
        # Avoid duplicate station entries
        if station_code in [sta.code for sta in network.stations]:
            continue

        latitude = row.latitude
        longitude = row.longitude
        serial_number = row.serial_number
        elevation = 0 if config["zero_elevation"] else row.elevation

        # Build sensor and station
        sensor = build_sensor(serial_number, config["start_date"], config["end_date"])
        station = build_station(
            station_code,
            latitude,
            longitude,
            elevation,
            config["start_date"],
            sensor,
            config["channel_prefix"],
            config["sampling_rate"],
            response,
            config["end_date"],
            config["location_code"],
        )

        network.stations.append(station)

    inventory.networks.append(network)
    return inventory


def main():
    # Configuration parameters (inline)
    config = {
        "source": "UNIGE CDFF",
        "station_location_file": "/home/users/s/savardg/aargau_ant/text_files/station_locations_noisepy.csv",
        "resp_file": "/home/users/s/savardg/NoisePy-ant/smartsolo/stationxml/RESP.XX.NS680..SPZ.DTSOLO.5.1850.43000.76_6_fromMilliVolts",
        "network_code": "RS",  # Two-letter network code
        "network_desc": "MIGRATE RoccaNodes deployment 2023",
        "start_date": UTCDateTime(2020, 12, 4),  # Experiment start date
        "end_date": UTCDateTime(2021, 1, 6),  # Experiment end date
        "sampling_rate": 250,  # Sampling rate in Hz
        "channel_prefix": "DP",  # Prefix for SEED channel code
        "location_code": "01",  # Customizable location code for all channels
        "zero_elevation": False,  # If true, elevation is fixed to zero
        "outdir": "/home/users/s/savardg/scratch/aargau/resp",  # Directory for one file per station
        "output_file": "RoccaNodes_stations_nodes.xml",  # Consolidated StationXML file
    }

    # Load station information and response data
    stainfo = read_station_data(config["station_location_file"])
    response_inventory = read_inventory(config["resp_file"], format="RESP")
    response = response_inventory.networks[0].stations[0].channels[0].response

    # Build inventory
    inventory = build_inventory(stainfo, response, config)

    # Write to output StationXML file
    inventory.write(config["output_file"], format="stationxml", validate=True)

    # Write one file per station
    os.makedirs(config["outdir"], exist_ok=True)
    for station in inventory[0]:
        filename = os.path.join(config["outdir"], f"{config['network_code']}.{station.code}.xml")
        inventory.select(station=station.code).write(filename, format="stationxml")
        print(f"Created: {filename}")


if __name__ == "__main__":
    main()
