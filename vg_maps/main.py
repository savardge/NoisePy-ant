import logging
import yaml
import numpy as np
import pandas as pd
from io.stations import get_stations_extent
from inversion.kernel import get_data_kernel
from inversion.tv_inversion import TV_inversion
from plotting.plot_map import plot_map
import os


def setup_logger():
    """Set up the logger for the application."""
    logger = logging.getLogger("main_logger")
    logger.setLevel(logging.INFO)

    # Create console handler to output logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter for log messages
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)
    return logger


def load_config(file_path):
    """Load configuration from a YAML file."""
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def filter_picks(data, period, std_percent_threshold, count_threshold, station_ids, logger):
    """Filter data picks based on given criteria and log summary details."""
    data["inst_period"] = data["inst_period"].round(2)
    filtered_data = data[
        (data["inst_period"] == period)
        & (data["std_percent"] <= std_percent_threshold)
        & (data["count"] >= count_threshold)
        ]
    filtered_data = filtered_data[
        filtered_data["stasrc"].isin(station_ids) & filtered_data["starcv"].isin(station_ids)
        ]
    logger.info(f"Filtered data: {len(filtered_data)} picks remain after filtering.")
    return filtered_data.sort_values("station_pair")


def create_grids(grid_config, logger):
    """Create x and y grids based on grid configuration."""
    min_lat = grid_config["min_lat"]
    max_lat = grid_config["max_lat"]
    min_lon = grid_config["min_lon"]
    max_lon = grid_config["max_lon"]
    dx_km = grid_config["dx_km"]
    dy_km = grid_config["dy_km"]

    logger.info(f"Creating grids with latitudes {min_lat} to {max_lat}, longitudes {min_lon} to {max_lon}, "
                f"dx={dx_km}, dy={dy_km}")

    x_grid = np.arange(0, max_lon - min_lon + dx_km, dx_km)
    y_grid = np.arange(0, max_lat - min_lat + dy_km, dy_km)
    return x_grid, y_grid


def main():
    # Setup logger
    logger = setup_logger()
    logger.info("Starting main application...")

    # Load YAML configuration
    logger.info("Loading configuration from YAML file.")
    config = load_config("config/example.yaml")

    # Extract values from the updated configuration
    grid = config["grid"]
    inversion = config["inversion"]
    thresholds = config["thresholds"]
    files = config["files"]

    # Unpacking values
    period = inversion["period"]
    sigma = inversion["sigma"]
    LC = inversion["LC"]

    max_std_percent = thresholds["max_std_percent"]
    min_count = thresholds["min_count"]

    station_file = files["station_file"]
    pick_file = files["pick_file"]
    background_map = files["background_map"]
    output_dir = files["output_dir"]

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    # Dynamically create the output map filename with a suffix showing the period value
    output_map = os.path.join(output_dir, f"velocity_map_period_{period}.png")

    # Load station data and determine extent
    logger.info("Loading station data and determining extent.")
    grid_origin, extent, stations = get_stations_extent(
        [
            grid["min_lat"],
            grid["max_lat"],
            grid["min_lon"],
            grid["max_lon"],
        ],
        station_file,
    )
    logger.info(f"Loaded stations. Grid origin: {grid_origin}, Extent: {extent}")

    # Load and filter pick data
    logger.info("Loading and filtering pick data.")
    data = pd.read_csv(pick_file)
    logger.info(f"Loaded pick data with {len(data)} rows.")
    station_ids = set(stations["id"])
    data = filter_picks(data, period, max_std_percent, min_count, station_ids, logger)
    logger.info(f"Number of picks/ray paths after filtering: {len(data)}")

    # Setup data vector
    V_dat = data["group_velocity"].values
    TAU = data["distance"].values / V_dat

    # Create grids
    x_grid, y_grid = create_grids(grid, logger)

    # Compute kernel and setup prior velocity model
    logger.info("Computing kernel and setting up prior velocity model.")
    G, mask = get_data_kernel(x_grid, y_grid, data, stations)
    v_moy = np.mean(V_dat)
    v_prior = v_moy * np.ones(len(x_grid) * len(y_grid))
    logger.info("Kernel computation complete. Starting inversion.")

    # Perform TV inversion and plot
    V_map, stats = TV_inversion(x_grid, y_grid, sigma, LC, TAU, v_prior, G)
    logger.info("TV inversion complete. Plotting results.")
    plot_map(
        V_map,
        V_dat,
        stats,
        mask,
        stations,
        x_grid,
        y_grid,
        background_map,
        fignum=1,
        period=period,
        sigma=sigma,
        LC=LC,
        output_file=output_map,
    )

    logger.info(f"Application completed successfully. Map saved at: {output_map}")



if __name__ == "__main__":
    main()
