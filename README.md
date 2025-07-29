# NoisePy-ant

!!! This repository is a work in progress... !!!

Collection of codes to run NoisePy to get stacked cross-correlations, do FTAN and pick dispersion curves automatically and apply ambient noise tomography. Includes some codes specific to working with SmartSolo nodal geophones and slurm job scheduling scripts to deploy codes on a HPC cluster.

Rough structure (to be improved...):
* `noisepy`: Python module containing functions needed to go from raw miniseed/sac data to dispersion curve picks. Many functions taken from the old version of NoisePy, but re-arranged and sometimes modified.
* `matlab`: Matlab codes to do group velocity map inversion and Vs depth inversion. See `scripts/postprocess_stacks/export_matlab.py` for preparing input files from the NoisePy outputs.
    * `vg_maps_inversion`: Group velocity map inversion (linearized inversion of Tarantola & Valette 1982)
    * `vs_depth_inversion`: Vs depth inversion for each 2D grid location using a set of group velocity maps, using the Neighborhood algorithm (Wathelet 2008).
    * `synthetic_tests_to_design_network`: Given a list of station coordinates, do a checkboard or spike resolution test (straight ray path assumption) for different periods. Useful for testing network configuration.
    * `checkerboard_resolution_tests`: Checkerboard/spike tests for real picked data.
    * `make_map_background`: utility script to create an image to use as background for plotting in Matlab
* `scripts`:
    * `raw2stack`: Follows old version of NoisePy: scripts S0B (miniseed -> pyasdf), S1 (raw data -> cross-correlations) and S2 (stacking).
    * `picking`: Do FTAN and dispersion curve picking.
    * `postprocess_stacks`:
        * `beamforming.py` Beamforming from stacked cross-correlation gather (Bowden 2021)
        * `compare_stacking` Compare stacking methods
        * `extract_ncfs.py/extract_ncts.py` Gather the stacked CCFs from the pyasdf H5 files into more convenient Numpy arrays
        * `export_matlab.py` Export dispersion curve picks and other to matlab inputs for inversion
        * `write_sac.py` Export pyasdf stacked CCF to SAC (taken from old NoisePy)
    * `QC`: PPSD analysis
    * `FK`: FK decomposition and polarisation analysis (Takagi, 2014)
    * `dvv`: dv/v monitoring scripts
    * `other`: download data from data centers, etc.
* `param_files`: Examples of parameter input YAML files for the `raw2stack` scripts and other scripts.
* `dispvelmaps`: module for group velocity map inversion in Python [WIP]
    * `dispvelmaps/prep_data.py`: prepares the input files in Matlab and Numpy format for group velocity map inversion.
* `HV`: Scripts for calculating ellipticity from cross-correlations (approach of Lin et al. 2014, Berg et al. 2018)
* `smartsolo`: scripts for handling SmartSolo nodal data

---
### Data preparation 

1. Create a text file of station locations (comma-separated, .csv) file containing station names and geographic locations. 
    * The CSV file must contain the following columns: `network, station, channel, latitude, longitude, [elevation], [serial number]`
    * **SmartSolo:** To get station coordinates from the SmartSolo log files, run `smartsolo/extract_coordinates_from_log.py`, which reads the DigiSolo.LOG files and extract the mean values of the GPS data logged.
2. Ensure the miniseed/sac files have the same headers as the station file.
    * **SmartSolo**: By default, when exporting the data, SmartSolo fill in the last digits of the serial number as the station name. If you want to use other station name, you must correct the headers and rename the files and directory structure. To do this, use the scripts in `smartsolo/fix-headers`. These scripts also make the file names start with the usual structure "NETWORK.STATION.LOCATION.CHANNEL"
3. Do some quality control to determine stations that were disturbed during the experiment or that have poor quality data.
    * **SmartSolo:** For SmartSolo nodes, run `smartsolo/QC/extract_QC_stats.py`. This creates a plot and two .csv files, that contain in a table format the GPS info and the positional angles logged every ~8 minutes (if Cycle On GPS mode). To identify bad stations, either inspect each plot or apply a threshold to the standard deviation of the ecompass North, tilt, roll or pitch angles measurements. (`smartsolo/QC/get_bad_stations.py`)
    * Calculate the PPSD distribution for each station using `scripts/QC/ppsd_1sta.py`
4. Gather instrument response info and create a directory with one StationXML per station including the instrument response.
    * **SmartSolo:** Run `smartsolo/create_stationXML.py` to create these files given the RESP file template and the station location table in csv format.



---
### From raw miniseed files to stacked cross-correlations
1. Create station location file. 
    - The CSV file must contain the following columns: `network, station, channel, latitude, longitude, [elevation], [serial number]`
    - Use `scripts/raw2stack/create_station_list.py` to create it from a merged StationXML file.
2. Create a text file of all raw data miniseed/sac files (`allfiles_time.csv`) using `scripts/raw2stack/create_allfiles_time_csv.py`
3. `S0` scripts: Run conversion from miniseed/sac to pyasdf .h5 files.
4. `S1` scripts: Calculate cross-correlations in frequency domain for given sub-window lengths.
5. `S2` scripts: Stack the sub-window cross-correlations into a total stack and sub-stacks, if needed.

### Postprocess stacks and plotting
1. To merge the pyasdf files into numpy arrays for easier manipulation, use `scripts/postprocess_stacks/extract_ncts.py`
2. Noise directivity analysis/ beamforming with script `scripts/postprocess_stacks/beamform.py` (requires the numpy files created with the script above)
3. See various plotting functions in `noisepy/plotting.py` and `noisepy/binstack.py` (binned stack gathers, FK decomposition, plotting the full tensor, etc)
4. Make plots of (ZR+RZ)/2 and (ZR-RZ)/2 with scripts in `scripts/FK/plot_takagi.py`

### FTAN and picking
Run scripts in `scripts/picking`. 
1. Run `dispersion_curves_V2.py` for each station pair (each stack H5 file), use the slurm script for efficient job scheduling (`dispersion.slurm`).
2. Merge the output files into one big data table (csv file) for analysis with Pandas using `step1_merge_picks.py`
3. Create histograms of picks given some filtering threshold with `step2_pick_histograms.py`

### Group velocity maps and 3D Vs inversion [MATLAB]

1. Filter picks for group velocity maps, and make the various input files for the Matlab scripts (grid definition etc.) with `matlab/noisepy_to_matlab/prep_data_noisepy2matlab.py`
2. Create the map background for plotting `matlab/make_map_background`
3. Vg map inversion with the scripts in module `matlab/vg_maps_inversion`
4. Depth inversion with module `matlab/vs_depth_inversion`

### 

