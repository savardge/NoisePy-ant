### README for `step1_rename_files.py` and `step2_fix-headers.sh`

---

## Overview

This repository is aimed at processing and managing seismic data files stored in MiniSEED format. It consists of two main scripts:

1. **`step1_rename_files.py`**: Handles the renaming of MiniSEED data files based on new codes for station, network, channel, location.
2. **`step2_fix-headers.sh`**: Updates MiniSEED headers using EarthScope's `msmod` tool via a SLURM array job.

These two scripts are designed to work sequentially to ensure consistent naming and metadata accuracy for seismic data files.

---

## Prerequisites

### Dependencies:

1. **Python Version**: Compatible with Python 3.x (for `step1_rename_files.py`).
   - **Packages**: Ensure `pandas` is installed (used in the Python script). Install with:
```shell script
pip install pandas
```

2. **System Requirements**: 
   - Install the `msmod` tool (v1.2 or higher) for header modifications in MiniSEED files: https://github.com/EarthScope/msmod# 

3. **SLURM Cluster**: The `step2_fix-headers.sh` script requires a SLURM-based computing environment for job scheduling.

---

## Scripts Details

### 1. `step1_rename_files.py`

This Python script is used to **rename MiniSEED data files** in a directory based on metadata or descriptive information extracted from the file.

#### Key Features:
- Extracts key metadata such as `network`, `station`, `location`, and `datatype` from file names or associated info.
- Renames files according to a configurable naming schema.
- Handles exceptions and keeps a log of failed entries that couldn't be processed.
- Performs checks with a `dry_run` mode to validate renaming without applying changes.

#### Input & Outputs:
- **Input**: Directory of existing MiniSEED files and configuration file (if required).
- **Output**: Renamed files in a specified directory, and logging of failures or processed files.

---

### 2. `step2_fix-headers.sh`

A **SLURM Array Job script** to update MiniSEED file headers with updated metadata using the `msmod` tool.

#### Key Features:
- Processes seismic data in batches, where each SLURM task handles a single station directory.
- Extracts metadata (network, station, location, channel) from MiniSEED file names.
- Uses `msmod` to apply the metadata to headers of all MiniSEED files in the directory.
- Provides detailed logs for each task.

#### Input & Outputs:
- **Input**: A text file (`station_dirs.txt`) containing paths to station directories, one per line.
- **Output**: Updated MiniSEED files, along with logs stored under the `msmod-output` directory.

#### Usage:
1. Update the input list path (`INPUTLIST`) and the `msmod` executable path in the script.
2. Submit the script for execution in a SLURM cluster:
```shell script
sbatch step2_fix-headers.sh
```

---

## Workflow Summary

1. **Run `step1_rename_files.py`**:
   - Use this script to properly rename MiniSEED files, ensuring consistency across file names and directories.

2. **Run `step2_fix-headers.sh`**:
   - Once files are renamed, submit the SLURM script to adjust file headers using `msmod`.

---

## Example

### Renaming Files:

```shell script
# Rename MiniSEED files in a directory
python step1_rename_files.py --datadir /path/to/data --log_dir ./logs --dry_run
```

### Fixing Headers:

```shell script
# Run SLURM job to update headers
sbatch step2_fix-headers.sh
```

---

## Contributing

Feel free to submit issues or pull requests for improvements or bug fixes.

**Author**: Geneviève Savard  
**Last Updated**: April 19, 2025

---

### Notes:
- Ensure paths and dependencies are correctly configured before running the scripts.
- Logs from both scripts provide detailed insights into processing results and errors.