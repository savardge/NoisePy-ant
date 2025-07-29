#!/bin/bash
#!/bin/bash
# -----------------------------------------------------------------------------
# SLURM Array Job Script to Update Miniseed Headers with msmod
#
# Author: Geneviève Savard
# Date: 2025-04-19
# Description:
#   This script runs msmod to modify metadata headers in miniseed files,
#   using a SLURM array job. Each task processes one station directory,
#   with directories listed in an input file. Metadata is parsed from file names.
#
# Usage:
#   sbatch this_script.sh
#
# Dependencies:
#   - msmod v1.2
#   - station_dirs.txt with full directory paths (one per line)
#
# -----------------------------------------------------------------------------


#SBATCH --array=1-189                   # One task per station directory (189 total)
#SBATCH --partition=shared-cpu         # Use shared CPU partition
#SBATCH --time=12:00:00                # Max wall time per task
#SBATCH --ntasks=1                     # One task per job
#SBATCH --cpus-per-task=1              # Single-threaded task
#SBATCH --mem=3G                       # Memory allocation per task
#SBATCH --output="outslurm/slurm-%A_%a.out"  # SLURM output log (per task)

# ------------------------
# User-defined inputs
# ------------------------

INPUTLIST="/path/to/station_dirs.txt"            # Text file with full paths to directories containing miniseed files (one per line)
msmod="/home/share/cdff/msmod-1.2/msmod"         # Full path to the msmod executable

# ------------------------
# Validate array index
# ------------------------

total_dirs=$(wc -l < "$INPUTLIST")
if [ "$SLURM_ARRAY_TASK_ID" -gt "$total_dirs" ]; then
    echo "[ERROR] SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds number of entries in $INPUTLIST ($total_dirs)"
    exit 1
fi

# ------------------------
# Log setup
# ------------------------

mkdir -p msmod-output
logfile="msmod-output/msmod_${SLURM_ARRAY_TASK_ID}.log"

{
    echo "=========="
    echo "[INFO] SLURM Job ID: $SLURM_JOB_ID | Array Task ID: $SLURM_ARRAY_TASK_ID"
    echo "[START] $(date)"

    # ------------------------
    # Get data directory to process
    # ------------------------

    datadir=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$INPUTLIST")

    if [ ! -d "$datadir" ]; then
        echo "[ERROR] Directory does not exist: $datadir"
        exit 1
    fi

    echo "[INFO] Processing directory: $datadir"

    # ------------------------
    # Extract metadata from first miniseed file
    # ------------------------

    firstfile=$(ls "$datadir"/*.miniseed 2>/dev/null | head -1)
    if [ ! -f "$firstfile" ]; then
        echo "[ERROR] No .miniseed files found in: $datadir"
        exit 1
    fi

    NET=$(basename "$firstfile" | awk -F. '{print $1}')
    STA=$(basename "$firstfile" | awk -F. '{print $2}')
    LOC=$(basename "$firstfile" | awk -F. '{print $3}')
    CHAN=$(basename "$firstfile" | awk -F. '{print $4}')

    if [[ -z "$NET" || -z "$STA" || -z "$CHAN" ]]; then
        echo "[ERROR] Failed to extract metadata from filename: $firstfile"
        exit 1
    fi

    echo "[INFO] Extracted metadata -> NET=$NET, STA=$STA, LOC=$LOC, CHAN=$CHAN"

    # ------------------------
    # Run msmod to update headers
    # ------------------------

    echo "[INFO] Running msmod..."
    "$msmod" "$datadir"/*.miniseed -i --net "$NET" --sta "$STA" --chan "$CHAN"

    echo "[INFO] Finished processing."
    echo "[END] $(date)"
} >> "$logfile" 2>&1

