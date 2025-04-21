#!/bin/bash
#SBATCH --job-name=ppsd_riehen
#SBATCH --partition=shared-cpu,public-cpu,public-bigmem,shared-bigmem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=1:00:00
#SBATCH --output="outslurm/slurm-%A_%a-%x.out"
#SBATCH --array=2-198
#SBATCH --mail-type=START,END
#SBATCH --mail-user=genevieve.savard@unige.ch

#set -euo pipefail
mkdir -p outslurm

fname="station_locations_riehen.csv"
configfile="ppsd_params.yaml"

# Check required files
[[ -f "$fname" ]] || { echo "File not found: $fname"; exit 1; }
[[ -f "$configfile" ]] || { echo "File not found: $configfile"; exit 1; }

# Get station name from CSV
station=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$fname" | awk -F, '{print $2}')
[[ -n "$station" ]] || { echo "No station for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"; exit 1; }

echo "Processing station: $station"

# Loop over channels if needed
for channel in DPZ; do
    python ./ppsd_station.py "$station" "$channel" "$configfile"
done
