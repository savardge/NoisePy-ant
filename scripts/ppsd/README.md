# PPSD Station Processing Script

This script computes the Probabilistic Power Spectral Density (PPSD) for seismic data from a specified station and channel. It uses the ObsPy library and reads input data from MiniSEED files, guided by configuration parameters provided in a YAML file.

---

## 🔧 Requirements
- Python 3.x
- ObsPy
- PyYAML

Install dependencies with:
```bash
pip install obspy pyyaml
```

---

## 🚀 Usage

### CLI
```bash
python ppsd_station.py [station] [channel] [config_file.yaml]
```

Example:
```bash
python ppsd_station.py 3006977 DPZ ppsd_params.yaml
```

### Required YAML Parameters
```yaml
# Input directory containing station/channel subfolders
datadir: "/path/to/miniseed/files"

# Directory where plots will be saved
figdir: "/path/to/output/figures"

# Directory for caching the computed PPSD
npzdir: "/path/to/output/ppsd_npz"

# Full path to the StationXML metadata
stationxml_file: "/path/to/station.xml"

# Time window for spectrogram plot
starttime: "2023-01-01T00:00:00"
endtime: "2023-12-31T23:59:59"

# Period range for plotting
minT: 0.02
maxT: 10.0

# Period bins for temporal evolution plot
period_bins: [0.1, 0.2, 1.0]

# File search pattern template (optional)
file_pattern: "{datadir}/*/{station}/{channel}/*.mseed"

# Plotting and overwrite options
makefig: true
overwrite: false
```

---

## ⚙️ Running with SLURM

### Batch Script Example (`run_ppsd.slurm`):
```bash
#!/bin/bash
#SBATCH --job-name=ppsd_riehen
#SBATCH --partition=shared-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=1:00:00
#SBATCH --output="outslurm/slurm-%A_%a-%x.out"
#SBATCH --array=2-198
#SBATCH --mail-type=START,END
#SBATCH --mail-user=your_email@example.com

set -euo pipefail
mkdir -p outslurm

fname="station_locations_riehen.csv"
configfile="ppsd_params.yaml"

station=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$fname" | awk -F, '{print $2}')
[[ -n "$station" ]] || { echo "No station found for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"; exit 1; }

for channel in DPZ DPN DPE; do
    python ppsd_station.py "$station" "$channel" "$configfile"
done
```

Make sure the `station_locations_riehen.csv` file contains station names in the second column.

---

## 📂 Output
- `.npz` files containing cached PPSD data.
- `.png` files for:
  - Standard PPSD plot
  - Temporal evolution at specified periods
  - Spectrogram view of PSD evolution

---

## 📫 Support
For help or suggestions, please contact:
**Geneviève Savard** - genevieve.savard@unige.ch

