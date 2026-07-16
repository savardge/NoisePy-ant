#!/bin/bash
# ---------------------------------------------------------------------------
# Grid Vs(z) inversion on the cluster, at the production convergence standard
# (10 chains x 200k burn-in x 100k main). SLURM array fan-out: each array task
# inverts a DISJOINT slice tasks[i::N_SHARDS] of the cell list, running the
# validated per-cell BayHunter runner (serial chains, BLAS pinned to 1) with a
# local worker pool of size cpus-per-task. Cells are parallelised across the
# array; after all shards finish, one --assemble-only step stitches the volume.
#
# This reuses the macOS-validated run_bayhunter_cell.py unchanged. It does NOT
# use BayHunter's native mp_inversion (parallel chains within a cell), which is
# documented to deadlock; array-sharding gives the same wall-time win by
# parallelising over cells. If you specifically want intra-cell mp_inversion,
# see README_vs_grid_cluster.md.
#
# >>> EDIT every line marked  # >>> EDIT  before submitting. <<<
# ---------------------------------------------------------------------------
#SBATCH --job-name=vsgrid
#SBATCH --partition=shared-cpu,public-cpu,public-bigmem,shared-bigmem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # local workers per shard = cells done in parallel
#SBATCH --mem-per-cpu=4G
#SBATCH --time=11:00:00
#SBATCH --output=outslurm/vsgrid-%A_%a.out
#SBATCH --array=0-29               # N_SHARDS-1 ; keep in sync with N_SHARDS below

set -euo pipefail
N_SHARDS=30                        # must equal (array max + 1)
NET=riehen                         # >>> EDIT: riehen | aargau  (submit once per net)

# ---- cluster paths / environment (all CLUSTER-side, NOT the macOS /Users paths) ----
REPO=/home/users/s/savard/Codes/NoisePy-ant            # >>> EDIT: repo on the cluster
PROJ=/home/users/s/savard/Codes/extract_higher_modes/Projects  # >>> EDIT: staged Projects data
BH_PY=/home/users/s/savard/.conda/envs/bayhunter/bin/python     # >>> EDIT: bayhunter env python
DRIVER_PY=/home/users/s/savard/.conda/envs/bayesbay_dev/bin/python  # >>> EDIT: driver env python
# >>> EDIT: load whatever your cluster needs (module load / conda activate / gfortran on PATH)
# module load GCC/12  Anaconda3
# source activate bayesbay_dev

# ---- per-network wiring (dx / LC are baked into the far-field config + production root) ----
if [ "$NET" = "riehen" ]; then
    PROD=$PROJ/riehen/tomo/swtomotv-output-200m-farfield/production
    CONFIG=$PROJ/riehen/tomo/riehen_swtomotv_200m_farfield.yaml
    OUTDIR=$PROJ/riehen/tomo/vs_inversion/grid_physical_200m_farfield
else
    PROD=$PROJ/aargau/tomo/swtomotv-output-500m-farfield-v2/production
    CONFIG=$PROJ/aargau/tomo/aargau_swtomotv_500m_farfield_v2.yaml
    OUTDIR=$PROJ/aargau/tomo/vs_inversion/grid_physical_500m_farfield_v2
fi

cd "$REPO/scripts/picking"
export PYTHONPATH="$REPO"

srun "$DRIVER_PY" grid_vs_inversion.py \
    --production "$PROD" --config "$CONFIG" --outdir "$OUTDIR" \
    --net "$NET" --criterion physical \
    --n-chains 10 --iter-burnin 200000 --iter-main 100000 \
    --cell-timeout 5400 \
    --n-workers "${SLURM_CPUS_PER_TASK}" \
    --shard "${SLURM_ARRAY_TASK_ID}/${N_SHARDS}" \
    --bayhunter-python "$BH_PY" \
    --bayhunter-runner run_bayhunter_cell.py
