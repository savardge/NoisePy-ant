#!/bin/bash
#SBATCH --job-name=dinver
#SBATCH --partition=public-cpu
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=4G
# Per-cell Dinver (Geopsy NA, SWinvert workflow) Vs inversion over one (NET, CFG) as a Slurm
# array of strided shards. Sibling of vs_prod3.sbatch; same resume semantics (cells/ is
# skip-if-exists) and same sharding.
#
# Required --export: NET, CFG (R0gR1g|R0g), N_SHARDS (= array size). Optional: NS, NS0, NR,
# NTRIALS (SWinvert budget; defaults 50000/10000/100/3), CELL_TIMEOUT, LIMIT, CELLS,
# STAGE=assemble, OUTDIR_OVERRIDE (smoke tests MUST redirect).
#
# Parallelism: dinver is single-threaded here (-j 1); the pool runs $SLURM_CPUS_PER_TASK cells
# concurrently, each walking its 8 params x NTRIALS runs sequentially.
#
# Disk / I/O (learned the hard way on the first submission): dinver STREAMS a ~560 MB .report
# per run; with ~200 cells writing to BeeGFS at once a 3-layer run took 20 min and every cell
# hit the 6 h timeout. So the transient report goes to node-local $TMPDIR
# (--dinver-report-dir), only the ~100 KB best-100 caches touch scratch, the work dir is removed
# when the cell finishes, and --dinver-lean keeps the npz at ~50 KB. Strict minimum on scratch:
# per-cell npz + volume.
#
# Resume: --work-tag cell keeps one work dir per cell across submissions and the driver keeps
# it when a cell times out, so a resubmission resumes at (param, trial) granularity. Therefore
# NEVER over-provision this arm (no booster arrays on the same outdir). CELL_TIMEOUT is a soft
# bound (default 24 h): a cell that hits it just continues next submission.
set -eu

. "${BAMBOO_VS_DIR:-$SLURM_SUBMIT_DIR}/env_bamboo.sh"
activate_dinver
net_paths "$NET"
dinver_cfg_flags "$CFG"

STAGE="${STAGE:-grid}"
OUTDIR=${OUTDIR_OVERRIDE:-$DINVER_RUNS/$NET/$CFG}
mkdir -p "$OUTDIR"
cd "$NOISEPY/scripts/picking"

# per-network priors (VS_INVERSION.md "Third engine"): riehen 6.5 / 0.5 / 4.2
case "$NET" in
    riehen)     DMAX=6.5;  VSMAX=4.2 ;;
    aargau)     DMAX=12.0; VSMAX=5.0 ;;
    hautesorne) DMAX=14.5; VSMAX=4.8 ;;
esac

COMMON=(--engine dinver --dinver-bin "$DINVER_BIN"
        --production "$PROD" --config "$CONFIG_YAML" --net "$NET"
        --outdir "$OUTDIR" --wavesets "$WS"
        --period-ranges "$INPUTS/period_ranges_DECISIONS_v1.csv"
        --criterion none --noise-regime free --phase-tmin 0
        --depth-max "$DMAX" --vs-min 0.5 --vs-max "$VSMAX"
        --dinver-lean --dinver-jobs 1 --work-tag cell
        --dinver-report-dir "${TMPDIR:-/tmp}")

if [ "$STAGE" = assemble ]; then
    srun "$BH_PY" grid_vs_inversion.py "${COMMON[@]}" $DEXTRA --assemble-only
    exit 0
fi

I=$(( ${SLURM_ARRAY_TASK_ID:-0} ))
srun "$BH_PY" grid_vs_inversion.py "${COMMON[@]}" $DEXTRA \
    --dinver-ns "${NS:-50000}" --dinver-ns0 "${NS0:-10000}" --dinver-nr "${NR:-100}" \
    --dinver-ntrials "${NTRIALS:-3}" \
    --n-workers "${SLURM_CPUS_PER_TASK}" \
    --cell-timeout "${CELL_TIMEOUT:-86400}" \
    ${LIMIT:+--limit "$LIMIT"} ${CELLS:+--cells "$CELLS"} \
    --shard "$I/${N_SHARDS}" --no-assemble
