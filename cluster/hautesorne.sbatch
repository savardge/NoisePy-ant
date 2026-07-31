#!/bin/bash
# Haute-Sorne -- ts-PWS production chain on yggdrasil.  Submit from THIS directory.
#
#   STAGE=build     job array, one shard of the ts-PWS stack build per task
#   STAGE=jackknife job array, per-pair substack sigma (feeds Tarantola-Valette C_d)
#   STAGE=finalize  single job: pick -> QC -> export -> attach sigma -> figures
#
# Submit (see README.md for how to pick the array width and throttle):
#   mkdir -p logs
#   B=$(sbatch --parsable --array=0-199%50 hautesorne.sbatch)
#   J=$(sbatch --parsable --array=0-199%50 --export=ALL,STAGE=jackknife \
#              --dependency=afterok:$B hautesorne.sbatch)
#   sbatch --dependency=afterok:$J --export=ALL,STAGE=finalize \
#          --partition=public-bigmem --time=1-00:00:00 --cpus-per-task=32 --mem=384G \
#          hautesorne.sbatch
#
# The %25 suffix caps CONCURRENT tasks -- lower it to fit your association's CPU limit
# (preflight.sh prints the limit). Every stage is skip-if-exists, so a task that dies on a
# time limit is fixed by resubmitting the same array; nothing is recomputed or corrupted.
#SBATCH --job-name=tspws_hautesorne
#SBATCH --partition=shared-cpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/hautesorne_%A_%a.out
#SBATCH --error=logs/hautesorne_%A_%a.err
set -eu

NET=hautesorne
STAGE="${STAGE:-build}"
. "${CLUSTER_DIR:-$SLURM_SUBMIT_DIR}/env.sh"
activate_env

# Shard from the array bounds (works with 0-49, 1-50 and any %throttle).
I=$(( ${SLURM_ARRAY_TASK_ID:-0} - ${SLURM_ARRAY_TASK_MIN:-0} ))
N=$(( ${SLURM_ARRAY_TASK_MAX:-0} - ${SLURM_ARRAY_TASK_MIN:-0} + 1 ))
[ "$N" -lt 1 ] && N=1
NPROC="${SLURM_CPUS_PER_TASK:-8}"
SRC=$(src_root $NET)
CFG=$NOISEPY/param_files/cluster/pipeline_${NET}_tspws.yaml

echo "[$NET/$STAGE] shard $I/$N | nproc $NPROC | $(date)"

case "$STAGE" in
build)
    python "$NOISEPY/scripts/picking/build_tspws_stacks.py" \
        --stack-root "$SRC" --out "$PROJ/$NET/stacks_tspws" \
        --nproc "$NPROC" --pre-block 2 --shard "$I/$N"
    ;;
jackknife)
    python "$NOISEPY/scripts/picking/substack_jackknife.py" \
        --stack-root "$SRC" --out "$PROJ/$NET/substack_jackknife_k2" \
        --k 2 --nproc "$NPROC" --shard "$I/$N"
    ;;
finalize)
    python "$NOISEPY/scripts/picking/run_pipeline.py" --config "$CFG" --stage all
    python "$NOISEPY/scripts/picking/attach_substack_sigma.py" \
        --picks-dir "$PROJ/$NET/tomo/1_velocity_maps/inputs_tspws" \
        --jackknife "$PROJ/$NET/substack_jackknife_k2"
    python "$NOISEPY/scripts/picking/plot_exported_picks_hist2d.py" \
        --dir "$PROJ/$NET/tomo/1_velocity_maps/inputs_tspws" \
        --title "$NET ts-PWS production (vmin=0.2, jackknife sigma)"
    python "$NOISEPY/scripts/picking/plot_sigma_census.py" \
        --jackknife "$PROJ/$NET/substack_jackknife_k2" --label "$NET"
    ;;
*)
    echo "unknown STAGE=$STAGE (build|jackknife|finalize)"; exit 2 ;;
esac
echo "[$NET/$STAGE] done $(date)"
