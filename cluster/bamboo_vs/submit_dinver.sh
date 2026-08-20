#!/bin/bash
# Submit a Dinver campaign arm: shard array + dependent assemble. Usage:
#   ./submit_dinver.sh riehen R0gR1g          # N_SHARDS=32 THROTTLE=8 by default
# Resume-safe: resubmitting continues (cells/ skip-if-exists AND per-cell work dirs resume
# half-done cells). Do NOT launch a second array on the same (net, cfg) while one runs:
# --work-tag cell shares work dirs, two tasks on one cell would corrupt each other.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/env_bamboo.sh"
NET=${1:?net}; CFG=${2:?cfg}
N_SHARDS=${N_SHARDS:-32}; THROTTLE=${THROTTLE:-8}
mkdir -p "$SCRATCH_DINVER/logs" "$DINVER_RUNS"
[ -e "$SCRATCH_DINVER/inputs" ] || ln -s "$INPUTS" "$SCRATCH_DINVER/inputs"   # same inputs, no copy
MANIFEST=$DINVER_RUNS/submitted_$(date +%Y%m%d_%H%M%S).tsv
echo -e "net\tcfg\tarray_jobid\tassemble_jobid\tn_shards" > "$MANIFEST"
JID=$(sbatch --parsable --array=0-$((N_SHARDS - 1))%$THROTTLE \
    --export=ALL,NET=$NET,CFG=$CFG,N_SHARDS=$N_SHARDS,BAMBOO_VS_DIR=$HERE \
    --output="$SCRATCH_DINVER/logs/${NET}_${CFG}_%A_%a.out" "$HERE/dinver_vs.sbatch")
AID=$(sbatch --parsable --dependency=afterany:$JID \
    --partition=shared-cpu --time=01:00:00 --cpus-per-task=2 --mem-per-cpu=8G \
    --export=ALL,NET=$NET,CFG=$CFG,N_SHARDS=$N_SHARDS,STAGE=assemble,BAMBOO_VS_DIR=$HERE \
    --output="$SCRATCH_DINVER/logs/${NET}_${CFG}_assemble_%j.out" "$HERE/dinver_vs.sbatch")
echo -e "$NET\t$CFG\t$JID\t$AID\t$N_SHARDS" | tee -a "$MANIFEST"
echo "manifest: $MANIFEST"
