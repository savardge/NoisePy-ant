#!/bin/bash
# Submit the full vs_prod3 campaign: for each (net, cfg) one shard array + a dependent
# assemble job. Usage:
#   ./submit_all.sh                 # all 3 networks x 6 configs
#   ./submit_all.sh riehen          # one network, all configs
#   ./submit_all.sh riehen R0g      # one (net, cfg)
# Shard arrays are resume-safe (cells/ skip-if-exists): resubmitting continues, never redoes.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/env_bamboo.sh"

N_SHARDS=${N_SHARDS:-32}
THROTTLE=${THROTTLE:-8}
NETS=${1:-$ALL_NETS}
CFGS=${2:-$ALL_CFGS}

mkdir -p "$SCRATCH_VS/logs" "$SCRATCH_VS/runs"
MANIFEST=$SCRATCH_VS/runs/submitted_$(date +%Y%m%d_%H%M%S).tsv
echo -e "net\tcfg\tarray_jobid\tassemble_jobid\tn_shards" > "$MANIFEST"

for NET in $NETS; do
  for CFG in $CFGS; do
    JID=$(sbatch --parsable \
        --array=0-$((N_SHARDS - 1))%$THROTTLE \
        --export=ALL,NET=$NET,CFG=$CFG,N_SHARDS=$N_SHARDS,BAMBOO_VS_DIR=$HERE \
        --output="$SCRATCH_VS/logs/${NET}_${CFG}_%A_%a.out" \
        "$HERE/vs_prod3.sbatch")
    AID=$(sbatch --parsable \
        --dependency=afterany:$JID \
        --partition=shared-cpu --time=01:00:00 --cpus-per-task=2 --mem-per-cpu=8G \
        --export=ALL,NET=$NET,CFG=$CFG,N_SHARDS=$N_SHARDS,STAGE=assemble,BAMBOO_VS_DIR=$HERE \
        --output="$SCRATCH_VS/logs/${NET}_${CFG}_assemble_%j.out" \
        "$HERE/vs_prod3.sbatch")
    echo -e "$NET\t$CFG\t$JID\t$AID\t$N_SHARDS" | tee -a "$MANIFEST"
  done
done
echo "manifest: $MANIFEST"
