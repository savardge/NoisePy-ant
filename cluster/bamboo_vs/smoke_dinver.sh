#!/bin/bash
# One-cell, tiny-NA-budget end-to-end smoke of a Dinver arm on debug-cpu. Writes to
# $SCRATCH_DINVER/smoke/<net>/<cfg> -- NEVER the production runs/ tree. Usage: ./smoke_dinver.sh [net] [cfg]
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/env_bamboo.sh"
NET=${1:-riehen}; CFG=${2:-R0gR1g}
mkdir -p "$SCRATCH_DINVER/logs"
sbatch --parsable --partition=debug-cpu --time=00:15:00 --cpus-per-task=4 --array=0 \
    --export=ALL,NET=$NET,CFG=$CFG,N_SHARDS=1,NS=500,NS0=100,NR=50,NTRIALS=1,CELL_TIMEOUT=600,CELLS=23_47,OUTDIR_OVERRIDE=$SCRATCH_DINVER/smoke/$NET/$CFG,BAMBOO_VS_DIR=$HERE \
    --output="$SCRATCH_DINVER/logs/smoke_${NET}_${CFG}_%A.out" "$HERE/dinver_vs.sbatch"
echo "smoke submitted; result under $SCRATCH_DINVER/smoke/$NET/$CFG"
