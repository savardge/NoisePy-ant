#!/bin/bash
# One-cell, tiny-iteration end-to-end smoke of all 6 configs on one network (default riehen),
# on debug-cpu. Writes to $SCRATCH_VS/smoke/<net>/<cfg> -- NEVER the production runs/ tree.
# Usage: ./smoke_test.sh [net]
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/env_bamboo.sh"
NET=${1:-riehen}
mkdir -p "$SCRATCH_VS/logs"
for CFG in $ALL_CFGS; do
  sbatch --parsable \
      --partition=debug-cpu --time=00:15:00 --cpus-per-task=8 \
      --array=0 \
      --export=ALL,NET=$NET,CFG=$CFG,N_SHARDS=1,NCHAINS=3,BURN=5000,MAIN=2000,CELL_TIMEOUT=600,LIMIT=1,OUTDIR_OVERRIDE=$SCRATCH_VS/smoke/$NET/$CFG,BAMBOO_VS_DIR=$HERE \
      --output="$SCRATCH_VS/logs/smoke_${NET}_${CFG}_%A.out" \
      "$HERE/vs_prod3.sbatch"
done
echo "smoke jobs submitted; results under $SCRATCH_VS/smoke/$NET/"
