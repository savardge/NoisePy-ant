#!/bin/bash
# Booster arrays: extra short-walltime workers for the SLOW (net,cfg) arrays.
#
# Why: each main array is capped by its own ArrayTaskThrottle, so when a fast array finishes
# its slots go back to the cluster, not to our slow ones -- the campaign ends when its slowest
# array ends. Meanwhile public-cpu is full while shared-cpu sits idle, unusable by the main
# arrays because it caps jobs at 12 h and they ask for 48 h. Boosters are 11h45 tasks that fit
# shared-cpu and chew the SAME outdir from the opposite end (--reverse).
#
# Safe to over-provision: cells/ is skip-if-exists, work dirs are per task, and the result npz
# is written atomically -- so a cell claimed by both arrays is duplicated effort, never
# corruption. Forward and reverse walkers only meet in the middle, at the very end.
#
# Usage: ./booster.sh                       # default slow set
#        ./booster.sh "riehen:RLg_radial aargau:R0g"
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/env_bamboo.sh"

# ranked by measured remaining hours at throttle 8 (2026-08-14)
TARGETS=${1:-"riehen:RLg_radial riehen:RLg_iso riehen:R0g hautesorne:RLg_radial riehen:R0p"}
N_SHARDS=${N_SHARDS:-32}
THROTTLE=${THROTTLE:-8}
WALL=${WALL:-11:45:00}

mkdir -p "$SCRATCH_VS/logs" "$SCRATCH_VS/runs"
MANIFEST=$SCRATCH_VS/runs/boosters_$(date +%Y%m%d_%H%M%S).tsv
echo -e "net\tcfg\tjobid\tshards\tthrottle" > "$MANIFEST"

for t in $TARGETS; do
    NET=${t%%:*}; CFG=${t##*:}
    JID=$(sbatch --parsable \
        --job-name=vsboost \
        --partition=shared-cpu,public-cpu \
        --time="$WALL" \
        --array=0-$((N_SHARDS - 1))%$THROTTLE \
        --export=ALL,NET=$NET,CFG=$CFG,N_SHARDS=$N_SHARDS,EXTRA_FLAGS=--reverse,CELL_TIMEOUT=5400,BAMBOO_VS_DIR=$HERE \
        --output="$SCRATCH_VS/logs/boost_${NET}_${CFG}_%A_%a.out" \
        "$HERE/vs_prod3.sbatch")
    echo -e "$NET\t$CFG\t$JID\t$N_SHARDS\t$THROTTLE" | tee -a "$MANIFEST"
done
echo "manifest: $MANIFEST"
echo "NOTE: volumes must be re-assembled after boosters finish (submit_all's assemble jobs"
echo "      depend on the MAIN arrays only) -- run the final --assemble-only sweep."
