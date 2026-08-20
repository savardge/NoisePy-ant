#!/bin/bash
# Launch ONE inversion arm into its own <cfg><suffix>/ directory.
#
#   ./rerun_arm.sh <net> <cfg> <suffix> "<extra driver flags>"
#
# The flags string may contain these tokens, substituted per network so one call site works for
# all three (their dx differs, so the production paths do):
#   @PHASE@   the phase production root      @GROUP@   the group production root
#
# Examples:
#   ./rerun_arm.sh riehen RLg_iso _modegate "--mode-gate-phase @PHASE@"
#   ./rerun_arm.sh aargau R0R1gp _new "--wavesets fundot --phase-root @PHASE@ --vs-max 4.5 \
#                                      --mode-gate-phase @PHASE@"
#
# A SEPARATE output directory is mandatory: writing into an existing arm would be skipped
# cell-for-cell by skip-if-exists and silently produce nothing.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/env_bamboo.sh"

NET=${1:?net}; CFG=${2:?cfg}; SUFFIX=${3?suffix (may be empty: "" tops up an original arm)}; FLAGS=${4:-}
N_SHARDS=${N_SHARDS:-32}
THROTTLE=${THROTTLE:-4}
WALL=${WALL:-2-00:00:00}
PART=${PART:-public-cpu}

net_paths "$NET"
FLAGS=${FLAGS//@PHASE@/$PHASE_PROD}
FLAGS=${FLAGS//@GROUP@/$GROUP_PROD}
# sbatch --export takes a COMMA-separated list, so a comma inside any value silently truncates
# it: "--vpvs-range 1.5,3.5" arrived as "--vpvs-range 1.5", giving a 1-element prior that
# BayHunter fails to unpack ("expected 2, got 1"). Ship commas as @C@ and let the job script
# put them back. Applies to any comma-bearing flag (--vpvs-range, --radial-prior, --vbounds).
FLAGS_ESC=${FLAGS//,/@C@}
OUT=$RUNS/$NET/${CFG}${SUFFIX}

# RESUME=1 is the deliberate top-up of an arm whose array finished short: skip-if-exists means
# the finished cells cost nothing and only the gaps are computed. Without it, an existing cells/
# dir is refused, because silently doing nothing is the more likely mistake.
if [ "${RESUME:-0}" != "1" ] && [ -d "$OUT/cells" ] \
   && [ "$(ls -A "$OUT/cells" 2>/dev/null | head -1)" ]; then
    echo "REFUSING: $OUT/cells already has results -- pick a new suffix, or set RESUME=1 to"
    echo "top up the gaps in this same arm."
    exit 1
fi
# An empty cells/ is NOT proof the directory is free: an array launched minutes ago has not
# written its first cell yet. Launching a second array on the same outdir is what corrupted
# riehen/R0g -- two arrays with identical shard slices raced cell-for-cell. Check the QUEUE.
for _a in $(squeue -u "$USER" -h -o "%i" | awk -F_ '{print $1}' | sort -u); do
    _o=$(scontrol show job "$_a" 2>/dev/null | tr ' ' '\n' \
         | grep -o 'OUTDIR_OVERRIDE=[^,]*' | head -1)
    if [ "${_o#OUTDIR_OVERRIDE=}" = "$OUT" ]; then
        echo "REFUSING: array $_a is already queued/running on $OUT."
        echo "Two arrays on one outdir race cell-for-cell and corrupt results (see"
        echo "runs/riehen/R0g/README_COLLISION.txt). Cancel it first, or use a new suffix."
        exit 1
    fi
done

JID=$(sbatch --parsable \
    --job-name=vsprod3 \
    --partition="$PART" \
    --time="$WALL" \
    --array=0-$((N_SHARDS - 1))%$THROTTLE \
    --export=ALL,NET=$NET,CFG=$CFG,N_SHARDS=$N_SHARDS,EXTRA_FLAGS="$FLAGS_ESC",OUTDIR_OVERRIDE=$OUT,BAMBOO_VS_DIR=$HERE \
    --output="$SCRATCH_VS/logs/${NET}_${CFG}${SUFFIX}_%A_%a.out" \
    "$HERE/vs_prod3.sbatch")
printf '%-12s %-14s %-11s job %s\n   flags: %s\n   out:   %s\n' \
    "$NET" "$CFG" "$SUFFIX" "$JID" "$FLAGS" "$OUT"
echo -e "$NET\t$CFG\t$SUFFIX\t$JID\t$OUT" >> "$RUNS/arms_launched.tsv"
