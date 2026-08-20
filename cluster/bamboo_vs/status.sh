#!/bin/bash
# One-line-per-fact status of the vs_prod3 campaign, for humans and for the watch loop.
# Usage: ./status.sh          # compact key=value summary
#        ./status.sh --full   # + per-(net,cfg) breakdown
. "$(cd "$(dirname "$0")" && pwd)/env_bamboo.sh" 2>/dev/null || true
SCRATCH_VS="${SCRATCH_VS:-/srv/beegfs/scratch/users/s/savardg/vs_prod3}"

# count MAIN (vsprod3) and BOOSTER (vsboost) tasks separately -- a vsprod3-only count hides
# the boosters entirely and badly understates the capacity actually in play.
main=$(squeue -u "$USER" -h -t R -o "%j" 2>/dev/null | grep -c vsprod3 || true)
boost=$(squeue -u "$USER" -h -t R -o "%j" 2>/dev/null | grep -c vsboost || true)
run=$((main + boost))
pend=$(squeue -u "$USER" -h -t PD -o "%j" 2>/dev/null | grep -cE "vsprod3|vsboost" || true)
# Count the ORIGINAL campaign separately from the vs_max re-runs: the re-runs write into
# <cfg>_vmax<X>/ under the same tree, so a single total would sail past the 53,235-cell target
# and stop meaning anything.
# The ORIGINAL campaign is exactly runs/<net>/<one of the 6 labels>/. Every re-run and
# diagnostic arm lives in a suffixed sibling (_vmax4.5, _modegate, _RLgp, _R0R1*, _T1.08).
# Counting them together makes "cells" sail past the 53,235 target and stop meaning anything --
# it already happened once with _vmax, so match the canonical names instead of blacklisting
# each new suffix as it appears.
_orig='/runs/[^/]*/\(R0g\|R0p\|L0g\|L0p\|RLg_radial\|RLg_iso\)/cells/'
_all=$(find "$SCRATCH_VS/runs" -name "cell_*.npz" 2>/dev/null | grep -v "\.tmp" || true)
cells=$(echo "$_all" | grep -c "$_orig" || true)
recells=$(echo "$_all" | grep -vc "$_orig" || true)
_volo='/runs/[^/]*/\(R0g\|R0p\|L0g\|L0p\|RLg_radial\|RLg_iso\)/volume_'
vols=$(find "$SCRATCH_VS/runs" -name "volume_*.npz" 2>/dev/null | grep -c "$_volo" || true)
revols=$(find "$SCRATCH_VS/runs" -name "volume_*.npz" 2>/dev/null | grep -vc "$_volo" || true)
# Terminal states, split by what they MEAN. Only genuine faults drive the alert:
# TIMEOUT is expected (long shards hit the 48 h wall and are resumed by resubmit) and
# CANCELLED is an operator action, so lumping them in would bury a real failure in noise.
_states=$(sacct -u "$USER" -S 2026-08-13 --name=vsprod3,vsboost -X -n -P \
          --format=State 2>/dev/null || true)
badstates=$(echo "$_states" | grep -cE "FAILED|NODE_FAIL|OUT_OF_ME" || true)
touts=$(echo "$_states" | grep -c "TIMEOUT" || true)
cancels=$(echo "$_states" | grep -c "CANCELLED" || true)
echo "running=$run (main=$main boost=$boost) pending=$pend cells=$cells volumes=$vols" \
     "badtasks=$badstates timeouts=$touts cancelled=$cancels" \
     "| extra_arms rr_ncell=$recells rr_nvol=$revols"
# NOTE the odd rr_ncell / rr_nvol names: the watch loop extracts fields with a GREEDY
# 's/.*cells=\(...\)/' style regex, so any second token ending in "cells=" or "volumes=" on this
# line (including "rerun_volumes=") would be the one it captured, and it would silently read the
# re-run counter as the campaign's. These names share no such suffix.

if [ "${1:-}" = "--full" ]; then
    for net in riehen aargau hautesorne; do
        for c in R0g R0p L0g L0p RLg_radial RLg_iso; do
            d="$SCRATCH_VS/runs/$net/$c"
            n=$(ls "$d/cells"/*.npz 2>/dev/null | grep -vc "\.tmp" || true)
            v=$(ls "$d"/volume_*.npz 2>/dev/null | wc -l || true)
            echo "  $net/$c cells=$n volume=$v"
        done
    done
    echo "  --- terminal-state tasks:"
    sacct -u "$USER" -S 2026-08-13 --name=vsprod3 -X -n -P --format=JobID,State,Elapsed \
        2>/dev/null | grep -E "TIMEOUT|FAILED|NODE_FAIL|OUT_OF_ME" | head -20
fi
