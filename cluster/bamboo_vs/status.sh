#!/bin/bash
# One-line-per-fact status of the vs_prod3 campaign, for humans and for the watch loop.
# Usage: ./status.sh          # compact key=value summary
#        ./status.sh --full   # + per-(net,cfg) breakdown
. "$(cd "$(dirname "$0")" && pwd)/env_bamboo.sh" 2>/dev/null || true
SCRATCH_VS="${SCRATCH_VS:-/srv/beegfs/scratch/users/s/savardg/vs_prod3}"

run=$(squeue -u "$USER" -h -t R -o "%j" 2>/dev/null | grep -c vsprod3 || true)
pend=$(squeue -u "$USER" -h -t PD -o "%j" 2>/dev/null | grep -c vsprod3 || true)
cells=$(find "$SCRATCH_VS/runs" -name "cell_*.npz" 2>/dev/null | grep -vc "\.tmp" || true)
vols=$(find "$SCRATCH_VS/runs" -name "volume_*.npz" 2>/dev/null | wc -l || true)
# terminal failure states across the campaign's array jobs (allocation level only)
badstates=$(sacct -u "$USER" -S 2026-08-13 --name=vsprod3 -X -n -P \
            --format=State 2>/dev/null \
            | grep -cE "TIMEOUT|FAILED|NODE_FAIL|OUT_OF_ME|CANCELLED" || true)
echo "running=$run pending=$pend cells=$cells volumes=$vols badtasks=$badstates"

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
