#!/bin/bash
# End-of-campaign recovery + acceptance for vs_prod3. Run the stages IN ORDER, waiting for the
# queue to drain between `recover` and `assemble`.
#
#   ./final_sweep.sh check      # integrity + collision scan, read-only, safe any time
#   ./final_sweep.sh quarantine # move damaged cells aside so they regenerate
#   ./final_sweep.sh recover    # resubmit all nets to fill every remaining gap
#   ./final_sweep.sh assemble   # re-assemble ALL 18 volumes (only after the queue is empty)
#   ./final_sweep.sh verify     # acceptance gate: every volume complete and populated
#
# What needs recovering, and why:
#   - 267 riehen/R0g cells quarantined after the duplicate-array collision (see that config's
#     README_COLLISION.txt)
#   - a handful of in-flight cells per array, lost whenever a task ended mid-cell
#   - hautesorne/R0g shard 1, killed at launch by a transient PIL import error
# Every one of these is simply a missing npz, so a resubmit picks them up and skips the rest.
#
# Why `assemble` must re-run for ALL configs, including ones that already have a volume:
# submit_all.sh's assemble jobs are `afterany` on the MAIN arrays only, so any volume written
# while a booster was still adding cells is provisional.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/env_bamboo.sh"
PY=$BH_PY
STAGE=${1:-}

srun_q() { srun --partition=shared-cpu --time="$1" --cpus-per-task=2 "${@:2}" 2>&1 \
           | grep -v "^srun:" || true; }

case "$STAGE" in
  check)
    echo "### truncated / unreadable npz"
    srun_q 01:00:00 "$PY" "$HERE/verify_cells.py" "$RUNS"
    echo
    echo "### collision scan (n_chains_used < 24) across every config"
    for net in $ALL_NETS; do for cfg in $ALL_CFGS; do
        d=$RUNS/$net/$cfg/cells
        [ -d "$d" ] || continue
        out=$(srun_q 00:30:00 "$PY" "$HERE/quarantine_collision.py" "$d" --dry-run | head -2)
        echo "$net/$cfg: $out"
    done; done
    ;;
  quarantine)
    for net in $ALL_NETS; do for cfg in $ALL_CFGS; do
        d=$RUNS/$net/$cfg/cells
        [ -d "$d" ] || continue
        echo "### $net/$cfg"
        srun_q 00:30:00 "$PY" "$HERE/quarantine_collision.py" "$d"
    done; done
    srun_q 01:00:00 "$PY" "$HERE/verify_cells.py" "$RUNS" --delete
    ;;
  recover)
    # resume-safe: finished cells are skipped, so this only fills gaps
    for net in $ALL_NETS; do "$HERE/submit_all.sh" "$net"; done
    echo
    echo "wait for the queue to drain, then: ./final_sweep.sh assemble"
    ;;
  assemble)
    n=$(squeue -u "$USER" -h -o "%j" | grep -cE "vsprod3|vsboost" || true)
    if [ "$n" -gt 0 ]; then
        echo "REFUSING: $n campaign tasks still queued/running -- assembling now would freeze"
        echo "volumes before the last cells land. Wait for the queue to empty."
        exit 1
    fi
    for net in $ALL_NETS; do for cfg in $ALL_CFGS; do
        echo "### assembling $net/$cfg"
        NET=$net CFG=$cfg N_SHARDS=32 STAGE=assemble BAMBOO_VS_DIR=$HERE \
          srun_q 00:30:00 bash "$HERE/vs_prod3.sbatch" || true
    done; done
    ;;
  verify)
    srun_q 00:30:00 "$PY" "$HERE/verify_volumes.py" "$RUNS"
    echo
    echo "### convergence gate: chain_disagree(Vs) per config"
    echo "### (gate on chain_disagree, NOT logL -- free noise sigma drives spurious logL basins)"
    srun_q 00:45:00 "$PY" "$HERE/convergence_report.py" "$RUNS"
    ;;
  *)
    sed -n '2,25p' "$0"
    exit 2
    ;;
esac
