#!/bin/bash
# Grid every Vs volume with GMT surface, then re-render its figures from the spline.
#
# Two environments by necessity: pygmt (and therefore GMT) lives in its own env so the analysis
# env is untouched, while the figure script runs in das-ambient-noise. Each arm is
# grid-then-render so a failure in one arm does not block the rest.
#
# usage: regrid_and_render.sh [--tension T] [--refine N] <net>:<arm>:<waveset>[:<label>] ...
set -u
PYGMT=/opt/anaconda3/envs/pygmt-env/bin/python
DAS=/opt/anaconda3/envs/das-ambient-noise/bin/python
NP=/Users/genevievesavard/Codes/NoisePy-ant
E=/Users/genevievesavard/Codes/extract_higher_modes
TENSION=0.35; REFINE=4
CAMPAIGN=${CAMPAIGN:-vs_prod3}     # 2026-10-02: vs_prod4 arms live under 2_vs_depth_inversion/vs_prod4/
while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --tension) TENSION=$2; shift 2;;
        --refine)  REFINE=$2; shift 2;;
        *) echo "unknown flag $1"; exit 2;;
    esac
done
ok=0; fail=0
for spec in "$@"; do
    IFS=: read -r net arm ws label <<< "$spec"
    label=${label:-$arm}
    gd=$E/Projects/$net/tomo/2_vs_depth_inversion/$CAMPAIGN/$arm
    if [ ! -f "$gd/volume_$ws.npz" ]; then
        echo "SKIP $net/$arm: no volume_$ws.npz"; continue
    fi
    echo "=== $net/$arm ($ws)"
    if ! $PYGMT $NP/scripts/picking/grid_model_surface.py --griddir "$gd" --waveset "$ws" \
            --tension "$TENSION" --refine "$REFINE" 2>&1 \
            | grep -viE "warning|pre-process|decimals|half-way|grdmask|compatible" | tail -3; then
        echo "  GRID FAILED"; fail=$((fail+1)); continue
    fi
    if ! PYTHONPATH=$NP $DAS $NP/scripts/picking/vs_model_figures.py \
            --griddir "$gd" --net "$net" --waveset "$ws" --label "$label" --gridded 2>&1 \
            | grep -viE "warning" | tail -3; then
        echo "  RENDER FAILED"; fail=$((fail+1)); continue
    fi
    ok=$((ok+1))
done
echo "done: $ok arms gridded+rendered, $fail failed"
