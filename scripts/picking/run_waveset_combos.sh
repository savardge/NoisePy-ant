#!/bin/bash
# Waveset-combination Vs inversions at the well cells (prod3_k3 maps, v1 period ranges).
#
# 8 combos x 6 wells x 2 engines. Convergence-first settings:
#   bayesbay : 16 chains, 400k iters, 150k burn-in
#   BayHunter: 16 chains, 150k burn-in + 120k main, fork multiprocessing (8 workers/run so
#              two concurrent streams fill the 16 cores without heavy oversubscription)
#
# Argument = stream (a | b): three wells each, run sequentially. Combos run sequentially
# within a well.
set -u
NP=/Users/genevievesavard/Codes/NoisePy-ant
EHM=/Users/genevievesavard/Codes/extract_higher_modes/Projects
PY_BB=/opt/anaconda3/envs/bayesbay_dev/bin/python
PY_BH=/opt/anaconda3/envs/bayhunter/bin/python
RANGES=$EHM/_period_validity/period_ranges_DECISIONS_v1.csv
TAG=test_2026-08-06_waveset_combos

# combo name | group waves | phase waves ('' = none)
COMBOS="
R0g|fund|
R0gR1g|fund,overtone|
R0gR0p|fund|fund
L0g|love|
L0gL0p|love|love
R0gL0g|fund,love|
R0pL0p||fund,love
R0gL0gR0pL0p|fund,love|fund,love
"

run_well () {  # net dx well ix iy
  local net=$1 dx=$2 well=$3 ix=$4 iy=$5
  local root=$EHM/$net/tomo/1_velocity_maps/1_production
  local wdir=$EHM/$net/tomo/2_vs_depth_inversion/tests/$TAG/${well}_cell_${ix}_${iy}
  echo "$COMBOS" | while IFS='|' read -r name gw pw; do
    [ -z "$name" ] && continue
    local out=$wdir/$name
    if [ -f "$out/engine_comparison.png" ]; then
      echo "=== $net/$well $name: already done, skip ==="; continue
    fi
    mkdir -p "$out"
    echo "=== $net / $well ($ix,$iy) combo $name  group=[$gw] phase=[$pw] ==="
    PYTHONPATH=$NP $PY_BB $NP/scripts/picking/run_vs_inversion.py \
      --production       "$root/tspws_group_scaled_dx${dx}_prod3_k3/production" \
      --production-phase "$root/tspws_phase_blanket_dx${dx}_prod3_k3/production" \
      --period-ranges "$RANGES" --net "$net" \
      --waves "$gw" --waves-phase "$pw" \
      --config "$NP/param_files/cluster/tomo/${net}_tspws_group_scaled_lccov.yaml" \
      --ix "$ix" --iy "$iy" --outdir "$out" \
      --engines bayesbay,bayhunter \
      --bayhunter-python "$PY_BH" \
      --bayhunter-runner "$NP/scripts/picking/run_bayhunter_cell.py" \
      --n-chains 16 --iterations 400000 --burnin 150000 \
      --bh-iter-burnin 150000 --bh-iter-main 120000 \
      --bh-use-mp --bh-mp-nthreads 8 \
      > "$out/run.log" 2>&1
    echo "=== $net/$well $name exit $? ==="
  done
}

case "$1" in
  a)
    run_well riehen 0.2 Basel-1     23 47
    run_well riehen 0.2 Otterbach-2 26 43
    run_well aargau 0.5 Boettstein  26 42
    ;;
  b)
    run_well riehen 0.2 Riehen-1    43 48
    run_well riehen 0.2 Riehen-2    46 52
    run_well aargau 0.5 Riniken     21 28
    ;;
  *) echo "usage: $0 a|b"; exit 1 ;;
esac
