#!/bin/bash
# Well-cell Vs inversion tests on the prod3_k3 maps with the v1 period ranges.
# First joint GROUP+PHASE runs (group = scaled Cd, phase = blanket Cd), both engines.
#
# One argument: the network stream to run (riehen | aargau). Wells run SEQUENTIALLY within
# a stream so at most two cells invert at once on this machine.
set -u
NP=/Users/genevievesavard/Codes/NoisePy-ant
EHM=/Users/genevievesavard/Codes/extract_higher_modes/Projects
PY_BB=/opt/anaconda3/envs/bayesbay_dev/bin/python
PY_BH=/opt/anaconda3/envs/bayhunter/bin/python
RANGES=$EHM/_period_validity/period_ranges_DECISIONS_v1.csv
TAG=test_2026-08-06_joint_v1ranges

run_one () {  # net dx well ix iy
  local net=$1 dx=$2 well=$3 ix=$4 iy=$5
  local root=$EHM/$net/tomo/1_velocity_maps/1_production
  local out=$EHM/$net/tomo/2_vs_depth_inversion/tests/$TAG/${well}_cell_${ix}_${iy}
  mkdir -p "$out"
  echo "=== $net / $well cell ($ix,$iy) -> $out ==="
  PYTHONPATH=$NP $PY_BB $NP/scripts/picking/run_vs_inversion.py \
    --production       "$root/tspws_group_scaled_dx${dx}_prod3_k3/production" \
    --production-phase "$root/tspws_phase_blanket_dx${dx}_prod3_k3/production" \
    --period-ranges "$RANGES" --net "$net" --waves fund,overtone,love \
    --config "$NP/param_files/cluster/tomo/${net}_tspws_group_scaled_lccov.yaml" \
    --ix "$ix" --iy "$iy" --outdir "$out" \
    --engines bayesbay,bayhunter \
    --bayhunter-python "$PY_BH" \
    --bayhunter-runner "$NP/scripts/picking/run_bayhunter_cell.py" \
    > "$out/run.log" 2>&1
  echo "=== $well exit $? ==="
}

case "$1" in
  riehen)
    run_one riehen 0.2 Basel-1     23 47
    run_one riehen 0.2 Otterbach-2 26 43
    run_one riehen 0.2 Riehen-1    43 48
    run_one riehen 0.2 Riehen-2    46 52
    ;;
  aargau)
    run_one aargau 0.5 Boettstein  26 42
    run_one aargau 0.5 Riniken     21 28
    ;;
  *) echo "usage: $0 riehen|aargau"; exit 1 ;;
esac
