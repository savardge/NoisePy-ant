#!/bin/bash
# GVL-1: R0g (Rayleigh FUNDAMENTAL GROUP ONLY) with --save-ensemble.
#
# This is the MANUSCRIPT's exact target configuration, which none of the earlier combo sweeps
# covered (they all included Love). Needed to answer "is the current workflow converging
# better than Fig 5 / S10" on a like-for-like target set rather than a richer one.
#
# Same everything else as the ensemble sweep: prod3 k2 blanket maps, v1 period ranges, 8 km
# box, 24 chains, 150k burn-in + 120k main, isotropic. Fewer mp workers because the 3-combo
# ensemble streams may still be running.
set -u
NP=/Users/genevievesavard/Codes/NoisePy-ant
EHM=/Users/genevievesavard/Codes/extract_higher_modes/Projects
PY_BB=/opt/anaconda3/envs/bayesbay_dev/bin/python
PY_BH=/opt/anaconda3/envs/bayhunter/bin/python
RANGES=$EHM/_period_validity/period_ranges_DECISIONS_v1.csv
TAG=test_2026-08-07_gvl1_iso_combos_ens
GROOT=$EHM/hautesorne/tomo/1_velocity_maps/1_production/tspws_group_blanket_dx0.5_prod3_k2/production

for cell in "41 21" "42 21" "41 22" "42 22" "41 20"; do
  set -- $cell; ix=$1; iy=$2
  out=$EHM/hautesorne/tomo/2_vs_depth_inversion/tests/$TAG/GVL1_cell_${ix}_${iy}/R0g
  [ -f "$out/bayhunter_result.npz" ] && { echo "=== R0g ($ix,$iy) already done ==="; continue; }
  mkdir -p "$out"
  echo "=== GVL1 ($ix,$iy) R0g  (Rayleigh fund group only) ==="
  PYTHONPATH=$NP $PY_BB $NP/scripts/picking/run_vs_inversion.py \
    --production "$GROOT" \
    --period-ranges "$RANGES" --net hautesorne \
    --waves fund --waves-phase "" \
    --config "$NP/param_files/cluster/tomo/hautesorne_tspws_group_scaled_lccov.yaml" \
    --ix "$ix" --iy "$iy" --outdir "$out" \
    --engines bayhunter --save-ensemble \
    --bayhunter-python "$PY_BH" \
    --bayhunter-runner "$NP/scripts/picking/run_bayhunter_cell.py" \
    --depth-max 8.0 \
    --n-chains 24 --bh-iter-burnin 150000 --bh-iter-main 120000 \
    --bh-use-mp --bh-mp-nthreads 4 \
    > "$out/run.log" 2>&1
  echo "=== R0g ($ix,$iy) exit $? ==="
done
