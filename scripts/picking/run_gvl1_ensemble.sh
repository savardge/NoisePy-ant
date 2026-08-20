#!/bin/bash
# GVL-1 ENSEMBLE re-run (isotropic only) with --save-ensemble, so the per-model layer
# boundaries survive into the npz and interface PROBABILITY P(z) can be computed -- the
# statistic the manuscript uses (its Fig 6b), which posterior percentiles cannot reconstruct.
#
# 5 grid cells within 0.7 km of GVL-1 (7.22020E, 47.33308N) x 3 Love+Rayleigh combos x
# {isotropic, continuous-zeta radial} = 30 BayHunter runs.
#
# DEPTH BOX 8 km (not the driver default 6): HS curves run to 8.6-9.1 s, the lambda-floor is
# ~12 km and union kernels show sensitivity past 7 km. The first pass (tags without _z8) used
# 6 km and TRUNCATED the resolved depth range into the half-space -- kept for comparison only. BOTH setups use the calibrated
# CZ config (24 chains, 150k burn-in + 120k main, fork-mp) so iso-vs-radial differences are
# not a chain-count artifact. Group AND phase both come from BLANKET Cd k2 trees (user call;
# note the network recommendation for group elsewhere is scaled).
#
# Argument = stream (a | b): cells split 3/2, sequential within a stream.
set -u
NP=/Users/genevievesavard/Codes/NoisePy-ant
EHM=/Users/genevievesavard/Codes/extract_higher_modes/Projects
PY_BB=/opt/anaconda3/envs/bayesbay_dev/bin/python
PY_BH=/opt/anaconda3/envs/bayhunter/bin/python
RANGES=$EHM/_period_validity/period_ranges_DECISIONS_v1.csv
ISO_TAG=test_2026-08-07_gvl1_iso_combos_ens
RAD_TAG=test_2026-08-07_gvl1_radial_combos_ens
GROOT=$EHM/hautesorne/tomo/1_velocity_maps/1_production/tspws_group_blanket_dx0.5_prod3_k2/production
PROOT=$EHM/hautesorne/tomo/1_velocity_maps/1_production/tspws_phase_blanket_dx0.5_prod3_k2/production

COMBOS="
R0gL0g|fund,love|
R0pL0p||fund,love
R0gL0gR0pL0p|fund,love|fund,love
"

run_cell () {  # ix iy
  local ix=$1 iy=$2
  for mode in iso; do
    local tag=$ISO_TAG rflag=""
    [ "$mode" = radial ] && { tag=$RAD_TAG; rflag="--radial"; }
    local wdir=$EHM/hautesorne/tomo/2_vs_depth_inversion/tests/$tag/GVL1_cell_${ix}_${iy}
    echo "$COMBOS" | while IFS='|' read -r name gw pw; do
      [ -z "$name" ] && continue
      local out=$wdir/$name
      if [ -f "$out/bayhunter_result.npz" ]; then
        echo "=== GVL1 ($ix,$iy) $mode $name: already done, skip ==="; continue
      fi
      mkdir -p "$out"
      echo "=== GVL1 ($ix,$iy) $mode combo $name  group=[$gw] phase=[$pw] ==="
      PYTHONPATH=$NP $PY_BB $NP/scripts/picking/run_vs_inversion.py \
        --production "$GROOT" --production-phase "$PROOT" \
        --period-ranges "$RANGES" --net hautesorne \
        --waves "$gw" --waves-phase "$pw" \
        --config "$NP/param_files/cluster/tomo/hautesorne_tspws_group_scaled_lccov.yaml" \
        --ix "$ix" --iy "$iy" --outdir "$out" \
        --engines bayhunter $rflag \
        --bayhunter-python "$PY_BH" \
        --bayhunter-runner "$NP/scripts/picking/run_bayhunter_cell.py" \
        --depth-max 8.0 --save-ensemble \
        --n-chains 24 --bh-iter-burnin 150000 --bh-iter-main 120000 \
        --bh-use-mp --bh-mp-nthreads 8 \
        > "$out/run.log" 2>&1
      echo "=== GVL1 ($ix,$iy) $mode $name exit $? ==="
    done
  done
}

case "$1" in
  a)
    run_cell 41 21
    run_cell 41 22
    run_cell 41 20
    ;;
  b)
    run_cell 42 21
    run_cell 42 22
    ;;
  *) echo "usage: $0 a|b"; exit 1 ;;
esac
