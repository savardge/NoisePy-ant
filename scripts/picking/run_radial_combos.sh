#!/bin/bash
# Continuous-zeta RADIAL ANISOTROPY re-runs of the Love+Rayleigh waveset combos at the well
# cells. BayHunter ONLY (bayesbay has no zeta): Love targets forward on Vsh, Rayleigh on Vsv,
# gamma(z)=(Vsh-Vsv)/Vsv continuous per layer, prior [-0.35, 0.35].
#
# Calibrated CZ config (cz-convergence-calibration): 24 chains, 150k burn-in, 120k main;
# convergence gated on chain_disagree(Vs). Fork-mp 8 workers so two streams share 16 cores.
#
# Argument = stream (a | b), three wells each, sequential; combos sequential within a well.
set -u
NP=/Users/genevievesavard/Codes/NoisePy-ant
EHM=/Users/genevievesavard/Codes/extract_higher_modes/Projects
PY_BB=/opt/anaconda3/envs/bayesbay_dev/bin/python
PY_BH=/opt/anaconda3/envs/bayhunter/bin/python
RANGES=$EHM/_period_validity/period_ranges_DECISIONS_v1.csv
TAG=test_2026-08-07_radial_cz_combos

# only combos mixing Love and Rayleigh constrain zeta
COMBOS="
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
    if [ -f "$out/bayhunter_result.npz" ]; then
      echo "=== $net/$well $name: already done, skip ==="; continue
    fi
    mkdir -p "$out"
    echo "=== $net / $well ($ix,$iy) RADIAL combo $name  group=[$gw] phase=[$pw] ==="
    PYTHONPATH=$NP $PY_BB $NP/scripts/picking/run_vs_inversion.py \
      --production       "$root/tspws_group_scaled_dx${dx}_prod3_k3/production" \
      --production-phase "$root/tspws_phase_blanket_dx${dx}_prod3_k3/production" \
      --period-ranges "$RANGES" --net "$net" \
      --waves "$gw" --waves-phase "$pw" \
      --config "$NP/param_files/cluster/tomo/${net}_tspws_group_scaled_lccov.yaml" \
      --ix "$ix" --iy "$iy" --outdir "$out" \
      --engines bayhunter --radial \
      --bayhunter-python "$PY_BH" \
      --bayhunter-runner "$NP/scripts/picking/run_bayhunter_cell.py" \
      --n-chains 24 --bh-iter-burnin 150000 --bh-iter-main 120000 \
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
