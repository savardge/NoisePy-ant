#!/bin/bash
#SBATCH --job-name=vsprod3
#SBATCH --partition=public-cpu
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=4G
# Per-cell 1D Vs inversion over one (NET, CFG) as a Slurm array of shards.
#
# Required --export vars: NET (riehen|aargau|hautesorne), CFG (R0g|R0p|L0g|L0p|RLg_radial|
# RLg_iso), N_SHARDS (must equal the array size). Optional overrides for smoke tests:
# NCHAINS, BURN, MAIN, CELL_TIMEOUT, LIMIT, CELLS, STAGE=assemble (stitch cells/ -> volume).
#
# One cell inverts at a time per task (--n-workers 1) with all chains forked concurrently
# (--use-mp --mp-nthreads = cpus): fewer threads than chains would run the chains in
# sequential waves -- identical core-hours, longer per-cell wall time, more timeout exposure.
set -eu

. "${BAMBOO_VS_DIR:-$SLURM_SUBMIT_DIR}/env_bamboo.sh"
activate_env
net_paths "$NET"
cfg_flags "$CFG"
# WS_OVERRIDE: the waveset for arms whose CFG does not imply it. The R0+R1 arms run
# CFG=R0g/R0p (for the production root and flags) but invert waveset `fundot`, and
# --assemble-only stitches whatever --wavesets says -- without this it would silently build a
# `fund` volume from `fundot` cells.
WS=${WS_OVERRIDE:-$WS}

# CELLS is a comma-separated cell list, but sbatch --export is ITSELF comma-separated, so
# passing it raw delivers only the first entry (a 5-cell sweep silently became 1 cell).
# Encode commas as @C@ at submit time, exactly as EXTRA_FLAGS does.
CELLS="${CELLS:-}"; CELLS=${CELLS//@C@/,}

STAGE="${STAGE:-grid}"
# OUTDIR_OVERRIDE: smoke tests MUST redirect output -- a tiny-iter cell written into the
# production cells/ dir would be skipped-as-done by the real run and silently ship.
OUTDIR=${OUTDIR_OVERRIDE:-$RUNS/$NET/$CFG}
mkdir -p "$OUTDIR"
cd "$NOISEPY/scripts/picking"

COMMON=(--production "$PROD" --config "$CONFIG_YAML" --net "$NET"
        --outdir "$OUTDIR" --wavesets "$WS"
        --period-ranges "$INPUTS/period_ranges_DECISIONS_v1.csv"
        --criterion none --noise-regime free --phase-tmin 0
        --bayhunter-python "$BH_PY" --bayhunter-runner run_bayhunter_cell.py)

if [ "$STAGE" = assemble ]; then
    # 2026-10-02: the assemble branch now also takes EXTRA_FLAGS (decoded as below), so assembly-time
    # flags such as --min-obs-rungs or an arm's --period-ranges can be delivered; before this the
    # branch passed only the cfg's $EXTRA and silently ignored anything else (see assemble_riehen_grids).
    AF=${EXTRA_FLAGS:-}; AF=${AF//@C@/,}
    srun "$BH_PY" grid_vs_inversion.py "${COMMON[@]}" $EXTRA ${AF:-} --assemble-only
    exit 0
fi

I=$(( ${SLURM_ARRAY_TASK_ID:-0} ))
# EXTRA_FLAGS: extra driver flags for booster/diagnostic arms (e.g. --reverse). Unquoted on
# purpose. @C@ decodes back to a comma: sbatch --export is itself comma-separated, so a
# literal comma in a value (--vpvs-range 1.5,3.5) is truncated in transit.
# default BEFORE the substitution: `${VAR//x/y}` on an unset VAR is itself an unbound-variable
# error under `set -u`, so submissions that set no EXTRA_FLAGS (submit_all.sh) died here.
EXTRA_FLAGS=${EXTRA_FLAGS:-}
EXTRA_FLAGS=${EXTRA_FLAGS//@C@/,}
srun "$BH_PY" grid_vs_inversion.py "${COMMON[@]}" $EXTRA ${EXTRA_FLAGS:-} \
    --n-chains "${NCHAINS:-24}" \
    --iter-burnin "${BURN:-500000}" --iter-main "${MAIN:-500000}" \
    --use-mp --mp-nthreads "${SLURM_CPUS_PER_TASK}" --n-workers 1 \
    --cell-timeout "${CELL_TIMEOUT:-43200}" \
    ${LIMIT:+--limit "$LIMIT"} ${CELLS:+--cells "$CELLS"} \
    --shard "$I/${N_SHARDS}" --no-assemble
