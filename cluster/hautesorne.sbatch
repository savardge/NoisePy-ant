#!/bin/bash
# Haute-Sorne -- ts-PWS production chain on yggdrasil.  Submit from THIS directory.
#
#   STAGE=build     job array, one shard of the ts-PWS stack build per task
#   STAGE=jackknife job array, per-pair substack sigma (feeds Tarantola-Valette C_d)
#   --res-drop-q 0 (2026-08-04): the default 0.25 discarded the worst-resolved
#           QUARTER of cells by RANK, not by any absolute standard. res_diag is
#           speckly cell-to-cell, so the discarded quarter was scattered and left
#           isolated holes inside well-imaged regions. 0 = the coverage/resolution
#           footprint alone defines the map.
#   VPLAUS  DISABLED 2026-08-04 -- do NOT set it. Masking inversion output on a
#           histogram of the input picks removed 111-144 Aargau cells per period,
#           100% of them one real unit (Plate-forme mesozoique epivarisque),
#           flagged only for being fast. Leave it at the 'off' default.
#   STAGE=sigma     re-attach substack sigma to the exported tables (no re-pick)
#   STAGE=finalize  single job: pick -> QC -> export -> attach sigma -> figures
#
# Submit (see README.md for how to pick the array width and throttle):
#   mkdir -p logs
#   B=$(sbatch --parsable --array=0-199%50 hautesorne.sbatch)
#   J=$(sbatch --parsable --array=0-199%50 --export=ALL,STAGE=jackknife \
#              --dependency=afterok:$B hautesorne.sbatch)
#   sbatch --dependency=afterok:$J --export=ALL,STAGE=finalize \
#          --partition=shared-cpu --time=12:00:00 --cpus-per-task=32 --mem=128G \
#          hautesorne.sbatch
#
# The %25 suffix caps CONCURRENT tasks -- lower it to fit your association's CPU limit
# (preflight.sh prints the limit). Every stage is skip-if-exists, so a task that dies on a
# time limit is fixed by resubmitting the same array; nothing is recomputed or corrupted.
#SBATCH --job-name=tspws_hautesorne
#SBATCH --partition=shared-cpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/hautesorne_%A_%a.out
#SBATCH --error=logs/hautesorne_%A_%a.err
set -eu

NET=hautesorne
STAGE="${STAGE:-build}"
. "${CLUSTER_DIR:-$SLURM_SUBMIT_DIR}/env.sh"
activate_env

# Shard from the array bounds (works with 0-49, 1-50 and any %throttle).
I=$(( ${SLURM_ARRAY_TASK_ID:-0} - ${SLURM_ARRAY_TASK_MIN:-0} ))
N=$(( ${SLURM_ARRAY_TASK_MAX:-0} - ${SLURM_ARRAY_TASK_MIN:-0} + 1 ))
[ "$N" -lt 1 ] && N=1
NPROC="${SLURM_CPUS_PER_TASK:-8}"
SRC=$(src_root $NET)
CFG=$NOISEPY/param_files/cluster/pipeline_${NET}_tspws.yaml

echo "[$NET/$STAGE] shard $I/$N | nproc $NPROC | $(date)"

case "$STAGE" in
build)
    python "$NOISEPY/scripts/picking/build_tspws_stacks.py" \
        --stack-root "$SRC" --out "$PROJ/$NET/stacks_tspws" \
        --nproc "$NPROC" --pre-block 2 --code "$(net_code $NET)" --shard "$I/$N"
    ;;
jackknife)
    python "$NOISEPY/scripts/picking/substack_jackknife.py" \
        --stack-root "$SRC" --out "$PROJ/$NET/substack_jackknife_k2" \
        --k 2 --nproc "$NPROC" --code "$(net_code $NET)" --shard "$I/$N"
    ;;
finalize)
    python "$NOISEPY/scripts/picking/run_pipeline.py" --config "$CFG" --stage all
    python "$NOISEPY/scripts/picking/attach_substack_sigma.py" \
        --picks-dir "$PROJ/$NET/tomo/1_velocity_maps/inputs_tspws" \
        --jackknife "$PROJ/$NET/substack_jackknife_k2"
    # Phase tables need their own pass: attach matches picks_<wave>_uni<suffix>.csv, so
    # without this the _phase tables keep the degenerate cross-method std (identically 0),
    # which a Tarantola-Valette inversion would read as zero data variance. --proxy labels
    # sigma_src *_group_proxy because the jackknife measures GROUP, not phase.
    python "$NOISEPY/scripts/picking/attach_substack_sigma.py" \
        --picks-dir "$PROJ/$NET/tomo/1_velocity_maps/inputs_tspws" \
        --jackknife "$PROJ/$NET/substack_jackknife_k2" --suffix _phase --proxy
    # One figure per MEASURE -- the group table has an empty suffix, so a single call wrote
    # a bare picks_hist2d.png that neither said which measure it was nor covered phase.
    for _m in "" "_phase"; do
        python "$NOISEPY/scripts/picking/plot_exported_picks_hist2d.py" \
            --dir "$PROJ/$NET/tomo/1_velocity_maps/inputs_tspws" --suffix "$_m" \
            --title "$NET ts-PWS production (vmin=0.2, jackknife sigma)"
    done
    python "$NOISEPY/scripts/picking/plot_sigma_census.py" \
        --jackknife "$PROJ/$NET/substack_jackknife_k2" --label "$NET"
    ;;
sigma)
    # Re-attach the substack sigma to the EXPORTED pick tables without re-picking. Use after
    # changing the sigma treatment (e.g. --winsor-pct); the finalize stage runs the same two
    # passes, but re-running finalize would redo QC and export for nothing.
    #   pass 1: group tables, measured jackknife sigma
    #   pass 2: _phase tables, same sigma as a labelled proxy (the jackknife re-picks GROUP
    #           velocity per block, so there is no phase sigma anywhere)
    python "$NOISEPY/scripts/picking/attach_substack_sigma.py" \
        --picks-dir "$PROJ/$NET/tomo/1_velocity_maps/inputs_tspws" \
        --jackknife "$PROJ/$NET/substack_jackknife_k2"
    python "$NOISEPY/scripts/picking/attach_substack_sigma.py" \
        --picks-dir "$PROJ/$NET/tomo/1_velocity_maps/inputs_tspws" \
        --jackknife "$PROJ/$NET/substack_jackknife_k2" --suffix _phase --proxy
    # One figure per MEASURE -- the group table has an empty suffix, so a single call wrote
    # a bare picks_hist2d.png that neither said which measure it was nor covered phase.
    for _m in "" "_phase"; do
        python "$NOISEPY/scripts/picking/plot_exported_picks_hist2d.py" \
            --dir "$PROJ/$NET/tomo/1_velocity_maps/inputs_tspws" --suffix "$_m" \
            --title "$NET ts-PWS production (vmin=0.2, jackknife sigma)"
    done
    ;;
maps)
    # Dense linear algebra (cm_exp_inverse does an explicit ncell^2 inverse; cho_factor and
    # the posterior solves are the same size), so BLAS threads matter: use every core Slurm
    # gave us instead of the hardcoded 4 this stage used to pin.
    export PYTHONPATH="$NOISEPY:$HOME/swtomotv/src${PYTHONPATH:+:$PYTHONPATH}"
    NTH="${SLURM_CPUS_PER_TASK:-4}"
    export OMP_NUM_THREADS=$NTH VECLIB_MAXIMUM_THREADS=$NTH \
           OPENBLAS_NUM_THREADS=$NTH MKL_NUM_THREADS=$NTH
    # Two Cd models, side by side into separate output roots:
    #   measured = per-datum tau_std^2 from the substack jackknife (what the sigma is FOR)
    #   blanket  = the legacy (0.10*tau)^2, every pick weighted equally
    # Until inversion.tv_two_step gained its cd argument, blanket was the only behaviour and
    # --rel-err changed only the reported chi2, so every earlier production map is blanket.
    # 12 combos = 2 measures x 2 Cd modes x 3 waves. Submitted WITHOUT --array they run
    # sequentially; submitted as an array each task takes a strided subset, which is the
    # faster way to use a lot of cores (BLAS threading saturates well before 32).
    # CDMODES lets a submission target a subset, e.g. --export=ALL,STAGE=maps,CDMODES=scaled
    # re-runs only the new variant instead of recomputing the finished ones.
    COMBOS=""
    for measure in group phase; do
        for cdmode in ${CDMODES:-measured blanket scaled}; do
            for wave in fund overtone love; do
                COMBOS="$COMBOS ${measure}:${cdmode}:${wave}"
            done
        done
    done
    k=0
    NFAIL=0
    for combo in $COMBOS; do
        if [ "$N" -gt 1 ] && [ $((k % N)) -ne "$I" ]; then k=$((k+1)); continue; fi
        k=$((k+1))
        measure=${combo%%:*}; rest=${combo#*:}; cdmode=${rest%%:*}; wave=${rest##*:}
        # Smoothing experiments write to a TAGGED root via a matching config, so the
        # production maps are never overwritten:
        #   LCTAG=_lc0.5  LCSCALE=0.5              relaxed Fresnel
        #   LCTAG=_lcfix1 LCMODE=fixed LCVAL=1.0   one LC for every period
        LCTAG="${LCTAG:-}"
        LCS="${LCSCALE:-1.0}"
        LCMODE="${LCMODE:-fresnel}"
        LCV="${LCVAL:-3.0}"
        CFG_T=$NOISEPY/param_files/cluster/tomo/${NET}_tspws_${measure}_${cdmode}${LCTAG}.yaml
        if [ "$LCMODE" = "fixed" ]; then
            LCFLAGS="--lc $LCV --lc-mode fixed"
        elif [ "$LCMODE" = "coverage" ]; then
            # LC per period = the smallest correlation length that still bridges the gaps
            # between adequately-sampled cells, measured over the imaging footprint.
            LCFLAGS="--lc $LCV --lc-mode coverage --lc-max 8.0 \
                     --lc-cov-frac ${LCCOVFRAC:-0.5} --lc-cov-q ${LCCOVQ:-98}"
        else
            LCFLAGS="--lc $LCV --lc-mode fresnel --lc-max 8.0 --lc-scale $LCS"
        fi
        echo "=== MAP $NET $measure/$cdmode :: $wave  ($NTH threads) ==="
        python "$NOISEPY/scripts/picking/run_production.py" --config "$CFG_T" \
            --wave "$wave" --cd-mode "$cdmode" $LCFLAGS \
            --mask-mode res --se 0.025 --fast --vplaus off --res-drop-q 0 \
            || { echo "FAILED $NET $measure $cdmode $wave"; NFAIL=$((NFAIL+1)); }
    done
    # `|| echo` alone made every task exit 0, so sacct reported COMPLETED for a run that had
    # actually died on its 4th period. Propagate the failure count to Slurm.
    [ "$NFAIL" -eq 0 ] || { echo "[$NET/maps] $NFAIL inversion(s) FAILED"; exit 1; }
    ;;
*)
    echo "unknown STAGE=$STAGE (build|jackknife|sigma|finalize|maps)"; exit 2 ;;
esac
echo "[$NET/$STAGE] done $(date)"
