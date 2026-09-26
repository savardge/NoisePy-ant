#!/bin/bash
# Rebuild the ts-PWS stack trees after the 2026-09-26 fixes, into a NEW tree per network:
#     $PROJ/<net>/stacks_tspws_v2      (Allstack_tspws with the inlined eq. 11 inverse + Allstack_linear)
# The production <net>.sbatch STAGE=build must NOT be used for this: it writes to
# $PROJ/<net>/stacks_tspws and skip-if-exists would keep every old-formula stack.
#
#   aargau, hautesorne : full rebuild (their stacks_tspws were cluster-built with pycwt 0.3.0a22,
#                        i.e. carry the sqrt(f) icwt tilt)
#   riehen             : --reuse-tspws $PROJ/riehen/stacks_tspws (laptop-built with eq. 11 = already
#                        correct); only Allstack_linear is computed. Only the pairs whose substacks
#                        are on the cluster (12,999 of 19,503); the rest is a laptop run.
#
# Submit from this directory (logs/ must exist):
#   sbatch --array=0-49%25   --export=ALL,NET=aargau     rebuild_tspws_v2.sbatch
#   sbatch --array=0-199%50  --export=ALL,NET=hautesorne rebuild_tspws_v2.sbatch
#   sbatch --array=0-19%20   --export=ALL,NET=riehen     rebuild_tspws_v2.sbatch
# Skip-if-exists per pair: a task that dies is fixed by resubmitting the same array.
# Evidence: extract_higher_modes/Projects/method_tests/2_pick_qc/test_2026-09-26_tspws_snr_noise_window/
#SBATCH --job-name=tspws_v2
#SBATCH --partition=shared-cpu
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/tspws_v2_%x_%A_%a.out
#SBATCH --error=logs/tspws_v2_%x_%A_%a.err
set -eu
. "${CLUSTER_DIR:-$SLURM_SUBMIT_DIR}/env.sh"
activate_env

I=$(( ${SLURM_ARRAY_TASK_ID:-0} - ${SLURM_ARRAY_TASK_MIN:-0} ))
N=$(( ${SLURM_ARRAY_TASK_MAX:-0} - ${SLURM_ARRAY_TASK_MIN:-0} + 1 ))
[ "$N" -lt 1 ] && N=1
NPROC="${SLURM_CPUS_PER_TASK:-8}"
OUT=$PROJ/$NET/stacks_tspws_v2
REUSE=""
[ "$NET" = riehen ] && REUSE="--reuse-tspws $PROJ/riehen/stacks_tspws"
echo "[$NET v2] shard $I/$N | nproc $NPROC | $(date) | pycwt $(python -c 'import pycwt;print(pycwt.__version__)') (inverse inlined) | out $OUT $REUSE"
python "$NOISEPY/scripts/picking/build_tspws_stacks.py" \
    --stack-root "$(src_root $NET)" --out "$OUT" --nproc "$NPROC" --pre-block 2 \
    --code "$(net_code $NET)" --shard "$I/$N" $REUSE
echo "[$NET v2] shard $I/$N done $(date)"
