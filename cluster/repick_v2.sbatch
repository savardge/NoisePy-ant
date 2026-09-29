#!/bin/bash
# Re-pick a network on the rebuilt stacks_tspws_v2 tree with one picking/SNR option, as a job array.
#   ARM=A  pick on Allstack_linear trimmed to the ts-PWS length (DISP_PICK_TRIM_VMIN=0.2); SNR = linear,
#          window max, 'after' noise window                                    -> ..._tspwsv2_A_linear
#   ARM=D  pick on Allstack_tspws; SNR = linear stack AT THE PICK (DISP_SNR_AT_PICK=1) -> ..._tspwsv2_D_snrpick
# Each task takes files[I::N] of the v2 tree via a node-local symlink tree (the picker globs its
# stack_root), so tasks never collide; the picker is skip-if-exists, so a failed task is fixed by
# resubmitting the same array. Picker parameters = the production picker YAML with only stack_root
# changed (riehen: the pickerKIN regional YAML = Rayleigh W/E KIN block only).
#   sbatch --array=0-49%50 --export=ALL,NET=hautesorne,ARM=A repick_v2.sbatch
# Evidence: extract_higher_modes/Projects/method_tests/2_pick_qc/test_2026-09-26_stack_method_and_snr_gate/
#SBATCH --job-name=repick_v2
#SBATCH --partition=shared-cpu
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/repick_v2_%x_%A_%a.out
#SBATCH --error=logs/repick_v2_%x_%A_%a.err
set -eu
. "${CLUSTER_DIR:-$SLURM_SUBMIT_DIR}/env.sh"
activate_env
I=$(( ${SLURM_ARRAY_TASK_ID:-0} - ${SLURM_ARRAY_TASK_MIN:-0} ))
N=$(( ${SLURM_ARRAY_TASK_MAX:-0} - ${SLURM_ARRAY_TASK_MIN:-0} + 1 ))
[ "$N" -lt 1 ] && N=1
NPROC="${SLURM_CPUS_PER_TASK:-16}"
CODE=$(net_code "$NET")
V2=$PROJ/$NET/stacks_tspws_v2
case "$NET" in
  riehen) BASE=$PROJ/$NET/modesep_riehen_pickerKIN_v2base.yaml ;;
  *)      BASE=$NOISEPY/param_files/cluster/modesep_${NET}_tspws.yaml ;;
esac
case "$ARM" in
  A) export DISP_STACK=linear DISP_PICK_TRIM_VMIN=0.2; OUT=$PROJ/$NET/dispersion_unified_vmin0.2_tspwsv2_A_linear ;;
  D) export DISP_STACK=tspws DISP_SNR_AT_PICK=1;        OUT=$PROJ/$NET/dispersion_unified_vmin0.2_tspwsv2_D_snrpick ;;
  *) echo "ARM must be A or D"; exit 2 ;;
esac
export DISP_VMIN=0.2
unset DISP_LOVE_OT DISP_SNR_STACK DISP_SNR_NOISE || true
T=${TMPDIR:-/tmp}/repick_${NET}_${ARM}_${SLURM_JOB_ID}_$I
mkdir -p "$T/stacks"
python - "$V2" "$CODE" "$I" "$N" "$T/stacks" <<'PY'
import glob, os, sys
v2, code, i, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
files = sorted(glob.glob(os.path.join(v2, "%s.*" % code, "%s.*_%s.*.h5" % (code, code))))[i::n]
for f in files:
    d = os.path.join(out, os.path.basename(os.path.dirname(f)))
    os.makedirs(d, exist_ok=True)
    os.symlink(f, os.path.join(d, os.path.basename(f)))
print("[shard] %d/%d: %d pairs" % (i, n, len(files)))
PY
# ref_dir is set EXPLICITLY: the aargau/hautesorne picker YAMLs have ref_dir: null, which resolves to
# {project_dir}/vsg_modesep on the old /srv/beegfs/.../cdff/savardg tree -- gone since the July move to
# $HOME. The picker then only WARNs and disables phase for every stream (2026-09-27: first A/D launch
# was group-only for AA and HS). Missing reference curves are now fatal.
REF=$PROJ/$NET/vsg_modesep
for r in ref_fundamental_phase.txt ref_overtone_phase.txt ref_love_phase.txt ref_love_overtone_phase.txt; do
    [ -s "$REF/$r" ] || { echo "FATAL: missing reference $REF/$r"; exit 3; }
done
python - "$BASE" "$T/stacks" "$REF" "$T/cfg.yaml" <<'PY'
import sys, yaml
base, root, ref, out = sys.argv[1:5]
cfg = yaml.safe_load(open(base))
cfg["paths"]["stack_root"] = root
cfg["paths"]["ref_dir"] = ref
yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
PY
echo "[$NET/$ARM] shard $I/$N | $(date) | out $OUT | base $BASE"
python "$NOISEPY/scripts/picking/dispersion_unified.py" --config "$T/cfg.yaml" --out "$OUT" --nproc "$NPROC" \
    > "$T/picker.log" 2>&1 || { cat "$T/picker.log"; exit 4; }
grep -v "^  [0-9]*/" "$T/picker.log"
if grep -q "could not load" "$T/picker.log"; then echo "FATAL: a reference curve failed to load"; exit 5; fi
rm -rf "$T"
echo "[$NET/$ARM] shard $I/$N done $(date)"
