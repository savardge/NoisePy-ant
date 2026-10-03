#!/bin/bash
# vs_prod4: the vs_prod3 PRODUCTION arms re-run with ONLY the map roots changed, on the option-D
# production maps (stacks_tspws_v2, ts-PWS picks, linear-stack SNR at the pick, --topo; user decision
# 2026-09-30). Plan: Projects/method_tests/5_vs_inversion/plan_2026-10-01_vs_campaign_on_option_D_maps/.
#
#   ./launch_vs_prod4.sh <net> <arm> <cluster> smoke
#   ./launch_vs_prod4.sh <net> <arm> <cluster> pilot              (well cells, --save-ensemble, 1 cell/task)
#   ./launch_vs_prod4.sh <net> <arm> <cluster> grid <lo> <hi> <throttle>   (1 cell/task, disjoint ranges)
#   ./launch_vs_prod4.sh <net> <arm> <cluster> assemble [EXP=<n>] [FORCE=1]  (cells/ -> volume_<ws>.npz)
#
# arms: aargau RLp_radial_Dp_ff15_domesc | hautesorne R0gp_g5_p6_g108_p03_hankel_ff1.5_topo_aniso2t,
#       hautesorne R0gp_g5_p6_g108_p03_hankel_ff1.5_topo (isotropic-root twin) | riehen R0p_pickerKIN
#       hautesorne R0gp_g5_p6_g108_p03_gcap2.5_hankel_ff1.5_topo_aniso2t (gate pilot: group band capped 2.5 s)
# Everything else (chains, iterations, free sigma, vs-max, bands, no rebasin) = the vs_prod3 twin.
# Scratch root vs_prod4 (never vs_prod3); outputs runs/<net>/<arm>; smoke -> smoke/<net>/<arm>.
set -eu
NET=${1:?net}; ARM=${2:?arm}; CL=${3:?bamboo|baobab|yggdrasil}; MODE=${4:?smoke|pilot|grid|assemble}
export SCRATCH_VS=/srv/beegfs/scratch/users/s/savardg/vs_prod4
case $CL in
  bamboo) H=$HOME/NoisePy-ant/cluster/bamboo_vs; NPY=$HOME/NoisePy-ant ;;
  *)      H=$HOME/NoisePy-ant_bamboo/cluster/bamboo_vs; NPY=$HOME/NoisePy-ant_bamboo ;;
esac
export NOISEPY=$NPY
. "$H/env_bamboo.sh"
DX=0.5; [ "$NET" = riehen ] && DX=0.2
case "$NET:$ARM" in
  aargau:RLp_radial_Dp_ff15_domesc)
    CFG=RLp_radial
    export GROUP_TREE=tspws_group_blanket_dx0.5_prod3_k3_tspwsv2_D_snrpick_topo
    export PHASE_TREE=tspws_phase_blanket_dx0.5_prod3_k3_hankel_ff1.5_tspwsv2_D_snrpick_topo
    FLAGS="--period-ranges $INPUTS/period_ranges_percell_Dp_ff15_ag.csv --phase-tmin 0"
    CHECK="phase:fund phase:love"; CELLS_PILOT="${AARGAU_WELL_CELLS:?set AARGAU_WELL_CELLS}" ;;   # phase-only: the driver reads the phase root only
  aargau:RLp_radial_Dp_ff15_Lp04_domesc)
    # 2026-10-02 one-knob pilot (tests/test_2026-10-02_riniken_shallow_mask): Love PHASE floor 0.30 -> 0.40 s
    CFG=RLp_radial
    export GROUP_TREE=tspws_group_blanket_dx0.5_prod3_k3_tspwsv2_D_snrpick_topo
    export PHASE_TREE=tspws_phase_blanket_dx0.5_prod3_k3_hankel_ff1.5_tspwsv2_D_snrpick_topo
    FLAGS="--period-ranges $INPUTS/period_ranges_percell_Dp_ff15_Lp04_ag.csv --phase-tmin 0"
    CHECK="phase:fund phase:love"; CELLS_PILOT="26_41,20_28,15_22,11_28,16_16,35_45,33_45,31_46,23_46,35_47" ;;
  hautesorne:R0gp_g5_p6_g108_p03_hankel_ff1.5_topo_aniso2t|hautesorne:R0gp_g5_p6_g108_p03_hankel_ff1.5_topo)
    CFG=R0gp_g5
    SFX=""; [ "${ARM##*_}" = aniso2t ] && SFX=_aniso2t
    export GROUP_TREE=tspws_group_blanket_dx0.5_prod3_k3_tspwsv2_D_snrpick_topo$SFX
    export PHASE_TREE=tspws_phase_blanket_dx0.5_prod3_k3_hankel_ff1.5_tspwsv2_D_snrpick_topo$SFX
    net_paths "$NET"
    # last-wins over cfg R0gp_g5's own --period-ranges/--phase-tmin 2 (as launch_hs_dprime_topo.sh)
    FLAGS="--production $GROUP_PROD --phase-root $PHASE_PROD --period-ranges $INPUTS/period_ranges_g108_p03_p6_hs.csv --phase-tmin 0"
    CHECK="group:fund group:overtone phase:fund phase:overtone"; CELLS_PILOT="41_21,41_20,40_21,42_21" ;;   # the driver loads fund+overtone from BOTH roots
  hautesorne:R0gp_g5_p6_g108_p03_gcap2.5_hankel_ff1.5_topo_aniso2t)
    # gate pilot 2026-10-01 (plan README §7 option 2): = production HS arm with the group band capped at 2.5 s
    CFG=R0gp_g5
    export GROUP_TREE=tspws_group_blanket_dx0.5_prod3_k3_tspwsv2_D_snrpick_topo_aniso2t
    export PHASE_TREE=tspws_phase_blanket_dx0.5_prod3_k3_hankel_ff1.5_tspwsv2_D_snrpick_topo_aniso2t
    net_paths "$NET"
    FLAGS="--production $GROUP_PROD --phase-root $PHASE_PROD --period-ranges $INPUTS/period_ranges_g108_p03_p6_gcap25_hs.csv --phase-tmin 0"
    CHECK="group:fund group:overtone phase:fund phase:overtone"; CELLS_PILOT="41_21,41_20,40_21,42_21" ;;
  riehen:R0p_pickerKIN)
    CFG=R0p
    export GROUP_TREE=tspws_group_blanket_dx0.2_prod3_k3_tspwsv2_D_snrpick_topo
    export PHASE_TREE=tspws_phase_blanket_dx0.2_prod3_k3_hankel_tspwsv2_D_snrpick_topo
    FLAGS="--vs-max 4.5"                      # bands = sbatch COMMON default period_ranges_DECISIONS_v1
    CHECK="phase:fund"; CELLS_PILOT="22_46,25_42,43_47,45_51" ;;
  *) echo "unknown net:arm $NET:$ARM"; exit 2 ;;
esac
net_paths "$NET"
for rw in $CHECK; do
  r=${rw%%:*}; w=${rw#*:}; d=$PHASE_PROD; [ "$r" = group ] && d=$GROUP_PROD
  [ -d "$d/$w" ] || { echo "MISSING on $CL: $d/$w"; exit 1; }
  n=$(ls $d/$w/map_T*.npz 2>/dev/null | wc -l); [ "$n" -ge 25 ] || { echo "only $n maps in $d/$w"; exit 1; }
  python3 -c "import zipfile,glob,sys; f=sorted(glob.glob('$d/$w/map_T*.npz'))[0]; sys.exit(0 if 'topo.npy' in zipfile.ZipFile(f).namelist() else 1)" \
    || { echo "NOT a --topo tree: $d/$w"; exit 1; }
done
[ -f "$CONFIG_YAML" ] || { echo "MISSING $CONFIG_YAML"; exit 1; }
FLAGS_ESC=${FLAGS//,/@C@}
PART="shared-cpu,public-cpu"; [ "$CL" = yggdrasil ] && PART="shared-cpu"
mkdir -p $SCRATCH_VS/logs
case $MODE in
  smoke)
    OUT=$SCRATCH_VS/smoke/$NET/$ARM; mkdir -p $OUT
    JID=$(sbatch --parsable --job-name=vp4smoke --partition=debug-cpu --time=00:15:00 --cpus-per-task=8 --mem-per-cpu=3G \
      --export=ALL,NOISEPY=$NPY,SCRATCH_VS=$SCRATCH_VS,GROUP_TREE=$GROUP_TREE,PHASE_TREE=$PHASE_TREE,NET=$NET,CFG=$CFG,N_SHARDS=1,NCHAINS=3,BURN=5000,MAIN=2000,CELL_TIMEOUT=600,LIMIT=1,EXTRA_FLAGS="$FLAGS_ESC",OUTDIR_OVERRIDE=$OUT,BAMBOO_VS_DIR=$H \
      --output=$SCRATCH_VS/logs/smoke_${NET}_${ARM}_${CL}_%j.out $H/vs_prod3.sbatch) ;;
  pilot)
    OUT=$SCRATCH_VS/runs/$NET/$ARM; mkdir -p $OUT
    [ -z "$(ls -A $OUT/cells 2>/dev/null)" ] || { echo "REFUSING: $OUT/cells not empty"; exit 1; }
    NC=$(echo $CELLS_PILOT | tr ',' '\n' | wc -l)
    JID=$(sbatch --parsable --job-name=vp4pilot --partition=$PART --time=06:00:00 --cpus-per-task=24 --mem-per-cpu=3G --array=0-$((NC-1)) \
      --export=ALL,NOISEPY=$NPY,SCRATCH_VS=$SCRATCH_VS,GROUP_TREE=$GROUP_TREE,PHASE_TREE=$PHASE_TREE,NET=$NET,CFG=$CFG,N_SHARDS=$NC,CELLS=${CELLS_PILOT//,/@C@},EXTRA_FLAGS="$FLAGS_ESC --save-ensemble",OUTDIR_OVERRIDE=$OUT,BAMBOO_VS_DIR=$H \
      --output=$SCRATCH_VS/logs/pilot_${NET}_${ARM}_${CL}_%A_%a.out $H/vs_prod3.sbatch) ;;
  grid)
    LO=${5:?lo}; HI=${6:?hi}; THR=${7:?throttle}; NCELLS=${NCELLS:?export NCELLS (>= cell count, identical on every cluster)}
    OUT=$SCRATCH_VS/runs/$NET/$ARM; mkdir -p $OUT
    JID=$(sbatch --parsable --job-name=vp4grid --partition=$PART --time=03:00:00 --cpus-per-task=24 --mem-per-cpu=3G --array=${LO}-${HI}%${THR} \
      --export=ALL,NOISEPY=$NPY,SCRATCH_VS=$SCRATCH_VS,GROUP_TREE=$GROUP_TREE,PHASE_TREE=$PHASE_TREE,NET=$NET,CFG=$CFG,N_SHARDS=$NCELLS,EXTRA_FLAGS="$FLAGS_ESC",OUTDIR_OVERRIDE=$OUT,BAMBOO_VS_DIR=$H \
      --output=$SCRATCH_VS/logs/grid_${NET}_${ARM}_${CL}_%A_%a.out $H/vs_prod3.sbatch) ;;
  assemble)
    # Cells written by this campaign already carry the cluster rule + DOMINANCE_LOGL=50, so no
    # --reselect-chains (it changed 0/1767 cells on the vs_prod3 twin). Aargau keeps the recipe's
    # assembly gate --min-obs-rungs 18 (29 rung-starved rim cells on vs_prod3). Expected cell count =
    # the largest "N cells x" the driver printed in this arm's grid logs (pilots/smoke print fewer).
    OUT=$SCRATCH_VS/runs/$NET/$ARM
    HAVE=$(ls "$OUT/cells" 2>/dev/null | grep -c '\.npz$')
    EXP=${EXP:-$(grep -h -o '[0-9]\+ cells x' $SCRATCH_VS/logs/grid_${NET}_${ARM}_*.out 2>/dev/null | cut -d' ' -f1 | sort -n | tail -1)}
    [ -n "${EXP:-}" ] || { echo "cannot determine expected cell count -- pass EXP=<n>"; exit 1; }
    echo "$NET/$ARM: $HAVE / $EXP cell npz"
    [ "${FORCE:-0}" = 1 ] || [ "$HAVE" -ge "$EXP" ] || { echo "REFUSING: $HAVE/$EXP cells (top up first, or FORCE=1)"; exit 1; }
    [ "$(ls $OUT/volume_*.npz 2>/dev/null | wc -l)" -eq 0 ] || { echo "REFUSING: $OUT already has a volume"; exit 1; }
    AFLAGS="$FLAGS_ESC"; [ "$NET" = aargau ] && AFLAGS="$AFLAGS --min-obs-rungs 18"
    JID=$(sbatch --parsable --job-name=vp4asm --partition=$PART --time=02:00:00 --cpus-per-task=2 --mem-per-cpu=8G \
      --export=ALL,NOISEPY=$NPY,SCRATCH_VS=$SCRATCH_VS,GROUP_TREE=$GROUP_TREE,PHASE_TREE=$PHASE_TREE,NET=$NET,CFG=$CFG,STAGE=assemble,EXTRA_FLAGS="$AFLAGS",OUTDIR_OVERRIDE=$OUT,BAMBOO_VS_DIR=$H \
      --output=$SCRATCH_VS/logs/asm_${NET}_${ARM}_${CL}_%j.out $H/vs_prod3.sbatch) ;;
  *) echo "MODE smoke|pilot|grid|assemble"; exit 2 ;;
esac
echo "$CL $NET $ARM $MODE: job $JID -> $OUT"
echo "   group $GROUP_PROD"; echo "   phase $PHASE_PROD"; echo "   flags $FLAGS"
echo -e "$NET\t$CFG\t$ARM($MODE $CL ${5:-} ${6:-})\t$JID\t$OUT" >> $SCRATCH_VS/runs/arms_launched.tsv
