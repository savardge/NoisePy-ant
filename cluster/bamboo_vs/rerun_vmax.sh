#!/bin/bash
# Re-run the PHASE arms with a wider Vs prior (user decision 2026-08-16: vs_max should have
# been 4.0, not the driver default 3.6).
#
# Why only the phase arms: measured on the finished volumes, with the reach mask applied, the
# fraction of cells touching 3.58 km/s is
#     aargau/R0p    median 53.4%   p84 84.1%   p975 93.3%
#     aargau/L0p    median 23.1%   p84 85.4%   p975 93.4%
#     hautesorne/L0p median 34.0%  p84 79.2%   p975 90.3%
#     riehen/R0p    median  3.3%   p84 17.5%   p975 39.6%
#     riehen/L0p    median  2.7%   p84 11.7%   p975 28.8%
#     every GROUP arm      0.0%         <=0.6%       <=4.2%
# so the group and joint arms are genuinely unaffected -- their posteriors never approach the
# bound and a wider prior would change nothing. Riehen's phase medians are mostly fine but its
# CREDIBLE BOUNDS are clipped in ~40% of cells, and a uniform prior across networks is required
# for any cross-network comparison, so all six phase arms are re-run.
#
# Output goes to a SEPARATE <cfg>_vmax4.0 directory: writing into the existing tree would be
# skipped cell-for-cell by skip-if-exists and would silently produce nothing, and the 3.6
# results are worth keeping as a direct truncation comparison.
#
# Deliberately throttled low so this does not starve the ~8% of the original campaign still
# finishing. Raise THROTTLE once the main arrays drain.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/env_bamboo.sh"

VSMAX=${VSMAX:-4.0}
N_SHARDS=${N_SHARDS:-32}
THROTTLE=${THROTTLE:-4}
TARGETS=${1:-"aargau:R0p aargau:L0p hautesorne:L0p hautesorne:R0p riehen:R0p riehen:L0p"}

mkdir -p "$SCRATCH_VS/logs" "$SCRATCH_VS/runs"
MANIFEST=$SCRATCH_VS/runs/rerun_vmax${VSMAX}_$(date +%Y%m%d_%H%M%S).tsv
echo -e "net\tcfg\tjobid\tvsmax\toutdir" > "$MANIFEST"

for t in $TARGETS; do
    NET=${t%%:*}; CFG=${t##*:}
    OUT=$RUNS/$NET/${CFG}_vmax${VSMAX}
    JID=$(sbatch --parsable \
        --job-name=vsprod3 \
        --partition=public-cpu \
        --time=2-00:00:00 \
        --array=0-$((N_SHARDS - 1))%$THROTTLE \
        --export=ALL,NET=$NET,CFG=$CFG,N_SHARDS=$N_SHARDS,EXTRA_FLAGS="--vs-max $VSMAX",OUTDIR_OVERRIDE=$OUT,BAMBOO_VS_DIR=$HERE \
        --output="$SCRATCH_VS/logs/vmax${VSMAX}_${NET}_${CFG}_%A_%a.out" \
        "$HERE/vs_prod3.sbatch")
    echo -e "$NET\t$CFG\t$JID\t$VSMAX\t$OUT" | tee -a "$MANIFEST"
done
echo "manifest: $MANIFEST"
echo "stop with: scancel \$(cut -f3 $MANIFEST | tail -n +2 | tr '\\n' ' ')"
