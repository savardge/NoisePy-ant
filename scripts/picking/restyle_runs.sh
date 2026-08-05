#!/bin/zsh
# Regenerate styled maps for velocity-map run directories (2026-08-05 workflow layout).
#
#   restyle_runs.sh <net> <dx> <subpath> [<subpath> ...]
#   restyle_runs.sh riehen 0.2 1_production
#   restyle_runs.sh aargau 0.5 2_superseded/prod2_masked
#
# Each <subpath> is a directory under Projects/<net>/tomo/1_velocity_maps/ that contains
# tspws_* run dirs. Styling always uses the CURRENT styled_maps.py conventions (inferno,
# corrected hillshade, distribution strip, veil merged back from vel_hidden, interior
# speckle holes closed from vel_full), so restyling an old run is the supported way to
# view it with current fixes -- the npz are never modified.
setopt NULL_GLOB
net=$1; dx=$2; shift 2
EHM=/Users/genevievesavard/Codes/extract_higher_modes
V=$EHM/Projects/$net/tomo/1_velocity_maps
export PYTHONPATH=/Users/genevievesavard/Codes/NoisePy-ant:/Users/genevievesavard/Codes/swtomotv/src
for sub in "$@"; do
  for R in $V/$sub/tspws_*_dx${dx}*; do
    [ -d "$R/production" ] || { echo "SKIP  $sub/$(basename $R)"; continue; }
    /opt/anaconda3/envs/nant/bin/python \
      /Users/genevievesavard/Codes/NoisePy-ant/scripts/picking/styled_maps.py $net \
      --run-root $net=$R/production \
      --picks-dir $V/0_inputs/exported_picks_tspws 2>&1 \
      | grep -E "no pick|Traceback|Error"
    echo "  done $sub/$(basename $R)"
  done
done
echo "FINISHED $net"
