#!/bin/sh
# Shared environment for the yggdrasil Slurm jobs. Sourced by <net>.sbatch and preflight.sh.
#
# EDIT THESE THREE if your layout differs, then re-run make_cluster_configs.py so the YAML
# paths match (the sbatch scripts read paths from here, the picker reads them from the YAML
# -- they must agree, and preflight.sh checks that they do).
CDFF=/srv/beegfs/scratch/shares/cdff
export NOISEPY="${NOISEPY:-$CDFF/savardg/NoisePy-ant}"
export PROJ="${PROJ:-$CDFF/savardg/extract_higher_modes/Projects}"
export CONDA_ENV="${CONDA_ENV:-das-ambient-noise}"

# SOURCE substack trees already on the cluster (inputs; never written to)
SRC_riehen=$CDFF/riehen/crosscorrelations/STACK_CHRI_normZ
SRC_aargau=$CDFF/aargau/crosscorrelations/STACK_CHAA_normZ
SRC_hautesorne=$CDFF/hautesorne/STACK_coh_60s_substacks

# station-code prefix; the pair glob is <code>.*/<code>.*_<code>.*.h5
CODE_riehen=RI
CODE_aargau=AA
CODE_hautesorne=SS

src_root()  { eval echo \$SRC_$1; }
net_code()  { eval echo \$CODE_$1; }

activate_env() {
    if [ -z "$CONDA_EXE" ] && command -v conda >/dev/null 2>&1; then
        CONDA_EXE=$(command -v conda)
    fi
    if [ -n "$CONDA_EXE" ]; then
        . "$("$CONDA_EXE" info --base)/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV" || { echo "FATAL: cannot activate $CONDA_ENV"; exit 1; }
    else
        echo "FATAL: conda not found; module load it first"; exit 1
    fi
    export PYTHONPATH="$NOISEPY${PYTHONPATH:+:$PYTHONPATH}"
    # Each Slurm task already owns its cores via the Pool; stop BLAS from oversubscribing.
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
}
