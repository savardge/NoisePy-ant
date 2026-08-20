# Shared paths + activation for the bamboo Vs-inversion campaign (vs_prod3).
# Sourced by vs_prod3.sbatch and submit_all.sh. All paths live HERE and nowhere else.

export NOISEPY="${NOISEPY:-$HOME/NoisePy-ant}"
export SCRATCH_VS="${SCRATCH_VS:-/srv/beegfs/scratch/users/s/savardg/vs_prod3}"
export INPUTS="$SCRATCH_VS/inputs"
export RUNS="$SCRATCH_VS/runs"
export BH_PY="${BH_PY:-$HOME/.conda/envs/bayhunter_aniso/bin/python}"

activate_env() {
    # conda hooks are NOT `set -u`/`set -e` clean; relax, activate, restore.
    _flags=$-
    set +eu
    source /opt/ebsofts/Anaconda3/2024.02-1/etc/profile.d/conda.sh
    conda activate bayhunter_aniso
    case "$_flags" in *e*) set -e ;; esac
    case "$_flags" in *u*) set -u ;; esac
    export PYTHONPATH="$NOISEPY${PYTHONPATH:+:$PYTHONPATH}"
    # chains are forked per cell; keep BLAS single-threaded (fork + threaded BLAS hazard)
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
}

# per-network inputs: production roots (group + phase) and the swtomotv dataset YAML
net_paths() {           # usage: net_paths <net>  -> sets GROUP_PROD, PHASE_PROD, CONFIG_YAML
    local net=$1 dx
    case "$net" in
        riehen)     dx=0.2; CONFIG_YAML=$INPUTS/riehen/riehen_unified_tomo_200m_prod.yaml ;;
        aargau)     dx=0.5; CONFIG_YAML=$INPUTS/aargau/aargau_unified_tomo_500m.yaml ;;
        hautesorne) dx=0.5; CONFIG_YAML=$INPUTS/hautesorne/hautesorne_unified_tomo_ffv2.yaml ;;
        *) echo "net_paths: unknown net '$net'" >&2; return 2 ;;
    esac
    GROUP_PROD=$INPUTS/$net/tspws_group_blanket_dx${dx}_prod3_k3/production
    PHASE_PROD=$INPUTS/$net/tspws_phase_blanket_dx${dx}_prod3_k3/production
}

# config -> production root + waveset + extra driver flags.
# The radial prior stays at the driver default (-0.35,0.35 from the CZ calibration).
cfg_flags() {           # usage: cfg_flags <cfg>  -> sets PROD, WS, EXTRA
    local cfg=$1
    case "$cfg" in
        R0g)        PROD=$GROUP_PROD; WS=fund;     EXTRA="" ;;
        R0p)        PROD=$PHASE_PROD; WS=fund;     EXTRA="--measure phase" ;;
        L0g)        PROD=$GROUP_PROD; WS=love;     EXTRA="" ;;
        L0p)        PROD=$PHASE_PROD; WS=love;     EXTRA="--measure phase" ;;
        RLg_radial) PROD=$GROUP_PROD; WS=fundlove; EXTRA="--radial" ;;
        RLg_iso)    PROD=$GROUP_PROD; WS=fundlove; EXTRA="" ;;
        *) echo "cfg_flags: unknown cfg '$cfg'" >&2; return 2 ;;
    esac
}

ALL_CFGS="R0g R0p L0g L0p RLg_radial RLg_iso"
ALL_NETS="riehen aargau hautesorne"

# ---------------------------------------------------------------- Dinver (SWinvert) campaign
# Sibling of vs_prod3 on scratch; inputs are the SAME blanket prod3_k3 trees (symlinked, not
# copied). Outputs: runs/<net>/<cfg>/cells/*.npz (lean, ~50 KB each) + volume; nothing else.
export SCRATCH_DINVER="${SCRATCH_DINVER:-/srv/beegfs/scratch/users/s/savardg/vs_dinver}"
export DINVER_RUNS="$SCRATCH_DINVER/runs"

activate_dinver() {
    activate_env                       # bayhunter_aniso: numpy/disba/swprepost + PYTHONPATH
    _flags=$-
    set +eu
    module load GCC/11.3.0 OpenMPI/4.1.4 geopsy/3.4.2
    _rc=$?
    case "$_flags" in *e*) set -e ;; esac
    case "$_flags" in *u*) set -u ;; esac
    [ "$_rc" -eq 0 ] || { echo "FATAL: cannot load geopsy/3.4.2"; exit 1; }
    export DINVER_BIN="$EBROOTGEOPSY/bin/dinver"
    export GPDCREPORT_BIN="$EBROOTGEOPSY/bin/gpdcreport"
    # dinver is GUI-linked. On any fatal error -- and on the SIGTERM the driver sends at
    # --cell-timeout -- CoreApplicationPrivate spawns a GUI copy of itself (-reportbug /
    # -reportint) to render a bug report. Without a usable Qt platform plugin that child
    # aborts in ~100 ms and the parent's diagnostic is lost, so a timed-out cell leaves
    # only an opaque abort. offscreen keeps timeouts readable; verified not to affect
    # -optimization or -plugin-list.
    export QT_QPA_PLATFORM=offscreen
}

# Dinver configs -> waveset + measure + sizing. R0gR1g = the arm the well study validated
# (both group curves at chi~1, basement recovered). Group runs size the SWinvert layering
# from the fundamental PHASE wavelength (--dinver-size-phase-root), phase is not inverted.
dinver_cfg_flags() {    # usage: dinver_cfg_flags <cfg> -> sets PROD, WS, DEXTRA
    local cfg=$1
    case "$cfg" in
        R0gR1g) PROD=$GROUP_PROD; WS=fundot; DEXTRA="--measure group --dinver-size-phase-root $PHASE_PROD" ;;
        R0g)    PROD=$GROUP_PROD; WS=fund;   DEXTRA="--measure group --dinver-size-phase-root $PHASE_PROD" ;;
        *) echo "dinver_cfg_flags: unknown cfg '$cfg'" >&2; return 2 ;;
    esac
}
