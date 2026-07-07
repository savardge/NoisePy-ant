"""
Dispersion picking V6 -- V5 + Nayak & Thurber (2020) multi-component mode separation.

Everything V5 does (group velocity by CWT-FTAN ridge tracking on ZZ/RR/RZ/ZR/TT single components
and amplitude-product combos, Shapiro/Levshin period corrections, Bensen phase velocity), plus a
phase-aware mode-separation step:

  * G_LR0 / G_LR1 -- the fundamental and 1st-higher-mode Rayleigh traces synthesized from the four
    radial-vertical cross-correlations via the +-pi/2 phase corrections of Nayak & Thurber (2020)
    eqs (3)/(4) (see noisepy.dispersion.synthesize_rayleigh_modes). These are fed through the SAME
    CWT-FTAN + picking pipeline as any component, so the ridge picker follows one mode at a time.
    Picks are written with component = 'G_LR0' (fundamental) / 'G_LR1' (1st higher), wave 'rayleigh'.

The amplitude-product combos (all4, ZZ-ZR, RR-RZ) are kept alongside G_LR0/G_LR1 for comparison.
Validate / QA-QC the modes against the single-component and combo picks with validate_modes.py.

Reading: uses pyasdf if installed, else falls back to plain h5py (ASDF is HDF5), so the script runs
in environments without pyasdf as long as pycwt + findpeaks are present.

Usage:  python dispersion_curves_V6_modesep.py <stacked_asdf_file> [--config <network_yaml>]
        (--config overrides only the reference-curve dir + output root; env DISP_OUTPUT_ROOT wins)
"""
import numpy as np
import os
import sys
from noisepy import dispersion

# ----------------------------------------------------------------------------- config
stack_methods = ['pws', 'linear', 'nroot']           # add 'robust', 'auto_covariance' if present
lag_types = ['neg', 'pos', 'sym']
pick_methods = ["argmax", "topology"]

# Components we pick on (single-component, full corrections + phase velocity) and their wave type
PICK_COMPONENTS = {"ZZ": "rayleigh", "RR": "rayleigh", "RZ": "rayleigh", "ZR": "rayleigh",
                   "TT": "love"}
# Extra components loaded only for Love SNR context (Bensen/V4 'other' columns)
LOVE_CONTEXT = ["TZ", "ZT", "RT", "TR"]

# Nayak & Thurber mode-separated Rayleigh traces (synthesized, then picked like a component).
GLR_LAG = 'sym'                                       # paper uses the symmetric component
# G_LR1 eigenfunction-sign assumption: False = eq (4), flip at BOTH stations (same-medium; right
# for Aargau where both stations sit in the Molasse). True = the paper's section-3.1 variant, flip
# at the receiver only (use for pairs straddling a strong structural boundary).
GLR_RECEIVER_SIDE_FLIP = False
# How to stack the four phase-corrected components into G_LR0/G_LR1:
#   'tfpws'  = t-f phase-weighted stack (Schimmel & Gallart 2007 / Ventosa 2017), what Nayak &
#              Thurber use on real data -- suppresses wave packets not in phase across the four
#              components (incoherent noise, non-elliptical arrivals);
#   'linear' = plain sum, eqs (3)/(4) exactly (the synthetic-test form).
GLR_STACK = 'tfpws'

# Period / velocity grid (matches V3/V4/V5)
Tmin = 0.2
dT = 0.1
vmin = 0.5
vmax = 4.5                                            # 4.0 clipped the overtone (c up to ~4.3)
dvel = 0.01
vave = 3.0                                            # Tmax = dist / vave
maxgap = int(0.2 / dvel)                              # 0.2 km/s max jump over dT
MIN_SEG = 5                                           # argmax picking: keep every continuous
#   segment >= MIN_SEG samples (segment-aware extract_dispersion) -- preserves both sides of a
#   steep real step of the curve (basin/bedrock transition) and drops few-sample blobs (e.g.
#   short-period body-wave arrivals).
min_score = 0.7                                       # topology persistence threshold
gauss_alpha = 5.0                                     # narrowband Gaussian SNR width

# Group-velocity wavelength cutoff: keep a pick only if at least MIN_LAMBDA_GROUP wavelengths
# fit in the interstation distance, i.e. dist / (period * U) >= MIN_LAMBDA_GROUP.
MIN_LAMBDA_GROUP = 1.0

# Phase-velocity options. Convention (validated on a synthetic of known c(T), see
# dispersion.phase_velocity): the analytic signal of e^{-i(kr-pi/4)} is W ~ e^{i(wt-kr+pi/4)},
# so k r = w t_u - phi + shift + 2*pi*N with shift = +pi/4 for ZZ-convention components; the
# measured phase enters NEGATED in the library formulas.
PHASE_OFFSET = 0.0                                    # Morlet phase convention constant [rad]
PHASE_SHIFT = {"rayleigh": +np.pi / 4.0,
               "love": +np.pi / 4.0}                  # Love sign less settled -- verify vs FK Love curve
# Component-wise initial phase terms for Rayleigh. Nayak & Thurber (2020, p. 1591) list the
# adjustments in their (+phi) convention as -pi/4 ZZ/RR, +pi/4 RZ, -3pi/4 ZR; in the (-phi)
# convention used here the constants are negated: the eq-2 offsets e^{-+i pi/2} of RZ/ZR shift
# their measured phase, compensated as below. G_LR stacks are ZZ-phase-aligned -> +pi/4.
PHASE_SHIFT_COMPONENT = {"ZZ": +np.pi / 4.0, "RR": +np.pi / 4.0,
                         "RZ": -np.pi / 4.0, "ZR": +3.0 * np.pi / 4.0,
                         "G_LR0": +np.pi / 4.0, "G_LR1": +np.pi / 4.0}


def phase_shift_for(comp, wave):
    """Initial phase term for a component (Rayleigh per-component; else the wave-type default)."""
    return PHASE_SHIFT_COMPONENT.get(comp, PHASE_SHIFT[wave])
MIN_LAMBDA_PHASE = 3.0                                # require dist >= 3*lambda for phase velocity
TAU_MAX_FACTOR = 12.0                                 # phase reliable for period <= dist / 12
PHASE_JOINT = False                                   # 2*pi*N resolution for the CSV output:
#   False    = per-period argmin against the reference. With a DENSE, DATA-DERIVED reference
#              (the VSG-picked curves) this is empirically the most accurate (13-pair test:
#              95% same-fringe vs 86-90% for 'unwrap') -- the reference's own smoothness
#              supplies the continuity, and each pick stays independently anchored.
#   'unwrap' = single-N physics (Bensen 2007: one 2*pi*N for the unwrapped phase SPECTRUM):
#              continuous frequency-unwrap per wave-packet segment (breaks at group-curve
#              jumps/period gaps), one global integer per segment. Ridge-hopping impossible by
#              construction, but sequential unwrapping ACCUMULATES phase/group measurement
#              noise along a segment. Use when the reference is coarse or model-based (the
#              Bensen scenario) rather than dense and data-derived.
#   True     = joint Viterbi (reference + smoothness; soft compromise, deprecated).
# Quicklooks plot per-period AND unwrap so the choice can be audited per pair.
PHASE_SMOOTH_WEIGHT = 3.0                             # curve-continuity weight (joint mode)

# Dispersion-image export (group + phase velocity images, a la Douglas' .dat exports).
SAVE_IMAGES = True
IMAGE_COMPONENTS = ["ZZ", "TT", "G_LR0", "G_LR1"]     # single comps + synthesized modes to export
IMAGE_LAGS = ["sym"]                                  # lags to export
IMAGE_STACKS = ["pws"]                                # stack methods to export
SAVE_PNG = True                                       # also write a quicklook PNG with overlays

# Optional network YAML: `--config <yaml>` overrides ONLY the reference-curve directory and
# OUTPUT_ROOT below. It is popped from argv here so `sfile = sys.argv[1]` still finds the stack
# file regardless of flag position; the rest of the PARAMETER SECTION stays inline (it is the
# validated algorithm configuration that dispersion_batch_modesep.py mirrors).
_CFG = None
if "--config" in sys.argv:
    _i = sys.argv.index("--config")
    _cfgpath = sys.argv[_i + 1]
    del sys.argv[_i:_i + 2]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import modesep_config
    _CFG = modesep_config.load_config(_cfgpath)

# Reference phase-velocity curves for 2*pi*N resolution (see V5 docstring for format).
# Data-derived network curves picked from the mode-separated VSG phase-shift stacks
# (pick_reference_ridges.py; see VSG_REFERENCE_METHODOLOGY.md).
_VSG_REF = (_CFG["paths"]["ref_dir"] if _CFG else
            "/Users/genevievesavard/Codes/extract_higher_modes/Projects/aargau/vsg_modesep")
REFERENCE_CURVES = {
    "rayleigh": os.path.join(_VSG_REF, "ref_fundamental_phase.txt"),
    "love": None,
}
# Per-component override: the overtone stack G_LR1 must be resolved against the OVERTONE branch
# (the fundamental curve would pick the wrong 2*pi*N rung). All other Rayleigh components ride
# the fundamental and use REFERENCE_CURVES['rayleigh'].
REFERENCE_CURVES_COMPONENT = {
    "G_LR1": os.path.join(_VSG_REF, "ref_overtone_phase.txt"),
}
# Raw reference tables (period, c) drawn on every quicklook phase panel for orientation.
REF_TABLES = []
for _lab, _fn, _col in (("VSG ref fundamental", "ref_fundamental_phase.txt", "w"),
                        ("VSG ref overtone", "ref_overtone_phase.txt", "0.2")):
    try:
        REF_TABLES.append((np.loadtxt(os.path.join(_VSG_REF, _fn)), _lab, _col))
    except Exception:
        pass

overwrite = True

# Output root for the dispersion_V6/ tree. Default None = write next to the input stacks (V5
# behaviour). Set this when the stacks live on a full/read-only drive. Precedence:
# env DISP_OUTPUT_ROOT > --config paths.project_dir > this constant.
OUTPUT_ROOT = (_CFG["paths"]["project_dir"] if _CFG else
               "/Users/genevievesavard/Codes/extract_higher_modes/Projects/aargau")

# ----------------------------------------------------------------------------- setup
sfile = sys.argv[1]
tmp = sfile.split('/')[-1].split('_')
station1 = tmp[0]
spair = tmp[0] + '_' + tmp[1][:-3]

rootpath = os.environ.get("DISP_OUTPUT_ROOT", OUTPUT_ROOT)
if not rootpath:                                      # fall back to co-locating with the stacks
    rootpath = "/".join(os.path.split(sfile)[0].split("/")[:-1])
output_dir_root = os.path.join(rootpath, "dispersion_V6", station1)
os.makedirs(output_dir_root, exist_ok=True)
print(f"Input file: {sfile}\nOutput directory: {output_dir_root}")

image_dir = os.path.join(output_dir_root, "images")
if SAVE_IMAGES:
    os.makedirs(image_dir, exist_ok=True)
    if SAVE_PNG:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

dcfile = os.path.join(output_dir_root, spair + '_dispersion_all.csv')
if os.path.exists(dcfile):
    if overwrite:
        os.remove(dcfile)
    else:
        print(f"File already exists. Skipping. {dcfile}")
        sys.exit()

HEADER = ("nominal_period,T_centroid,T_inst,group_velocity,phase_velocity,N_ambiguity,"
          "U_from_phase,score,snr_nbG,snr_bb,ratio_d_lambda,azimuth,backazimuth,distance,"
          "lag,component,wave_type,stack_method,pick_method,snr_bb_other,snr_nbG_other\n")

# Load reference curves once (per wave type), keyed by this pair (falls back to 'default')
cref = {}
for wave, src in REFERENCE_CURVES.items():
    if src is None:
        cref[wave] = None
        continue
    try:
        cref[wave] = dispersion.load_reference_curve(src, key=spair)
    except Exception as e:
        print(f"Could not load {wave} reference curve ({e}); phase velocity disabled for {wave}.")
        cref[wave] = None

cref_comp = {}
for _c, _src in REFERENCE_CURVES_COMPONENT.items():
    try:
        cref_comp[_c] = dispersion.load_reference_curve(_src, key=spair)
    except Exception as e:
        print(f"Could not load {_c} reference curve ({e}); falling back to wave default.")


def cref_for(comp, wave):
    """Reference curve for a component: per-component override, else the wave-type default."""
    return cref_comp.get(comp, cref.get(wave))


def read_stack_components(sfile, dtype, comps):
    """Read params (dist, dt, azi, baz) + component waveforms for one stack type.

    Uses pyasdf if available; otherwise falls back to plain h5py since an ASDF file is just HDF5
    with the data under /AuxiliaryData/<dtype>/<comp> and the parameters stored as dataset attrs.
    """
    try:
        import pyasdf
        with pyasdf.ASDFDataSet(sfile, mode='r') as ds:
            p = ds.auxiliary_data[dtype]['ZZ'].parameters
            params = {k: float(p[k]) for k in ('dist', 'dt', 'azi', 'baz')}
            raw = {}
            for c in comps:
                try:
                    raw[c] = ds.auxiliary_data[dtype][c].data[:]
                except Exception:
                    pass
            return params, raw
    except ImportError:
        import h5py
        with h5py.File(sfile, 'r') as f:
            g = f['AuxiliaryData'][dtype]
            a = g['ZZ'].attrs
            params = {k: float(a[k]) for k in ('dist', 'dt', 'azi', 'baz')}
            raw = {c: np.asarray(g[c][:], dtype=float) for c in comps if c in g}
            return params, raw


def split_lags(tdata):
    """Return dict of neg/pos/sym lags from a two-sided CCF."""
    npts = len(tdata)
    i = npts // 2
    return {"neg": tdata[:i + 1][::-1],
            "pos": tdata[i:],
            "sym": 0.5 * (tdata[i:] + np.flip(tdata[:i + 1], axis=0))}


def snr_at(bank, period):
    """Index a cached (periods, snr_nbG) bank at the closest period."""
    pers, snr = bank
    return float(snr[int(np.argmin(np.abs(pers - period)))])


def save_images(tag, amp, cwt_data, per, vel, coi, dist, wave, c_ref, shift=None):
    """
    Export the group-velocity image (FTAN amplitude) and the phase-velocity image as .dat
    matrices (transposed velocity x period), plus axes, COI mask, and an optional PNG quicklook
    with the argmax group curve and extracted phase-velocity points overlaid.
    """
    phase_img, _, _ = dispersion.phase_image_from_cwt(
        cwt_data, dist, Tmin=Tmin, dT=dT, vmin=vmin, vmax=vmax, dvel=dvel, vave=vave)
    mask = dispersion.coi_mask(coi, per, vel)
    base = os.path.join(image_dir, tag)
    np.savetxt(base + "_group.dat", np.transpose(amp))
    np.savetxt(base + "_phase.dat", np.transpose(phase_img))
    np.savetxt(base + "_coimask.dat", np.transpose(mask), fmt="%d")
    np.savez(base + "_axes.npz", period=per, velocity=vel, dist=dist, coi=coi)

    if not SAVE_PNG:
        return
    gp, gv, gsc = dispersion.extract_dispersion(amp, per, vel, dist, vmax=vmax, maxgap=maxgap,
                                                minlambda=MIN_LAMBDA_GROUP,
                                                segments=True, min_seg=MIN_SEG)
    gp, gv, gsc = dispersion.remove_picks_coi(gp, gv, gsc, vel, coi)
    v_cut = dist / (MIN_LAMBDA_GROUP * per)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
    ext = [per[0], per[-1], vel[0], vel[-1]]
    ax[0].imshow(np.transpose(amp), cmap="jet", extent=ext, aspect="auto", origin="lower")
    ax[0].plot(coi, vel, "k--", lw=1, label="COI")
    ax[0].plot(per, v_cut, "m-", lw=1, label=f"{MIN_LAMBDA_GROUP:g}λ cutoff")
    if len(gp):
        ax[0].plot(gp, gv, "w.", ms=4, label="group pick")
    ax[0].set(title=f"Group image  {tag}", xlabel="Period [s]", ylabel="Group velocity [km/s]")
    ax[0].legend(loc="upper left", fontsize=8)
    ax[1].imshow(np.transpose(phase_img), cmap="RdBu", extent=ext, aspect="auto", origin="lower",
                 vmin=-1, vmax=1)
    ax[1].plot(coi, vel, "k--", lw=1)
    ax[1].plot(per, v_cut, "m-", lw=1)
    # Phase velocity predicted from the picked group curve. The fringe ladder (faint dots) is
    # the per-pick measurement c(N) for each integer N (Bensen eqs 10-11) -- ambiguous by
    # construction. The UNIQUE curve (solid line) integrates k(w) = k_ref + int dw'/U from the
    # group picks (U = dw/dk) and resolves the integration constant k_ref by anchoring the rigid
    # curve to those measured fringes (dispersion.phase_from_group).
    if len(gp):
        sh = PHASE_SHIFT[wave] if shift is None else shift
        Ns = np.arange(-4, 5)
        c_lad = np.full((len(Ns), len(gp)), np.nan)
        phis = np.full(len(gp), np.nan)
        Uref = np.asarray(gv, dtype=float).copy()
        for i in range(len(gp)):
            m = dispersion.measure_point(cwt_data, float(gp[i]), float(gv[i]), dist)
            if not np.isfinite(m["phase"]):
                continue
            phis[i] = m["phase"]
            Uref[i] = m["U"]
            w_i = 2.0 * np.pi / float(gp[i])
            # measured phase NEGATED (see dispersion.phase_velocity convention note)
            s_c = 1.0 / m["U"] + (-m["phase"] + PHASE_OFFSET + sh + 2.0 * np.pi * Ns) / (w_i * dist)
            c_lad[:, i] = np.where(s_c > 0, 1.0 / s_c, np.nan)
        c_lad[(c_lad < vel[0]) | (c_lad > vel[-1])] = np.nan
        for k in range(len(Ns)):
            if np.any(np.isfinite(c_lad[k])):
                ax[1].plot(gp, c_lad[k], "k.", ms=3.0, alpha=0.85,
                           label="c(N) fringe ladder" if k == 0 else None)
        c_pred = dispersion.phase_from_group(gp, Uref, dist, phis, phase_shift=sh,
                                             phase_offset=PHASE_OFFSET,
                                             cmin=vel[0], cmax=vel[-1])
        if np.any(np.isfinite(c_pred)):
            ax[1].plot(gp, c_pred, "-", color="limegreen", lw=2.2,
                       label="c(T) from group curve (fringe-anchored)")
        ax[1].legend(loc="upper left", fontsize=8)
    # VSG reference curves for orientation
    for _tbl, _lab, _col in REF_TABLES:
        ax[1].plot(_tbl[:, 0], _tbl[:, 1], ls="--", lw=1.4, color=_col, label=_lab)
    if len(gp) and c_ref is not None:
        # BOTH 2*pi*N resolutions, so joint-Viterbi vs per-period can be compared per pair:
        for joint, mk, colr, lab in ((False, "x", "lime", "phase pick (per-period)"),
                                     ("unwrap", "o", "magenta", "phase pick (unwrap, single N)")):
            corr = dispersion.measure_corrections_and_phase(
                cwt_data, gp, gv, dist, c_ref=c_ref,
                phase_shift=PHASE_SHIFT[wave] if shift is None else shift,
                phase_offset=PHASE_OFFSET, use_period="nominal",
                joint=joint, smooth_weight=PHASE_SMOOTH_WEIGHT)
            ax[1].plot(gp, corr["phase_velocity"], mk, ms=5, mfc="none", color=colr, label=lab,
                       linestyle="none")
    ax[1].legend(loc="upper left", fontsize=8)
    ax[1].set(title=f"Phase image (apparent c)  {tag}", xlabel="Period [s]",
              ylabel="Phase velocity [km/s]")
    ax[1].set_xlim(per[0], per[-1])
    ax[1].set_ylim(vel[0], vel[-1])
    fig.tight_layout()
    fig.savefig(base + "_quicklook.png", dpi=120)
    plt.close(fig)


rows = []

# ----------------------------------------------------------------------------- main loop
for stack_method in stack_methods:
    dtype = 'Allstack_' + stack_method

    # ---- single read: params + every available component (pyasdf or h5py) ----
    try:
        params, raw = read_stack_components(sfile, dtype, list(PICK_COMPONENTS) + LOVE_CONTEXT)
        dist = params['dist']
        dt = params['dt']
        azi = params['azi']
        baz = params['baz']
    except Exception as e:
        print(f"[{stack_method}] missing ZZ parameters ({e}); skipping stack method.")
        continue

    Tmax = dist / vave
    tau_max = dist / TAU_MAX_FACTOR
    per_grid = np.arange(Tmin, Tmax, dT)
    if len(per_grid) == 0:
        print(f"[{stack_method}] distance too short for period grid; skipping.")
        continue

    ccf = {comp: split_lags(td) for comp, td in raw.items()}

    # ---- per (component, lag) caches computed ONCE ----
    cwt_cache = {}     # (comp, lag) -> (cwt_data, coi)   (only for picked single components)
    ftan_cache = {}    # (comp, lag) -> amp image         (single components, for combos)
    snr_cache = {}     # (comp, lag) -> (periods, snr_nbG)
    snrbb_cache = {}   # (comp, lag) -> snr_bb
    per = vel = None

    all_comps = [c for c in list(PICK_COMPONENTS) + LOVE_CONTEXT if c in ccf]
    for comp in all_comps:
        for lag in lag_types:
            sig = ccf[comp][lag]
            try:
                snr_nbG, snr_bb, _, _ = dispersion.nb_filt_gauss(
                    sig, dt, 1.0 / per_grid, dist, alpha=gauss_alpha, vmin=vmin, vmax=vmax)
                snr_cache[(comp, lag)] = (per_grid, snr_nbG)
                snrbb_cache[(comp, lag)] = snr_bb
            except Exception as e:
                print(f"[{stack_method}] SNR failed {comp}/{lag}: {e}")
            if comp in PICK_COMPONENTS:
                try:
                    cwt_data = dispersion.compute_cwt(
                        sig, dist, dt, Tmin=Tmin, vmin=vmin, vmax=vmax, vave=vave)
                    amp, per, vel, coi = dispersion.disp_image_from_cwt(
                        cwt_data, dist, Tmin=Tmin, dT=dT, vmin=vmin, vmax=vmax, dvel=dvel, vave=vave)
                    cwt_cache[(comp, lag)] = (cwt_data, coi)
                    ftan_cache[(comp, lag)] = amp
                except Exception as e:
                    print(f"[{stack_method}] CWT failed {comp}/{lag}: {e}")

    def pick(amp_img):
        """Run both pick methods on an FTAN image; yield (method, nper, gv, score)."""
        for pm in pick_methods:
            try:
                if pm == "argmax":
                    np_, gv_, sc_ = dispersion.extract_dispersion(amp_img, per, vel, dist,
                                                                  vmax=vmax, maxgap=maxgap,
                                                                  minlambda=MIN_LAMBDA_GROUP,
                                                                  segments=True, min_seg=MIN_SEG)
                else:
                    np_, gv_, sc_ = dispersion.extract_curves_topology(amp_img, per, vel,
                                                                       limit=min_score)
                yield pm, np.asarray(np_), np.asarray(gv_), np.asarray(sc_)
            except Exception as e:
                print(f"[{stack_method}] picking ({pm}) failed: {e}")

    def emit(nper, gv, score, corr, comp, wave, lag, pm,
             snr_bank, snr_bb, snr_bb_other=np.nan, snr_other_bank=None):
        """Append CSV rows for one pick set."""
        for i in range(len(nper)):
            T = float(nper[i])
            U = float(gv[i])
            if T <= 0 or U <= 0:
                continue
            ratio = dist / (T * U)
            if ratio < MIN_LAMBDA_GROUP:
                continue
            snrnb = snr_at(snr_bank, T) if snr_bank is not None else np.nan
            if corr is not None:
                Tc = corr['T_centroid'][i]
                Ti = corr['T_inst'][i]
                cph = corr['phase_velocity'][i]
                Namb = int(corr['N_ambiguity'][i])
                Uph = corr['U_from_phase'][i]
                if ratio < MIN_LAMBDA_PHASE or T > tau_max:
                    cph, Namb, Uph = np.nan, 0, np.nan
            else:
                Tc = Ti = cph = Uph = np.nan
                Namb = 0
            snro = snr_at(snr_other_bank, T) if snr_other_bank is not None else np.nan
            rows.append(
                f"{T:.2f},{Tc:.3f},{Ti:.3f},{U:.2f},{cph:.3f},{Namb:d},{Uph:.2f},"
                f"{float(score[i]):.2f},{snrnb:.2f},{snr_bb:.2f},{ratio:.2f},"
                f"{azi:.2f},{baz:.2f},{dist:.3f},{lag},{comp},{wave},{stack_method},{pm},"
                f"{snr_bb_other:.2f},{snro:.2f}\n")

    # ---- single components: group + corrections + phase velocity ----
    for comp, wave in PICK_COMPONENTS.items():
        for lag in lag_types:
            if (comp, lag) not in cwt_cache:
                continue
            cwt_data, coi = cwt_cache[(comp, lag)]
            amp = ftan_cache[(comp, lag)]
            if (SAVE_IMAGES and comp in IMAGE_COMPONENTS and lag in IMAGE_LAGS
                    and stack_method in IMAGE_STACKS):
                try:
                    tag = f"{spair}_{stack_method}_{comp}_{lag}"
                    save_images(tag, amp, cwt_data, per, vel, coi, dist, wave, cref_for(comp, wave),
                                shift=phase_shift_for(comp, wave))
                except Exception as e:
                    print(f"[{stack_method}] image export failed {comp}/{lag}: {e}")
            if wave == "love":
                bb = [snrbb_cache[(c, lag)] for c in LOVE_CONTEXT if (c, lag) in snrbb_cache]
                snr_bb_other = float(np.max(bb)) if bb else np.nan
                other_banks = [snr_cache[(c, lag)] for c in LOVE_CONTEXT if (c, lag) in snr_cache]
            else:
                snr_bb_other, other_banks = np.nan, None
            for pm, nper, gv, score in pick(amp):
                nper, gv, score = dispersion.remove_picks_coi(nper, gv, score, vel, coi)
                if len(nper) == 0:
                    continue
                corr = dispersion.measure_corrections_and_phase(
                    cwt_data, nper, gv, dist, c_ref=cref_for(comp, wave),
                    phase_shift=phase_shift_for(comp, wave), phase_offset=PHASE_OFFSET,
                    use_period='nominal', joint=PHASE_JOINT, smooth_weight=PHASE_SMOOTH_WEIGHT)
                if other_banks:
                    other_combined = (other_banks[0][0],
                                      np.max([b[1] for b in other_banks], axis=0))
                else:
                    other_combined = None
                emit(nper, gv, score, corr, comp, wave, lag, pm,
                     snr_cache.get((comp, lag)), snrbb_cache.get((comp, lag), np.nan),
                     snr_bb_other=snr_bb_other, snr_other_bank=other_combined)

    # ---- Rayleigh component-product images: group velocity + SNR only ----
    def prod(comps, lag, root):
        try:
            img = np.ones_like(ftan_cache[(comps[0], lag)])
            for c in comps:
                img = img * ftan_cache[(c, lag)]
            return img ** (1.0 / root)
        except KeyError:
            return None

    combos = []
    img = prod(["ZZ", "ZR", "RZ", "RR"], "sym", 4)
    if img is not None:
        combos.append((img, "sym", "all4", "ZZ"))
    img = prod(["ZZ", "ZR"], "sym", 2)
    if img is not None:
        combos.append((img, "sym", "ZZ-ZR", "ZZ"))
    img = prod(["RR", "RZ"], "sym", 2)
    if img is not None:
        combos.append((img, "sym", "RR-RZ", "RZ"))

    for amp, lag, label, snr_comp in combos:
        for pm, nper, gv, score in pick(amp):
            if len(nper) == 0:
                continue
            emit(nper, gv, score, None, label, "rayleigh", lag, pm,
                 snr_cache.get((snr_comp, lag)), snrbb_cache.get((snr_comp, lag), np.nan))

    # ---- Nayak & Thurber (2020) mode-separated Rayleigh traces (G_LR0 fund, G_LR1 1st higher) ----
    # Synthesize from the symmetric folds, then run the SAME CWT-FTAN + picking pipeline; picks are
    # labelled G_LR0 / G_LR1 in the component column. Validated against ZZ/RR/products downstream.
    if all((c in ccf) for c in ("ZZ", "RR", "RZ", "ZR")):
        comps0, comps1 = dispersion.phase_corrected_components(
            ccf["ZZ"][GLR_LAG], ccf["RR"][GLR_LAG], ccf["RZ"][GLR_LAG], ccf["ZR"][GLR_LAG],
            receiver_side_flip=GLR_RECEIVER_SIDE_FLIP)
        if GLR_STACK == 'tfpws':
            g0 = dispersion.tf_pws(comps0, dt)
            g1 = dispersion.tf_pws(comps1, dt)
        else:                                          # 'linear' = eqs (3)/(4) exactly
            g0 = np.sum(comps0, axis=0)
            g1 = np.sum(comps1, axis=0)
        for label, sig in (("G_LR0", g0), ("G_LR1", g1)):
            try:
                cwt_data = dispersion.compute_cwt(
                    sig, dist, dt, Tmin=Tmin, vmin=vmin, vmax=vmax, vave=vave)
                amp, per, vel, coi = dispersion.disp_image_from_cwt(
                    cwt_data, dist, Tmin=Tmin, dT=dT, vmin=vmin, vmax=vmax, dvel=dvel, vave=vave)
                snr_nbG, snr_bb, _, _ = dispersion.nb_filt_gauss(
                    sig, dt, 1.0 / per_grid, dist, alpha=gauss_alpha, vmin=vmin, vmax=vmax)
            except Exception as e:
                print(f"[{stack_method}] G_LR synth/CWT failed {label}: {e}")
                continue
            if (SAVE_IMAGES and label in IMAGE_COMPONENTS and GLR_LAG in IMAGE_LAGS
                    and stack_method in IMAGE_STACKS):
                try:
                    tag = f"{spair}_{stack_method}_{label}_{GLR_LAG}"
                    save_images(tag, amp, cwt_data, per, vel, coi, dist, "rayleigh",
                                cref_for(label, "rayleigh"), shift=phase_shift_for(label, "rayleigh"))
                except Exception as e:
                    print(f"[{stack_method}] image export failed {label}: {e}")
            for pm, nper, gv, score in pick(amp):
                nper, gv, score = dispersion.remove_picks_coi(nper, gv, score, vel, coi)
                if len(nper) == 0:
                    continue
                corr = dispersion.measure_corrections_and_phase(
                    cwt_data, nper, gv, dist, c_ref=cref_for(label, "rayleigh"),
                    phase_shift=phase_shift_for(label, "rayleigh"), phase_offset=PHASE_OFFSET,
                    use_period='nominal', joint=PHASE_JOINT, smooth_weight=PHASE_SMOOTH_WEIGHT)
                emit(nper, gv, score, corr, label, "rayleigh", GLR_LAG, pm,
                     (per_grid, snr_nbG), snr_bb)

    print(f"[{stack_method}] done ({len(rows)} rows so far)")

# ----------------------------------------------------------------------------- write
with open(dcfile, 'w') as f:
    f.write(HEADER)
    f.writelines(rows)
print(f"Wrote {len(rows)} rows to {dcfile}")
