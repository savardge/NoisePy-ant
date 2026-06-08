"""
Dispersion picking V5 -- Rayleigh + Love in one pass, with the Shapiro/Levshin period
corrections and Bensen phase-velocity extraction added to the group-velocity measurement.

What it does per station-pair ASDF file:
  * group velocity by tracking the CWT (Morlet) FTAN ridge (argmax + topology), as before;
  * Shapiro centroid period (T_centroid) and Levshin instantaneous period (T_inst), both
    derived directly from the complex CWT coefficients (see noisepy/dispersion.py);
  * phase velocity with the 2*pi*N ambiguity resolved against a reference curve (e.g. from
    FK analysis), per station-pair / region;
  * Rayleigh single components (ZZ, RR, RZ, ZR) + component-product images, and Love (TT)
    with TZ/ZT/RT/TR SNR context.

Efficiency vs V3/V4: the ASDF file is opened once; each CWT and each narrowband Gaussian
SNR bank is computed once per (component, lag) and reused across pick methods, picks and the
correction/phase measurements; CSV rows are batched.

Usage:  python dispersion_curves_V5.py <stacked_asdf_file>
"""
import numpy as np
import os
import sys
import pyasdf
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

# Period / velocity grid (matches V3/V4)
Tmin = 0.2
dT = 0.1
vmin = 0.5
vmax = 4.0
dvel = 0.01
vave = 3.0                                            # Tmax = dist / vave
maxgap = int(0.2 / dvel)                              # 0.2 km/s max jump over dT
min_score = 0.7                                       # topology persistence threshold
gauss_alpha = 5.0                                     # narrowband Gaussian SNR width

# Group-velocity wavelength cutoff: keep a pick only if at least MIN_LAMBDA_GROUP wavelengths
# fit in the interstation distance, i.e. dist / (period * U) >= MIN_LAMBDA_GROUP. With 1.0 this
# stops the curve once the wavelength exceeds the station spacing (less than one wavelength fits).
MIN_LAMBDA_GROUP = 1.0

# Phase-velocity options
PHASE_OFFSET = 0.0                                    # Morlet phase convention constant [rad];
#   calibrate with scripts/picking/synthetic_phase_calibration.py. At typical inter-station
#   distances the result is weakly sensitive to this (the 2*pi*N reference dominates).
PHASE_SHIFT = {"rayleigh": -np.pi / 4.0,             # Snieder 2004 / Lin 2007, vertical Rayleigh
               "love": -np.pi / 4.0}                  # Love sign less settled -- verify vs FK Love curve
MIN_LAMBDA_PHASE = 3.0                                # require dist >= 3*lambda for phase velocity
TAU_MAX_FACTOR = 12.0                                 # phase reliable for period <= dist / 12
PHASE_JOINT = True                                    # resolve 2*pi*N jointly across the curve
#   (Viterbi: stay near reference + keep c(T) smooth -> tracks one branch). False = per-period argmin.
PHASE_SMOOTH_WEIGHT = 3.0                             # curve-continuity weight (joint mode)

# Dispersion-image export (group + phase velocity images, a la Douglas' .dat exports).
SAVE_IMAGES = True
IMAGE_COMPONENTS = ["ZZ", "TT"]                       # single components to export images for
IMAGE_LAGS = ["sym"]                                  # lags to export
IMAGE_STACKS = ["pws"]                                # stack methods to export
SAVE_PNG = True                                       # also write a quicklook PNG with overlays

# Reference phase-velocity curves for 2*pi*N resolution. Each entry may be:
#   - a path to a 2-column file (period[s]  c[km/s]);
#   - a dict {pair_or_region_key: path-or-(periods,velocities), 'default': ...};
#   - None to skip phase velocity for that wave type.
REFERENCE_CURVES = {
    "rayleigh": None,   # e.g. "/path/to/rayleigh_ref_phase.txt" or a {region: path, 'default': path} dict
    "love": None,       # e.g. "/path/to/love_ref_phase.txt"
}

overwrite = True

# ----------------------------------------------------------------------------- setup
sfile = sys.argv[1]
tmp = sfile.split('/')[-1].split('_')
station1 = tmp[0]
spair = tmp[0] + '_' + tmp[1][:-3]

dum = os.path.split(sfile)[0].split("/")[:-1]
rootpath = "/".join(dum)
output_dir_root = os.path.join(rootpath, "dispersion_V5", station1)
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


def save_images(tag, amp, cwt_data, per, vel, coi, dist, wave, c_ref):
    """
    Export the group-velocity image (FTAN amplitude) and the phase-velocity image (CWT real
    part mapped to apparent phase velocity) as .dat matrices (transposed velocity x period,
    matching Douglas' format), plus the period/velocity axes, the COI mask, and an optional
    PNG quicklook with the argmax group curve and extracted phase-velocity points overlaid.
    """
    phase_img, _, _ = dispersion.phase_image_from_cwt(
        cwt_data, dist, Tmin=Tmin, dT=dT, vmin=vmin, vmax=vmax, dvel=dvel, vave=vave)
    mask = dispersion.coi_mask(coi, per, vel)
    base = os.path.join(image_dir, tag)
    # .dat matrices: transpose to (velocity, period) like the douglas exports
    np.savetxt(base + "_group.dat", np.transpose(amp))
    np.savetxt(base + "_phase.dat", np.transpose(phase_img))
    np.savetxt(base + "_coimask.dat", np.transpose(mask), fmt="%d")
    np.savez(base + "_axes.npz", period=per, velocity=vel, dist=dist, coi=coi)

    if not SAVE_PNG:
        return
    # overlays: argmax group curve and (if available) extracted phase velocity
    gp, gv, gsc = dispersion.extract_dispersion(amp, per, vel, dist, vmax=vmax, maxgap=maxgap,
                                                minlambda=MIN_LAMBDA_GROUP)
    gp, gv, gsc = dispersion.remove_picks_coi(gp, gv, gsc, vel, coi)
    # 1-wavelength cutoff curve: v = dist / (MIN_LAMBDA_GROUP * T); picks must lie below it
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
    if len(gp) and c_ref is not None:
        corr = dispersion.measure_corrections_and_phase(
            cwt_data, gp, gv, dist, c_ref=c_ref,
            phase_shift=PHASE_SHIFT[wave], phase_offset=PHASE_OFFSET, use_period="nominal",
            joint=PHASE_JOINT, smooth_weight=PHASE_SMOOTH_WEIGHT)
        cph = corr["phase_velocity"]
        ax[1].plot(gp, cph, "k.", ms=5, label="phase pick")
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

    # ---- single ASDF open: read params + every available component ----
    raw = {}
    with pyasdf.ASDFDataSet(sfile, mode='r') as ds:
        try:
            params = ds.auxiliary_data[dtype]['ZZ'].parameters
            dist = params['dist']
            dt = params['dt']
            azi = params['azi']
            baz = params['baz']
        except Exception as e:
            print(f"[{stack_method}] missing ZZ parameters ({e}); skipping stack method.")
            continue
        for comp in list(PICK_COMPONENTS) + LOVE_CONTEXT:
            try:
                raw[comp] = ds.auxiliary_data[dtype][comp].data[:]
            except Exception:
                pass

    Tmax = dist / vave
    tau_max = dist / TAU_MAX_FACTOR
    per_grid = np.arange(Tmin, Tmax, dT)
    if len(per_grid) == 0:
        print(f"[{stack_method}] distance too short for period grid; skipping.")
        continue

    ccf = {comp: split_lags(td) for comp, td in raw.items()}

    # ---- per (component, lag) caches computed ONCE ----
    cwt_cache = {}     # (comp, lag) -> cwt_data         (only for picked single components)
    ftan_cache = {}    # (comp, lag) -> amp image        (single components, for combos)
    snr_cache = {}     # (comp, lag) -> (periods, snr_nbG)
    snrbb_cache = {}   # (comp, lag) -> snr_bb
    per = vel = None

    all_comps = [c for c in list(PICK_COMPONENTS) + LOVE_CONTEXT if c in ccf]
    for comp in all_comps:
        for lag in lag_types:
            sig = ccf[comp][lag]
            # narrowband Gaussian SNR bank over the full period grid (used by everything)
            try:
                snr_nbG, snr_bb, _, _ = dispersion.nb_filt_gauss(
                    sig, dt, 1.0 / per_grid, dist, alpha=gauss_alpha, vmin=vmin, vmax=vmax)
                snr_cache[(comp, lag)] = (per_grid, snr_nbG)
                snrbb_cache[(comp, lag)] = snr_bb
            except Exception as e:
                print(f"[{stack_method}] SNR failed {comp}/{lag}: {e}")
            # CWT + FTAN image only for components we actually pick on
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
                                                                  minlambda=MIN_LAMBDA_GROUP)
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
            ratio = dist / (T * U)                       # dist / wavelength (group)
            if ratio < MIN_LAMBDA_GROUP:                 # 1-wavelength cutoff: stop the curve
                continue
            snrnb = snr_at(snr_bank, T) if snr_bank is not None else np.nan
            if corr is not None:
                Tc = corr['T_centroid'][i]
                Ti = corr['T_inst'][i]
                cph = corr['phase_velocity'][i]
                Namb = int(corr['N_ambiguity'][i])
                Uph = corr['U_from_phase'][i]
                # phase-velocity QC: drop where far-field/ambiguity criteria are not met
                if ratio < MIN_LAMBDA_PHASE or T > tau_max:
                    cph, Namb, Uph = np.nan, 0, np.nan
            else:
                Tc = Ti = cph = Uph = np.nan
                Namb = 0
            snro = snr_at(snr_other_bank, T) if snr_other_bank is not None else np.nan
            rows.append(
                f"{T:5.2f},{Tc:6.3f},{Ti:6.3f},{U:5.2f},{cph:6.3f},{Namb:d},{Uph:5.2f},"
                f"{float(score[i]):5.2f},{snrnb:7.2f},{snr_bb:7.2f},{ratio:6.2f},"
                f"{azi:6.2f},{baz:6.2f},{dist:7.3f},{lag},{comp},{wave},{stack_method},{pm},"
                f"{snr_bb_other:7.2f},{snro:7.2f}\n")

    # ---- single components: group + corrections + phase velocity ----
    for comp, wave in PICK_COMPONENTS.items():
        for lag in lag_types:
            if (comp, lag) not in cwt_cache:
                continue
            cwt_data, coi = cwt_cache[(comp, lag)]
            amp = ftan_cache[(comp, lag)]
            # Export group + phase velocity images (Douglas-style) for the selected subset
            if (SAVE_IMAGES and comp in IMAGE_COMPONENTS and lag in IMAGE_LAGS
                    and stack_method in IMAGE_STACKS):
                try:
                    tag = f"{spair}_{stack_method}_{comp}_{lag}"
                    save_images(tag, amp, cwt_data, per, vel, coi, dist, wave, cref.get(wave))
                except Exception as e:
                    print(f"[{stack_method}] image export failed {comp}/{lag}: {e}")
            # Love SNR context (max over TZ/ZT/RT/TR)
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
                    cwt_data, nper, gv, dist, c_ref=cref.get(wave),
                    phase_shift=PHASE_SHIFT[wave], phase_offset=PHASE_OFFSET, use_period='nominal',
                    joint=PHASE_JOINT, smooth_weight=PHASE_SMOOTH_WEIGHT)
                # max narrowband SNR over the Love-context components, per pick
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

    print(f"[{stack_method}] done ({len(rows)} rows so far)")

# ----------------------------------------------------------------------------- write
with open(dcfile, 'w') as f:
    f.write(HEADER)
    f.writelines(rows)
print(f"Wrote {len(rows)} rows to {dcfile}")
