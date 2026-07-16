"""Unified surface-wave dispersion picking: Rayleigh + Love, group + phase, fundamental + overtone.

One entry point (`pick_all_modes`) returns all EIGHT pick types for a station pair:

    (rayleigh, fundamental) | (rayleigh, overtone) | (love, fundamental) | (love, overtone)
    each with BOTH group_velocity and phase_velocity.

This is a thin orchestration layer over the validated primitives in `noisepy.dispersion`; it adds no
new signal processing. The two things it *does* add are (1) an explicit `mode` label so every row is
one of the eight cells, and (2) the Love-specific ridge labeling that the physics forces on us.

Theory note (why Love is not symmetric with Rayleigh):
  * Phase MEASUREMENT is identical. The far-field 2-D surface-wave Green's function is
    ~ e^{-i(kr - pi/4)} for BOTH Love and Rayleigh -- the -pi/4 is the stationary-phase term, not a
    polarization effect -- so Love phase uses the same `measure_corrections_and_phase` chain and the
    same +pi/4 shift as ZZ/RR (TT is an auto-component with NO cross-component 90-deg term). Wave type
    enters the library only through the scalar `phase_shift` and the callable `c_ref`.
  * Mode SEPARATION is fundamentally different. Rayleigh's G_LR0/G_LR1 synthesis
    (phase_corrected_components + tf_pws) needs two orthogonal components 90 deg apart (elliptical Z-R
    coupling). Love is single-component SH; there is no orthogonal partner, hence NO G_LR analog. Love
    fundamental/overtone separate only in the image/array domain: here we pick TT ridges (argmax +
    topology) and LABEL them fundamental/overtone against data-derived Love reference curves. Where the
    two modes overlap in velocity, the labeling is flagged (`mode_overlap=1`) and the Love-overtone
    phase -- read off the MIXED TT wavelet transform -- is unreliable by construction.

The unified CSV schema (see HEADER) is the V6 schema plus an explicit `mode` column and three Love QC
columns (`env_ratio`, `ot_flag`, `mode_overlap`); Rayleigh rows leave the Love QC columns blank/NaN.
"""
import numpy as np
from scipy.interpolate import interp1d

from noisepy import dispersion


# --------------------------------------------------------------------------- configuration
class Config:
    """Algorithm parameters. Defaults mirror dispersion_curves_V6_modesep.py, except the phase gate,
    which follows the validated phase step-2 methodology (phase is valid to ~1 lambda; Luo 2015 /
    Ekstrom 2009) so Love overtone and long-period phase are not thrown away."""
    # period / velocity grid
    Tmin = 0.2
    dT = 0.1
    vmin = 0.5
    vmax = 5.0                                   # 4.5 clipped the Riehen overtone (phase ref reaches
    #   4.51 km/s) and extract_dispersion deletes picks within 3 samples of vmax -- keep headroom.
    dvel = 0.01
    vave = 3.0                                   # Tmax = dist / vave
    # group picking
    maxgap = int(0.2 / dvel)
    MIN_SEG = 5
    min_score = 0.6                              # topology persistence (0.5-0.7; Love ridges are weak)
    MIN_LAMBDA_GROUP = 1.0
    gauss_alpha = 5.0
    # phase resolution
    PHASE_OFFSET = 0.0
    PHASE_SHIFT = {"rayleigh": +np.pi / 4.0, "love": +np.pi / 4.0}  # far-field -pi/4 -> +pi/4 (neg-phi)
    MIN_LAMBDA_PHASE = 1.0                        # phase valid to ~1 lambda (step-2 validated)
    TAU_MAX_FACTOR = None                        # None = no long-period cap (keep the deep band)
    PHASE_USE_PERIOD = "nominal"                 # keep group & phase on the same period axis per row
    PHASE_JOINT = "unwrap"                        # branch-tracking; safest across dense & sparse refs
    PHASE_SMOOTH_WEIGHT = 3.0
    # Rayleigh mode synthesis
    GLR_LAG = "sym"
    GLR_RECEIVER_SIDE_FLIP = False
    GLR_STACK = "tfpws"                           # 'tfpws' (paper, real data) or 'linear' (eqs 3/4)
    # Love mode labeling
    LOVE_LAGS = ("sym",)                          # lags to pick Love on (sym is the production choice)
    LOVE_SEP_TOL = 0.15                          # |U0_ref - U1_ref| below this -> mode_overlap flag
    LOVE_OVERTONE_MARGIN = 0.25                  # overtone must exceed the fundamental group ref by
    #   this [km/s]: higher modes are FASTER, so a slower ridge is never the overtone (guards against
    #   reference-edge noise driving U1_ref down toward U0_ref).
    LOVE_CONTEXT = ("TZ", "ZT", "RT", "TR")      # cross-terms for env_ratio (leakage) context
    PICK_LOVE_OVERTONE = False                   # Love overtone judged not credible on Aargau/Riehen;
    #   when False, overtone-labeled TT ridge points are DROPPED (never relabeled fundamental, which
    #   would contaminate the fundamental stream with fast ridges). Re-enable via DISP_LOVE_OT=1.
    LOVE_RAY_T_TOL = 0.15                        # [s] period tolerance for the dU_rayfund/dU_rayot
    #   lookup against the pair's own G_LR0/G_LR1 argmax curves (batch_love REF_T_TOL pattern).
    # Rayleigh overtone resolution/leakage flags (constants from validate_modes.py):
    RAYLEIGH_SEP_MIN = 0.30                      # [km/s] floor of the fund<->overtone separation
    RAYLEIGH_RES_FACTOR = 1.0                    # envelope-widths: sep_req = max(SEP_MIN, F*U0^2*T/d)
    RAYLEIGH_SLOW_TOL = 0.10                     # [km/s] U1 < U0 - tol -> ot_flag='slow' (leakage
    #   candidate; group curves CAN cross at Airy phases, so this is a flag, not a drop)


# Components that must be present to synthesize the Rayleigh modes.
RAYLEIGH_COMPS = ("ZZ", "RR", "RZ", "ZR")

HEADER = ("nominal_period,T_centroid,T_inst,group_velocity,phase_velocity,N_ambiguity,U_from_phase,"
          "score,snr_nbG,snr_bb,ratio_d_lambda,azimuth,backazimuth,distance,lag,component,"
          "wave_type,mode,stack_method,pick_method,snr_bb_other,snr_nbG_other,"
          "env_ratio,ot_flag,mode_overlap,xmode_amp,dU_rayfund,dU_rayot,dUdT_local,T_scale,scale_j")

_COLS = HEADER.split(",")


# --------------------------------------------------------------------------- IO helpers
def read_stack_components(sfile, dtype, comps):
    """Read params (dist, dt, azi, baz) + component waveforms for one stack type.

    pyasdf if available, else plain h5py (an ASDF file is HDF5 with data under
    /AuxiliaryData/<dtype>/<comp> and parameters as dataset attrs). Ported from V6."""
    try:
        import pyasdf
        with pyasdf.ASDFDataSet(sfile, mode="r") as ds:
            p = ds.auxiliary_data[dtype]["ZZ"].parameters
            params = {k: float(p[k]) for k in ("dist", "dt", "azi", "baz")}
            raw = {}
            for c in comps:
                try:
                    raw[c] = ds.auxiliary_data[dtype][c].data[:]
                except Exception:
                    pass
            return params, raw
    except ImportError:
        import h5py
        with h5py.File(sfile, "r") as f:
            g = f["AuxiliaryData"][dtype]
            a = g["ZZ"].attrs
            params = {k: float(a[k]) for k in ("dist", "dt", "azi", "baz")}
            raw = {c: np.asarray(g[c][:], dtype=float) for c in comps if c in g}
            return params, raw


def split_lags(tdata):
    """neg/pos/sym lags from a two-sided CCF (ported from V6)."""
    npts = len(tdata)
    i = npts // 2
    return {"neg": tdata[:i + 1][::-1],
            "pos": tdata[i:],
            "sym": 0.5 * (tdata[i:] + np.flip(tdata[:i + 1], axis=0))}


def load_pair(sfile, stack_method, comps):
    """Read one stack method and fold every component into {comp: {neg,pos,sym}}."""
    params, raw = read_stack_components(sfile, "Allstack_" + stack_method, comps)
    ccf = {comp: split_lags(td) for comp, td in raw.items()}
    return params, ccf


# --------------------------------------------------------------------------- reference helpers
def phase_ref_to_group_ref(c_ref, per_grid):
    """Turn a PHASE reference callable into a GROUP reference callable via the dispersion relation
    (dispersion.group_from_phase). Needed because the FTAN topology ridges are GROUP velocity but the
    Love references are PHASE. Returns a callable U_ref(period)->km/s (NaN outside range), or None.

    The phase reference is first fit with a low-order polynomial over its valid band and resampled on
    a fine grid before differentiation. Raw data-derived references (especially the narrow, sparse
    Love overtone curve) are slightly non-monotonic, and directly differentiating them drives the
    group slowness negative (U -> NaN) at most periods. Smoothing yields a stable group reference,
    which is all the ridge labeling needs -- the exact phase reference is still used for the 2*pi*N
    resolution downstream."""
    if c_ref is None:
        return None
    T = np.asarray(per_grid, float)
    c = np.array([float(c_ref(t)) for t in T])
    good = np.isfinite(c) & (c > 0)
    if good.sum() < 3:
        return None
    Tg, cg = T[good], c[good]
    # fine, smooth resampling over the reference's valid band
    Tfine = np.linspace(Tg.min(), Tg.max(), max(40, 4 * len(Tg)))
    deg = int(min(2, good.sum() - 1))
    try:
        cfit = np.polyval(np.polyfit(Tg, cg, deg), Tfine)
    except Exception:
        cfit = np.interp(Tfine, np.sort(Tg), cg[np.argsort(Tg)])
    U = dispersion.group_from_phase(Tfine, cfit)
    # Clamp to a physical band: group velocity is below but comparable to phase velocity. Where the
    # differentiation is unstable (near-flat / sparse references drive it out of [0.5c, 1.02c] or to
    # NaN), fall back to 0.92*c. Labeling only needs the fast overtone ridge separated from the slow
    # fundamental ridge -- exact group values are not required, and the true phase reference is still
    # used for 2*pi*N resolution.
    bad = ~np.isfinite(U) | (U < 0.5 * cfit) | (U > 1.02 * cfit)
    U = np.where(bad, 0.92 * cfit, U)
    order = np.argsort(Tfine)
    return interp1d(Tfine[order], U[order], bounds_error=False, fill_value=np.nan)


# --------------------------------------------------------------------------- picking helpers
def pick_group_ridges(amp, per, vel, dist, cfg):
    """Both pick methods on one FTAN image; yields (method, nper, gv, score)."""
    out = []
    try:
        np_, gv_, sc_ = dispersion.extract_dispersion(
            amp, per, vel, dist, vmax=cfg.vmax, maxgap=cfg.maxgap,
            minlambda=cfg.MIN_LAMBDA_GROUP, segments=True, min_seg=cfg.MIN_SEG)
        out.append(("argmax", np.asarray(np_), np.asarray(gv_), np.asarray(sc_)))
    except Exception as e:
        print(f"  argmax pick failed: {e}")
    try:
        np_, gv_, sc_ = dispersion.extract_curves_topology(amp, per, vel, limit=cfg.min_score)
        out.append(("topology", np.asarray(np_), np.asarray(gv_), np.asarray(sc_)))
    except Exception as e:
        print(f"  topology pick failed: {e}")
    return out


def label_love_ridges(nper, gv, score, U0_ref, U1_ref, sep_tol, overtone_margin=0.25):
    """Assign each TT group-ridge point to fundamental/overtone by nearest GROUP reference.

    A point is labeled overtone only if it is (a) closer to the overtone reference U1 than to the
    fundamental reference U0 AND (b) genuinely faster than the fundamental by `overtone_margin`
    (higher modes are faster -- this guards against reference-edge noise pulling U1 down toward U0).
    Where only the fundamental reference is defined (the overtone reference covers a narrow band),
    everything labels fundamental. `mode_overlap=1` marks periods where the two references are within
    `sep_tol` (modes not separable -> label ambiguous, overtone phase unreliable).

    Returns {mode: (nper, gv, score, overlap)} with mode in {'fundamental','overtone'}."""
    acc = {"fundamental": ([], [], [], []), "overtone": ([], [], [], [])}
    for T, U, s in zip(np.asarray(nper, float), np.asarray(gv, float), np.asarray(score, float)):
        if T <= 0 or U <= 0:
            continue
        u0 = float(U0_ref(T)) if U0_ref is not None else np.nan
        u1 = float(U1_ref(T)) if U1_ref is not None else np.nan
        has0, has1 = np.isfinite(u0), np.isfinite(u1)
        if not has0 and not has1:
            mode, overlap = "fundamental", 1          # no reference: cannot separate -> flag
        else:
            d0 = abs(U - u0) if has0 else np.inf
            d1 = abs(U - u1) if has1 else np.inf
            faster = (not has0) or (U > u0 + overtone_margin)
            mode = "overtone" if (has1 and d1 < d0 and faster) else "fundamental"
            overlap = int(has0 and has1 and abs(u0 - u1) < sep_tol)
        acc[mode][0].append(T)
        acc[mode][1].append(U)
        acc[mode][2].append(s)
        acc[mode][3].append(overlap)
    return {m: (np.asarray(a[0]), np.asarray(a[1]), np.asarray(a[2]), np.asarray(a[3]))
            for m, a in acc.items()}


# --------------------------------------------------------------------------- QC helpers
def _norm_image(amp):
    """Per-period-max normalized copy of an FTAN image (rows = periods), the normalization the
    validate_modes.py mutual-suppression test is defined on."""
    mx = np.nanmax(amp, axis=1, keepdims=True)
    return amp / np.where(mx > 0, mx, 1.0)


def _amp_at(per, vel, img, T, U):
    """Nearest-node normalized image amplitude at (period T, velocity U) (validate_modes._amp_at)."""
    return float(img[int(np.argmin(np.abs(per - T))), int(np.argmin(np.abs(vel - U)))])


def _curve_at(nper, gv, T, t_tol):
    """Velocity of a picked (nper, gv) curve at the period nearest T, NaN beyond t_tol."""
    if nper is None or len(nper) == 0:
        return np.nan
    i = int(np.argmin(np.abs(nper - T)))
    return float(gv[i]) if abs(float(nper[i]) - T) <= t_tol else np.nan


def _local_slopes(nper, gv):
    """Per-pick |dU/dT| along an argmax curve (sorted by period). Report-only mode-mixing indicator;
    big values at real geological steps are expected -- threshold downstream, if at all."""
    n = len(nper)
    out = np.full(n, np.nan)
    if n < 2:
        return out
    order = np.argsort(nper)
    Ts, Us = np.asarray(nper, float)[order], np.asarray(gv, float)[order]
    dT = np.gradient(Ts)
    with np.errstate(divide="ignore", invalid="ignore"):
        sl = np.abs(np.gradient(Us) / np.where(dT != 0, dT, np.nan))
    out[order] = sl
    return out


def _rayleigh_ot_flags(nper, gv, u0_at, dist, cfg):
    """Per-pick overtone flags vs the fundamental curve: 'slow' (leakage candidate), 'unresolved'
    (within the resolution-adaptive separation sep_req -- osculation guard), 'sep' (resolved)."""
    flags = []
    for T, U in zip(np.asarray(nper, float), np.asarray(gv, float)):
        u0 = u0_at(T)
        if not np.isfinite(u0):
            flags.append("")
            continue
        sep_req = max(cfg.RAYLEIGH_SEP_MIN, cfg.RAYLEIGH_RES_FACTOR * u0 * u0 * T / dist)
        if U < u0 - cfg.RAYLEIGH_SLOW_TOL:
            flags.append("slow")
        elif (U - u0) <= sep_req:
            flags.append("unresolved")
        else:
            flags.append("sep")
    return flags


# --------------------------------------------------------------------------- row emission
def _blank_row(defaults):
    r = {c: np.nan for c in _COLS}
    r.update({"N_ambiguity": 0, "component": "", "wave_type": "", "mode": "",
              "lag": "", "stack_method": "", "pick_method": "", "ot_flag": "", "mode_overlap": 0,
              "scale_j": -1})
    r.update(defaults)
    return r


def _emit_rows(rows, nper, gv, score, corr, *, dist, azi, baz, comp, wave, mode, lag, pm,
               stack_method, snr_bank, snr_bb, snr_bb_other, snr_other_bank, env_bank,
               overlap, cfg, tau_max, extra=None):
    """Append unified rows for one labeled, measured pick set (group + optional phase).

    extra: optional {column: per-pick sequence} of QC values aligned with nper (xmode_amp,
    ot_flag, dU_rayfund, dU_rayot, dUdT_local, ...); applied verbatim per row."""
    for i in range(len(nper)):
        T = float(nper[i])
        U = float(gv[i])
        if T <= 0 or U <= 0:
            continue
        ratio = dist / (T * U)
        if ratio < cfg.MIN_LAMBDA_GROUP:
            continue
        r = _blank_row({"nominal_period": T, "group_velocity": U, "score": float(score[i]),
                        "ratio_d_lambda": ratio, "azimuth": azi, "backazimuth": baz, "distance": dist,
                        "component": comp, "wave_type": wave, "mode": mode, "lag": lag,
                        "stack_method": stack_method, "pick_method": pm})
        r["snr_nbG"] = _bank_at(snr_bank, T)
        r["snr_bb"] = snr_bb
        r["snr_bb_other"] = snr_bb_other
        r["snr_nbG_other"] = _bank_at(snr_other_bank, T)
        # Love QC: env_ratio (TT vs cross-term SNR), overlap flag, overtone-branch tag
        if wave == "love":
            r["env_ratio"] = _bank_at(env_bank, T)
            ov = int(overlap[i]) if overlap is not None and i < len(overlap) else 0
            r["mode_overlap"] = ov
            if mode == "overtone":
                r["ot_flag"] = "overlap" if ov else "sep"
        if corr is not None:
            r["T_centroid"] = corr["T_centroid"][i]
            r["T_inst"] = corr["T_inst"][i]
            r["T_scale"] = corr["T_scale"][i]
            r["scale_j"] = int(corr["scale_j"][i])
            cph = corr["phase_velocity"][i]
            namb = int(corr["N_ambiguity"][i])
            uph = corr["U_from_phase"][i]
            gated = ratio < cfg.MIN_LAMBDA_PHASE or (tau_max is not None and T > tau_max)
            if gated:
                cph, namb, uph = np.nan, 0, np.nan
            r["phase_velocity"] = cph
            r["N_ambiguity"] = namb
            r["U_from_phase"] = uph
        if extra:
            for col, vals in extra.items():
                if vals is not None and i < len(vals):
                    r[col] = vals[i]
        rows.append(r)


def _bank_at(bank, period):
    if bank is None:
        return np.nan
    pers, vals = bank
    return float(vals[int(np.argmin(np.abs(pers - period)))])


def rows_to_csv(rows):
    """Format row dicts to CSV text (header + lines). Plain float formatting (NaN -> 'nan')."""
    lines = [HEADER]
    for r in rows:
        cells = []
        for c in _COLS:
            v = r[c]
            if isinstance(v, str):
                cells.append(v)
            elif c in ("N_ambiguity", "mode_overlap"):
                cells.append(str(int(v)) if np.isfinite(v) else "0")
            elif c == "scale_j":
                cells.append(str(int(v)) if np.isfinite(v) else "-1")
            else:
                cells.append(f"{float(v):.4f}" if np.isfinite(v) else "nan")
        lines.append(",".join(cells))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- main entry point
def pick_all_modes(params, ccf, refs, stack_method, cfg=Config):
    """Return the full 8-way pick set for one station pair / stack method as a list of row dicts.

    params: {'dist','dt','azi','baz'}.
    ccf: {comp: {'neg','pos','sym': trace}} for at least ZZ/RR/RZ/ZR and TT (+ Love context comps).
    refs: {(wave, mode): phase_c_ref_callable} for the four (wave,mode) combos; missing/None disables
          phase for that stream (group still emitted). Love group references for ridge labeling are
          derived internally from the Love phase references via phase_ref_to_group_ref.
    """
    dist, dt, azi, baz = params["dist"], params["dt"], params["azi"], params["baz"]
    Tmax = dist / cfg.vave
    per_grid = np.arange(cfg.Tmin, Tmax, cfg.dT)
    if len(per_grid) == 0:
        return []
    tau_max = (dist / cfg.TAU_MAX_FACTOR) if cfg.TAU_MAX_FACTOR else None
    rows = []

    def snr_bank(sig):
        try:
            snr_nbG, snr_bb, _, _ = dispersion.nb_filt_gauss(
                sig, dt, 1.0 / per_grid, dist, alpha=cfg.gauss_alpha, vmin=cfg.vmin, vmax=cfg.vmax)
            return (per_grid, snr_nbG), float(snr_bb)
        except Exception:
            return None, np.nan

    def image_and_cwt(sig):
        cwt = dispersion.compute_cwt(sig, dist, dt, Tmin=cfg.Tmin, vmin=cfg.vmin, vmax=cfg.vmax,
                                     vave=cfg.vave)
        amp, per, vel, coi = dispersion.disp_image_from_cwt(
            cwt, dist, Tmin=cfg.Tmin, dT=cfg.dT, vmin=cfg.vmin, vmax=cfg.vmax, dvel=cfg.dvel,
            vave=cfg.vave)
        return cwt, amp, per, vel, coi

    # ---------------------------------------------------------------- Rayleigh: G_LR0 / G_LR1
    # Two passes: (1) synthesize + image BOTH modes, (2) pick & emit. The two-pass split exists so
    # every pick can be checked against the OTHER mode's image (in-memory mutual suppression, the
    # validate_modes.py anchor test: a genuine mode is WEAK in the other stack at its (T, U)).
    ray_curves = {}          # mode -> (nper, gv) argmax curve; consumed by the Love dU_ray* columns
    if all(c in ccf for c in RAYLEIGH_COMPS):
        comps0, comps1 = dispersion.phase_corrected_components(
            ccf["ZZ"][cfg.GLR_LAG], ccf["RR"][cfg.GLR_LAG], ccf["RZ"][cfg.GLR_LAG],
            ccf["ZR"][cfg.GLR_LAG], receiver_side_flip=cfg.GLR_RECEIVER_SIDE_FLIP)
        if cfg.GLR_STACK == "tfpws":
            g0, g1 = dispersion.tf_pws(comps0, dt), dispersion.tf_pws(comps1, dt)
        else:
            g0, g1 = np.sum(comps0, axis=0), np.sum(comps1, axis=0)
        glr = {}
        for comp, sig, mode in (("G_LR0", g0, "fundamental"), ("G_LR1", g1, "overtone")):
            try:
                cwt, amp, per, vel, coi = image_and_cwt(sig)
            except Exception as e:
                print(f"  G_LR CWT failed {comp}: {e}")
                continue
            bank, sbb = snr_bank(sig)
            glr[comp] = dict(cwt=cwt, amp=amp, per=per, vel=vel, coi=coi, bank=bank, sbb=sbb,
                             mode=mode, norm=_norm_image(amp))
        cross_of = {"G_LR0": "G_LR1", "G_LR1": "G_LR0"}
        # fundamental group curve for the overtone flags: the pair's own G_LR0 argmax, with the
        # reference-derived curve as fallback when G_LR0 yields nothing
        U0_ray_ref = phase_ref_to_group_ref(refs.get(("rayleigh", "fundamental")), per_grid)

        def u0_at(T):
            u = _curve_at(*ray_curves.get("fundamental", (None, None)), T=T, t_tol=cfg.LOVE_RAY_T_TOL)
            if not np.isfinite(u) and U0_ray_ref is not None:
                u = float(U0_ray_ref(T))
            return u

        for comp in ("G_LR0", "G_LR1"):          # G_LR0 first: its curve anchors the G_LR1 flags
            if comp not in glr:
                continue
            d = glr[comp]
            mode = d["mode"]
            c_ref = refs.get(("rayleigh", mode))
            xnorm = glr[cross_of[comp]]["norm"] if cross_of[comp] in glr else None
            for pm, nper, gv, score in pick_group_ridges(d["amp"], d["per"], d["vel"], dist, cfg):
                nper, gv, score = dispersion.remove_picks_coi(nper, gv, score, d["vel"], d["coi"])
                if len(nper) == 0:
                    continue
                if comp == "G_LR0" and pm == "argmax":
                    ray_curves["fundamental"] = (nper, gv)
                if comp == "G_LR1" and pm == "argmax":
                    ray_curves["overtone"] = (nper, gv)
                corr = dispersion.measure_corrections_and_phase(
                    d["cwt"], nper, gv, dist, c_ref=c_ref, phase_shift=cfg.PHASE_SHIFT["rayleigh"],
                    phase_offset=cfg.PHASE_OFFSET, use_period=cfg.PHASE_USE_PERIOD,
                    joint=cfg.PHASE_JOINT, smooth_weight=cfg.PHASE_SMOOTH_WEIGHT)
                extra = {
                    "xmode_amp": ([_amp_at(d["per"], d["vel"], xnorm, T, U)
                                   for T, U in zip(nper, gv)] if xnorm is not None else None),
                    "dUdT_local": _local_slopes(nper, gv) if pm == "argmax" else None,
                    "ot_flag": (_rayleigh_ot_flags(nper, gv, u0_at, dist, cfg)
                                if mode == "overtone" else None),
                }
                _emit_rows(rows, nper, gv, score, corr, dist=dist, azi=azi, baz=baz, comp=comp,
                           wave="rayleigh", mode=mode, lag=cfg.GLR_LAG, pm=pm,
                           stack_method=stack_method, snr_bank=d["bank"], snr_bb=d["sbb"],
                           snr_bb_other=np.nan, snr_other_bank=None, env_bank=None, overlap=None,
                           cfg=cfg, tau_max=tau_max, extra=extra)

    # ---------------------------------------------------------------- Love: TT fund + overtone
    if "TT" in ccf:
        love_fund_cref = refs.get(("love", "fundamental"))
        love_ot_cref = refs.get(("love", "overtone"))
        U0_ref = phase_ref_to_group_ref(love_fund_cref, per_grid)
        U1_ref = phase_ref_to_group_ref(love_ot_cref, per_grid)
        for lag in cfg.LOVE_LAGS:
            if lag not in ccf["TT"]:
                continue
            sig = ccf["TT"][lag]
            try:
                cwt, amp, per, vel, coi = image_and_cwt(sig)
            except Exception as e:
                print(f"  TT CWT failed {lag}: {e}")
                continue
            bank, sbb = snr_bank(sig)
            # cross-term context: env_ratio = TT snr / max(cross snr), snr_bb_other = max cross bb
            other_banks, other_bbs = [], []
            for c in cfg.LOVE_CONTEXT:
                if c in ccf and lag in ccf[c]:
                    ob, obb = snr_bank(ccf[c][lag])
                    if ob is not None:
                        other_banks.append(ob)
                        other_bbs.append(obb)
            snr_bb_other = float(np.max(other_bbs)) if other_bbs else np.nan
            other_combined = None
            env_bank = None
            if other_banks:
                other_max = np.max([b[1] for b in other_banks], axis=0)
                other_combined = (per_grid, other_max)
                if bank is not None:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        env_bank = (per_grid, bank[1] / np.where(other_max > 0, other_max, np.nan))
            for pm, nper, gv, score in pick_group_ridges(amp, per, vel, dist, cfg):
                nper, gv, score = dispersion.remove_picks_coi(nper, gv, score, vel, coi)
                if len(nper) == 0:
                    continue
                if cfg.PICK_LOVE_OVERTONE:
                    labeled = label_love_ridges(nper, gv, score, U0_ref, U1_ref, cfg.LOVE_SEP_TOL,
                                                overtone_margin=cfg.LOVE_OVERTONE_MARGIN)
                else:
                    # Overtone extraction off: do NOT label/drop fast ridges -- that carved a notch
                    # out of the fundamental distribution over the overtone-reference band. All TT
                    # ridges stay in the fundamental stream; Rayleigh-overtone-coincident picks are
                    # gated at merge (ot_leak on the dU_rayot column), not silently dropped here.
                    labeled = {"fundamental": (np.asarray(nper, float), np.asarray(gv, float),
                                               np.asarray(score, float),
                                               np.zeros(len(nper), dtype=int))}
                for mode, (lnp, lgv, lsc, lov) in labeled.items():
                    if len(lnp) == 0:
                        continue
                    c_ref = love_fund_cref if mode == "fundamental" else love_ot_cref
                    corr = dispersion.measure_corrections_and_phase(
                        cwt, lnp, lgv, dist, c_ref=c_ref, phase_shift=cfg.PHASE_SHIFT["love"],
                        phase_offset=cfg.PHASE_OFFSET, use_period=cfg.PHASE_USE_PERIOD,
                        joint=cfg.PHASE_JOINT, smooth_weight=cfg.PHASE_SMOOTH_WEIGHT)
                    # anti-leakage discriminators vs the pair's OWN Rayleigh branches: a Love pick
                    # sitting on the Rayleigh fundamental (|dU_rayfund| small) is the R-fund-on-TT
                    # leakage fingerprint; on the Rayleigh overtone -> overtone leakage.
                    extra = {
                        "dU_rayfund": [lU - _curve_at(*ray_curves.get("fundamental", (None, None)),
                                                      T=lT, t_tol=cfg.LOVE_RAY_T_TOL)
                                       for lT, lU in zip(lnp, lgv)],
                        "dU_rayot": [lU - _curve_at(*ray_curves.get("overtone", (None, None)),
                                                    T=lT, t_tol=cfg.LOVE_RAY_T_TOL)
                                     for lT, lU in zip(lnp, lgv)],
                        "dUdT_local": _local_slopes(lnp, lgv) if pm == "argmax" else None,
                    }
                    _emit_rows(rows, lnp, lgv, lsc, corr, dist=dist, azi=azi, baz=baz, comp="TT",
                               wave="love", mode=mode, lag=lag, pm=pm, stack_method=stack_method,
                               snr_bank=bank, snr_bb=sbb, snr_bb_other=snr_bb_other,
                               snr_other_bank=other_combined, env_bank=env_bank, overlap=lov,
                               cfg=cfg, tau_max=tau_max, extra=extra)
    return rows
