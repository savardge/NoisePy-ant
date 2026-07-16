"""Depth-resolved reliability of a trans-D Vs(z) posterior from per-chain summaries.

Multimodality in a group/phase-velocity Vs inversion is depth-dependent: the shallow
structure is pinned by the data (independent chains agree), while below the sensitivity
floor the chains wander into different basins.  A single scalar `chain_disagree` cannot
see that.  This module derives a PER-DEPTH agreement diagnostic from compact per-chain
summaries (median + 16/84 percentiles of Vs(z), a few KB/cell) so the raw per-chain
traces can be deleted after a run.

Everything here is pure numpy (no BayHunter, no disba) so it runs in any env, both inside
run_bayhunter_cell (bayhunter env) and in post-processing (bayesbay_dev env).

Primary diagnostic -- a per-depth, simplified Gelman-Rubin ratio:

    rho(z) =  std_j( median_j Vs(z) )              (between-chain spread of chain medians)
              --------------------------------------
              mean_j( [p84_j(z) - p16_j(z)] / 2 )   (typical within-chain 1-sigma)

    rho <~ 0.5  -> chains converge to the same Vs within their own uncertainty  -> RESOLVED
    rho >~ 1.0  -> chains separated by more than their width -> marginal posterior is
                   multimodal, the median is meaningless there                  -> UNRESOLVED

Combined with an (optional) data-sensitivity depth floor -- a period no longer constrains
structure below ~0.4 of its wavelength -- to give a single reliable-depth mask + interval,
and a cell-level confidence flag from the kept-chain fraction and the worst rho in-band.
"""
import numpy as np

DELTA_LOGL = 5.0       # absolute log-likelihood units; chains within this of the best chain's
                       # median count as the same basin.  Replaces BayHunter's RELATIVE `dev`
                       # (|1 - logL/best| <= dev), whose absolute tolerance is dev*|best| and so
                       # depends entirely on the likelihood SCALE: across runs of identical
                       # construction here, best spans -34..+97, making dev=0.05 range from 0.11
                       # to 4.84 logL.  Near best~0 it explodes (best=-5.4 discarded a chain
                       # 0.4 logL away, keeping 1/16).  A likelihood RATIO is the meaningful
                       # quantity; a relative deviation is not.  Tightening dev does not help --
                       # it is the same artifact scaled down.
RHO_MAX = 1.0          # rho <= 1 is the natural boundary: kept chains' medians spread by no
                       # more than their own posterior width (between <= within), i.e. the
                       # per-chain posteriors overlap.  (0.7 proved overly strict: it cut bands
                       # on noise excursions, esp. when few kept chains make rho itself noisy.)
ABS_TOL = 0.05         # km/s; chains this close in absolute terms count as resolved regardless
                       # of rho (guards the near-surface where within-chain width -> 0)
WL_FRAC = 0.4          # wavelength fraction for the data-sensitivity depth floor (~1/3-1/2 lambda)
MAX_GAP = 4            # samples; close unresolved gaps up to this thick (0.2 km on the 50 m
                       # grid) -- adjacent depth samples are correlated, so a 2-4-sample rho
                       # blip is sampling noise, not independent evidence of multimodality
EPS = 1e-6


def _medfilt(x, k=5):
    """Small odd-window median filter (edge-replicated) to de-noise rho(z) before thresholding."""
    if k < 3 or x.size < k:
        return x
    k |= 1
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.array([np.nanmedian(xp[i:i + k]) for i in range(len(x))])


def per_depth_rho(p16, p50, p84, kept=None):
    """Between-chain/within-chain Vs(z) agreement ratio from per-chain (p16,p50,p84).

    p16/p50/p84: (nchain, ndepth) per-chain posterior percentiles of Vs(z).
    kept: optional bool (nchain,) -- restrict to the outlier-filtered (good-basin) chains.
    Returns (rho, between, within) each (ndepth,)."""
    p16, p50, p84 = map(np.asarray, (p16, p50, p84))
    if kept is not None:
        kept = np.asarray(kept, bool)
        if kept.sum() >= 2:                       # need >=2 chains to have a between-spread
            p16, p50, p84 = p16[kept], p50[kept], p84[kept]
    between = np.nanstd(p50, axis=0)
    within = np.nanmean((p84 - p16) / 2.0, axis=0)
    rho = between / np.where(within > EPS, within, np.nan)
    return rho, between, within


def wavelength_floor(periods, velocities, frac=WL_FRAC):
    """Approximate max resolvable depth [km]: frac * (the LONGEST WAVELENGTH in the data).

    periods [s], velocities [km/s] of the observed dispersion (any mode/measure). A period no
    longer resolves structure much below ~1/3-1/2 of its wavelength; frac~0.4 is the usual rule.
    (A rigorous version uses disba GroupSensitivity; this needs no disba.)

    Uses max(v*T) rather than the velocity at argmax(T). Those differ whenever several
    measurements share the longest period -- e.g. a JOINT group+phase inversion, where at T_max
    both a group U and a phase c exist and c > U by ~30-50%. argmax(T) returns whichever was
    concatenated first (group, in run_bayhunter_cell's wave order), so the floor was computed
    from the SHORTER group wavelength and the joint run's depth reach was under-reported as
    equal to group-only (measured 2.45 km vs phase-only 4.38 km) even though it strictly
    contains more information. The depth reach is set by the longest wavelength present."""
    T = np.asarray(periods, float)
    U = np.asarray(velocities, float)
    m = np.isfinite(T) & np.isfinite(U) & (T > 0) & (U > 0)
    if not m.any():
        return np.inf
    return float(frac * np.max(U[m] * T[m]))


def _close_gaps(mask, max_gap=2):
    """Fill runs of <=max_gap False samples that sit between True samples (binary closing)."""
    m = mask.copy()
    n = len(m)
    i = 0
    while i < n:
        if m[i]:
            i += 1; continue
        j = i
        while j < n and not m[j]:
            j += 1
        if i > 0 and j < n and (j - i) <= max_gap:      # gap bounded by True on both sides
            m[i:j] = True
        i = j
    return m


def _longest_run(mask):
    """(start, end) inclusive indices of the longest contiguous True run; (None, None) if none."""
    best = (None, None); best_len = 0
    i = 0; n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1; continue
        j = i
        while j < n and mask[j]:
            j += 1
        if j - i > best_len:
            best_len = j - i; best = (i, j - 1)
        i = j
    return best


def reliability(depth, p16, p50, p84, kept=None, z_floor=None, rho_max=RHO_MAX,
                abs_tol=ABS_TOL, medfilt_k=5, max_gap=MAX_GAP):
    """Full per-depth reliability assessment.

    A depth is chain-resolved if the kept chains agree either RELATIVELY (rho <= rho_max) or
    ABSOLUTELY (between-chain median spread <= abs_tol) -- the abs_tol arm guards the near
    surface, where the within-chain width collapses and the ratio blows up on a few-m/s spread.
    reliable(z) = chain-resolved(z) AND (z <= z_floor).  Long-period curves have little shallow
    sensitivity, so the resolved zone is generally a MID-DEPTH band, not anchored at the surface:
    the reported interval is the longest contiguous reliable run (small gaps closed).

    reln_frac = fraction of the data-sensitive band (z <= z_floor) that is reliable -- how much
    of what the data can see the chains actually agree on.

    Returns dict(rho, rho_smooth, between, within, resolved, reliable, z_reliable_min,
    z_reliable_max, reln_frac, z_floor).
    """
    depth = np.asarray(depth, float)
    rho, between, within = per_depth_rho(p16, p50, p84, kept)
    rho_s = _medfilt(rho, medfilt_k)
    between_s = _medfilt(between, medfilt_k)
    resolved = (rho_s <= rho_max) | (between_s <= abs_tol)
    resolved = np.where(np.isnan(rho_s), False, resolved)
    in_floor = (depth <= z_floor) if (z_floor is not None and np.isfinite(z_floor)) \
        else np.ones(len(depth), bool)
    reliable = _close_gaps(resolved & in_floor, max_gap)
    s, e = _longest_run(reliable)
    z_reliable_min = float(depth[s]) if s is not None else 0.0
    z_reliable_max = float(depth[e]) if e is not None else 0.0
    n_floor = int(in_floor.sum())
    reln_frac = float((reliable & in_floor).sum() / n_floor) if n_floor else 0.0
    return dict(rho=rho, rho_smooth=rho_s, between=between, between_smooth=between_s,
                within=within, resolved=resolved, reliable=reliable,
                z_reliable_min=z_reliable_min, z_reliable_max=z_reliable_max,
                reln_frac=reln_frac,
                z_floor=(float(z_floor) if z_floor is not None else np.inf))


def confidence(frac_kept, reln_frac, n_kept):
    """Cell-level trust flag.

    frac_kept  : fraction of chains in the good (outlier-filtered) basin -- posterior support.
    reln_frac  : fraction of the data-sensitive depth band that is chain-resolved
                 (z_reliable_max / min(z_floor, depth_max)) -- how much of what the data can
                 see the chains actually agree on.

    UNDER PARALLEL TEMPERING, `n_kept` counts chains that were EVER at T=1, not the instantaneous
    ladder width -- and that is the right definition, not a workaround. Temperatures SWAP between
    chains (mcmcOptimizer._swap_temperatures), so with t1chains=3 of 16 it is not 3 fixed chains
    that stay cold: over a full run nearly every chain visits T=1 and contributes valid cold
    samples to the posterior. This falls out for free because the upstream `loglike_med` is
    bh_diagnostics.chain_medians_t1, i.e. a median over each chain's T=1 samples only, which
    returns -inf for a chain that never went cold -- so such a chain fails `best - ll <= delta` and
    is excluded, while every chain that did go cold is counted. The `n_kept >= 8` gate below is
    therefore NOT a function of t1chains, and PT does not silently downgrade cells.
    """
    if n_kept >= 8 and frac_kept >= 0.5 and reln_frac >= 0.8:
        return "high"
    if n_kept >= 3 and frac_kept >= 0.25 and reln_frac >= 0.5:
        return "marginal"
    return "low"


def assess(depth, p16, p50, p84, loglike_med, periods=None, velocities=None,
           delta=DELTA_LOGL, rho_max=RHO_MAX):
    """Convenience: kept-mask (absolute Delta-logL cut) -> reliability -> confidence, one call.

    loglike_med: (nchain,) per-chain post-burnin median log-likelihood (best = max).
    periods/velocities: observed dispersion for the wavelength depth floor (optional).
    delta: keep chains within this many log-likelihood units of the best chain -- scale-free,
        unlike BayHunter's relative `dev` (see DELTA_LOGL).
    Returns a dict with rho(z), reliable mask/interval, and scalar QC (frac_kept, confidence).
    """
    loglike_med = np.asarray(loglike_med, float)
    best = np.nanmax(loglike_med) if loglike_med.size else np.nan
    if np.isfinite(best):
        # -inf entries (chains that never sampled at T=1 under PT, per
        # bh_diagnostics.median_at_t1) give inf > delta and are correctly dropped here.
        kept = (best - loglike_med) <= delta
    elif loglike_med.size and np.all(np.isneginf(loglike_med)):
        # every chain -inf => not "no information, keep everything" but "NO chain ever sampled at
        # T=1", which is a broken run. Keeping all of them would launder it into a confident cell.
        kept = np.zeros(loglike_med.size, bool)
    else:
        kept = np.ones(len(loglike_med), bool)
    n_kept = int(kept.sum())
    frac_kept = n_kept / max(1, len(loglike_med))
    z_floor = wavelength_floor(periods, velocities) if periods is not None else None
    rel = reliability(depth, p16, p50, p84, kept, z_floor, rho_max)
    conf = confidence(frac_kept, rel["reln_frac"], n_kept)
    rel.update(kept=kept, n_kept=n_kept, frac_kept=frac_kept, confidence=conf)
    return rel
