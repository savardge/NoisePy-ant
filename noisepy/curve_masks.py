"""Adaptive, per-cell exclusion masks for well/cell dispersion curves.

Replaces the blanket trims (phase T >= 2.5 s; any fixed group cut) with two data-driven
criteria whose cut periods are OUTCOMES, not inputs. Design agreed 2026-07-16 (see
fund-phase-branch-diagnosis memory and the two test_2026-07-16_noise_regime_pair READMEs).

CRITERION 1 -- kinematic consistency (phase), per cell and wave type
    Group U and phase c at the same cell are two views of one dispersion relation:
    U = c / (1 + (T/c) dc/dT). Exclude phase where the measured pair is kinematically
    inconsistent:
      core     : U/c >= 1 while the (heavily smoothed) c(T) slope is >= 0 -- impossible in any
                 layered medium (the slope guard closes the anomalous-dispersion loophole).
      extension (v2, integral-domain): |c - c_pred| > 3 sqrt(S_c^2 + S_pred^2), with c_pred from
                 integrating the measured group slowness (k(w) = k_a + int 1/U dw; Bensen) anchored
                 on the longest-period overlap points. Integration smooths group noise instead of
                 amplifying it; a wrong 2piN phase sheet displaces measured k by a near-constant
                 the anchored prediction does not follow, so the 1-2 s fast band is flagged by
                 construction. (v1 used the derivative form U_implied = c/(1+(T/c)dc/dT) on a
                 heavily smoothed spline and UNDER-TRIMMED: the smoothing flattened the dispersion
                 step, the 1-2 s band went unflagged, and the adaptive-v1 arm railed fund_phase
                 0.98 at Basel-1 where the blanket cut gave 0.07.)
      trigger  : contiguous runs of (core | extension) flags EXCLUDE only if they contain >= 1
                 core violation. A 3-sigma-only run never cuts (spline-wiggle guard).
      cut      : for a triggering run touching the short-period end of the overlap, exclude all
                 phase points with T <= run's upper edge (below the run, phase sits on the same
                 wrong 2piN sheet). A detached triggering run excludes only itself, loudly.
      no-overlap rule (user decision 2026-07-16): phase BELOW the shortest group period is
      excluded (unvalidatable, and the artifact lives at short T); phase ABOVE the longest group
      period is kept (that is where group cannot sense and phase is the only constraint).

CRITERION 2 -- near-field + azimuthal coverage (group), per cell and period
    From the tomography's own input pick table, the picks whose straight inter-station chord
    passes within R_KM of the cell are the data that constrained this cell's map value. Per
    period T (each pick's own lambda = U_pick * T):
      A: frac(r < FARFIELD_FACTOR * lambda) <= 0.5      (majority far-field)
      B: far-field picks occupy >= MIN_AZI_BINS of NBINS 30-degree azimuth bins (mod 180), AND
         the largest contiguous empty arc is <= MAX_GAP_DEG (three adjacent occupied bins pass
         the count but mean a single 90-degree corridor -- then the "isotropic" cell value is a
         directional apparent velocity aliasing azimuthal anisotropy; the gap condition rejects
         exactly that). Near-field picks never count toward coverage.
      C: n_crossing >= 3. Periods with NO crossing picks are excluded outright: a map value
         there is regularization borrowed from neighbours, not a measurement of this cell.
    keep(T) = A and B and C. Applied uniformly to fund, overtone and love group.

Parameters and their defense (all surfaced as module constants; sensitivity belongs in the
supplement): SIGMA_LEVEL=3 (standard); SPLINE_S_FACTOR=2 (heavy smoothing, cuts move <~ one
period sample at 1x/4x); FARFIELD_FACTOR=2.5 (the project's existing pick-gate factor);
R_KM=1.0 (about two 0.5-km grid cells; user decision); NBINS=6/MIN_AZI_BINS=3/MAX_GAP_DEG=90
(standard coverage heuristics).

Pure numpy on purpose: runs in both the bayhunter and bayesbay envs.
"""
import csv
import os

import numpy as np
from scipy.interpolate import UnivariateSpline

SIGMA_LEVEL = 3.0
SPLINE_S_FACTOR = 2.0
FARFIELD_FACTOR = 2.5
R_KM = 1.0
NBINS = 6                     # azimuth bins over [0, 180)
MIN_AZI_BINS = 3
MAX_GAP_DEG = 90.0
MIN_CROSSING = 3
FRAC_NEAR_MAX = 0.5
PERIOD_TOL = 0.051            # curve period -> pick-table inst_period matching


# --------------------------------------------------------------------------- criterion 1
def phase_consistency_mask(Tu, U, Su, Tc, c, Sc,
                           sigma_level=SIGMA_LEVEL, s_factor=SPLINE_S_FACTOR):
    """Keep-mask over the PHASE points (Tc, c) from the kinematic-consistency criterion.

    Returns dict(keep, T_cut, n_core, runs, note). T_cut is None when nothing triggers.
    """
    Tu, U, Su = (np.asarray(x, float) for x in (Tu, U, Su))
    Tc, c, Sc = (np.asarray(x, float) for x in (Tc, c, Sc))
    keep = np.ones(Tc.size, bool)
    out = dict(keep=keep, T_cut=None, n_core=0, runs=[], note="")

    if Tu.size < 4 or Tc.size < 4:
        out["note"] = "too few points; no exclusion"
        return out

    # no-overlap rule: phase below the group band is unvalidatable -> excluded;
    # phase above the group band is kept unconditionally.
    below = Tc < Tu.min() - 1e-9
    keep[below] = False

    ov = (Tc >= Tu.min() - 1e-9) & (Tc <= Tu.max() + 1e-9)
    if ov.sum() < 4:
        out["note"] = "overlap <4 points; only the below-group rule applied"
        out["T_cut"] = float(Tu.min()) if below.any() else None
        return out

    o = np.argsort(Tc)
    # spline used ONLY for the core test's slope guard (anomalous-dispersion loophole); the
    # extension test is integral-domain (below), NOT derivative-based. v1 used
    # |U - c/(1+(T/c)dc/dT)| here and UNDER-TRIMMED: the heavy smoothing that makes dc/dT usable
    # also flattens the basin dispersion step, so the mis-branch band (phase too FAST at 1-2 s,
    # which does NOT violate U/c>=1) went unflagged -- measured empirically 2026-07-16, adaptive
    # v1 arm, Basel-1 fund_phase rail 0.98 vs 0.07 under the blanket 2.5 s cut.
    sp = UnivariateSpline(Tc[o], c[o], w=1.0 / np.maximum(Sc[o], 1e-4), k=3,
                          s=Tc.size * s_factor)
    idx = np.where(ov)[0]
    T = Tc[idx]
    Ui = np.interp(T, Tu, U)
    Si = np.interp(T, Tu, Su)
    dcdT = sp.derivative()(T)

    core = (Ui / c[idx] >= 1.0) & (dcdT >= 0.0)

    # EXTENSION v2 -- integral-domain consistency (Bensen: k(w) = k(w_a) + int s_U dw, s_U = 1/U;
    # c_pred = w/k). Integration SMOOTHS group noise instead of amplifying it, and the 1-2 s
    # fast band stands out by construction: a wrong 2piN sheet displaces k by a near-constant,
    # while the U-integral prediction stays on the true sheet anchored at long T.
    # Anchor: the N_ANCHOR longest-period overlap points (where phase is consistent -- they are
    # the far side of any short-T artifact), constant fitted as the median of
    # k_meas - int(s_U) there, so anchor noise enters as a robust average, not one point.
    w = 2.0 * np.pi / T                                  # ascending T = descending w
    ow = np.argsort(w)                                   # integrate in ascending w
    ws, Ts = w[ow], T[ow]
    sU = 1.0 / np.interp(Ts, Tu, U)
    dw = np.diff(ws)
    dk = 0.5 * (sU[1:] + sU[:-1]) * dw                  # trapezoid increments
    kint = np.concatenate([[0.0], np.cumsum(dk)])        # int_{w0}^{w} s_U dw'
    kmeas = ws / c[idx][ow]
    n_anchor = max(3, min(5, ws.size // 4))
    k0 = np.median((kmeas - kint)[:n_anchor])            # smallest w = longest T = anchor side
    cpred = ws / (k0 + kint)
    # sigma of the prediction: independent-noise propagation of the group term through the
    # integral + anchor scatter (conservative, stated in the paper as such)
    varints = np.concatenate([[0.0], np.cumsum((0.5 * dw) ** 2 * (
        (np.interp(Ts, Tu, Su)[1:] / np.interp(Ts, Tu, U)[1:] ** 2) ** 2 +
        (np.interp(Ts, Tu, Su)[:-1] / np.interp(Ts, Tu, U)[:-1] ** 2) ** 2))])
    var_anchor = np.var((kmeas - kint)[:n_anchor]) / n_anchor
    scpred = (cpred ** 2 / ws) * np.sqrt(varints + var_anchor)
    ext_w = np.abs(c[idx][ow] - cpred) > sigma_level * np.sqrt(Sc[idx][ow] ** 2 + scpred ** 2)
    ext = np.empty_like(ext_w)
    ext[ow] = ext_w                                      # back to ascending-T order
    flag = core | ext
    out["n_core"] = int(core.sum())
    if not core.any():
        out["T_cut"] = float(Tu.min()) if below.any() else None
        return out

    # contiguous runs of flags on the overlap axis; only runs containing a core violation trigger
    runs, i = [], 0
    while i < flag.size:
        if flag[i]:
            j = i
            while j + 1 < flag.size and flag[j + 1]:
                j += 1
            if core[i:j + 1].any():
                runs.append((i, j))
            i = j + 1
        else:
            i += 1
    out["runs"] = [(float(T[a]), float(T[b])) for a, b in runs]

    for a, b in runs:
        if a <= 1:                                   # anchored at the short end of the overlap
            keep[Tc <= T[b] + 1e-9] = False
            out["T_cut"] = max(out["T_cut"] or 0.0, float(T[b]))
        else:                                        # detached run: exclude only itself, loudly
            keep[(Tc >= T[a] - 1e-9) & (Tc <= T[b] + 1e-9)] = False
            out["note"] += (f"DETACHED inconsistent run {T[a]:.2f}-{T[b]:.2f}s excluded "
                            f"(not the diagnosed short-T mis-branch -- inspect!); ")
    return out


# --------------------------------------------------------------------------- criterion 2
def _load_stations(stations_csv):
    st = {}
    with open(stations_csv) as fh:
        for row in csv.DictReader(fh):
            st[row["id"]] = (float(row["longitude"]), float(row["latitude"]))
    return st


def _seg_point_dist_km(p1, p2, q, lat0):
    """Distance (km) from point q to segment p1-p2; all (lon, lat), local flat projection."""
    kx = 111.0 * np.cos(np.deg2rad(lat0))
    a = np.array([(p1[0] - q[0]) * kx, (p1[1] - q[1]) * 111.0])
    b = np.array([(p2[0] - q[0]) * kx, (p2[1] - q[1]) * 111.0])
    d = b - a
    L2 = float(d @ d)
    t = 0.0 if L2 == 0 else float(np.clip(-(a @ d) / L2, 0.0, 1.0))
    return float(np.hypot(*(a + t * d)))


def group_nearfield_mask(picks_csv, stations_csv, cell_lon, cell_lat, periods,
                         r_km=R_KM, farfield_factor=FARFIELD_FACTOR,
                         frac_near_max=FRAC_NEAR_MAX, nbins=NBINS,
                         min_azi_bins=MIN_AZI_BINS, max_gap_deg=MAX_GAP_DEG,
                         min_crossing=MIN_CROSSING):
    """Keep-mask over `periods` for a cell's GROUP curve, from the picks that built the cell.

    Returns dict(keep, frac_near, n_cross, n_far, azi_bins, max_gap, note) -- arrays over
    `periods` (nan where no picks).
    """
    st = _load_stations(stations_csv)
    periods = np.asarray(periods, float)

    # one geometry pass per unique pair (tables repeat the pair per period row)
    pair_cross = {}
    rows = {"T": [], "U": [], "r": [], "azi": [], "pair": []}
    with open(picks_csv) as fh:
        for row in csv.DictReader(fh):
            pair = row["station_pair"]
            hit = pair_cross.get(pair)
            if hit is None:
                s1, s2 = st.get(row["stasrc"]), st.get(row["starcv"])
                hit = (s1 is not None and s2 is not None and
                       _seg_point_dist_km(s1, s2, (cell_lon, cell_lat), cell_lat) <= r_km)
                pair_cross[pair] = hit
            if not hit:
                continue
            rows["T"].append(float(row["inst_period"]))
            rows["U"].append(float(row["group_velocity"]))
            rows["r"].append(float(row["distance"]))
            rows["azi"].append(float(row["azimuth"]))
    Tp = np.array(rows["T"]); Up = np.array(rows["U"])
    rp = np.array(rows["r"]); az = np.array(rows["azi"]) % 180.0

    n = periods.size
    out = dict(keep=np.zeros(n, bool), frac_near=np.full(n, np.nan),
               n_cross=np.zeros(n, int), n_far=np.zeros(n, int),
               azi_bins=np.zeros(n, int), max_gap=np.full(n, np.nan),
               note=f"{int(sum(pair_cross.values()))} crossing pairs of {len(pair_cross)}")
    binw = 180.0 / nbins
    for k, T in enumerate(periods):
        m = np.abs(Tp - T) < PERIOD_TOL
        nc = int(m.sum())
        out["n_cross"][k] = nc
        if nc == 0:
            continue                                  # keep stays False: not a measurement here
        lam = Up[m] * Tp[m]
        near = rp[m] < farfield_factor * lam
        fn = float(np.mean(near))
        out["frac_near"][k] = fn
        far_az = az[m][~near]
        out["n_far"][k] = far_az.size
        occ = np.zeros(nbins, bool)
        occ[np.floor(far_az / binw).astype(int) % nbins] = True
        out["azi_bins"][k] = int(occ.sum())
        # largest contiguous empty arc on the circular (mod-180) bin ring
        if occ.all():
            gap = 0.0
        elif not occ.any():
            gap = 180.0
        else:
            ring = np.concatenate([occ, occ])
            best, run = 0, 0
            for v in ring:
                run = run + 1 if not v else 0
                best = max(best, run)
            gap = min(best, nbins) * binw
        out["max_gap"][k] = gap
        out["keep"][k] = (fn <= frac_near_max and out["azi_bins"][k] >= min_azi_bins
                          and gap <= max_gap_deg and nc >= min_crossing)
    return out


# --------------------------------------------------------------------------- batch criterion 2
def build_c2_table(picks_csv, stations_csv, cells_lonlat, periods=None,
                   r_km=R_KM, farfield_factor=FARFIELD_FACTOR, frac_near_max=FRAC_NEAR_MAX,
                   nbins=NBINS, min_azi_bins=MIN_AZI_BINS, max_gap_deg=MAX_GAP_DEG,
                   min_crossing=MIN_CROSSING, cache=None, verbose=True):
    """Vectorized criterion 2 for MANY cells at once: keep[ncells, nperiods] + diagnostics.

    Same rules as group_nearfield_mask (single-cell reference implementation), restructured for
    grid rollout: one geometry pass over unique pairs x all cells, then per-(cell, period)
    aggregation. `cells_lonlat` is (ncells, 2) [lon, lat]; `periods` defaults to the pick table's
    own grid. Pass `cache` (an .npz path) to persist/reuse -- the table depends only on the pick
    table + station coords + cell list + parameters, all of which are hashed into the cache key.
    """
    cells_lonlat = np.asarray(cells_lonlat, float)
    key = None
    if cache is not None:
        import hashlib
        h = hashlib.sha256()
        for fp in (picks_csv, stations_csv):
            h.update(fp.encode())
            h.update(str(os.path.getmtime(fp)).encode())
        h.update(cells_lonlat.tobytes())
        h.update(np.asarray([r_km, farfield_factor, frac_near_max, nbins, min_azi_bins,
                             max_gap_deg, min_crossing], float).tobytes())
        key = h.hexdigest()[:16]
        if os.path.exists(cache):
            z = np.load(cache, allow_pickle=True)
            if str(z.get("key")) == key:
                if verbose:
                    print(f"    c2 table cache hit: {os.path.basename(cache)}")
                return {k: z[k] for k in z.files}

    st = _load_stations(stations_csv)
    # one pass over the table: per-row arrays + unique-pair endpoint geometry
    pair_id, pair_xy = {}, []
    rT, rU, rr, raz, rp = [], [], [], [], []
    lat0 = float(np.mean(cells_lonlat[:, 1]))
    kx = 111.0 * np.cos(np.deg2rad(lat0))
    with open(picks_csv) as fh:
        for row in csv.DictReader(fh):
            pair = row["station_pair"]
            pid = pair_id.get(pair)
            if pid is None:
                s1, s2 = st.get(row["stasrc"]), st.get(row["starcv"])
                if s1 is None or s2 is None:
                    pair_id[pair] = -1
                    continue
                pid = len(pair_xy)
                pair_id[pair] = pid
                pair_xy.append([s1[0] * kx, s1[1] * 111.0, s2[0] * kx, s2[1] * 111.0])
            elif pid == -1:
                continue
            rT.append(float(row["inst_period"])); rU.append(float(row["group_velocity"]))
            rr.append(float(row["distance"])); raz.append(float(row["azimuth"]) % 180.0)
            rp.append(pid)
    P = np.array(pair_xy)                                 # (npairs, 4) km
    rT, rU, rr, raz = (np.array(x) for x in (rT, rU, rr, raz))
    rp = np.array(rp, int)
    if periods is None:
        periods = np.unique(np.round(rT, 3))
    periods = np.asarray(periods, float)
    peridx = np.full(rT.size, -1, int)                    # row -> period bin
    for k, T in enumerate(periods):
        peridx[np.abs(rT - T) < PERIOD_TOL] = k

    M, K = cells_lonlat.shape[0], periods.size
    out = dict(keep=np.zeros((M, K), bool), frac_near=np.full((M, K), np.nan),
               n_cross=np.zeros((M, K), int), azi_bins=np.zeros((M, K), int),
               max_gap=np.full((M, K), np.nan), periods=periods, cells_lonlat=cells_lonlat)
    binw = 180.0 / nbins
    a1 = P[:, :2]; a2 = P[:, 2:]; seg = a2 - a1
    L2 = np.maximum(np.einsum("ij,ij->i", seg, seg), 1e-12)
    lam_rows = rU * rT
    near_rows = rr < farfield_factor * lam_rows
    azbin_rows = np.floor(raz / binw).astype(int) % nbins
    for m in range(M):
        q = np.array([cells_lonlat[m, 0] * kx, cells_lonlat[m, 1] * 111.0])
        t = np.clip(np.einsum("ij,ij->i", q - a1, seg) / L2, 0.0, 1.0)
        d = np.hypot(*(a1 + t[:, None] * seg - q).T)
        crossing = d <= r_km                              # per pair
        rowm = crossing[rp]
        for k in range(K):
            sel = rowm & (peridx == k)
            nc = int(sel.sum())
            out["n_cross"][m, k] = nc
            if nc == 0:
                continue
            near = near_rows[sel]
            fn = float(near.mean())
            out["frac_near"][m, k] = fn
            occ = np.zeros(nbins, bool)
            occ[azbin_rows[sel][~near]] = True
            nb = int(occ.sum())
            out["azi_bins"][m, k] = nb
            if occ.all():
                gap = 0.0
            elif not occ.any():
                gap = 180.0
            else:
                ring = np.concatenate([occ, occ])
                best = run = 0
                for v in ring:
                    run = run + 1 if not v else 0
                    best = max(best, run)
                gap = min(best, nbins) * binw
            out["max_gap"][m, k] = gap
            out["keep"][m, k] = (fn <= frac_near_max and nb >= min_azi_bins
                                 and gap <= max_gap_deg and nc >= min_crossing)
        if verbose and (m + 1) % 100 == 0:
            print(f"    c2 table: {m + 1}/{M} cells", flush=True)
    if cache is not None:
        out["key"] = key
        np.savez_compressed(cache, **out)
        if verbose:
            print(f"    c2 table cached -> {cache}")
    return out


def c2_keep_for_periods(table, cell_index, T):
    """Boolean keep-mask for curve periods T (nearest table period within PERIOD_TOL;
    periods with no table entry are DROPPED -- no crossing picks means no measurement)."""
    tp = np.asarray(table["periods"], float)
    keep_row = np.asarray(table["keep"][cell_index], bool)
    T = np.asarray(T, float)
    out = np.zeros(T.size, bool)
    for i, t in enumerate(T):
        j = int(np.argmin(np.abs(tp - t)))
        if abs(tp[j] - t) < PERIOD_TOL:
            out[i] = keep_row[j]
    return out


# --------------------------------------------------------------------------- application
def apply_masks_to_workdir(src_work, dst_work, net_inputs, cell_lon, cell_lat,
                           waves=("fund", "overtone", "love"), verbose=True):
    """Read disp_*.txt from src_work, write adaptively-masked copies into dst_work.

    Order: criterion 2 on each group curve first (pick geometry only), then criterion 1 on each
    phase curve against the C2-SURVIVING group band (untrusted group never judges phase).
    Returns a report dict per wave. net_inputs = the network's 1_velocity_maps/inputs dir.
    """
    os.makedirs(dst_work, exist_ok=True)
    report = {}
    kept_group = {}
    for w in waves:
        fp = os.path.join(src_work, f"disp_{w}.txt")
        if not os.path.exists(fp):
            continue
        d = np.loadtxt(fp)
        g = group_nearfield_mask(os.path.join(net_inputs, f"picks_{w}_uni.csv"),
                                 os.path.join(net_inputs, "stations.csv"),
                                 cell_lon, cell_lat, d[:, 0])
        dk = d[g["keep"]]
        report[w] = dict(n0=len(d), n1=len(dk), crit2=g)
        if verbose:
            drop = d[~g["keep"], 0]
            print(f"    {w:<9} group: {len(d)} -> {len(dk)}"
                  + (f"  dropped T={np.round(drop, 2).tolist()}" if drop.size else ""))
        if len(dk) >= 4:
            np.savetxt(os.path.join(dst_work, f"disp_{w}.txt"), dk, fmt="%.6f")
            kept_group[w] = dk
        elif verbose:
            print(f"    {w:<9} group: <4 points survive -> curve DROPPED")

    for w in waves:
        fp = os.path.join(src_work, f"disp_{w}_phase.txt")
        if not os.path.exists(fp):
            continue
        d = np.loadtxt(fp)
        if w not in kept_group:
            np.savetxt(os.path.join(dst_work, f"disp_{w}_phase.txt"), d, fmt="%.6f")
            report[f"{w}_phase"] = dict(n0=len(d), n1=len(d), crit1=None)
            if verbose:
                print(f"    {w:<9} phase: no surviving group curve -> kept untouched ({len(d)})")
            continue
        gk = kept_group[w]
        r1 = phase_consistency_mask(gk[:, 0], gk[:, 1], gk[:, 2], d[:, 0], d[:, 1], d[:, 2])
        dk = d[r1["keep"]]
        report[f"{w}_phase"] = dict(n0=len(d), n1=len(dk), crit1=r1)
        if verbose:
            tcut = r1["T_cut"]
            print(f"    {w:<9} phase: {len(d)} -> {len(dk)}"
                  f"  T_cut={'%.2f s' % tcut if tcut else 'none'}"
                  f"  n_core={r1['n_core']}  runs={[(round(a,2),round(b,2)) for a,b in r1['runs']]}"
                  + (f"  {r1['note']}" if r1["note"] else ""))
        if len(dk) >= 4:
            np.savetxt(os.path.join(dst_work, f"disp_{w}_phase.txt"), dk, fmt="%.6f")
        elif verbose:
            print(f"    {w:<9} phase: <4 points survive -> curve DROPPED")
    return report
