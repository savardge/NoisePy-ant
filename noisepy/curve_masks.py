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
      extension: |U - U_implied(c)| > 3 sqrt(S_U^2 + S_c^2), U_implied from an error-weighted,
                 deliberately over-smoothed spline of c (an under-smoothed derivative is noise --
                 the historical failure of naive U-c checks).
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
    # heavy, error-weighted smoothing of c(T); derivative of an under-smoothed spline is noise
    sp = UnivariateSpline(Tc[o], c[o], w=1.0 / np.maximum(Sc[o], 1e-4), k=3,
                          s=Tc.size * s_factor)
    idx = np.where(ov)[0]
    T = Tc[idx]
    Ui = np.interp(T, Tu, U)
    Si = np.interp(T, Tu, Su)
    cs, dcdT = sp(T), sp.derivative()(T)
    Uimp = cs / (1.0 + (T / cs) * dcdT)

    core = (Ui / c[idx] >= 1.0) & (dcdT >= 0.0)
    ext = np.abs(Ui - Uimp) > sigma_level * np.sqrt(Si ** 2 + Sc[idx] ** 2)
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
