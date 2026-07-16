"""Per-cell, per-period reliability metrics and dispersion-curve trimming criteria.

Motivation: swtomotv is Tarantola-Valette, so the stored slowness uncertainty is prior-bounded
and stays *low even where there is little data* — hiding poor reliability at the array edge and
long periods. This module quantifies period reliability per cell and trims each cell's dispersion
curve to its well-resolved periods before 1D Vs inversion, via three selectable criteria:

  - "tomographic": keep T iff res_diag(cell,T) >= R_min          (resolution-matrix diagonal)
  - "physical"   : keep T iff d_edge(cell) >= alpha*lambda(T)     (wavelength vs distance-to-edge)
                   AND kernel sampling depth z_eff(T) <= beta*depth_max   (disba GroupSensitivity)
  - "combined"   : intersection of the two

res_diag is read from the same production npz maps as `vs_inversion.load_cell_curves`; lambda uses
the VSG reference phase curves; the array edge is the station convex hull (grid-km frame); z_eff is
the |kernel|-weighted mean depth of the disba group-velocity sensitivity kernel.
"""
import glob
import os

import numpy as np

PROJ = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
_REF = {"fund": "fundamental", "overtone": "overtone", "love": "love"}
# wave key -> (disba/surf96 wave type, mode index); mirrors vs_inversion.WAVEDEF.
WAVEDEF = {"fund": ("rayleigh", 0), "overtone": ("rayleigh", 1), "love": ("love", 0)}
DISBA_MODE = {w: m for w, (_, m) in WAVEDEF.items()}
# Per-net path overrides for alternate tomography grids (e.g. finer dx into a new output dir).
# Station positions in grid-km (xstat/ystat) don't depend on dx, so the default cache is reusable.
# Keys are either `net` (applies to Rayleigh waves) or `(net, wave)` (per-wave, e.g. Love maps
# live under a separate swtomotv-output-love-<dx> tree with a net-specific dx).
PROD_ROOT = {}   # net or (net, wave) -> production root (holds {wave}/map_T*.npz)
CACHE_CSV = {}   # net -> stations_in_grid.csv


# The hardcoded fallbacks point at swtomotv-output/, which is NOT production (production is
# swtomotv-output-uni) and vanishes after the tomo/ reorg. Silently returning a stale/wrong root
# would trim periods against the wrong maps, so a fallback that doesn't exist raises instead of
# being used. Callers that set PROD_ROOT[net] / CACHE_CSV[net] explicitly (e.g. well_vs_qc.py's
# --production-root) bypass this entirely.
def _prod_root(net, wave="fund"):
    if (net, wave) in PROD_ROOT:
        return PROD_ROOT[(net, wave)]
    if net in PROD_ROOT:
        return PROD_ROOT[net]
    fallback = f"{PROJ}/{net}/tomo/swtomotv-output/production"
    if not os.path.isdir(fallback):
        raise FileNotFoundError(
            f"period_resolution: no production root registered for net '{net}' and the legacy "
            f"fallback '{fallback}' does not exist. Set PROD_ROOT['{net}'] (or pass "
            f"--production-root upstream) to the current production maps.")
    return fallback


def _cache_csv(net):
    if net in CACHE_CSV:
        return CACHE_CSV[net]
    fallback = f"{PROJ}/{net}/tomo/swtomotv-output/cache/stations_in_grid.csv"
    if not os.path.exists(fallback):
        raise FileNotFoundError(
            f"period_resolution: no station cache registered for net '{net}' and the legacy "
            f"fallback '{fallback}' does not exist. Set CACHE_CSV['{net}'] to the current "
            f"production cache's stations_in_grid.csv.")
    return fallback
# R_frac: keep periods whose res_diag >= R_frac * (this cell's peak res_diag) -- relative, because
# the Tarantola-Valette res_diag is low in absolute terms everywhere (heavy regularization). R_min:
# optional absolute floor (0 = off). alpha: wavelengths of clearance from the array edge. beta:
# fraction of depth_max the sensitivity kernel may reach.
DEFAULTS = {"R_frac": 0.5, "R_min": 0.0, "alpha": 0.5, "beta": 1.0,
            "depth_max": 6.0, "n_min_keep": 3}


# ---------------------------------------------------------------- wavelength
def _ref_phase(net, wave):
    f = f"{PROJ}/{net}/vsg_modesep/ref_{_REF[wave]}_phase.txt"
    d = np.loadtxt(f, comments="#")
    d = d[np.argsort(d[:, 0])]
    return d[:, 0], d[:, 1]                      # period[s], phase velocity[km/s]


def phase_wavelength(net, wave, T):
    """lambda(T) = c_phase(T)*T [km], c from the VSG reference phase curve (clamped at ends)."""
    Tr, c = _ref_phase(net, wave)
    cc = np.interp(np.asarray(T, float), Tr, c, left=c[0], right=c[-1])
    return cc * np.asarray(T, float)


# ---------------------------------------------------------------- edge distance
def _hull_xy(net):
    f = _cache_csv(net)
    a = np.genfromtxt(f, delimiter=",", names=True)
    return np.column_stack([a["xstat"], a["ystat"]])         # grid-km frame (matches cell.x_km)


def _pt_seg(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return float(np.hypot(px - (ax + t * dx), py - (ay + t * dy)))


def edge_distance_xy(x, y, hullpts):
    """Signed distance [km] from (x,y) to the station convex-hull boundary; + inside, - outside."""
    from scipy.spatial import ConvexHull
    from matplotlib.path import Path
    V = hullpts[ConvexHull(hullpts).vertices]
    d = min(_pt_seg(x, y, V[i, 0], V[i, 1], V[(i + 1) % len(V), 0], V[(i + 1) % len(V), 1])
            for i in range(len(V)))
    return d if Path(V).contains_point((x, y)) else -d


def edge_distance(cell, net):
    return edge_distance_xy(float(cell.x_km), float(cell.y_km), _hull_xy(net))


# ---------------------------------------------------------------- sensitivity kernel depth
def reference_model(depth_max=6.0, dz=0.25, vs_top=1.0, vs_bot=3.0):
    """Smooth layered Vs(z) reference for the sensitivity kernel (thickness[km], vs[km/s])."""
    ztop = np.arange(0.0, depth_max, dz)
    th = np.full(len(ztop), dz)
    th[-1] = 0.0                                              # bottom half-space
    vs = vs_top + (vs_bot - vs_top) * ztop / max(depth_max, 1e-9)
    return th, vs


_ZEFF_CACHE = {}       # (wave, depth_max) -> (dense periods, z_eff) interpolation table


def kernel_depth(wave, T, ref_model=None, depth_max=6.0):
    """|kernel|-weighted mean depth z_eff(T) [km] of the disba group-velocity Vs sensitivity.
    z_eff is cell-independent, so it is precomputed once per (wave, depth_max) on a dense period
    grid and interpolated -- avoids recomputing the disba kernel for every grid cell."""
    key = (wave, round(float(depth_max), 4), ref_model is not None)
    if key not in _ZEFF_CACHE:
        from disba import GroupSensitivity
        th, vs = ref_model if ref_model is not None else reference_model(depth_max)
        vp = 1.73 * vs
        rho = 0.32 * vp + 0.77
        gs = GroupSensitivity(th, vp, vs, rho)
        dwave, dmode = WAVEDEF[wave]
        Tg = np.round(np.arange(0.2, 8.001, 0.1), 4)
        zg = []
        for t in Tg:
            try:
                sk = gs(float(t), mode=dmode, wave=dwave, parameter="velocity_s")
                z = np.asarray(sk.depth, float); k = np.abs(np.asarray(sk.kernel, float))
                zg.append(float(np.sum(z * k) / np.sum(k)) if k.sum() > 0 else np.nan)
            except Exception:
                zg.append(np.nan)
        _ZEFF_CACHE[key] = (Tg, np.array(zg))
    Tg, zg = _ZEFF_CACHE[key]
    out = np.interp(np.atleast_1d(np.asarray(T, float)), Tg, zg)
    return out if np.ndim(T) else float(out[0])


# ---------------------------------------------------------------- resolution diagonal
_MAP_CACHE = {}        # (production_root, wave) -> (periods, res_diag[nT,nx,ny], mask[nT,nx,ny])


def _load_maps_rd(production_root, wave):
    key = (production_root, wave)
    if key not in _MAP_CACHE:
        files = sorted(glob.glob(os.path.join(production_root, wave, "map_T*.npz")),
                       key=lambda f: float(np.load(f)["period"]))
        T, R, M = [], [], []
        for f in files:
            z = np.load(f)
            T.append(float(z["period"])); R.append(z["res_diag"]); M.append(z["mask"])
        _MAP_CACHE[key] = (np.array(T), np.stack(R), np.stack(M))
    return _MAP_CACHE[key]


def res_diag_curve(production_root, ix, iy, wave):
    """(T, res_diag, mask) per period for one cell (maps cached in memory across cells)."""
    T, R, M = _load_maps_rd(production_root, wave)
    return T, R[:, ix, iy], M[:, ix, iy]


# ---------------------------------------------------------------- criteria
def metrics_table(cell, net, wave, params=None):
    """Per-period metrics for one cell/wave: dict of T, res_diag, lam, d_edge, n_lambda, z_eff."""
    p = {**DEFAULTS, **(params or {})}
    T = np.asarray(cell.curves[wave][0], float)
    prod = _prod_root(net, wave)
    Tr, R, _ = res_diag_curve(prod, cell.ix, cell.iy, wave)
    rmap = {round(float(t), 4): r for t, r in zip(Tr, R)}
    res = np.array([rmap.get(round(float(t), 4), np.nan) for t in T])
    lam = phase_wavelength(net, wave, T)
    de = edge_distance(cell, net)
    ze = kernel_depth(wave, T, depth_max=p["depth_max"])
    return {"T": T, "res_diag": res, "lam": lam, "d_edge": float(de),
            "n_lambda": de / lam, "z_eff": ze}


def keep_mask(cell, net, wave, criterion, params=None):
    """Boolean keep-mask over cell.curves[wave] periods for the given criterion.
    criterion in {"tomographic","physical","combined"}. Requires attach_cell_coords for physical."""
    p = {**DEFAULTS, **(params or {})}
    m = metrics_table(cell, net, wave, p)
    keep = np.ones(len(m["T"]), bool)
    if criterion in ("tomographic", "combined"):
        res = np.nan_to_num(m["res_diag"], nan=0.0)
        thr = max(p["R_min"], p["R_frac"] * (np.nanmax(res) if res.size else 0.0))
        keep &= res >= thr
    if criterion in ("physical", "combined"):
        keep &= m["d_edge"] >= p["alpha"] * m["lam"]
        keep &= np.nan_to_num(m["z_eff"], nan=np.inf) <= p["beta"] * p["depth_max"]
    return keep


def trim_reliable(cell, net, criterion, params=None):
    """Deep-copy `cell` with each wave trimmed to its reliable periods (criterion=None -> unchanged).
    If a wave drops below n_min_keep periods, keep its n_min_keep most reliable instead of emptying."""
    if criterion in (None, "none"):
        return cell
    import copy
    p = {**DEFAULTS, **(params or {})}
    c = copy.deepcopy(cell)
    for w in list(c.curves):
        T, U, S = c.curves[w]
        keep = keep_mask(c, net, w, criterion, p)
        if keep.sum() < p["n_min_keep"]:
            # fall back to the n_min_keep best-resolved periods so the inversion still has data
            res = np.nan_to_num(metrics_table(c, net, w, p)["res_diag"], nan=0.0)
            keep = np.zeros(len(T), bool)
            keep[np.argsort(res)[-p["n_min_keep"]:]] = True
        c.curves[w] = (T[keep], U[keep], S[keep])
    return c
