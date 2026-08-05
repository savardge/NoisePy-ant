"""
1D Vs DEPTH INVERSION FROM MODE-SEPARATED GROUP-VELOCITY MAPS

Trans-dimensional Bayesian inversion of a tomography cell's *effective* fundamental +
1st-overtone Rayleigh GROUP-velocity dispersion curves (from swtomotv production maps,
themselves from the V6 mode-separated picks) for a 1D Vs(z) profile.

Two engines behind one interface, to compare performance:
  * bayesbay  (trans-D Voronoi1D + disba GroupDispersion forward)  -- runs in-process
  * BayHunter (surf96 forward, native lvz/hvz)  -- runs via a subprocess in its own env,
    because BayHunter needs numpy<1.26 (numpy.distutils build) while bayesbay needs numpy>=2.

Physical prior: layered Vs that ALLOWS low- and high-velocity zones (no monotonic-increase
requirement) but caps the velocity contrast between adjacent layers at MAX_ADJ_FRAC (0.5 =
50%). In BayHunter this is native (initparams lvz=hvz=0.5); in bayesbay it is enforced as a
hard prior support (rejection in the forward + a constraint-respecting initializer).

Group velocity of the fundamental is disba mode 0 (BayHunter mode 1); the 1st overtone is
disba mode 1 (BayHunter mode 2). vp = vpvs*vs, rho = 0.32*vp + 0.77 (Brocher).

Heavy engine imports (bayesbay / disba / BayHunter) are lazy so this module imports in any env.

Usage: see scripts/picking/run_vs_inversion.py.
"""
from __future__ import annotations
import glob
import os
import time
from dataclasses import dataclass, field

import numpy as np

MAX_ADJ_FRAC = 0.5          # max |dVs/Vs| between adjacent layers (LVZ and HVZ allowed)
VPVS = 1.73                 # fixed Vp/Vs (both engines) for a clean comparison
# pipeline wave key -> (disba/surf96 wave type, mode index, 0-based). "fund"/"overtone" are the
# Rayleigh fundamental/1st-higher branches; "love" is the Love fundamental. The wave key stays the
# dict key everywhere; this map is the single place the wave TYPE and mode are resolved.
WAVEDEF = {"fund": ("rayleigh", 0), "overtone": ("rayleigh", 1), "love": ("love", 0),
           "love_ot": ("love", 1)}
DISBA_MODE = {w: m for w, (_, m) in WAVEDEF.items()}    # back-compat: wave key -> mode index

# CURVE KEYS carry the measure as a suffix: "fund" is the GROUP curve (unsuffixed = group, so
# every pre-existing caller and result file keeps working) and "fund_phase" the PHASE one. The
# convention is not invented here -- it is what run_bayhunter_cell.py already writes for its
# phase targets (`key = w if meas == "group" else f"{w}_phase"`), so CellData keys, BayHunter
# target names and the result npz all agree. Group and phase for the same wave can therefore
# coexist in one CellData and be inverted jointly.


def parse_curve_key(key):
    """'fund' -> ('fund', 'group');  'fund_phase' -> ('fund', 'phase')."""
    if key.endswith("_phase"):
        return key[: -len("_phase")], "phase"
    return key, "group"


def curve_key(wave, measure="group"):
    """Inverse of parse_curve_key."""
    return wave if measure == "group" else f"{wave}_phase"


def curve_def(key):
    """(disba_wave, mode, measure) for a curve key."""
    base, meas = parse_curve_key(key)
    disba_wave, mode = WAVEDEF[base]
    return disba_wave, mode, meas


# --------------------------------------------------------------------- data
@dataclass
class CellData:
    """One tomography cell's fund/overtone group-velocity dispersion curves."""
    ix: int
    iy: int
    lon: float = np.nan
    lat: float = np.nan
    x_km: float = np.nan
    y_km: float = np.nan
    curves: dict = field(default_factory=dict)   # {"fund": (T, U, sigmaU), "overtone": (...)}

    def has(self, wave):
        return wave in self.curves and len(self.curves[wave][0]) > 0


def load_cell_curves(production_root, ix, iy, waves=("fund", "overtone"),
                     res_min=None, min_periods=5, wave_roots=None, measure="group",
                     into=None):
    """Assemble a cell's dispersion curves from the per-period production npz maps.

    production_root : .../swtomotv-output/production  (holds {wave}/map_T*.npz)
    measure  : "group" or "phase". Curves are stored under `curve_key(wave, measure)`, i.e.
        "fund" for group and "fund_phase" for phase, so both can coexist in one CellData.
    into     : an existing CellData to add these curves to (returned, mutated) instead of a
        fresh one. This is how a joint group+phase cell is built -- call twice with the two
        measures and their two production roots. NOTE group and phase are recommended from
        DIFFERENT Cd runs (group=scaled, phase=blanket), so the roots genuinely differ.
    wave_roots : optional {wave: root} overriding production_root per wave. Use it to merge a
        wave whose maps live under a DIFFERENT production tree (e.g. Love, in
        .../swtomotv-output-love-<dx>/production) into the same CellData as the Rayleigh waves.
    Per period the npz stores vel (nx,ny), unc_s (slowness std, s/km), mask, res_diag.
    Velocity 1-sigma = unc_s * U^2 (slowness->velocity Jacobian).  res_min optionally
    tightens the per-cell resolution gate beyond the production mask.
    Returns a CellData (curves only for waves with >= min_periods valid periods).
    """
    cell = into if into is not None else CellData(ix=int(ix), iy=int(iy))
    if into is not None and (int(ix), int(iy)) != (cell.ix, cell.iy):
        raise ValueError("load_cell_curves(into=...): cell index mismatch (%d,%d) vs (%d,%d)"
                         % (ix, iy, cell.ix, cell.iy))
    for w in waves:
        w = parse_curve_key(w)[0]           # accept either "fund" or "fund_phase" as input
        root = (wave_roots or {}).get(w, production_root)
        # Fail LOUD on a stale/wrong root. Without this a bad path globs to [] and the wave is
        # silently dropped from the inversion (empty CellData returns cleanly, .has(w) skips it) --
        # a wave vanishing unnoticed, the classic failure after a directory move. A root that
        # EXISTS but has no maps for wave w is legitimate (a tree may simply lack that wave), so
        # only the missing-directory case raises; the empty-cell/empty-wave cases stay silent.
        wdir = os.path.join(root, w)
        if not os.path.isdir(wdir):
            raise FileNotFoundError(
                f"load_cell_curves: no '{w}' map directory under production root '{root}' "
                f"({wdir} does not exist). Check the production_root/wave_roots path -- a stale "
                f"path here silently drops the wave instead of erroring.")
        files = sorted(glob.glob(os.path.join(wdir, "map_T*.npz")),
                       key=lambda f: float(np.load(f)["period"]))
        T, U, S = [], [], []
        for f in files:
            z = np.load(f)
            if not bool(z["mask"][ix, iy]):
                continue
            if res_min is not None and float(z["res_diag"][ix, iy]) < res_min:
                continue
            u = float(z["vel"][ix, iy])
            if not np.isfinite(u):
                continue
            T.append(float(z["period"]))
            U.append(u)
            S.append(float(z["unc_s"][ix, iy]) * u ** 2)     # s/km -> km/s
        if len(T) >= min_periods:
            o = np.argsort(T)
            cell.curves[curve_key(w, measure)] = (np.array(T)[o], np.array(U)[o],
                                                  np.array(S)[o])
    return cell


def load_reference_curves(paths, waves=("fund", "overtone"), sigma_frac=0.03, sigma_floor=0.03,
                          label=None):
    """Build a CellData from network-averaged reference dispersion-curve txt files.

    paths : {wave: filepath}, each a 2-column (period[s], velocity[km/s]) text file (e.g. the
    mode-separated VSG PHASE-velocity references `ref_fundamental_phase.txt` /
    `ref_overtone_phase.txt` under Projects/{net}/vsg_modesep/ -- see [[aargau-dataset]] /
    [[riehen-dataset]]). No per-pick uncertainty is stored in those files, so sigma is assigned
    as max(sigma_floor, sigma_frac*c) -- a generic phase-picking-precision estimate, not a
    measured error bar; BayHunter's hierarchical noise model (swdnoise_sigma) still lets the
    data override this initial guess.
    """
    cell = CellData(ix=-1, iy=-1)
    if label is not None:
        cell.label = label
    for w in waves:
        if w not in paths:
            continue
        d = np.loadtxt(paths[w])
        T, U = d[:, 0], d[:, 1]
        o = np.argsort(T)
        T, U = T[o], U[o]
        S = np.maximum(sigma_floor, sigma_frac * U)
        cell.curves[w] = (T, U, S)
    return cell


def decimate_periods(cell, max_periods=55):
    """Return a copy of `cell` with each wave's curve thinned to <= max_periods points.

    BayHunter's surf96 (SurfDisp) forward hard-caps observed data at 60 periods per target
    (`Your observed data vector exceeds the maximum of 60 periods`). Dense reference curves
    (e.g. the ~225-period network phase-velocity picks) must be thinned before inversion.
    Picks the nearest actual data point to each of `max_periods` log-spaced period targets
    (no interpolation -- every kept point is a real measurement)."""
    import copy
    c = copy.deepcopy(cell)
    for w, (T, U, S) in list(c.curves.items()):
        if len(T) <= max_periods:
            continue
        targets = np.geomspace(T.min(), T.max(), max_periods)
        idx = sorted(set(int(np.argmin(np.abs(T - t))) for t in targets))
        c.curves[w] = (T[idx], U[idx], S[idx])
    return c


def restrict_periods(cell, period_ranges):
    """Return a copy of `cell` with each wave's curve trimmed to a period range.

    period_ranges: {wave: (tmin, tmax)}; None bound = open. E.g. {"overtone": (1.0, None)}
    to drop the suspect short-period overtone (data-quality test)."""
    import copy
    c = copy.deepcopy(cell)
    for w, (tmin, tmax) in period_ranges.items():
        if not c.has(w):
            continue
        T, U, S = c.curves[w]
        m = np.ones(len(T), bool)
        if tmin is not None:
            m &= T >= tmin
        if tmax is not None:
            m &= T <= tmax
        c.curves[w] = (T[m], U[m], S[m])
    return c


def attach_cell_coords(cell, swtomotv_yaml):
    """Fill lon/lat/x_km/y_km on a CellData from the swtomotv grid (lazy import).

    Uses swtomotv.make_grid for the exact legacy grid, then inverts the midpoint-latitude
    cosine projection (geometry.ll2xy) at the cell center. If swtomotv is unavailable the
    coords are left NaN (inversion still runs; only labels are missing)."""
    try:
        from swtomotv.config import DatasetConfig
        from swtomotv.geometry import make_grid
    except Exception:
        return cell
    ds = DatasetConfig.from_yaml(swtomotv_yaml)
    grid = make_grid(ds.bounds, ds.dx_km)
    xc = float(grid.x[cell.ix] + grid.dx / 2)
    yc = float(grid.y[cell.iy] + grid.dx / 2)
    olat, olon = grid.origin
    R = 6371.0                                    # geometry.R_EARTH_KM
    cell.x_km, cell.y_km = xc, yc
    cell.lat = float(olat + yc / R * 180.0 / np.pi)                       # invert y = R*(lat-olat)
    cell.lon = float(olon + xc / (R * np.cos(np.radians((cell.lat + olat) / 2))) * 180.0 / np.pi)
    return cell


# --------------------------------------------------------------------- forward + constraint
def adjacent_contrast_ok(vs, maxfrac=MAX_ADJ_FRAC):
    """True if every adjacent-layer |dVs/Vs| <= maxfrac (LVZ and HVZ both allowed)."""
    vs = np.asarray(vs, float)
    if vs.size < 2:
        return True
    return bool(np.all(np.abs(np.diff(vs)) / vs[:-1] <= maxfrac))


def dispersion_velocity(thickness, vs, periods, mode, measure="group", vpvs=VPVS,
                        disba_wave="rayleigh"):
    """disba GROUP or PHASE velocity for one mode; NaN where the mode is undefined.

    thickness: layer thicknesses km (last = half-space, use a large value).
    measure: "group" or "phase".  disba_wave: "rayleigh" or "love".
    Returns an array aligned to `periods` (NaN where disba does not return that period).
    """
    from disba import GroupDispersion, PhaseDispersion
    Disp = {"group": GroupDispersion, "phase": PhaseDispersion}[measure]
    vs = np.asarray(vs, float)
    vp = vpvs * vs
    rho = 0.32 * vp + 0.77
    try:
        gd = Disp(np.asarray(thickness, float), vp, vs, rho)
        cp = gd(periods, mode=mode, wave=disba_wave)
    except Exception:
        return np.full(len(periods), np.nan)
    out = np.full(len(periods), np.nan)
    # disba returns the subset of periods where the mode exists, in its own order
    idx = {round(float(p), 6): i for i, p in enumerate(periods)}
    for p, v in zip(cp.period, cp.velocity):
        j = idx.get(round(float(p), 6))
        if j is not None:
            out[j] = v
    return out


def group_velocity(thickness, vs, periods, mode, vpvs=VPVS, disba_wave="rayleigh"):
    """Back-compat alias: disba GROUP velocity for a wave type (see dispersion_velocity)."""
    return dispersion_velocity(thickness, vs, periods, mode, measure="group", vpvs=vpvs,
                               disba_wave=disba_wave)


# --------------------------------------------------------------------- bayesbay engine
def run_bayesbay(cell, waves=("fund", "overtone"), depth_max=6.0,
                 vs_bounds=(0.3, 3.6), n_layers=(3, 12), voronoi_perturb=1.0,
                 vs_perturb=0.12, n_chains=8, n_iterations=200_000, burnin=60_000,
                 save_every=200, maxfrac=MAX_ADJ_FRAC, seed=None, verbose=False,
                 noise_std=(0.01, 0.5), noise_perturb=0.03, constraint="project"):
    """Trans-dimensional 1D Vs inversion of the cell's group-velocity curves with bayesbay.

    Returns a result dict: engine, depth grid, posterior Vs samples (n_models x n_depth),
    per-wave predicted-curve posterior, inferred noise, runtime, acceptance, config.
    """
    import bayesbay as bb
    from bayesbay.discretization import Voronoi1D
    waves = [w for w in waves if cell.has(w)]
    if not waves:
        raise ValueError("cell has no usable curves")
    HALFSPACE = 100.0                       # km half-space thickness sentinel

    def _project(vs):
        """Clip each layer's Vs to within (1±maxfrac)x the layer above (depth order)."""
        vs = vs.copy()
        for i in range(1, len(vs)):
            vs[i] = min(max(vs[i], vs[i - 1] * (1 - maxfrac)), vs[i - 1] * (1 + maxfrac))
        return vs

    def model_from_state(state):
        """Return (thickness, vs) sorted by depth; applies the projection constraint here so
        the forward AND the reported posterior use the same (valid) model."""
        v = state["voronoi"]
        nuclei = np.asarray(v["discretization"], float)
        vs = np.asarray(v["vs"], float)
        o = np.argsort(nuclei)               # depth order (required for a correct layered model)
        nuclei, vs = nuclei[o], vs[o]
        if constraint == "project":
            vs = _project(vs)
        th = np.asarray(Voronoi1D.compute_cell_extents(nuclei), float)
        th[-1] = HALFSPACE
        return th, vs

    REJECT = {}                             # per-wave absurd sentinel (huge misfit -> rejected)
    for w in waves:
        REJECT[w] = np.full(len(cell.curves[w][0]), 0.0)

    def make_fwd(w):
        periods = cell.curves[w][0]
        # per-CURVE measure: a joint cell holds "fund" (group) and "fund_phase" (phase), which
        # must forward through GroupDispersion and PhaseDispersion respectively.
        disba_wave, mode, meas = curve_def(w)

        def fwd(state):
            th, vs = model_from_state(state)          # already projected if constraint=="project"
            if constraint != "project" and not adjacent_contrast_ok(vs, maxfrac):
                return REJECT[w]
            g = dispersion_velocity(th, vs, periods, mode, measure=meas,
                                    disba_wave=disba_wave)
            if not np.all(np.isfinite(g)):   # model cannot produce this branch/period -> reject
                return REJECT[w]
            return g
        return fwd

    # constraint-respecting initializer: bounded random walk (each layer within maxfrac of prev)
    rng = np.random.default_rng(seed)

    def init_vs(param, positions=None):
        vmin, vmax = param.get_vmin_vmax(positions)
        n = np.asarray(positions).size
        vlo = float(np.atleast_1d(vmin)[0]); vhi = float(np.atleast_1d(vmax)[0])
        out = np.empty(n)
        out[0] = rng.uniform(vlo, vhi)
        for i in range(1, n):
            lo = max(vlo, out[i - 1] * (1 - maxfrac))
            hi = min(vhi, out[i - 1] * (1 + maxfrac))
            out[i] = rng.uniform(lo, hi)
        return out

    vs_prior = bb.prior.UniformPrior(name="vs", vmin=vs_bounds[0], vmax=vs_bounds[1],
                                     perturb_std=vs_perturb)
    vs_prior.set_custom_initialize(init_vs)
    voronoi = Voronoi1D(name="voronoi", vmin=0.0, vmax=depth_max, perturb_std=voronoi_perturb,
                        n_dimensions=None, n_dimensions_min=n_layers[0],
                        n_dimensions_max=n_layers[1], parameters=[vs_prior],
                        birth_from="neighbour")
    parameterization = bb.parameterization.Parameterization(voronoi)

    # Hierarchical (inferred) data noise with a WIDE prior: the tomographic formal sigma
    # (~0.04-0.16 km/s) underestimates true error (1-D theory error, mode-ID, regularization),
    # and forcing it tight makes the trans-D over-fit (too many layers). A wide prior lets the
    # data set the effective noise -> parsimonious, smooth models (matches BayHunter's wide
    # swdnoise_sigma). Same policy for both engines.
    targets, fwds = [], []
    for w in waves:
        T, U, S = cell.curves[w]
        targets.append(bb.likelihood.Target(w, U, std_min=float(noise_std[0]),
                                            std_max=float(noise_std[1]),
                                            std_perturb_std=float(noise_perturb)))
        fwds.append(make_fwd(w))
    log_like = bb.likelihood.LogLikelihood(targets=targets, fwd_functions=fwds)

    inv = bb.BayesianInversion(parameterization=parameterization, log_likelihood=log_like,
                               n_chains=n_chains)
    t0 = time.time()
    inv.run(n_iterations=n_iterations, burnin_iterations=burnin, save_every=save_every,
            verbose=verbose)
    runtime = time.time() - t0

    def _sorted_proj(nuc_list, vs_list):
        """Depth-sort (+ project if constraint=='project') each posterior sample, matching the
        forward, so the reported Vs(z) is the model that actually fit the data."""
        ext, vsp = [], []
        for n, v in zip(nuc_list, vs_list):
            n = np.asarray(n, float); v = np.asarray(v, float)
            o = np.argsort(n); n, v = n[o], v[o]
            if constraint == "project":
                v = _project(v)
            ext.append(Voronoi1D.compute_cell_extents(n)); vsp.append(v)
        return ext, vsp

    res = inv.get_results(concatenate_chains=True)
    nuclei = res["voronoi.discretization"]
    vs_s = res["voronoi.vs"]
    dep = np.linspace(0, depth_max, 121)
    ext, vs_proj = _sorted_proj(nuclei, vs_s)
    stats = Voronoi1D.get_tessellation_statistics(ext, vs_proj, dep, input_type="extents",
                                                  percentiles=(2.5, 16, 84, 97.5))
    p = np.asarray(stats["percentiles"])              # (4, ndepth): 95% and 68% band edges
    acc = _mean_acceptance(inv)
    # convergence: between-chain agreement of the posterior median Vs(z). Small => converged.
    chain_disagree = np.nan
    try:
        pc = inv.get_results(concatenate_chains=False)
        meds = []
        for cn, cv in zip(pc["voronoi.discretization"], pc["voronoi.vs"]):
            e, vp = _sorted_proj(cn, cv)
            s = Voronoi1D.get_tessellation_statistics(e, vp, dep, input_type="extents")
            meds.append(np.asarray(s["median"]))
        meds = np.array(meds)                          # (n_chains, ndepth)
        chain_disagree = float(np.nanmax(np.nanstd(meds, axis=0)))   # km/s, worst depth
    except Exception:
        pass
    out = dict(engine="bayesbay", depth=dep,
               vs_mean=np.asarray(stats["mean"]), vs_median=np.asarray(stats["median"]),
               vs_p025=p[0], vs_p16=p[1], vs_p84=p[2], vs_p975=p[3],
               n_models=len(vs_s), runtime_s=runtime, acceptance=acc,
               chain_disagree=chain_disagree,
               n_layers_post=np.array([len(v) for v in vs_s]),
               waves=waves, cell=cell, pred={})
    for w in waves:
        dp = res.get(f"{w}.dpred")
        if dp is not None:
            out["pred"][w] = (cell.curves[w][0], np.array(dp))
        std = res.get(f"{w}.std")
        if std is not None:
            out.setdefault("noise", {})[w] = float(np.median(std))
    return out


def _mean_acceptance(inv):
    """Overall accepted/proposed across chains (best-effort)."""
    ta = tp = 0.0
    def _s(x):
        return float(sum(x.values())) if isinstance(x, dict) else float(x or 0)
    for ch in inv.chains:
        st = getattr(ch, "statistics", {})
        ta += _s(st.get("n_accepted_models_total"))
        tp += _s(st.get("n_proposed_models_total"))
    return ta / tp if tp else float("nan")


# --------------------------------------------------------------------- result I/O
_RESULT_KEYS = ("engine", "depth", "vs_mean", "vs_median", "vs_p025", "vs_p16",
                "vs_p84", "vs_p975", "runtime_s", "acceptance", "chain_disagree", "n_models")


def save_result(result, path):
    """Persist an engine result (Vs(z) posterior + predicted curves) to npz, engine-agnostic."""
    d = {k: result[k] for k in _RESULT_KEYS if k in result}
    d["n_layers_post"] = result.get("n_layers_post", np.array([]))
    d["waves"] = np.array(result.get("waves", []))
    for w, (T, dp) in result.get("pred", {}).items():
        d[f"predT_{w}"] = T
        d[f"pred_{w}"] = np.asarray(dp)          # (n_models, nT) or (nT,)
    c = result.get("cell")
    if c is not None:
        d["cell_ixiy"] = np.array([c.ix, c.iy])
        d["cell_lonlat"] = np.array([c.lon, c.lat])
        for w in result.get("waves", []):
            if c.has(w):
                T, U, S = c.curves[w]
                d[f"obsT_{w}"], d[f"obs_{w}"], d[f"obssig_{w}"] = T, U, S
    np.savez_compressed(path, **d)
    return path


def load_result(path):
    """Load a saved engine result npz back into the result-dict schema (no engine imports)."""
    z = np.load(path, allow_pickle=True)
    r = {k: (z[k].item() if z[k].shape == () else z[k]) for k in z.files
         if not k.startswith(("pred", "predT", "obs"))}
    r["engine"] = str(r.get("engine"))
    r["waves"] = list(z["waves"]) if "waves" in z.files else []
    r["pred"] = {w: (z[f"predT_{w}"], z[f"pred_{w}"]) for w in r["waves"] if f"pred_{w}" in z.files}
    r["obs"] = {w: (z[f"obsT_{w}"], z[f"obs_{w}"], z[f"obssig_{w}"])
                for w in r["waves"] if f"obs_{w}" in z.files}
    return r


def _obs(result, w):
    if "obs" in result and w in result["obs"]:
        return result["obs"][w]
    c = result.get("cell")
    return c.curves[w]


def _pred_band(result, w):
    """Median and 2.5/97.5 predicted-curve percentiles for wave w (or (T, median, None))."""
    T, dp = result["pred"][w]
    dp = np.asarray(dp)
    if dp.ndim == 1:
        return T, dp, None, None
    return T, np.nanmedian(dp, 0), np.nanpercentile(dp, 2.5, 0), np.nanpercentile(dp, 97.5, 0)


def data_misfit(result):
    """Normalized rms misfit (chi) per wave: rms((obs - pred_median)/sigma_obs)."""
    out = {}
    for w in result.get("waves", []):
        if w not in result.get("pred", {}):
            continue
        T, U, S = _obs(result, w)
        _, med, _, _ = _pred_band(result, w)
        m = np.isfinite(med)
        out[w] = float(np.sqrt(np.mean(((U[m] - med[m]) / S[m]) ** 2)))
    return out


# --------------------------------------------------------------------- plotting
def plot_inversion(result, path, title=None, measure="group"):
    """Single-engine figure: posterior Vs(z) (median + 68/95% bands) + dispersion fit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    waves = result.get("waves", [])
    vlabel = f"{measure} velocity [km/s]"
    fig, axs = plt.subplots(1, 1 + len(waves), figsize=(4.5 * (1 + len(waves)), 6))
    axs = np.atleast_1d(axs)
    dep = result["depth"]
    ax = axs[0]
    if "vs_p025" in result:
        ax.fill_betweenx(dep, result["vs_p025"], result["vs_p975"], color="0.85", label="95%")
        ax.fill_betweenx(dep, result["vs_p16"], result["vs_p84"], color="0.65", label="68%")
    ax.plot(result["vs_median"], dep, "b-", lw=2, label="median")
    ax.invert_yaxis(); ax.set(xlabel="Vs [km/s]", ylabel="depth [km]",
                              title=f"{result['engine']} posterior Vs(z)")
    ax.legend(fontsize=8)
    for ax, w in zip(axs[1:], waves):
        T, U, S = _obs(result, w)
        ax.errorbar(T, U, yerr=S, fmt="k.", ms=4, lw=0.8, label="observed", zorder=3)
        if w in result.get("pred", {}):
            Tp, med, lo, hi = _pred_band(result, w)
            if lo is not None:
                ax.fill_between(Tp, lo, hi, color="C0", alpha=0.25, label="pred 95%")
            ax.plot(Tp, med, "C0-", lw=1.6, label="pred median")
        ax.set(xlabel="period [s]", ylabel=vlabel, title=f"{w} dispersion fit")
        ax.legend(fontsize=8)
    if title:
        fig.suptitle(title, y=1.0)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def compare_engines(results, path, cell=None):
    """Overlay N engine posteriors: Vs(z), dispersion fits, and a metrics/QC table."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    waves = sorted({w for r in results for w in r.get("waves", [])})
    fig, axs = plt.subplots(1, 2 + len(waves), figsize=(4.6 * (2 + len(waves)), 6))
    colors = {"bayesbay": "C0", "bayhunter": "C3"}
    ax = axs[0]
    for r in results:
        c = colors.get(r["engine"], "C2")
        if "vs_p025" in r:
            ax.fill_betweenx(r["depth"], r["vs_p16"], r["vs_p84"], color=c, alpha=0.2)
        ax.plot(r["vs_median"], r["depth"], color=c, lw=2, label=f"{r['engine']} median")
    ax.invert_yaxis(); ax.set(xlabel="Vs [km/s]", ylabel="depth [km]",
                              title="posterior Vs(z) (68% band)")
    ax.legend(fontsize=8)
    for ax, w in zip(axs[1:1 + len(waves)], waves):
        obs_done = False
        for r in results:
            if w not in r.get("waves", []):
                continue
            if not obs_done:
                T, U, S = _obs(r, w)
                ax.errorbar(T, U, yerr=S, fmt="k.", ms=4, lw=0.8, label="observed", zorder=3)
                obs_done = True
            Tp, med, lo, hi = _pred_band(r, w)
            ax.plot(Tp, med, color=colors.get(r["engine"], "C2"), lw=1.6, label=r["engine"])
        ax.set(xlabel="period [s]", ylabel="group velocity [km/s]", title=f"{w} fit")
        ax.legend(fontsize=8)
    # metrics/QC table
    ax = axs[-1]; ax.axis("off")
    rows = [["engine", "runtime", "n_layers", "chain_std", "chi(fund)", "chi(ot)", "chi(love)"]]
    for r in results:
        mis = data_misfit(r)
        nl = r.get("n_layers_post", np.array([np.nan]))
        cd = r.get("chain_disagree", np.nan)
        cd = float(cd) if cd is not None else np.nan
        rows.append([r["engine"], f"{r.get('runtime_s', np.nan):.0f}s",
                     f"{np.mean(nl):.1f}±{np.std(nl):.1f}",
                     ("%.2f" % cd) if np.isfinite(cd) else "-",
                     f"{mis.get('fund', np.nan):.2f}", f"{mis.get('overtone', np.nan):.2f}",
                     f"{mis.get('love', np.nan):.2f}"])
    tb = ax.table(cellText=rows, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.6)
    ax.set_title("performance / QC", fontsize=10)
    ttl = "1D Vs inversion engine comparison"
    if cell is not None:
        ttl += f"  — cell (ix,iy)=({cell.ix},{cell.iy})"
        if np.isfinite(cell.lon):
            ttl += f"  lon,lat=({cell.lon:.3f},{cell.lat:.3f})"
    fig.suptitle(ttl, y=1.0)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


# --------------------------------------------------------------------- BayHunter (subprocess)
def run_bayhunter(cell, out_npz, runner, bayhunter_python, waves=("fund", "overtone"),
                  depth_max=6.0, vs_bounds=(0.3, 3.6), n_layers=(1, 20),
                  maxfrac=MAX_ADJ_FRAC, nchains=8, iter_burnin=120_000, iter_main=60_000,
                  workdir=None, timeout=None, measure="group"):
    """Run BayHunter in its own env via subprocess; return the loaded result dict.

    runner            : path to run_bayhunter_cell.py (executed with bayhunter_python)
    bayhunter_python  : interpreter of the env where BayHunter is built (numpy<1.26)
    measure           : "group" or "phase" -- selects RayleighDispersionGroup/Phase targets
                        and the disba GroupDispersion/PhaseDispersion posterior-predictive forward.
    Writes the cell's fund/overtone curves to txt, invokes the runner (which writes out_npz
    with the shared result schema), then loads it. maxfrac=None disables the LVZ/HVZ constraint
    entirely (BayHunter initparams lvz=hvz=None); otherwise -> initparams lvz=hvz=maxfrac.
    """
    import json
    import subprocess
    import tempfile
    workdir = workdir or tempfile.mkdtemp(prefix="bayhunter_cell_")
    os.makedirs(workdir, exist_ok=True)
    # The runner takes TWO curve dicts, both keyed by BASE wave: `curves` (interpreted with
    # cfg["measure"]) and an optional `curves_phase`. Split the cell's keys accordingly --
    # writing a "fund_phase" key into `curves` would make the runner look up WAVE_DISBA
    # ["fund_phase"] and silently skip it.
    curvefiles, phasefiles = {}, {}
    for w in waves:
        if not cell.has(w):
            continue
        base, meas = parse_curve_key(w)
        T, U, S = cell.curves[w]
        fp = os.path.join(workdir, f"disp_{w}.txt")
        np.savetxt(fp, np.column_stack([T, U, S]), fmt="%.6f")
        (phasefiles if meas == "phase" else curvefiles)[base] = fp
    if not curvefiles and phasefiles:
        # phase-only run: hand it over as the primary set with measure="phase"
        curvefiles, phasefiles, measure = phasefiles, {}, "phase"
    cfg = dict(curves=curvefiles, curves_phase=phasefiles,
               out_npz=out_npz, depth_max=depth_max,
               vs_bounds=list(vs_bounds), n_layers=list(n_layers), maxfrac=maxfrac,
               nchains=nchains, iter_burnin=iter_burnin, iter_main=iter_main, measure=measure,
               cell=[cell.ix, cell.iy, cell.lon, cell.lat])
    cfgpath = os.path.join(workdir, "config.json")
    with open(cfgpath, "w") as f:
        json.dump(cfg, f)
    # macOS: BayHunter forks worker chains; fork + multithreaded BLAS/Accelerate deadlocks,
    # so pin BLAS to 1 thread (chains are already parallel) and set objc fork safety.
    env = dict(os.environ, OBJC_DISABLE_INITIALIZE_FORK_SAFETY="YES",
               VECLIB_MAXIMUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    subprocess.run([bayhunter_python, runner, cfgpath], check=True, timeout=timeout, env=env)
    return load_result(out_npz)
