"""Per-well posterior Vs QC plots at the tomography cell closest to each deep borehole.

For each well (deeper than 1 km and inside the station convex hull), inverts the nearest
inverted cell's fund+overtone group-velocity curves with BayHunter (full posterior ensemble)
and makes one figure with:
  * Vs posterior distribution (2-D density Vs vs depth) + median/68/95% + the well log overlay
  * interface-probability vertical strip (posterior layer-boundary depth density)
  * dispersion data fit: observed fund/overtone + posterior predictive band
  * misfit distribution histogram + a QC text panel (n_layers, chi, coords, cell distance)

Well coords are WGS84 from swisstopo deep_wells.csv. Overlay: Riehen = Michel2016 well Vs
(Basel-1/Otterbach); Aargau = Nagra sonic Vp/1.73 per well.

Run with the bayesbay env python (shells out to the BayHunter env for the inversion):
  PYTHONPATH=~/Codes/Noisepy-ant /opt/anaconda3/envs/bayesbay_dev/bin/python well_vs_qc.py \
    --net riehen --bayhunter-python /opt/anaconda3/envs/bayhunter/bin/python \
    --bayhunter-runner run_bayhunter_cell.py
"""
import argparse
import json
import os
import subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull

from noisepy import vs_inversion as vi
from noisepy import period_resolution as pr
from noisepy import vs_reliability as vr
from noisepy import pt_defaults

# per-cell period-trimming criteria (see noisepy/period_resolution.py); "none" = untrimmed baseline
CRIT_LABEL = {"none": "untrimmed", "combined": "A: combined", "physical": "B: physical",
              "tomographic": "C: tomographic"}
CRIT_ORDER = ["none", "combined", "physical", "tomographic"]

# waveset key -> wave tuple. "love" = Love fundamental (TT); joint sets add it to the Rayleigh waves.
WAVESETS = {"fund": ("fund",), "fundot": ("fund", "overtone"), "love": ("love",),
            "fundlove": ("fund", "love"), "fundotlove": ("fund", "overtone", "love")}
WS_LABEL = {"fund": "R fund", "fundot": "R fund+ot", "love": "Love",
            "fundlove": "R fund + Love", "fundotlove": "R fund+ot + Love"}
# Love production root per net -- LEGACY split-root fallback (V6 era, now archived). Production
# uses the unified single root via --production-root; these only fire when it's omitted.
_EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
_ARC = "tomo/1_velocity_maps/_archive"
LOVE_PROD = {"riehen": f"{_EHM}/riehen/{_ARC}/swtomotv-output-love-500m/production",
             "aargau": f"{_EHM}/aargau/{_ARC}/swtomotv-output-love-1km/production"}

# name, lat, lon, depth_m  (WGS84; from swisstopo deep_wells.csv)
WELLS = {
    "riehen": [("Basel-1", 47.585413, 7.595614, 5009), ("Otterbach-2", 47.577748, 7.603832, 2745),
               ("Riehen-1", 47.587100, 7.649485, 1547), ("Riehen-2", 47.593696, 7.657156, 1247)],
    "aargau": [("Boettstein", 47.565033, 8.227163, 1501), ("Riniken", 47.504507, 8.189936, 1800),
               ("Leuggern", 47.589033, 8.205224, 1689), ("Kaisten", 47.539828, 8.031539, 1306),
               ("Schafisheim", 47.369472, 8.148685, 2006), ("Weiach-1", 47.563788, 8.458407, 2482),
               ("Weiach-2", 47.565144, 8.453530, 2013), ("Benken", 47.644915, 8.649547, 1007)],
}
NAGRA_VP = "/Users/genevievesavard/Data/aargau/nagra-wells-vp"   # {well}-geoIntervalVp.csv (Vp,depth)
MICHEL_MODEL = "/Volumes/T7Shield/riehen/well-data/Michel2016_gpdc.model"
VPVS_RATIOS = [(1.73, "tab:orange", "--"), (1.90, "tab:purple", ":"), (2.50, "tab:brown", "-.")]


def stations_hull(net):
    st = np.genfromtxt(f"/Users/genevievesavard/Codes/extract_higher_modes/Projects/{net}/tomo/1_velocity_maps/inputs/stations.csv",
                       delimiter=",", names=True)
    pts = np.column_stack([st["longitude"], st["latitude"]])
    return MplPath(pts[ConvexHull(pts).vertices])


def _read_geopsy_model(path):
    """geopsy .model (n; then thickness Vp Vs rho per layer, m & m/s) -> (tops_km, vp_kms, vs_kms)."""
    toks = [ln.split() for ln in open(path).read().splitlines() if ln.strip()]
    n = int(toks[0][0])
    th = np.array([float(t[0]) for t in toks[1:1 + n]])
    vp = np.array([float(t[1]) for t in toks[1:1 + n]])
    vs = np.array([float(t[2]) for t in toks[1:1 + n]])
    tops = np.concatenate([[0.0], np.cumsum(th)[:-1]])
    return tops / 1000.0, vp / 1000.0, vs / 1000.0


def _staircase(tops, vals, zmax=6.0):
    """Layer tops + values -> (val, depth) staircase points for plotting a blocky log."""
    bots = np.concatenate([tops[1:], [max(zmax, tops[-1] + 0.1)]])
    zz, vv = [], []
    for t, b, v in zip(tops, bots, vals):
        zz += [t, b]; vv += [v, v]
    return np.array(vv), np.array(zz)


def overlay_curves(net, wellname):
    """List of (vs_kms, depth_km, label, color, linestyle) reference well-log overlays.
    Vp logs are shown as Vs = Vp/1.73 and Vp/1.90 to bracket the Vp/Vs conversion."""
    out = []
    if net == "riehen" and os.path.exists(MICHEL_MODEL):
        tops, vp, vs = _read_geopsy_model(MICHEL_MODEL)
        v, z = _staircase(tops, vs)
        out.append((v, z, "Michel2016 Vs (in-situ)", "green", "-"))
        for ratio, col, ls in VPVS_RATIOS:
            v, z = _staircase(tops, vp / ratio)
            out.append((v, z, f"Michel Vp/{ratio:g}", col, ls))
    elif net == "aargau":
        # blocky geological-interval Vp model (header Vp,depth in m/s,m); filenames are lowercase
        # and use a plain 'o' (bottstein) vs the well name 'Boettstein'.
        stem = wellname.lower().replace("oe", "o").replace("ö", "o")
        fp = os.path.join(NAGRA_VP, f"{stem}-geoIntervalVp.csv")
        if os.path.exists(fp):
            d = np.genfromtxt(fp, delimiter=",", skip_header=1)
            dep, vp = d[:, 1] / 1000.0, d[:, 0] / 1000.0
            for ratio, col, ls in VPVS_RATIOS:
                out.append((vp / ratio, dep, f"Nagra {wellname} Vp/{ratio:g}", col, ls))
    return out


def run_cell_ensemble(net, ix, iy, out_npz, args, waves=("fund", "overtone"), criterion="none"):
    """Run BayHunter (full ensemble) for one cell if out_npz absent; return path.
    waves selects the inverted curves: ("fund",) = fundamental only.
    criterion trims each curve to its reliable periods (period_resolution.trim_reliable)."""
    if os.path.exists(out_npz):
        return out_npz
    cfgp = f"/Users/genevievesavard/Codes/extract_higher_modes/Projects/{net}/tomo/1_velocity_maps/inputs/{net}_swtomotv.yaml"
    if getattr(args, "production_root", None):
        # single production root carrying ALL wave subdirs (the unified rebuild:
        # swtomotv-output-uni/production/{fund,overtone,love}) -- no wave_roots juggling
        prod = args.production_root.format(net=net)
        pr.PROD_ROOT[net] = prod
        pr.PROD_ROOT[(net, "love")] = prod
        # Also register the station cache next to the production root. period_resolution's legacy
        # fallback (tomo/swtomotv-output/cache/...) was removed in the reorg and now RAISES, so
        # without this the driver aborts before writing a config. The cache lives at
        # <production_root>/../cache/stations_in_grid.csv (swtomotv writes it beside output_root).
        _cache = os.path.join(os.path.dirname(prod.rstrip("/")), "cache", "stations_in_grid.csv")
        if os.path.exists(_cache):
            pr.CACHE_CSV[net] = _cache
        cell = vi.load_cell_curves(prod, ix, iy, waves=("fund", "overtone", "love"))
    else:
        prod = f"/Users/genevievesavard/Codes/extract_higher_modes/Projects/{net}/tomo/1_velocity_maps/_archive/swtomotv-output/production"
        love_root = LOVE_PROD[net]
        pr.PROD_ROOT[(net, "love")] = love_root            # Love res_diag for the trimming criteria
        cell = vi.load_cell_curves(prod, ix, iy, waves=("fund", "overtone", "love"),
                                   wave_roots={"love": love_root})
    if "overtone" in waves:
        cell = vi.restrict_periods(cell, {"overtone": (args.overtone_min_t, None)})
    vi.attach_cell_coords(cell, cfgp)                       # coords needed by the physical criterion
    cell = pr.trim_reliable(cell, net, criterion,
                            {"alpha": args.alpha, "R_frac": args.rfrac, "depth_max": args.depth_max})
    # optional PHASE curves for the SAME cell (invert group AND phase jointly: phase is valid to
    # ~1 lambda vs the group 2 lambda gate, so it reaches deeper). Loaded from the phase tomography
    # root; trimmed by the same criterion. run_bayhunter_cell consumes cfg["curves_phase"].
    cell_ph = None
    if getattr(args, "phase_root", None):
        pr.PROD_ROOT[(net, "phase")] = args.phase_root.format(net=net)
        cell_ph = vi.load_cell_curves(args.phase_root.format(net=net), ix, iy,
                                      waves=("fund", "overtone", "love"))
        vi.attach_cell_coords(cell_ph, cfgp)
        if "overtone" in waves:
            cell_ph = vi.restrict_periods(cell_ph, {"overtone": (args.overtone_min_t, None)})
        # trim against the same net's group references (phase refs share the vsg_modesep dir);
        # PROD_ROOT[(net,'phase')] points res_diag at the phase maps.
        pr.PROD_ROOT[net] = args.phase_root.format(net=net)
        # phase-validity alpha (0.2 = group 0.5 x 1lambda/2.5lambda): the group edge-distance
        # factor applied to phase killed all long-T phase at hull-edge cells and collapsed
        # their depth reach (Basel-1 floor 1.8 vs 5.6 km) -- see grid_vs_inversion --phase-alpha
        cell_ph = pr.trim_reliable(cell_ph, net, criterion,
                                   {"alpha": getattr(args, "phase_alpha", 0.2),
                                    "R_frac": args.rfrac,
                                    "depth_max": args.depth_max})
        pr.PROD_ROOT[net] = (args.production_root.format(net=net)
                             if getattr(args, "production_root", None) else prod)
    # the measure/tag MUST be in the workdir name: parallel group/phase/both runs of the same
    # (well, waveset, criterion) would otherwise share one bh_results dir and collide.
    _wtag = f"_{getattr(args, 'tag', None) or getattr(args, 'measure', 'group')}"
    # bounded noise runs are PAIRED against free runs of the same cell -- keep them collision-free
    # even when the caller forgot a distinct --tag
    if getattr(args, "noise_regime", "free") != "free":
        _wtag += f"_{args.noise_regime}"
    workdir = os.path.join(os.path.dirname(out_npz),
                           f"work_{'_'.join(waves)}_{criterion}{_wtag}_{ix}_{iy}")
    os.makedirs(workdir, exist_ok=True)
    curves, curves_phase = {}, {}
    for w in waves:
        if cell.has(w):
            fp = os.path.join(workdir, f"disp_{w}.txt")
            np.savetxt(fp, np.column_stack(cell.curves[w]), fmt="%.6f")
            curves[w] = fp
        if cell_ph is not None and cell_ph.has(w):
            fp = os.path.join(workdir, f"disp_{w}_phase.txt")
            np.savetxt(fp, np.column_stack(cell_ph.curves[w]), fmt="%.6f")
            curves_phase[w] = fp
    # --measure selects WHICH measurement set is inverted (all with the same waves/radial config):
    #   group = group curves only (legacy);  phase = phase curves only (keys come out as *_phase);
    #   both  = joint group+phase. Comparing the three isolates what each measurement buys in depth.
    meas = getattr(args, "measure", "group")
    if meas == "phase":
        if not curves_phase:
            raise SystemExit("--measure phase needs --phase-root (no phase curves for this cell)")
        curves, curves_phase = curves_phase, {}
    cfg = dict(measure=("phase" if meas == "phase" else "group"),
               curves=curves, out_npz=out_npz, savepath=os.path.join(workdir, "bh_results"),
               cell=[cell.ix, cell.iy, cell.lon, cell.lat], depth_max=args.depth_max,
               vs_bounds=[args.vs_min, args.vs_max], n_layers=[1, 20], maxfrac=vi.MAX_ADJ_FRAC,
               nchains=args.n_chains, iter_burnin=args.iter_burnin, iter_main=args.iter_main,
               maxmodels=args.maxmodels, pred_nsub=args.pred_nsub, save_ensemble=True)
    if meas == "both" and curves_phase:          # joint group+phase inversion
        cfg["curves_phase"] = curves_phase
    if getattr(args, "vpvs_range", None):        # [lo,hi] => BayHunter searches Vp/Vs (free)
        cfg["vpvs"] = list(args.vpvs_range)
    if getattr(args, "radial", False):           # per-layer signed gamma=(Vsh-Vsv)/Vsv (fork ext.)
        cfg["radial_anisotropy"] = True
        cfg["radial_prior"] = list(getattr(args, "radial_prior", None) or (-0.35, 0.35))
    if getattr(args, "noise_regime", "free") != "free":
        # bounded: sigma ~ U(0.5*S_min, 3*S_min) per target instead of the free U(1e-4, 0.5).
        # Run PAIRED with a free run of the same cell; see run_bayhunter_cell's noise-regime
        # block for the rationale (free = silent data exclusion, bounded = loud rail verdict).
        cfg["noise_regime"] = args.noise_regime
    if getattr(args, "use_mp", False):
        # REAL multiprocessing via fork. BayHunter's mp_inversion does NOT deadlock on macOS (the
        # old comment was wrong): it fails under the macOS-default SPAWN because Process(target=)
        # is a closure and spawn must pickle it. Forcing fork works -- measured 3.5x on 8 chains
        # with nthreads=4, posterior statistically identical to serial (vs_median agrees to
        # 1.2e-3 km/s; only the unseeded ensemble-thinning at SingleChain.py:812 differs).
        cfg["use_mp"] = True
        cfg["mp_nthreads"] = int(getattr(args, "mp_nthreads", 0) or 0)
    if getattr(args, "parallel_tempering", False):
        # PT costs posterior samples: only t1chains/nchains of them sit at T=1 and BayHunter keeps
        # ONLY those (Plotting.py:271) -- ~19% at the t1chains=3 default, so ~2.7x fewer than the
        # old 8/16. Budget --n-chains / --iter-main accordingly. Defaults + rationale live in
        # noisepy.pt_defaults; `None` (not 0) means "unset", so --t1chains 0 is no longer silently
        # swallowed by an `or`.
        t1, mt = pt_defaults.resolve(getattr(args, "t1chains", None),
                                     getattr(args, "maxtemp", None), args.n_chains)
        cfg["parallel_tempering"] = True
        cfg["t1chains"] = t1
        cfg["maxtemp"] = mt
    cfgpath = os.path.join(workdir, "config.json")
    json.dump(cfg, open(cfgpath, "w"))
    env = dict(os.environ, OBJC_DISABLE_INITIALIZE_FORK_SAFETY="YES", VECLIB_MAXIMUM_THREADS="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    print(f"  inverting cell ({ix},{iy}) ...", flush=True)
    subprocess.run([args.bayhunter_python, args.bayhunter_runner, cfgpath], check=True, env=env)
    return out_npz


def _chain_kept(r, legacy_dev=None):
    """Which chains are actually IN the posterior, using the rule that actually built it.

    The point of this function is to mirror the ensemble, so it must track what the runner does.
    run_bayhunter_cell._use_abs_outlier_cut replaces BayHunter's get_outliers with an ABSOLUTE
    Delta-logL cut (`best - median <= outlier_delta`), so since then the posterior has NOT been
    built with `dev` and reproducing the relative rule here misdescribed the ensemble it claims to
    mirror. It now reads the run's own `outlier_delta` (saved into the npz), falling back to
    vs_reliability.DELTA_LOGL for npz written before that key existed.

    `dev` is RELATIVE (|1 - loglike/best|), so its tolerance in real log units is dev*|best| and
    its strictness depends entirely on the likelihood SCALE:
      * best near zero -> the ratio explodes. Boettstein group+phase has best = -5.4, so a chain
        at -5.8 -- 0.4 log units away, plainly the same basin -- scores dev = 0.075 and is cut;
        dev=0.05 kept 1/16 chains, an artifact, not a finding.
      * best large -> dev is very permissive and lets stranded chains through.
    A likelihood RATIO is the meaningful quantity; a relative deviation is not.

    Pass `legacy_dev` (e.g. 0.05) ONLY to regenerate a pre-2026-07 figure with its original rule.

    Returns (loglike, kept_mask, label) -- label names the rule, for stamping on the figure so a
    regenerated panel is distinguishable from an old one on sight.
    """
    if "chain_loglike_med" not in r.files:
        return None, None, ""
    ll = np.asarray(r["chain_loglike_med"], float)
    if ll.size == 0:
        return None, None, ""
    best = np.nanmax(ll)
    if not np.isfinite(best):
        return ll, np.ones(ll.size, bool), "all (best non-finite)"
    if legacy_dev is not None:
        kept = (np.abs(1.0 - ll / best) <= legacy_dev if best != 0
                else np.ones(ll.size, bool))
        return ll, kept, f"legacy dev={legacy_dev:g}"
    delta = float(r["outlier_delta"]) if "outlier_delta" in r.files else float(vr.DELTA_LOGL)
    return ll, (best - ll) <= delta, f"ΔlogL ≤ {delta:g}"


def _temp_note(r):
    """Short label describing the temperature provenance of chain_like_p2, for figure titles.

    Three distinguishable states, and the distinction matters: a legacy npz predates the
    temperature companion, so we CANNOT know whether it was a PT run -- saying nothing would let a
    temperature-mixed trace pass as a clean one.
    """
    if "chain_temps_p2" not in r.files:
        return "" if "pt_enabled" in r.files and not int(r["pt_enabled"]) \
            else "  [temperature-mixed? legacy npz]"
    t = np.asarray(r["chain_temps_p2"], float)
    if not np.isfinite(t).any():
        return ""                                  # no tempering info recorded => PT was off
    return "  [T=1 samples only]" if np.nanmax(t) > 1.0 else ""


def _t1_mask(r, i):
    """Boolean mask selecting chain i's T=1 samples in chain_like_p2 (all True if not tempered).

    Under PT a sample at T>1 is drawn from a FLATTENED posterior and fits worse BY CONSTRUCTION,
    so any statistic over a raw p2 trace penalises a chain for having been hot. Only T=1 samples
    enter the posterior (Plotting.py:271), so only they should enter its diagnostics.
    """
    if "chain_temps_p2" not in r.files:
        return None
    t = np.asarray(r["chain_temps_p2"], float)
    if i >= t.shape[0] or not np.isfinite(t[i]).any():
        return None
    return t[i] == 1.0


def _chain_clusters(r, dlog=None):
    """Scale-free chain grouping: absolute Delta(log-likelihood) from the best chain.

    A log-likelihood DIFFERENCE is the statistically meaningful quantity (a likelihood ratio);
    a relative deviation is not. Chains within a few log units of the best are sampling the same
    posterior; a chain tens of log units down is in a different basin.

    Default threshold: max(5.0, 3 x the median spacing of the top half) -- adapts to how tightly
    this particular ensemble clusters without inheriting `dev`'s scale bug. Also reports the
    LARGEST GAP, which is what actually separates basins when one exists.

    Returns dict(ll, dlog_from_best, kept, n_kept, gap_idx, gap_size, gap_ratio, thresh).
    """
    if "chain_loglike_med" not in r.files:
        return None
    ll = np.asarray(r["chain_loglike_med"], float)
    if ll.size == 0:
        return None
    best = np.nanmax(ll)
    d = best - ll                                  # >= 0, in log-likelihood units
    order = np.argsort(d)
    ds = d[order]
    gaps = np.diff(ds)
    gap_idx = int(np.argmax(gaps)) if gaps.size else -1
    gap_size = float(gaps[gap_idx]) if gaps.size else 0.0
    med_gap = float(np.median(gaps[gaps > 0])) if (gaps > 0).any() else 0.0
    gap_ratio = gap_size / med_gap if med_gap > 0 else np.inf
    if dlog is None:
        dlog = 5.0            # e^5 ~ 150x likelihood ratio: same-basin chains sit well inside
    # A gap marks a basin boundary only if it is BOTH relatively prominent (>=5x typical
    # spacing) AND absolutely large (>= dlog). Requiring only the ratio fires on noise: a 1.3
    # logL gap inside an ensemble spanning 4 logL total is not a basin boundary, but it is 7.8x
    # the typical spacing and would cut 16 healthy chains down to 3.
    if gaps.size and gap_ratio >= 5.0 and gap_size >= dlog:
        thresh = float((ds[gap_idx] + ds[gap_idx + 1]) / 2.0)
    else:
        thresh = float(dlog)
    kept = d <= thresh
    return dict(ll=ll, dlog_from_best=d, kept=kept, n_kept=int(kept.sum()),
                gap_idx=gap_idx, gap_size=gap_size, gap_ratio=gap_ratio, thresh=thresh)


def _noise_panel(ax, r):
    """Posterior of the hierarchical noise sigma, one curve per wave/measure target.

    Read this ALONGSIDE the Vs/gamma posteriors and the curve fits -- it says how much of each
    curve the sampler chose to BELIEVE. BayHunter uses our per-period sigmas only relatively
    (scaled_err = yerr/yerr.min()) and inverts the absolute level, so the effective std at point i
    is (S_i/S_min)*sigma. Hence `sigma` IS the effective noise at the best-measured point, and

        inflation = sigma / S_min

    is the factor by which the sampler inflated our stated uncertainty. Inflation ~1 = our error
    bars were taken at face value. Inflation >> 1 = the sampler decided that curve is noise and
    quietly stopped fitting it -- which is exactly how a joint group+phase run reports chi_eff ~ 1
    on every diagnostic while missing the phase data by 7 sigma.

    A sigma pressed against the prior's UPPER bound means the sampler wanted to disbelieve the
    curve even more than the prior allows: the fit is then constrained by the prior, not the data.
    """
    if "noise_post" not in r.files:
        ax.text(0.5, 0.5, "no noise posterior saved\n(rerun to record it)", ha="center",
                va="center", fontsize=8, transform=ax.transAxes, color="0.4")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("noise sigma posterior", fontsize=10)
        return
    N = np.asarray(r["noise_post"], float)
    waves = [str(w) for w in r["waves"]] if "waves" in r.files else []
    if N.ndim != 2 or N.size == 0 or not waves:
        ax.set_title("noise sigma posterior", fontsize=10); return
    smin = np.asarray(r["obssig_min"], float) if "obssig_min" in r.files else np.full(len(waves), np.nan)
    prior = np.asarray(r["noise_sigma_prior"], float) if "noise_sigma_prior" in r.files else None
    cols = {"fund": "tab:blue", "overtone": "tab:red", "love": "tab:green"}
    for i, w in enumerate(waves):
        col = 4 * i + 1                                   # [corr, sigma, sigma_c1, sigma_c2]
        if col >= N.shape[1]:
            break
        s = N[:, col]; s = s[np.isfinite(s)]
        if s.size == 0:
            continue
        base = w.replace("_phase", "")
        c = cols.get(base, "0.5")
        ls = "--" if w.endswith("_phase") else "-"
        infl = (np.median(s) / smin[i]) if (i < len(smin) and np.isfinite(smin[i]) and smin[i] > 0) else np.nan
        ax.hist(s, bins=50, histtype="step", lw=1.3, color=c, ls=ls, density=True,
                label=f"{w} ({np.median(s):.3f}, {infl:.1f}x)")
    if prior is not None and prior.ndim == 1 and prior.size == 2:
        for b in prior:
            ax.axvline(b, color="k", lw=0.8, ls=":")
        ax.set_xlim(0, prior[1] * 1.05)
        ax.text(prior[1], ax.get_ylim()[1], " prior\n bound", fontsize=6, va="top", color="0.3")
    elif prior is not None and prior.ndim == 2:
        # BOUNDED regime: per-target (lo, hi) around each curve's own S_min. Draw each target's
        # bounds in its colour; a posterior pressed against its upper bound is the loud
        # "cannot fit this curve within plausible errors" verdict the regime exists to produce.
        for i, w in enumerate(waves[:len(prior)]):
            c = cols.get(w.replace("_phase", ""), "0.5")
            ls = "--" if w.endswith("_phase") else "-"
            for b in prior[i]:
                ax.axvline(b, color=c, lw=0.7, ls=":", alpha=0.7)
        ax.set_xlim(0, np.nanmax(prior) * 1.1)
        ax.text(0.98, 0.98, "bounded regime\n(per-target bounds)", fontsize=6, va="top",
                ha="right", transform=ax.transAxes, color="0.3")
    ax.set(xlabel=r"noise $\sigma$ [km/s]  (median, inflation vs our $S_{min}$)", ylabel="density")
    ax.legend(fontsize=6, frameon=False)
    ax.set_title(r"hierarchical noise $\sigma$ posterior", fontsize=10)


def _convergence_panel(ax, r, dev=0.05):
    """Per-chain log-likelihood history: burn-in then main, kept chains vs discarded.

    This is the panel that answers 'has it converged?'. Three things to read off it:
      1. Do the discarded (red) chains sit at a visibly worse plateau? -> multimodal; the
         posterior is one basin chosen by an outlier filter, not the full solution space.
      2. Is the MAIN phase still climbing? -> not converged; iter_main is too short and the
         'posterior' is still a transient.
      3. Do chains only reach the good level LATE in burn-in? -> iter_burnin is too short.
    """
    p1 = np.asarray(r["chain_like_p1"], float) if "chain_like_p1" in r.files else None
    p2 = np.asarray(r["chain_like_p2"], float) if "chain_like_p2" in r.files else None
    if p1 is None or p2 is None or p1.size == 0:
        ax.text(0.5, 0.5, "no chain traces saved\n(rerun to record them)", ha="center",
                va="center", fontsize=8, transform=ax.transAxes, color="0.4")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("convergence", fontsize=10)
        return
    ll, kept, rule = _chain_kept(r, dev)
    nb = int(r["iter_burnin"]) if "iter_burnin" in r.files else p1.shape[1]
    nm = int(r["iter_main"]) if "iter_main" in r.files else p2.shape[1]
    x1 = np.linspace(0, nb, p1.shape[1])
    x2 = np.linspace(nb, nb + nm, p2.shape[1])
    for i in range(p1.shape[0]):
        good = kept is None or (i < len(kept) and kept[i])
        c, a, lw = ("tab:blue", 0.75, 0.7) if good else ("tab:red", 0.5, 0.6)
        ax.plot(x1[np.isfinite(p1[i])], p1[i][np.isfinite(p1[i])], color=c, alpha=a, lw=lw)
        ax.plot(x2[np.isfinite(p2[i])], p2[i][np.isfinite(p2[i])], color=c, alpha=a, lw=lw)
    ax.axvline(nb, color="k", ls="--", lw=0.9)
    # robust y-range: the good basin, not the -1e9 rejected excursions
    fin = np.concatenate([p2[i][np.isfinite(p2[i])] for i in range(p2.shape[0])
                          if np.isfinite(p2[i]).any()]) if p2.size else np.array([])
    if fin.size:
        hi = np.nanmax(fin); lo = np.nanpercentile(fin, 2)
        pad = 0.05 * max(hi - lo, 1e-6)
        ax.set_ylim(lo - pad, hi + pad)
    # las etiquetas van DESPUES de set_ylim: colocadas antes, get_ylim() devuelve el rango
    # autoescalado que incluye las excursiones rechazadas (~-1e15) y el texto queda anclado
    # a esa coordenada de datos. set_ylim recorta la VISTA pero no la extension del artista,
    # asi que savefig(bbox_inches="tight") calcula un lienzo de ~1e15 px y revienta
    # (TypeError en RendererAgg). Solo se disparaba con trazas guardadas (chain_like_p1/p2).
    ax.text(nb, ax.get_ylim()[0], " main ", fontsize=7, va="bottom", ha="left", color="0.3")
    ax.text(nb, ax.get_ylim()[0], "burn-in ", fontsize=7, va="bottom", ha="right", color="0.3")
    ax.set(xlabel="iteration", ylabel="log-likelihood")
    nk = int(kept.sum()) if kept is not None else -1
    # Stamp the RULE on the panel: the kept/discarded split moved from a relative dev to an
    # absolute Delta-logL cut, and a regenerated figure is otherwise indistinguishable from an old
    # one. Also flag a temperature-mixed trace: under PT a chain's p2 mixes every temperature it
    # held, so hot excursions are plotted next to cold samples and read as wander.
    tmix = _temp_note(r)
    ax.set_title(f"convergence — {nk}/{p1.shape[0]} chains kept ({rule})"
                 f"{tmix}\n(blue=kept, red=discarded)", fontsize=8)


def _convergence_text(r, dev=0.05):
    """Convergence scalars for the QC text block, with an explicit verdict.

    burnin_delta_frac = |mean(last 10% of burn-in) - mean(first 10% of main)| / |scale|.
    Large => the chain was still moving when burn-in ended, i.e. burn-in was too short and the
    main phase inherits a transient. The 2:1 burnin:main ratio is only meaningful once this
    is small -- ratio alone guarantees nothing.
    """
    ll, kept, rule = _chain_kept(r, dev)
    def g(k):
        return float(np.asarray(r[k]).ravel()[0]) if k in r.files else np.nan
    nb, nm = g("iter_burnin"), g("iter_main")
    cd, bd = g("chain_disagree"), g("burnin_delta_frac")
    nk = int(kept.sum()) if kept is not None else -1
    nc = int(len(kept)) if kept is not None else -1
    # main-phase drift: is the likelihood still climbing after burn-in? Computed over T=1 samples
    # only where the temperature companion exists -- a hot sample sits far below the cold ones by
    # construction, so a temperature-mixed trace manufactures "drift" out of healthy tempering.
    drift = np.nan
    if "chain_like_p2" in r.files:
        p2 = np.asarray(r["chain_like_p2"], float)
        rows = []
        for i in range(p2.shape[0]):
            if kept is not None and i < len(kept) and not kept[i]:
                continue
            row = p2[i]
            m = _t1_mask(r, i)
            if m is not None:
                row = np.where(m, row, np.nan)
            if np.isfinite(row).any():
                rows.append(row)
        if rows:
            m = np.nanmean(np.vstack(rows), axis=0)
            m = m[np.isfinite(m)]
            if m.size > 20:
                h = m.size // 4
                scale = abs(np.nanmedian(m)) or 1.0
                drift = float((np.nanmean(m[-h:]) - np.nanmean(m[:h])) / scale)
    cl = _chain_clusters(r)
    flags = []
    # Judge on the SCALE-FREE grouping, never on dev (see _chain_kept docstring).
    if cl is not None and len(cl["kept"]) and cl["n_kept"] / len(cl["kept"]) < 0.5:
        flags.append(f"ONLY {cl['n_kept']}/{len(cl['kept'])} CHAINS IN BEST BASIN")
    # burnin_delta_frac is normalised by the MAIN-PHASE STD, so it is already scale-free and
    # values < 1 mean the burn-in->main jump is SMALLER than the chain's own fluctuation, i.e.
    # no detectable jump. Only >~1 indicates burn-in ended before the chain settled. (An earlier
    # >0.10 gate here was invented and flagged healthy runs as broken.)
    if np.isfinite(bd) and bd > 1.0:
        flags.append("BURN-IN TOO SHORT")
    if np.isfinite(drift) and abs(drift) > 0.02:
        flags.append("MAIN STILL DRIFTING")
    verdict = ("  !! " + "; ".join(flags)) if flags else "  ok"
    ratio = f" ({nb/nm:.1f}:1)" if (np.isfinite(nb) and np.isfinite(nm) and nm > 0) else ""
    lines = ["convergence"]
    if cl is not None:
        lines.append(f"  chains:   {cl['n_kept']}/{len(cl['kept'])} within "
                     f"{cl['thresh']:.1f} logL of best")
        lines.append(f"            ({nk}/{nc} by BayHunter dev={dev} -- relative, scale-biased)")
        if np.isfinite(cl["gap_ratio"]):
            lines.append(f"  largest logL gap:  {cl['gap_size']:.2f} "
                         f"({cl['gap_ratio']:.1f}x typical)")
    else:
        lines.append(f"  chains:   {nk}/{nc} kept (dev={dev})")
    if np.isfinite(nb) and np.isfinite(nm) and (nb or nm):
        lines.append(f"  iters:    {nb:.0f} burn-in : {nm:.0f} main{ratio}")
    lines += [f"  chain_disagree:    {cd:.3f} km/s",
              f"  burnin_delta_frac: {bd:.3f}  (std units; >1 = bad)",
              f"  main drift:        {drift:+.4f}",
              verdict]
    return "\n".join(lines)


def plot_cell_posterior(npz_path, out_png, net="", crit_label=""):
    """Posterior figure for an arbitrary GRID cell npz (no well needed).

    Everything comes from the npz itself: coords, depth_max, vs bounds, wave set. This is the
    per-cell analogue of the well figure (Vs density, gamma strip + P(gamma>0), dispersion fits
    group+phase, noise-sigma posterior vs bounds, misfit + convergence panels), and it is called
    by run_bayhunter_cell at the END OF EVERY CELL RUN so posterior figures appear alongside the
    convergence diagnostics as the grid progresses -- not in a post-hoc batch (user decision
    2026-07-17). Failures must never kill a cell run: callers wrap in try/except.
    """
    r = np.load(npz_path, allow_pickle=True)
    ix, iy = (int(v) for v in r["cell_ixiy"])
    lon, lat = (float(v) for v in r["cell_lonlat"])
    z = np.asarray(r["depth"], float)
    vs_bounds = (float(np.nanmin(r["vs_p025"])), float(np.nanmax(r["vs_p975"])))
    waveset = "_".join(str(w) for w in r["waves"] if not str(w).endswith("_phase"))
    well = (f"cell {ix}_{iy}", lat, lon, float("nan"))
    plot_well(net, well, ix, iy, 0.0, npz_path, [], out_png,
              vs_bounds, float(np.nanmax(z)), waveset, crit_label)


def plot_well(net, well, ix, iy, dist_km, npz, overlay, out_png, vs_bounds, depth_max, waveset,
              crit_label=""):
    r = np.load(npz, allow_pickle=True)
    z = r["depth"]
    # grid cells do not carry the model ensemble (save_ensemble off: ~10 MB/cell x 864 cells);
    # fall back to the saved posterior percentiles for the density panel in that case.
    ens = r["ens_vs"] if "ens_vs" in r.files else None      # (nmodel, ndepth) or None
    name, lat, lon, welldep = well
    vmin, vmax = vs_bounds

    # radial runs carry a per-layer gamma=(Vsh-Vsv)/Vsv posterior -> add a gamma(z) strip
    has_gamma = ("gamma_median" in r.files and np.isfinite(r["gamma_median"]).any()
                 and np.nanmax(np.abs(r["gamma_p84"] - r["gamma_p16"])) > 1e-6)
    # a phase fit panel is shown whenever the run carried phase targets (obs_*_phase keys)
    has_phase = any(k.endswith("_phase") and k.startswith("obs_") for k in r.files)
    if has_gamma:
        fig = plt.figure(figsize=(16.5, 10.6))
        gs = fig.add_gridspec(4, 4, width_ratios=[3.0, 0.55, 1.6, 3.2],
                              height_ratios=[1, 1, 1, 0.9], wspace=0.34, hspace=0.52)
        ax_ga = fig.add_subplot(gs[:3, 2])
        ax_ft = fig.add_subplot(gs[0, 3]); ax_fp = fig.add_subplot(gs[1, 3])
        ax_ms = fig.add_subplot(gs[2, 3])
        ax_cv = fig.add_subplot(gs[3, :2]); ax_ns = fig.add_subplot(gs[3, 2:])
    else:
        fig = plt.figure(figsize=(13.5, 10.6))
        gs = fig.add_gridspec(4, 3, width_ratios=[3.0, 0.55, 3.2],
                              height_ratios=[1, 1, 1, 0.9], wspace=0.32, hspace=0.52)
        ax_ga = None
        ax_ft = fig.add_subplot(gs[0, 2]); ax_fp = fig.add_subplot(gs[1, 2])
        ax_ms = fig.add_subplot(gs[2, 2])
        ax_cv = fig.add_subplot(gs[3, 0]); ax_ns = fig.add_subplot(gs[3, 1:])
    ax_vs = fig.add_subplot(gs[:3, 0]); ax_if = fig.add_subplot(gs[:3, 1], sharey=ax_vs)

    # --- Vs posterior density (per-depth histogram; percentile-band fallback without ens) ---
    vsb = np.linspace(vmin, vmax, 121)
    if ens is not None:
        dens = np.zeros((len(vsb) - 1, len(z)))
        for k in range(len(z)):
            col = ens[:, k]; col = col[np.isfinite(col)]
            if col.size:
                dens[:, k] = np.histogram(col, bins=vsb, density=True)[0]
        cmax = dens.max(axis=0, keepdims=True)                # per-depth marginal posterior
        dens = dens / np.where(cmax > 0, cmax, 1.0)
        ax_vs.pcolormesh(0.5 * (vsb[1:] + vsb[:-1]), z, dens.T, cmap="hot_r", vmin=0, vmax=1,
                         shading="auto")
        med_style = ("c-", "c--", "c:")
    else:
        ax_vs.fill_betweenx(z, r["vs_p025"], r["vs_p975"], color="orange", alpha=0.20,
                            lw=0, label="95%")
        ax_vs.fill_betweenx(z, r["vs_p16"], r["vs_p84"], color="orange", alpha=0.45,
                            lw=0, label="68%")
        med_style = ("r-", "r--", "r:")
    ax_vs.plot(r["vs_median"], z, med_style[0], lw=1.6, label="posterior median")
    ax_vs.plot(r["vs_p16"], z, med_style[1], lw=0.8); ax_vs.plot(r["vs_p84"], z, med_style[1], lw=0.8)
    ax_vs.plot(r["vs_p025"], z, med_style[2], lw=0.6); ax_vs.plot(r["vs_p975"], z, med_style[2], lw=0.6)
    for v, zc, lab, col, ls in overlay:
        m = zc <= depth_max
        ax_vs.plot(v[m], zc[m], color=col, ls=ls, lw=1.9, label=lab)
    ax_vs.set(xlim=(vmin, vmax), ylim=(0, depth_max), xlabel="Vs [km/s]", ylabel="depth [km]")
    ax_vs.invert_yaxis(); ax_vs.legend(fontsize=6.5, loc="lower left")
    ax_vs.set_title("Vs posterior distribution", fontsize=10)

    # --- interface probability as a horizontal histogram (count on x-axis) ---
    ifd = r["iface_depths"] if "iface_depths" in r.files else np.array([])
    nmod = int(r["n_models"])
    bin_dz = 0.1                                              # 100 m depth bins
    edges = np.arange(0, depth_max + bin_dz, bin_dz)
    cnt, _ = np.histogram(ifd, bins=edges)
    zc = 0.5 * (edges[1:] + edges[:-1])
    ax_if.barh(zc, cnt, height=bin_dz, color="steelblue", edgecolor="none")
    ax_if.set_xlabel("interface\ncount", fontsize=8)
    ax_if.set_title("interface prob.", fontsize=9)
    ax_if.tick_params(labelleft=False, labelsize=7)

    # --- radial anisotropy gamma(z) = (Vsh-Vsv)/Vsv: 68/95% bands + sign significance ---
    # CONTINUOUS gamma (post CONTINUOUS_ZETA_PLAN.md): every layer carries gamma, so significance
    # is the SIGN posterior P(gamma>0) -- ~0.5 where the data do not constrain the sign, near 0/1
    # where they do -- and the median is a real estimate (no spike at 0 pinning it). Legacy
    # spike-and-slab npz instead carry gamma_frac_nonzero = P(gamma!=0) (occupancy), which is only
    # plotted for those files and is labelled as such; on continuous runs it would be ~1
    # everywhere and mean nothing.
    if ax_ga is not None:
        gm, g16, g84 = r["gamma_median"], r["gamma_p16"], r["gamma_p84"]
        ax_ga.fill_betweenx(z, r["gamma_p025"], r["gamma_p975"], color="tab:orange", alpha=0.15,
                            lw=0, label="95%")
        ax_ga.fill_betweenx(z, g16, g84, color="tab:orange", alpha=0.38, lw=0, label="68%")
        ax_ga.plot(gm, z, color="darkorange", lw=1.8, label="median")
        ax_ga.axvline(0, color="k", lw=0.8)
        ax_ga.set(ylim=(0, depth_max), xlabel="$\\gamma=(V_{SH}-V_{SV})/V_{SV}$")
        ax_ga.invert_yaxis(); ax_ga.tick_params(labelleft=False, labelsize=7)
        ax_ga.legend(fontsize=6, loc="lower left")
        if "gamma_p_pos" in r.files:                           # P(gamma>0) on a twin x-axis
            axp = ax_ga.twiny()
            axp.plot(r["gamma_p_pos"], z, color="tab:blue", lw=1.2, ls=":")
            axp.axvline(0.5, color="tab:blue", lw=0.5, alpha=0.4)
            axp.set_xlim(0, 1); axp.set_xlabel("P($\\gamma>0$)", fontsize=7, color="tab:blue")
            axp.tick_params(axis="x", labelsize=6, colors="tab:blue")
            axp.invert_yaxis()
        elif "gamma_frac_nonzero" in r.files:                  # legacy spike-and-slab npz
            axp = ax_ga.twiny()
            axp.plot(r["gamma_frac_nonzero"], z, color="tab:blue", lw=1.2, ls=":")
            axp.set_xlim(0, 1)
            axp.set_xlabel("P($\\gamma\\neq0$) (legacy)", fontsize=7, color="tab:blue")
            axp.tick_params(axis="x", labelsize=6, colors="tab:blue")
            axp.invert_yaxis()
        ax_ga.set_title("radial anisotropy", fontsize=9)

    # --- dispersion data fit: group (ax_ft) and, if inverted jointly, phase (ax_fp) ---
    colors = {"fund": "tab:blue", "overtone": "tab:red", "love": "tab:green"}

    def _fit_panel(ax, suffix, ylab):
        """obs + posterior-predictive band per wave. suffix='' -> group keys, '_phase' -> phase."""
        drawn = False
        for w in ("fund", "overtone", "love"):
            key = f"{w}{suffix}"
            if f"obs_{key}" not in r.files:
                continue
            T, U, S = r[f"obsT_{key}"], r[f"obs_{key}"], r[f"obssig_{key}"]
            ax.errorbar(T, U, yerr=S, fmt="o", ms=4, color=colors[w], label=f"{w} obs", zorder=3)
            drawn = True
            if f"pred_{key}" in r.files:
                q = np.nanpercentile(r[f"pred_{key}"], [16, 50, 84], axis=0)
                o = np.argsort(r[f"predT_{key}"]); Tp = r[f"predT_{key}"][o]
                ax.fill_between(Tp, q[0][o], q[2][o], color=colors[w], alpha=0.25, zorder=1)
                ax.plot(Tp, q[1][o], "-", color=colors[w], lw=1.3, zorder=2)
        ax.set(xlabel="period [s]", ylabel=ylab)
        if drawn:
            ax.legend(fontsize=6.5)
        return drawn

    _fit_panel(ax_ft, "", "group velocity [km/s]")
    ax_ft.set_title("group-velocity fit", fontsize=10)
    if has_phase:
        _fit_panel(ax_fp, "_phase", "phase velocity [km/s]")
        ax_fp.set_title("phase-velocity fit", fontsize=10)
    else:
        ax_fp.text(0.5, 0.5, "no phase targets\n(--phase-root to add)", ha="center", va="center",
                   transform=ax_fp.transAxes, fontsize=8, color="0.5"); ax_fp.axis("off")

    # --- misfit distribution + QC text ---
    # BayHunter stores misfits as (nmodels, ntargets+1): one column per target plus a JOINT
    # column that is their SUM. Histogramming the flat array superimposes 4 different
    # quantities and produces a spurious "bimodal" shape (fund+love cluster low, overtone+joint
    # cluster high) that reads as chain multimodality but is nothing of the sort. Split by
    # column: each target gets its own curve, and the joint is drawn separately.
    mis = np.asarray(r["ens_misfit"], float) if "ens_misfit" in r.files else np.array([])
    nmod = int(r["ens_vs"].shape[0]) if "ens_vs" in r.files else 0
    if mis.size and nmod and mis.size % nmod == 0:
        ncol = mis.size // nmod
        M = mis.reshape(nmod, ncol)
        wv = [str(w) for w in r["waves"]] if "waves" in r.files else []
        labs = (wv + ["joint"]) if ncol == len(wv) + 1 else [f"col{j}" for j in range(ncol)]
        cols = {"fund": "tab:blue", "overtone": "tab:red", "love": "tab:green"}
        for j, lab in enumerate(labs):
            c = "0.35" if lab == "joint" else cols.get(lab.replace("_phase", ""), "0.5")
            ax_ms.hist(M[:, j], bins=40, histtype="step", lw=1.3, color=c,
                       label=f"{lab} ({np.median(M[:, j]):.2f})")
        ax_ms.legend(fontsize=6, frameon=False, loc="upper right")
        ax_ms.set(xlabel="misfit (per target; joint = sum)", ylabel="count")
    elif mis.size:
        ax_ms.hist(mis, bins=40, color="0.5")
        ax_ms.set(xlabel="model misfit", ylabel="count")
    ax_ms.set_title("misfit distribution", fontsize=10)
    chi = {}
    for key in ("fund", "overtone", "love", "fund_phase", "overtone_phase", "love_phase"):
        if f"pred_{key}" in r.files and f"obs_{key}" in r.files:
            predmed = np.nanmedian(r[f"pred_{key}"], axis=0)
            obs, sig = r[f"obs_{key}"], r[f"obssig_{key}"]
            m = np.isfinite(predmed) & np.isfinite(obs) & (sig > 0)
            if m.any():
                chi[key] = float(np.sqrt(np.mean(((obs[m] - predmed[m]) / sig[m]) ** 2)))
    nlay = r["ens_nlayers"] if "ens_nlayers" in r.files else r.get("n_layers_post")
    def _c(k):
        return f"{chi[k]:.2f}" if k in chi else "—"
    txt = (f"well: {name}  (depth {welldep} m)\n"
           f"well lon,lat: {lon:.4f}, {lat:.4f}\n"
           f"cell ({ix},{iy}) — {dist_km:.2f} km from well\n"
           f"posterior models: {nmod}\n"
           f"n_layers: {np.mean(nlay):.1f} ± {np.std(nlay):.1f}\n"
           f"chi (group  | phase)\n"
           f"  fund:     {_c('fund'):>5} | {_c('fund_phase'):>5}\n"
           f"  overtone: {_c('overtone'):>5} | {_c('overtone_phase'):>5}\n"
           f"  love:     {_c('love'):>5} | {_c('love_phase'):>5}")
    txt += "\n" + _convergence_text(r)
    ax_ms.text(1.02, 1.0, txt, transform=ax_ms.transAxes, va="top", ha="left", fontsize=8,
               family="monospace", bbox=dict(boxstyle="round", fc="0.95", ec="0.6"))
    _convergence_panel(ax_cv, r)
    _noise_panel(ax_ns, r)
    wslab = WS_LABEL.get(waveset, waveset)
    ct = f"  |  {crit_label}" if crit_label else ""
    fig.suptitle(f"{net.capitalize()} — {name} well  |  cell ({ix},{iy}), {dist_km:.2f} km  "
                 f"[{wslab}]{ct}", fontsize=12, x=0.42)
    fig.savefig(out_png, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("wrote", out_png)


def _chi(r):
    out = {}
    for w in ("fund", "overtone", "love"):
        if f"pred_{w}" in r.files and f"obs_{w}" in r.files:
            pm = np.nanmedian(r[f"pred_{w}"], axis=0)
            obs, sig = r[f"obs_{w}"], r[f"obssig_{w}"]
            m = np.isfinite(pm) & np.isfinite(obs) & (sig > 0)
            if m.any():
                out[w] = float(np.sqrt(np.mean(((obs[m] - pm[m]) / sig[m]) ** 2)))
    return out


def _vs_density_panel(ax, r, overlay, vs_bounds, depth_max, show_legend=False):
    """Draw one Vs posterior-density panel (per-depth marginal) with overlays + median band."""
    vmin, vmax = vs_bounds
    z = r["depth"]; ens = r["ens_vs"]
    vsb = np.linspace(vmin, vmax, 121)
    dens = np.zeros((len(vsb) - 1, len(z)))
    for k in range(len(z)):
        col = ens[:, k]; col = col[np.isfinite(col)]
        if col.size:
            dens[:, k] = np.histogram(col, bins=vsb, density=True)[0]
    cmax = dens.max(axis=0, keepdims=True)
    dens = dens / np.where(cmax > 0, cmax, 1.0)
    ax.pcolormesh(0.5 * (vsb[1:] + vsb[:-1]), z, dens.T, cmap="hot_r", vmin=0, vmax=1, shading="auto")
    ax.plot(r["vs_median"], z, "c-", lw=1.6, label="posterior median")
    ax.plot(r["vs_p16"], z, "c--", lw=0.7); ax.plot(r["vs_p84"], z, "c--", lw=0.7)
    for v, zc, lab, col, ls in overlay:
        mm = zc <= depth_max
        ax.plot(v[mm], zc[mm], color=col, ls=ls, lw=1.7, label=lab)
    ax.set(xlim=(vmin, vmax), ylim=(0, depth_max))
    ax.invert_yaxis()
    if show_legend:
        ax.legend(fontsize=6, loc="lower left")


def plot_criteria_comparison(net, well, ix, iy, dist_km, npz_by_crit, overlay, out_png,
                             vs_bounds, depth_max, waveset):
    """One row of Vs-posterior panels, one per criterion, for side-by-side judgement vs the well."""
    name, lat, lon, welldep = well
    crits = [c for c in CRIT_ORDER if c in npz_by_crit and os.path.exists(npz_by_crit[c])]
    fig, axs = plt.subplots(1, len(crits), figsize=(3.4 * len(crits), 5.2), sharey=True, squeeze=False)
    for a, crit in zip(axs[0], crits):
        r = np.load(npz_by_crit[crit], allow_pickle=True)
        _vs_density_panel(a, r, overlay, vs_bounds, depth_max, show_legend=(crit == crits[0]))
        Tf = r["obsT_fund"] if "obsT_fund" in r.files else np.array([np.nan])
        chi = _chi(r)
        cst = "  ".join(f"χ{k[0]} {chi[k]:.2f}" for k in ("fund", "overtone", "love") if k in chi)
        a.set_title(f"{CRIT_LABEL[crit]}\nfund T {np.nanmin(Tf):.1f}–{np.nanmax(Tf):.1f}s ({len(Tf)})\n{cst}",
                    fontsize=9)
        a.set_xlabel("Vs [km/s]")
    axs[0][0].set_ylabel("depth [km]")
    wslab = WS_LABEL.get(waveset, waveset)
    fig.suptitle(f"{net.capitalize()} — {name} well  |  cell ({ix},{iy}), {dist_km:.2f} km  "
                 f"[{wslab}] — period-trim criteria", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("wrote", out_png)


def plot_waveset_overlap(net, well, ix, iy, dist_km, npz_by_ws, overlay, out_png,
                         vs_bounds, depth_max, crit_label=""):
    """Decisive Love-vs-Rayleigh consistency figure for one well cell + criterion.

    LEFT: overlay the posterior median + 68% Vs(z) band of each available waveset
    (love / fundot / fundotlove / ...). Overlapping bands => the wave types agree and the joint
    inversion is well-behaved; a Love-only band separating from the Rayleigh (fundot) band beyond
    their 68% envelopes => the two data types carry contradictory information.
    RIGHT: per-waveset per-wave chi table (is chi_love inflated when chi_fund is low?)."""
    name, lat, lon, welldep = well
    vmin, vmax = vs_bounds
    order = ["fund", "love", "fundot", "fundlove", "fundotlove"]
    wss = [w for w in order if w in npz_by_ws and os.path.exists(npz_by_ws[w])]
    if len(wss) < 2:
        return
    wscol = {"fund": "tab:blue", "love": "tab:green", "fundot": "tab:red",
             "fundlove": "tab:purple", "fundotlove": "black"}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 6), gridspec_kw={"width_ratios": [3, 2]})
    for v, zc, lab, col, ls in overlay:
        mm = zc <= depth_max
        axL.plot(v[mm], zc[mm], color=col, ls=ls, lw=1.6, label=lab, zorder=1)
    rows = [["waveset", "χ fund", "χ ot", "χ love"]]
    for w in wss:
        r = np.load(npz_by_ws[w], allow_pickle=True)
        z = r["depth"]; c = wscol.get(w, "0.4")
        axL.fill_betweenx(z, r["vs_p16"], r["vs_p84"], color=c, alpha=0.18, zorder=2)
        axL.plot(r["vs_median"], z, color=c, lw=2.0, label=WS_LABEL.get(w, w), zorder=3)
        chi = _chi(r)
        rows.append([WS_LABEL.get(w, w)] + [f"{chi.get(k, float('nan')):.2f}"
                                            for k in ("fund", "overtone", "love")])
    axL.set(xlim=(vmin, vmax), ylim=(0, depth_max), xlabel="Vs [km/s]", ylabel="depth [km]")
    axL.invert_yaxis(); axL.legend(fontsize=7, loc="lower left")
    axL.set_title("posterior Vs(z): median + 68% band per waveset", fontsize=10)
    axR.axis("off")
    tb = axR.table(cellText=rows, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.7)
    axR.set_title("per-wave misfit χ (obs vs posterior-predictive median)", fontsize=9)
    ct = f"  |  {crit_label}" if crit_label else ""
    fig.suptitle(f"{net.capitalize()} — {name} well  |  cell ({ix},{iy}), {dist_km:.2f} km  "
                 f"— Love vs Rayleigh consistency{ct}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("wrote", out_png)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", required=True, choices=("riehen", "aargau"))
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--max-cell-dist", type=float, default=1.5, help="km; skip wells with no cell within")
    ap.add_argument("--overtone-min-t", type=float, default=1.0)
    ap.add_argument("--depth-max", type=float, default=6.0)
    ap.add_argument("--vs-min", type=float, default=0.3)
    ap.add_argument("--vs-max", type=float, default=4.0)   # 3.6 railed the basement posterior (Riniken/Boettstein/Riehen-2 p97.5 at the bound)
    ap.add_argument("--n-chains", type=int, default=4)
    ap.add_argument("--iter-burnin", type=int, default=50_000)
    ap.add_argument("--iter-main", type=int, default=30_000)
    ap.add_argument("--maxmodels", type=int, default=20_000)
    ap.add_argument("--pred-nsub", type=int, default=300)
    ap.add_argument("--wavesets", default="fund,fundot",
                    help="comma list: fund, fundot (fund+overtone), love, fundlove (fund+Love), "
                         "fundotlove (fund+overtone+Love)")
    ap.add_argument("--criterion", default="all",
                    help="per-cell period-trim criterion: 'all' (none+combined+physical+"
                         "tomographic), a single one, or a comma list (e.g. physical,combined)")
    ap.add_argument("--alpha", type=float, default=0.5, help="physical: wavelengths of edge clearance")
    ap.add_argument("--rfrac", type=float, default=0.5, help="tomographic: frac of peak res_diag")
    ap.add_argument("--vpvs-range", default=None,
                    help="'lo,hi' => free Vp/Vs (BayHunter searches it); omit = fixed 1.73")
    ap.add_argument("--wells", default=None,
                    help="comma list of well names to restrict to (default: all in-hull)")
    ap.add_argument("--production-root", default=None,
                    help="single production root with ALL wave subdirs, '{net}' interpolated "
                         "(e.g. .../Projects/{net}/tomo/swtomotv-output-uni/production); "
                         "omit = legacy split roots (Rayleigh production + LOVE_PROD)")
    ap.add_argument("--phase-root", default=None,
                    help="PHASE production root (same layout, '{net}' interpolated); when set, the "
                         "cell's phase curves are inverted JOINTLY with the group curves (phase is "
                         "valid to ~1 lambda vs the group 2 lambda gate -> deeper). Adds a "
                         "phase-fit panel + group|phase chi columns.")
    ap.add_argument("--radial", action="store_true",
                    help="invert per-layer signed radial anisotropy gamma=(Vsh-Vsv)/Vsv "
                         "(BayHunter fork radial_anisotropy mode)")
    ap.add_argument("--radial-prior", default=None,
                    help="'lo,hi' gamma prior bounds (default -0.35,0.35; NB gamma railed at the old -0.20 floor at several cells -- a clipped gamma is not an estimate)")
    ap.add_argument("--noise-regime", choices=["free", "bounded"], default="free",
                    help="hierarchical noise sigma prior: 'free' = U(1e-4,0.5) per target "
                         "(historical; the sampler may silently inflate a curve's noise away, "
                         "measured 6.4x on joint phase), 'bounded' = U(0.5*S_min, 3*S_min) per "
                         "target (still hierarchical; a sigma railed at the upper bound is a "
                         "LOUD cannot-fit verdict). Run PAIRED with free; the difference is the "
                         "diagnostic. Auto-tags the workdir/outputs.")
    ap.add_argument("--use-mp", action="store_true",
                    help="run chains with REAL multiprocessing (forces the fork start method). "
                         "~nthreads x faster; posterior statistically identical to serial. Keep "
                         "OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES and *_NUM_THREADS=1 set, as "
                         "fork + multithreaded BLAS genuinely hangs on macOS")
    ap.add_argument("--mp-nthreads", type=int, default=0,
                    help="worker processes for --use-mp (0 = cpu_count)")
    ap.add_argument("--parallel-tempering", action="store_true",
                    help="enable parallel tempering. NB the runner drives chains in LOCKSTEP when "
                         "this is on (BayHunter's swap needs all chains at the same iteration); "
                         "and only the t1chains at T=1 contribute to the posterior, so PT buys "
                         f"mixing at the cost of samples (~{pt_defaults.PT_T1CHAINS}/nchains kept). "
                         "Judge the ladder by pt_round_trips in the npz, NOT by swap acceptance")
    ap.add_argument("--t1chains", type=int, default=None,
                    help=f"chains held at T=1 (default {pt_defaults.PT_T1CHAINS}; see "
                         f"noisepy/pt_defaults.py). Equal-T chains never swap with each other, so "
                         f"extra T=1 chains add no mixing")
    ap.add_argument("--maxtemp", type=float, default=None,
                    help=f"top of the temperature ladder (default {pt_defaults.PT_MAXTEMP}). Set "
                         f"this from the BARRIER HEIGHT: maxtemp=2 only halves logL differences, "
                         f"so it melts nothing. High swap acceptance is a symptom of a too-tight "
                         f"ladder, not health")
    ap.add_argument("--measure", default="group", choices=("group", "phase", "both"),
                    help="which measurement set to invert (waves/radial unchanged): group (legacy) "
                         "| phase (needs --phase-root) | both (joint). Run all three to compare "
                         "what each buys in depth.")
    ap.add_argument("--tag", default=None,
                    help="suffix appended to output npz/png names (keeps parallel configs apart)")
    ap.add_argument("--bayhunter-python", required=True)
    ap.add_argument("--bayhunter-runner", required=True)
    args = ap.parse_args()
    net = args.net
    if args.vpvs_range:
        args.vpvs_range = [float(x) for x in args.vpvs_range.split(",")]
    if args.radial_prior:
        args.radial_prior = [float(x) for x in args.radial_prior.split(",")]
    TAG = f"_{args.tag}" if args.tag else ""
    # bounded-noise runs are paired against free runs of the SAME cell: without a distinct tag
    # the bounded run would find the free run's npz and silently return it (run_cell_ensemble
    # skips existing out_npz), so the "pair" would be one run read twice.
    if args.noise_regime != "free" and args.noise_regime not in TAG:
        TAG += f"_{args.noise_regime}"
    well_filter = set(w.strip() for w in args.wells.split(",")) if args.wells else None
    # The hardcoded default only exists if nothing has moved. After the tomo reorg the old path is
    # gone, and `makedirs(exist_ok=True)` would silently RECREATE it and write a fresh, empty,
    # authoritative-looking wells tree there -- forking the results. Require an explicit --outdir
    # once the default no longer exists, instead of resurrecting it.
    default_outdir = f"/Users/genevievesavard/Codes/extract_higher_modes/Projects/{net}/tomo/vs_inversion/wells"
    if args.outdir:
        outdir = args.outdir
    elif os.path.isdir(default_outdir):
        outdir = default_outdir
    else:
        raise SystemExit(
            f"--outdir is required: the legacy default '{default_outdir}' does not exist "
            f"(the tomo/ tree was reorganized). Pass --outdir explicitly so results are not "
            f"written to a resurrected legacy path.")
    os.makedirs(outdir, exist_ok=True)

    hull = stations_hull(net)
    # cell mesh (ix,iy -> lon,lat) only; the grid volume was archived in the reorg but its mesh is
    # still the reference cell layout. It is set by the swtomotv bounds+dx, not the Vs result.
    vol = np.load(f"/Users/genevievesavard/Codes/extract_higher_modes/Projects/{net}/tomo/2_vs_depth_inversion/_archive/grid/volume_fundot.npz")
    cells, ll = vol["cells"], vol["lonlat"]

    crits = (CRIT_ORDER if args.criterion == "all"
             else [c.strip() for c in args.criterion.split(",") if c.strip()])
    bad = [c for c in crits if c not in CRIT_LABEL]
    if bad:
        raise SystemExit(f"unknown criterion(s): {bad}; valid: {list(CRIT_LABEL)} or 'all'")
    ws_keys = [w.strip() for w in args.wavesets.split(",") if w.strip()]
    # npz index[(well_name, crit)][wskey] -> ensemble npz, for the cross-waveset overlap figure
    idx = {}
    for wskey in ws_keys:
        waves = WAVESETS[wskey]
        for well in WELLS[net]:
            if well_filter and well[0] not in well_filter:
                continue
            name, lat, lon, welldep = well
            if welldep < 1000:
                print(f"skip {name}: depth {welldep} m < 1 km"); continue
            if not hull.contains_point((lon, lat)):
                print(f"skip {name}: outside station convex hull"); continue
            d = np.hypot((ll[:, 0] - lon) * np.cos(np.deg2rad(lat)), ll[:, 1] - lat) * 111.0
            j = int(np.argmin(d))
            if d[j] > args.max_cell_dist:
                print(f"skip {name}: nearest cell {d[j]:.2f} km > {args.max_cell_dist} km"); continue
            ix, iy = int(cells[j, 0]), int(cells[j, 1])
            overlay = overlay_curves(net, name)
            npz_by_crit = {}
            for crit in crits:
                npz = os.path.join(outdir, f"well_{name}_{wskey}_{crit}{TAG}_cell_{ix}_{iy}.npz")
                run_cell_ensemble(net, ix, iy, npz, args, waves, crit)
                npz_by_crit[crit] = npz
                idx.setdefault((name, crit), {"ixiy": (ix, iy), "d": d[j],
                                              "well": well, "overlay": overlay})[wskey] = npz
                # full detail figure (dispersion fit + misfit-histogram bimodality) per waveset+crit
                plot_well(net, well, ix, iy, d[j], npz, overlay,
                          os.path.join(outdir, f"well_{name}_{wskey}_{crit}{TAG}.png"),
                          (args.vs_min, args.vs_max), args.depth_max, wskey, CRIT_LABEL[crit])
            if len(npz_by_crit) > 1:           # all criteria -> side-by-side comparison
                plot_criteria_comparison(net, well, ix, iy, d[j], npz_by_crit, overlay,
                                         os.path.join(outdir, f"well_{name}_{wskey}_criteria{TAG}.png"),
                                         (args.vs_min, args.vs_max), args.depth_max, wskey)
    # cross-waveset Love-vs-Rayleigh consistency figure, per well per criterion
    for (name, crit), info in idx.items():
        npz_by_ws = {k: v for k, v in info.items() if k in WAVESETS}
        if len(npz_by_ws) >= 2:
            ix, iy = info["ixiy"]
            plot_waveset_overlap(net, info["well"], ix, iy, info["d"], npz_by_ws, info["overlay"],
                                 os.path.join(outdir, f"well_{name}_{crit}{TAG}_wsoverlap.png"),
                                 (args.vs_min, args.vs_max), args.depth_max, CRIT_LABEL[crit])
    print("done.")


if __name__ == "__main__":
    main()
