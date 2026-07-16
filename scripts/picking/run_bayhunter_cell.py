"""BayHunter runner for ONE tomography cell -- executed in the BayHunter conda env
(numpy<1.26) as a subprocess of noisepy.vs_inversion.run_bayhunter.

Reads a JSON config (curve txt paths, priors, iters), inverts the fundamental (surf96 mode 1)
+ 1st-overtone (mode 2) Rayleigh GROUP-velocity curves jointly, with the LVZ/HVZ-permitting
<=maxfrac adjacent-layer contrast enforced natively via initparams lvz=hvz=maxfrac, and writes
a result npz in the engine-agnostic schema that noisepy.vs_inversion.load_result expects.

Usage: python run_bayhunter_cell.py config.json
"""
import json
import os
import sys
import time

# CRITICAL (macOS): force the 'fork' start method BEFORE importing BayHunter/numpy so the
# multiprocessing.Manager BayHunter creates in __init__ uses fork, not spawn. Setting it after
# the import (or leaving the macOS spawn default) deadlocks/crawls the Manager+shared arrays.
import multiprocessing as mp
try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass

import types

import numpy as np
from BayHunter import Targets, MCMC_Optimizer, PlotFromStorage, ModelMatrix, Model

from noisepy.vs_reliability import DELTA_LOGL as vr_DELTA_LOGL
from noisepy import pt_defaults

BH_MODE = {"fund": 1, "overtone": 2, "love": 1}     # surf96 mode: 1=fundamental, 2=first higher
# wave key -> (disba/surf96 wave type, disba mode index); mirrors vs_inversion.WAVEDEF.
WAVE_DISBA = {"fund": ("rayleigh", 0), "overtone": ("rayleigh", 1), "love": ("love", 0)}
# (wave type, measure) -> BayHunter surf96 target class. LoveDispersionGroup exists in the
# BayHunter_Aniso fork (surf96 'ldispgr'); no fork change needed.
BH_TARGET = {("rayleigh", "group"): Targets.RayleighDispersionGroup,
             ("rayleigh", "phase"): Targets.RayleighDispersionPhase,
             ("love", "group"): Targets.LoveDispersionGroup,
             ("love", "phase"): Targets.LoveDispersionPhase}


def _chain_median_at_t1(likefile):
    """Median post-burnin log-likelihood of a chain, counting ONLY its T=1 samples.

    Chain-level outlier detection MUST be temperature-aware once parallel tempering is on. A
    chain's p2likes mixes every temperature it held (temperatures SWAP between chains), and a
    sample drawn at T>1 comes from a FLATTENED posterior, so it fits worse BY CONSTRUCTION. Taking
    the median over the raw array therefore penalises a chain for having spent time hot, drops it
    as an "outlier", and discards its perfectly valid T=1 models -- exactly the samples
    Plotting.py's `alltemps==1` filter would have kept. Observed: a PT run kept 11/16 chains where
    the identical non-PT run kept 16/16.

    NB BayHunter's own get_outliers (Plotting.py:130-133) has the same flaw, so this is not just
    an artefact of our absolute-delta override.

    STRICT NO-OP when tempering is off: BayHunter fills sharedtemperatures with 1.0 in that case
    (mcmcOptimizer._init_parallel_tempering else-branch), so the mask keeps every sample. Also a
    no-op if the temperatures file is missing (older runs).
    """
    likes = np.load(likefile)
    tfile = likefile.replace("likes", "temperatures")
    if not os.path.exists(tfile):
        return np.median(likes)
    temps = np.load(tfile)
    if temps.shape != likes.shape:
        return np.median(likes)
    m = temps == 1.0
    if not m.any():
        return -np.inf          # never sampled cold -> a genuine outlier, not a silent nan
    return np.median(likes[m])


def _git(repo, *args):
    """git output for `repo`, or None if unavailable (not a repo, no git, detached env...)."""
    try:
        import subprocess
        return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True,
                              check=True, timeout=10).stdout.strip()
    except Exception:
        return None


def _sha256(path):
    import hashlib
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return ""


def _provenance(initparams, opt, outlier_delta):
    """Everything needed to reconstruct, from the npz alone, WHAT ran and HOW it was tempered.

    Consumers MUST treat a MISSING key as "legacy run, unknown" -- never as "PT off". A pre-2026-07
    npz has no pt_enabled and may well have been a PT run; silently reading it as non-PT is exactly
    the error this block exists to prevent.
    """
    import importlib
    import BayHunter
    # the MODULE, not the class: `from BayHunter import SingleChain` binds the class (re-exported
    # by BayHunter/__init__.py), which has no __file__.
    _sc_mod = importlib.import_module("BayHunter.SingleChain")
    bh_dir = os.path.dirname(os.path.abspath(BayHunter.__file__))
    prov = {}
    pt = bool(initparams.get("parallel_tempering", False))
    prov["pt_enabled"] = int(pt)
    prov["pt_t1chains"] = int(initparams.get("t1chains", -1)) if pt else -1
    prov["pt_maxtemp"] = float(initparams.get("maxtemp", np.nan)) if pt else np.nan
    # The REALISED ladder, not just the two knobs: _create_temperature_ladder
    # (mcmcOptimizer.py:184-198) silently clamps t1chains > nchains to all-ones with only a logger
    # warning, so PT can be a complete no-op while pt_enabled=1. Only the ladder itself shows that.
    lad = getattr(opt, "temperatures", None)
    prov["pt_ladder"] = (np.asarray(lad[:, 0], np.float32) if lad is not None
                         else np.array([], np.float32))
    prov["pt_swaps_accepted"] = int(getattr(opt, "accepted_temperature_swaps", -1))
    prov["pt_swaps_total"] = int(getattr(opt, "total_temperature_swaps", -1))
    # Thinned per-chain temperature history -> ROUND-TRIP RATE (T=1 -> T_max -> T=1), which is the
    # only statistic that says whether the ladder actually TRANSPORTS states from the hot end down
    # to T=1. Swap acceptance cannot: a ladder crammed into [1,2] posts a healthy-looking rate
    # while melting nothing. Raw is (nchains x iterations) ~15 MB at 16x240k, so stride it to
    # ~1000 columns (~64 kB) -- round trips are slow, so nothing is lost.
    if pt and lad is not None:
        stride = max(1, lad.shape[1] // 1000)
        hist = np.asarray(lad[:, ::stride], np.float32)
        prov["pt_temp_history"] = hist
        prov["pt_temp_history_stride"] = int(stride)
        rt, tot = pt_defaults.round_trips(hist)
        prov["pt_round_trips"] = np.asarray(rt, int)
        print(f"  PT ladder: T={np.round(np.unique(hist[:, 0]), 2)}  "
              f"round trips {tot} total ({tot / max(1, hist.shape[0]):.2f}/chain)", flush=True)
    else:
        prov["pt_temp_history"] = np.empty((0, 0), np.float32)
        prov["pt_temp_history_stride"] = -1
        prov["pt_round_trips"] = np.array([], int)
    prov["outlier_delta"] = float(outlier_delta)
    # the module that was IMPORTED -- not src/ -- because that is what actually executed
    prov["singlechain_sha256"] = _sha256(os.path.abspath(_sc_mod.__file__).replace(".pyc", ".py"))
    # The BayHunter commit comes from the stamp deploy.sh leaves in site-packages, NOT from asking
    # git about the import path: the fork is installed as a COPY, so the deployed tree lives
    # outside the checkout and is not a git repo at all -- `git -C site-packages rev-parse HEAD`
    # simply fails, which is why this used to record an empty sha. If the stamp is absent the code
    # was deployed by hand rather than by deploy.sh, and "" is the honest answer.
    info = {}
    try:
        with open(os.path.join(bh_dir, ".deploy_info.json")) as f:
            info = json.load(f)
    except Exception:
        pass
    prov["bayhunter_git_sha"] = str(info.get("git_sha", ""))
    prov["bayhunter_dirty"] = int(bool(info["dirty"])) if "dirty" in info else -1
    prov["bayhunter_deployed_utc"] = str(info.get("deployed_utc", ""))
    prov["driver_git_sha"] = _git(os.path.dirname(os.path.abspath(__file__)),
                                  "rev-parse", "HEAD") or ""
    return prov


def _use_abs_outlier_cut(obj, delta):
    """Replace BayHunter's RELATIVE outlier filter with an absolute Delta-logL cut.

    BayHunter's get_outliers scores chains as 1 - median/best (best>0) or 1 - best/median
    (best<0) and cuts at `dev`, so the tolerance in real log-likelihood units is dev*|best|
    (and in the best<0 branch the denominator is each chain's OWN median, varying chain to
    chain). Identical runs therefore get wildly different strictness purely from the
    likelihood scale -- see noisepy.vs_reliability.DELTA_LOGL. Two consequences here:
    posteriors built from 1-3 of 16 healthy chains near best~0, and no-op filtering when
    |best| is large. It also raises UnboundLocalError when best == 0 exactly (neither branch
    assigns `scores`). Cutting on best - median <= delta is scale-free and has none of that.

    Bound onto the instance because save_final_distribution calls self.get_outliers(dev=dev)
    internally; the signature keeps `dev` so that call site still works, and ignores it.
    """
    def get_outliers(self, dev=None):
        idxs, medians = [], []
        for likefile in self.likefiles[1]:
            cidx, _, _ = self._return_c_p_t(likefile)
            idxs.append(cidx)
            medians.append(_chain_median_at_t1(likefile))
        idxs, medians = np.array(idxs, float), np.array(medians, float)
        best = np.nanmax(medians)
        dlog = best - medians
        outliers = idxs[dlog > delta]
        print(f"> outlier cut: keeping {len(idxs) - len(outliers)}/{len(idxs)} chains within "
              f"{delta:g} logL of best ({best:.2f}); logL spread {np.ptp(medians):.2f}")
        if len(outliers) > 0:
            with open(os.path.join(self.datapath, "outliers.dat"), "w") as f:
                f.write(f"# Outlier chainindices with absolute delta-logL > {delta:g}\n")
                for i, o in zip(idxs[dlog > delta], dlog[dlog > delta]):
                    f.write("%d\t%.3f\n" % (i, o))
        return outliers

    outlierfile = os.path.join(obj.datapath, "outliers.dat")
    if os.path.exists(outlierfile):        # save_final_distribution removes this itself, but
        os.remove(outlierfile)             # only after get_outliers has already run
    obj.get_outliers = types.MethodType(get_outliers, obj)


def main(cfgpath):
    cfg = json.load(open(cfgpath))
    depth_max = float(cfg["depth_max"])
    measure = cfg.get("measure", "group")
    workdir = os.path.dirname(os.path.abspath(cfg["out_npz"]))
    # per-run savepath (grid runs share one out dir, so a shared bh_results would collide);
    # honour cfg["savepath"] when given, else keep the single-cell default.
    savepath = cfg.get("savepath") or os.path.join(workdir, "bh_results")
    os.makedirs(savepath, exist_ok=True)

    # ---- targets: Rayleigh fund (mode1)/overtone (mode2) + Love fund, group and/or phase ----
    # `curves` (with `measure`) is the group set (back-compat). An optional `curves_phase` dict
    # adds Rayleigh PHASE targets to the SAME JointTarget -> a joint group+phase inversion, which
    # forwards both from one model via surf96 and tightens the deep (long-period) structure.
    sources = [(cfg["curves"], measure)]
    if cfg.get("curves_phase"):
        sources.append((cfg["curves_phase"], "phase"))
    tlist, waves, obs, wmeta = [], [], {}, {}
    for curves, meas in sources:
        for w in ("fund", "overtone", "love"):
            if w not in curves:
                continue
            d = np.loadtxt(curves[w]); d = d[d[:, 0].argsort()]
            T, U, S = d[:, 0], d[:, 1], d[:, 2]
            wtype, dmode = WAVE_DISBA[w]
            tgt = BH_TARGET[(wtype, meas)](T, U, yerr=S ** 2)        # yerr is VARIANCE
            tgt.moddata.plugin.set_modelparams(mode=BH_MODE[w])      # overtone / Love extension
            key = w if meas == "group" else f"{w}_phase"             # distinct output key for phase
            tlist.append(tgt); waves.append(key); obs[key] = (T, U, S)
            wmeta[key] = (wtype, dmode, meas)
    target = Targets.JointTarget(targets=tlist)

    # ---- priors / initparams (complete AzAniso key set; isotropic run) ----
    # lvz/hvz are FRACTIONAL: deeper layer velocity >= (1-lvz)*above and <= (1+hvz)*above,
    # i.e. lvz=hvz=maxfrac allows LVZ and HVZ but caps the adjacent contrast at maxfrac.
    # maxfrac=None disables the constraint entirely (BayHunter treats lvz/hvz=None as "off").
    vmin, vmax = cfg["vs_bounds"]
    N = int(cfg["nchains"])
    frac = cfg["maxfrac"]
    frac = None if frac is None else float(frac)
    # Vp/Vs: scalar (default 1.73) = fixed; a [lo,hi] list = BayHunter searches it per layer
    # (propfixed[4]=0 already enables the vpvs proposal). Free Vp/Vs is used to test whether the
    # graben Love/Rayleigh tension is a fixed-ratio artifact (saturated fill -> high Vp/Vs).
    vpvs_prior = cfg.get("vpvs", 1.73)
    vpvs_prior = tuple(vpvs_prior) if isinstance(vpvs_prior, (list, tuple)) else float(vpvs_prior)
    priors = {"vpvs": vpvs_prior, "layers": tuple(cfg["n_layers"]), "vs": (vmin, vmax),
              "z": (0.0, depth_max), "triangular_zprop": False, "mohoest": None,
              "mantle": None, "rfnoise_corr": 0.9, "swdnoise_corr": 0.0,
              "rfnoise_sigma": (1e-5, 0.1), "swdnoise_sigma": (1e-4, 0.5),
              "swdnoise_sigma_c1": (1e-5, 0.02), "swdnoise_sigma_c2": (1e-5, 0.02)}
    initparams = {"nchains": N, "iter_burnin": int(cfg["iter_burnin"]),
                  "iter_main": int(cfg["iter_main"]),
                  "propdist": (0.05, 0.3, 0.1, 0.005, 0.01),   # vs, z_move, vs_birth, noise, vpvs
                  "propfixed": (0, 0, 1, 0, 0), "acceptance": (40, 48),
                  "thickmin": 0.02, "relative_thickmin": True,
                  "lvz": frac, "hvz": frac,                     # <- the <=maxfrac constraint
                  "rcond": 1e-5, "station": "cell", "savepath": savepath, "maxmodels": 50000,
                  "parallel_tempering": bool(cfg.get("parallel_tempering", False)),
                  # defaults live in noisepy.pt_defaults (single source of truth -- they used to be
                  # duplicated here and in well_vs_qc.py and could diverge silently)
                  **dict(zip(("t1chains", "maxtemp"),
                             pt_defaults.resolve(cfg.get("t1chains"), cfg.get("maxtemp"), N))),
                  "azimuthal_anisotropy": False}                # isotropic (Vs) or radial below
    # RADIAL anisotropy: psi2amp block = signed per-layer gamma=(Vsh-Vsv)/Vsv (fork extension);
    # Love targets forward on Vsh, Rayleigh on Vsv. cfg["radial_prior"] = [lo, hi].
    if cfg.get("radial_anisotropy", False):
        initparams["radial_anisotropy"] = True
        priors["radial"] = tuple(cfg.get("radial_prior", (-0.15, 0.25)))

    # ---- optional REAL multiprocessing -------------------------------------------------------
    # BayHunter's mp_inversion does NOT deadlock on macOS -- the long-standing comment here was
    # wrong. It launches mp.Process(target=gochain), where gochain is a CLOSURE inside
    # mp_inversion. macOS has defaulted to the SPAWN start method since Python 3.8 (fork is unsafe
    # with the ObjC runtime), spawn must PICKLE the process target, and closures are unpicklable
    # -> "AttributeError: Can't pickle local object". It CRASHES instantly, it does not hang.
    # BayHunter never calls set_start_method, so it inherits spawn; on Linux (fork) the child
    # inherits memory, nothing is pickled, and the same code is fine. Forcing fork fixes macOS:
    # measured 8 chains x 60k iters, rc=0, no hang, even with ~40 competing processes.
    # The real macOS hazard is fork + MULTITHREADED BLAS (Accelerate/vecLib): the child inherits
    # mutexes held by threads that don't survive the fork. That is almost certainly the original
    # "deadlock" -- and it is already mitigated by the env this runner is launched with
    # (OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES, VECLIB_MAXIMUM_THREADS/OMP_NUM_THREADS=1).
    # KEEP THOSE SET when use_mp is on.
    USE_MP = bool(cfg.get("use_mp", False))
    if USE_MP:
        import multiprocessing as _mp
        try:
            _mp.set_start_method("fork", force=True)      # must precede any mp object (Manager)
        except RuntimeError as e:
            print(f"  (could not force fork: {e}; falling back to serial)", flush=True)
            USE_MP = False

    t0 = time.time()
    opt = MCMC_Optimizer(target, initparams=initparams, priors=priors, random_seed=42)
    PT = bool(initparams["parallel_tempering"])
    # BayHunter's mp_inversion (Manager + Process + shared arrays) deadlocks/crawls on macOS, so
    # we drive the chains ourselves. Same on-disk output (config.pkl from __init__ + per-chain
    # finalize()) that PlotFromStorage reads.
    if USE_MP:
        # Chain seeds are drawn in MCMC_Optimizer.__init__ (random_seed=self.rstate.randint(1000)),
        # i.e. BEFORE any parallelism, so without PT the chains are independent and deterministic:
        # fork must reproduce the serial result exactly. Each child finalize()s to disk and
        # PlotFromStorage reads from disk, so the parent never needs the chain objects back.
        nth = int(cfg.get("mp_nthreads", 0)) or 0        # 0 -> BayHunter uses mp.cpu_count()
        print(f"  multiprocessing ON (fork, nthreads={nth or 'auto'})", flush=True)
        opt.mp_inversion(nthreads=nth)
    elif not PT:
        # Chains are independent without tempering -> run each to completion (best cache locality).
        for k, chain in enumerate(opt.chains):
            while chain.iiter < chain.iter_phase2:
                chain.iterate()
            chain.finalize()
            print(f"  chain {k} done ({time.time()-t0:.0f}s elapsed)", flush=True)
    else:
        # PARALLEL TEMPERING requires LOCKSTEP, not parallelism. _swap_temperatures(iiter) lives on
        # the OPTIMIZER (mcmcOptimizer.py:327) and is called right after barrier.wait(): it reads
        # temperatures[:, iiter-1] and every chain's live likelihood, so all chains must sit at the
        # SAME iteration. Running chain 0 to completion first (the loop above) would therefore set
        # a temperature ladder and NEVER swap -- hot chains would sample a flattened posterior
        # forever and pollute the ensemble. That is strictly WORSE than PT off while looking like
        # healthy chain diversity. So replicate BayHunter's own gochain() round-robin, minus the
        # threads: advance every chain ONE step, then swap. The chain re-reads its temperature each
        # iteration (SingleChain.py:738 -> temperatures[iiter + iter_phase1]) and publishes its
        # likelihood (SingleChain.py:708 -> currentlike_shared[chainidx]), so this is exact.
        # Cost is ~nil: the total iterate() count is unchanged, only the order.
        print(f"  parallel tempering ON: {initparams['t1chains']}/{N} chains at T=1, "
              f"maxtemp={initparams['maxtemp']}", flush=True)
        iiter = 0
        while True:
            done = 0
            for chain in opt.chains:
                if chain.iiter < chain.iter_phase2:
                    chain.iterate()
                else:
                    done += 1
            iiter += 1
            if done == len(opt.chains):
                break
            if iiter < opt.iterations:
                opt._swap_temperatures(iiter)
            if iiter % 20000 == 0:
                sw = opt.accepted_temperature_swaps, opt.total_temperature_swaps
                print(f"  iiter {iiter}/{opt.iterations} ({time.time()-t0:.0f}s) "
                      f"swaps {sw[0]}/{sw[1]}", flush=True)
        for chain in opt.chains:
            chain.finalize()
        print(f"  PT done: accepted {opt.accepted_temperature_swaps}/"
              f"{opt.total_temperature_swaps} swaps ({time.time()-t0:.0f}s)", flush=True)
    runtime = time.time() - t0

    _prune_incomplete_chains(savepath)

    # ---- posterior ----
    obj = PlotFromStorage(os.path.join(savepath, "data", "cell_config.pkl"))
    _use_abs_outlier_cut(obj, float(cfg.get("outlier_delta", vr_DELTA_LOGL)))
    obj.save_final_distribution(maxmodels=int(cfg.get("maxmodels", 50000)))
    models, = obj._get_posterior_data(["models"], final=True)
    vpvs, = obj._get_posterior_data(["vpvs"], final=True)
    vpvs = np.atleast_1d(vpvs)
    if vpvs.size == 1:
        vpvs = np.full(len(models), float(vpvs[0]))

    # ---- hierarchical NOISE posterior, (nmodels, 4*ntargets): [corr, sigma, sigma_c1, sigma_c2]
    # per target. This is first-class information, not a nuisance: BayHunter uses our per-period
    # sigmas only RELATIVELY (scaled_err = yerr/yerr.min()) and INVERTS the absolute level, so
    # `sigma` IS the effective noise at each curve's best-measured point. sigma/S_min is therefore
    # the factor by which the sampler inflated our stated uncertainty -- i.e. how much of a curve
    # it chose to disbelieve. A joint group+phase run that cannot fit both silently inflates the
    # losing curve's sigma until chi_eff ~ 1, so the conflict is INVISIBLE in the likelihood and
    # in every convergence diagnostic. Without this array that decision is unrecoverable.
    try:
        noise_post, = obj._get_posterior_data(["noise"], final=True)
        noise_post = np.atleast_2d(np.asarray(noise_post, float))
    except Exception as e:
        print(f"  (noise posterior unavailable: {e})", flush=True)
        noise_post = np.empty((0, 0))

    # ---- convergence diagnostics: burn-in trace, post-burnin distribution, trans-D layer
    # trace, inter-chain Vs(z) agreement -- see noisepy.bh_diagnostics docstring.
    from noisepy.bh_diagnostics import build_diagnostics, chain_medians_t1, plot_diagnostics
    dep_diag = np.linspace(0, depth_max, 121)
    diag = build_diagnostics(os.path.join(savepath, "data"), dep_diag, obj.mantle)
    # Name the figure after out_npz (unique per cell): a fixed name here means every cell writing
    # into the same workdir silently overwrites the previous cell's diagnostics, leaving one file
    # that shows only the LAST run while sitting next to per-cell results.
    diag_png = cfg.get("diag_png") or os.path.join(
        workdir, os.path.splitext(os.path.basename(cfg["out_npz"]))[0] + "_diagnostics.png")
    conv = plot_diagnostics(diag, diag_png,
                            title=f"convergence diagnostics -- {measure} {cfg.get('cell')}")
    print(f"  convergence: {conv} -> {diag_png}", flush=True)

    # per-chain arrays for downstream chain-count studies + depth-resolved reliability
    # (all read in envs without BayHunter): chain index, post-burnin median log-like, and
    # per-chain (p16,p50,p84) Vs(z) on dep_diag -- the compact within/between-chain summary.
    chain_idx = np.asarray(diag["chain_idx"], int)
    # Temperature-aware: counts only T=1 samples, so a chain is not penalised for the time it
    # spent hot under PT (a strict no-op without tempering). This value is saved to the npz AND
    # drives vr.assess's kept-mask below, so a raw median here would corrupt every downstream
    # convergence/reliability number even though the posterior itself is built correctly.
    chain_loglike_med = chain_medians_t1(diag) if diag["likes_p2"] else np.array([])

    # Compact per-chain log-likelihood TRACES (burn-in p1 + main p2), thinned to ~400 samples
    # each (~50 kB/cell). The raw per-chain .npy under savepath/data is deleted at cleanup, so
    # without this the convergence HISTORY is unrecoverable and only endpoint scalars survive --
    # which is how runs keeping 5/16 (and 1/16) chains reached publication figures unnoticed.
    # Saved unconditionally: whether the main phase is still climbing is the single most
    # important thing to know about a posterior, and it must be visible on the plot itself.
    def _thin_traces(seq, n=400, companion=None):
        """Thin each chain's trace to n samples. If `companion` is given (same chain order), it
        is thinned with the SAME indices, so companion[i][j] still describes seq[i][j].

        The pairing is why this takes a companion instead of being called twice: chain_like_p2 and
        chain_temps_p2 are only meaningful together, and independently-computed index sets would
        silently mislabel which samples were hot -- worse than not saving temperatures at all.
        """
        out, out_c = [], []
        for i, l in enumerate(seq):
            l = np.asarray(l, float)
            c = None if companion is None else companion[i]
            c = None if c is None else np.asarray(c, float)
            if l.size == 0:
                out.append(np.full(n, np.nan))
                out_c.append(np.full(n, np.nan))
                continue
            idx = np.linspace(0, l.size - 1, min(n, l.size)).astype(int)
            v = l[idx]
            # a temperatures file that is absent (pre-PT run) or shape-mismatched yields NaN, which
            # downstream reads as "unknown", never as "cold"
            vc = c[idx] if (c is not None and c.shape == l.shape) else np.full(idx.size, np.nan)
            if v.size < n:
                v = np.pad(v, (0, n - v.size), constant_values=np.nan)
                vc = np.pad(vc, (0, n - vc.size), constant_values=np.nan)
            out.append(v)
            out_c.append(vc)
        arr = np.array(out, dtype=np.float32) if out else np.empty((0, n), np.float32)
        if companion is None:
            return arr
        arr_c = np.array(out_c, dtype=np.float32) if out_c else np.empty((0, n), np.float32)
        return arr, arr_c
    chain_vs_profiles = np.array(diag["vs_profile"]) if diag["vs_profile"] \
        else np.empty((0, len(dep_diag)))
    chain_vs_p16 = np.array(diag["vs_p16"]) if diag.get("vs_p16") \
        else np.empty((0, len(dep_diag)))
    chain_vs_p84 = np.array(diag["vs_p84"]) if diag.get("vs_p84") \
        else np.empty((0, len(dep_diag)))

    # depth-resolved reliability from the per-chain summaries (per-depth between/within-chain
    # agreement rho(z) + wavelength depth floor -> reliable interval + cell confidence flag).
    from noisepy import vs_reliability as vr
    allT = np.concatenate([obs[w][0] for w in waves]) if waves else np.array([])
    allU = np.concatenate([obs[w][1] for w in waves]) if waves else np.array([])
    if len(chain_loglike_med) and len(chain_vs_profiles):
        rel = vr.assess(dep_diag, chain_vs_p16, chain_vs_profiles, chain_vs_p84,
                        chain_loglike_med, periods=allT, velocities=allU,
                        delta=float(cfg.get("outlier_delta", vr_DELTA_LOGL)),
                        rho_max=float(cfg.get("rho_max", vr.RHO_MAX)))
        print(f"  reliability: confidence={rel['confidence']} frac_kept={rel['frac_kept']:.2f} "
              f"reliable={rel['z_reliable_min']:.2f}-{rel['z_reliable_max']:.2f} km "
              f"(reln_frac={rel['reln_frac']:.2f}, floor {rel['z_floor']:.2f} km)", flush=True)
    else:
        rel = None

    dep = np.linspace(0, depth_max, 121)
    # ensemble Vs(z) (+ gamma(z) when radial): interpolate each posterior model to the depth grid
    RADIAL = bool(cfg.get("radial_anisotropy", False))
    prof, gprof = [], []
    for m, vp in zip(models, vpvs):
        try:
            out = Model.get_interpmodel(m, dep, vpvs=float(vp), mantle=obj.mantle)
            vs_step = out[1] if isinstance(out, (tuple, list)) else out
            prof.append(np.asarray(vs_step, float))
            if RADIAL:
                _v, _s, h_g, _a, _b, g_lay = _vp_vs_h(m, obj, float(vp))
                gprof.append(_gamma_step(np.asarray(g_lay, float), np.asarray(h_g, float), dep))
        except Exception:
            continue
    prof = np.array(prof)
    p = np.nanpercentile(prof, [2.5, 16, 50, 84, 97.5], axis=0)
    gprof = np.array(gprof) if gprof else np.zeros((0, len(dep)))
    gp = (np.nanpercentile(gprof, [2.5, 16, 50, 84, 97.5], axis=0) if len(gprof)
          else np.full((5, len(dep)), np.nan))

    # predicted dispersion for a subsample (posterior predictive band), per wave.
    # NOTE: computed with disba (not surf96) -- surf96's group-velocity root-finder can HANG
    # (infinite loop) on some LVZ posterior models; disba has iteration limits and returns NaN
    # instead. disba and surf96 group/phase velocities agree to ~1%, so the fit band is faithful.
    from disba import GroupDispersion, PhaseDispersion
    pred = {}
    isub = np.random.default_rng(0).choice(
        len(models), min(int(cfg.get("pred_nsub", 150)), len(models)), replace=False)
    sub, sub_vpvs = models[isub], vpvs[isub]
    for w in waves:
        T = obs[w][0]
        wtype, dmode, meas = wmeta[w]                    # per-target: group vs phase
        DDisp = GroupDispersion if meas == "group" else PhaseDispersion
        preds = []
        for m, vpv in zip(sub, sub_vpvs):
            vp, vs, h, _, _, g_lay = _vp_vs_h(m, obj, float(vpv))
            if RADIAL and wtype == "love":
                vs = np.asarray(vs, float) * (1.0 + g_lay)   # Love forwards on Vsh
            th = np.asarray(h, float).copy()
            if th.size:
                th[-1] = 100.0                          # half-space thickness
            rho = np.asarray(vp) * 0.32 + 0.77
            try:
                gd = DDisp(th, np.asarray(vp), np.asarray(vs), rho)
                cp = gd(np.asarray(T, float), mode=dmode, wave=wtype)
            except Exception:
                continue
            y = np.full(len(T), np.nan)
            idx = {round(float(p), 6): i for i, p in enumerate(T)}
            for pp, vv in zip(cp.period, cp.velocity):
                j = idx.get(round(float(pp), 6))
                if j is not None:
                    y[j] = vv
            if np.isfinite(y).any():
                preds.append(y)
        if preds:
            pred[w] = np.array(preds)

    d = dict(engine="bayhunter", depth=dep,
             vs_mean=np.nanmean(prof, 0), vs_median=p[2],
             vs_p025=p[0], vs_p16=p[1], vs_p84=p[3], vs_p975=p[4],
             n_models=len(prof), runtime_s=runtime, acceptance=np.nan,
             n_layers_post=_layer_counts(models), waves=np.array(waves),
             cell_ixiy=np.array(cfg["cell"][:2]), cell_lonlat=np.array(cfg["cell"][2:]),
             vpvs_post=np.asarray(vpvs, float),      # posterior Vp/Vs (const array if fixed)
             noise_post=noise_post.astype(np.float32),   # (nmodels, 4*ntargets) hierarchical noise
             noise_sigma_prior=np.asarray(priors["swdnoise_sigma"], float),
             obssig_min=np.array([np.nanmin(obs[w][2]) for w in waves], float),  # per-wave S_min
             radial=int(RADIAL),
             gamma_p025=gp[0], gamma_p16=gp[1], gamma_median=gp[2],
             gamma_p84=gp[3], gamma_p975=gp[4],     # gamma(z)=(Vsh-Vsv)/Vsv (NaN if not radial)
             gamma_frac_nonzero=(np.nanmean(np.abs(gprof) > 1e-6, axis=0)
                                 if len(gprof) else np.full(len(dep), np.nan)),
             chain_disagree=conv["chain_disagree"], frac_chains_ok=conv["frac_chains_ok"],
             burnin_delta_frac=conv["burnin_delta_frac"], n_chains_used=conv["n_chains"])
    for w in waves:
        d[f"obsT_{w}"], d[f"obs_{w}"], d[f"obssig_{w}"] = obs[w]
        if w in pred:
            d[f"predT_{w}"] = obs[w][0]
            d[f"pred_{w}"] = pred[w]

    # compact per-chain summary + depth-resolved reliability -- ALWAYS saved (a few KB), so
    # the raw per-chain traces can be dropped. chain_vs_p16/p50/p84 support per-depth rho(z).
    d["chain_idx"] = chain_idx
    d["chain_loglike_med"] = chain_loglike_med
    d["chain_like_p1"] = _thin_traces(diag["likes_p1"])   # (nchain, 400) burn-in trace
    # main-phase trace WITH its per-sample temperature, thinned on the same indices. Under PT a
    # chain's p2likes mixes every temperature it held (temperatures SWAP between chains) and a
    # T>1 sample comes from a FLATTENED posterior, so it fits worse BY CONSTRUCTION. Without the
    # companion, plots and drift statistics show hot and cold samples indistinguishably and a
    # healthy chain looks like it is wandering. NaN where the run predates PT.
    d["chain_like_p2"], d["chain_temps_p2"] = _thin_traces(
        diag["likes_p2"], companion=diag.get("temps_p2"))
    d["iter_burnin"] = int(initparams.get("iter_burnin", 0))
    d["iter_main"] = int(initparams.get("iter_main", 0))
    d["chain_vs_depth"] = dep_diag
    d["chain_vs_profiles"] = chain_vs_profiles.astype(np.float32)      # per-chain median
    d["chain_vs_p16"] = chain_vs_p16.astype(np.float32)
    d["chain_vs_p84"] = chain_vs_p84.astype(np.float32)
    if rel is not None:
        d["rho"] = rel["rho"].astype(np.float32)
        d["rho_smooth"] = rel["rho_smooth"].astype(np.float32)
        d["reliable_mask"] = rel["reliable"]
        d["z_reliable_min"] = rel["z_reliable_min"]
        d["z_reliable_max"] = rel["z_reliable_max"]
        d["reln_frac"] = rel["reln_frac"]
        d["z_floor"] = rel["z_floor"]
        d["chains_kept"] = rel["kept"]
        d["n_chains_kept"] = rel["n_kept"]
        d["frac_kept"] = rel["frac_kept"]
        d["confidence"] = rel["confidence"]

    # ---- PROVENANCE: what code, and what tempering, actually produced this file --------------
    # Without this you cannot tell from a result npz whether PT was on, let alone which ladder or
    # which acceptance ratio ran -- and the raw chain dir (with p2temperatures) is deleted below,
    # so nothing else survives to tell you. Two shas, deliberately:
    #   bayhunter_git_sha  describes the REPO
    #   singlechain_sha256 describes the module that was actually IMPORTED
    # They differ whenever src/ was edited but not redeployed -- the fork is pip-installed as a
    # COPY, so that failure is silent (see BayHunter_Aniso/check_deploy.py). The git sha alone
    # would happily certify a run that used stale code; the module sha256 is what catches it.
    d.update(_provenance(initparams, opt, float(cfg.get("outlier_delta", vr_DELTA_LOGL))))

    # optional full-ensemble dump for per-well QC (Vs density, interface prob, misfit hist)
    if cfg.get("save_ensemble"):
        ifaces, nlay_each = [], []
        for m, vp in zip(models, vpvs):
            _vp, vsm, h, _, _, _g = _vp_vs_h(m, obj, float(vp))
            h = np.asarray(h, float)
            if h.size > 1:
                zc = np.cumsum(h[:-1])          # interface depths (exclude half-space base)
                ifaces.append(zc[zc <= depth_max])
            nlay_each.append(len(np.asarray(vsm)))
        try:
            misf, = obj._get_posterior_data(["misfits"], final=True)
            misf = np.asarray(misf, float).ravel()
        except Exception:
            misf = np.array([])
        d["ens_vs"] = prof.astype(np.float32)                     # (nmodel, ndepth)
        if RADIAL and len(gprof):
            d["ens_gamma"] = gprof.astype(np.float32)             # (nmodel, ndepth) gamma(z)
        d["iface_depths"] = (np.concatenate(ifaces) if ifaces else np.array([])).astype(np.float32)
        d["ens_misfit"] = misf
        d["ens_nlayers"] = np.asarray(nlay_each)
    np.savez_compressed(cfg["out_npz"], **d)
    print(f"BayHunter cell done: {len(prof)} posterior models, {runtime:.0f}s -> {cfg['out_npz']}")

    # storage: the raw per-chain trace dir is the big cost (~10-40 MB/cell); the compact
    # per-chain summary + reliability are now in the npz, so drop it unless kept for debugging.
    if not cfg.get("keep_chain_files", False):
        import shutil
        shutil.rmtree(os.path.join(savepath, "data"), ignore_errors=True)
        print(f"  removed raw chain dir {os.path.join(savepath, 'data')} "
              f"(set keep_chain_files=True to retain)", flush=True)


def _prune_incomplete_chains(savepath):
    """Delete any chain's files if its per-chain (phase, type) file set is incomplete.

    BayHunter's SingleChain.save_finalmodels() wraps each phase's np.save loop in a bare
    `except:` -- if writing ANY of [models,likes,misfits,noise,vpvs,temperatures] for a phase
    raises (seen intermittently under tight LVZ/HVZ + joint fund+overtone targets), earlier
    files in that loop are already on disk but later ones are silently missing. This leaves a
    per-filetype file-count mismatch that PlotFromStorage.init_filelists() detects but does not
    raise on -- it just skips setting self.likefiles/self.modfiles/etc, which then crashes
    get_outliers() with an opaque AttributeError. Drop the offending chain's remaining files
    entirely (excluding it from the posterior) rather than losing the whole run.
    """
    import glob
    import re
    datapath = os.path.join(savepath, "data")
    types = ("models", "likes", "misfits", "noise", "vpvs", "temperatures")
    expected = {(p, t) for p in "12" for t in types}
    chains = {}
    for f in glob.glob(os.path.join(datapath, "c*_p[12]*.npy")):
        m = re.match(r"c(\d+)_p([12])(\w+)\.npy$", os.path.basename(f))
        if not m:
            continue
        chains.setdefault(m.group(1), set()).add((m.group(2), m.group(3)))
    for cidx, have in chains.items():
        if have != expected:
            for p, t in have:
                fp = os.path.join(datapath, f"c{cidx}_p{p}{t}.npy")
                if os.path.exists(fp):
                    os.remove(fp)
            print(f"  dropped incomplete chain c{cidx} ({len(have)}/{len(expected)} files "
                  f"-- likely a partial write from BayHunter's save_finalmodels())", flush=True)


def _gamma_step(gamma, h, dep):
    """Per-layer gamma -> step profile on the depth grid (layer bottoms from cumsum(h);
    the halfspace has h=0, so depths beyond the last interface map to the last layer)."""
    bottoms = np.cumsum(h)
    idx = np.clip(np.searchsorted(bottoms, dep, side="right"), 0, len(gamma) - 1)
    return gamma[idx]


def _vp_vs_h(model, obj, vpvs):
    """Return vp, vs, h, c1, c2, gamma for one posterior model (AzAniso Model.get_vp_vs_h).
    gamma = the raw psi2amp block: azimuthal amplitude in azimuthal runs, SIGNED radial
    gamma=(Vsh-Vsv)/Vsv in radial_anisotropy runs, zeros in isotropic runs."""
    out = Model.get_vp_vs_h(model, vpvs, obj.mantle)
    if len(out) == 5:
        vp, vs, h, psi2amp, psi2azi = out
        c1 = psi2amp * np.cos(2 * psi2azi); c2 = psi2amp * np.sin(2 * psi2azi)
        gamma = np.asarray(psi2amp, float)
    else:
        vp, vs, h = out[:3]; c1 = c2 = gamma = np.zeros_like(vs)
    return vp, vs, h, c1, c2, gamma


def _layer_counts(models):
    n = []
    for m in models:
        mm = np.asarray(m, float)
        n.append(int(np.sum(np.isfinite(mm)) // 4))    # 4 params/layer (AzAniso)
    return np.array(n)


if __name__ == "__main__":
    main(sys.argv[1])
