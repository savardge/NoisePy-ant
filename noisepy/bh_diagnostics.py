"""Convergence diagnostics for a BayHunter run: burn-in trace, post-burnin log-likelihood
distribution, trans-D layer-count trace, and inter-chain Vs(z) agreement.

Operates directly on the raw per-chain files BayHunter's SingleChain.save_finalmodels() leaves
in <savepath>/data/ (c###_p1<type>.npy = burn-in / phase 1, c###_p2<type>.npy = main / phase 2).
Those arrays are the FULL per-iteration trace (BayHunter repeats each accepted model's value by
its iteration weight before thinning -- see Models.get_weightedvalues), only lightly thinned for
disk size, so they are a faithful trace/history plot, not just a final-sample summary.

Self-contained (numpy + matplotlib + BayHunter.Model only) so it runs in either the bayhunter
env (inline, from run_bayhunter_cell.py) or retroactively against an already-finished run's
still-on-disk chain files (scripts/picking/regen_bh_diagnostics.py).
"""
import glob
import os
import re

import numpy as np

from .vs_reliability import DELTA_LOGL      # pure numpy; keeps the basin threshold in one place


def find_chains(datapath):
    """Chain indices with a phase-2 (main) likelihood file -- i.e. survived
    run_bayhunter_cell._prune_incomplete_chains."""
    idxs = set()
    for f in glob.glob(os.path.join(datapath, "c*_p2likes.npy")):
        m = re.match(r"c(\d+)_p2likes\.npy$", os.path.basename(f))
        if m:
            idxs.add(m.group(1))
    return sorted(idxs)


def _load(datapath, cidx, name):
    fp = os.path.join(datapath, f"c{cidx}_{name}.npy")
    return np.load(fp, allow_pickle=True) if os.path.exists(fp) else None


def median_at_t1(likes, temps):
    """Median log-likelihood over a chain's T=1 samples only.

    Under parallel tempering a chain's phase-2 likes mix every temperature it held (temperatures
    SWAP between chains), and a sample drawn at T>1 comes from a FLATTENED posterior, so it fits
    worse BY CONSTRUCTION. Only T=1 samples enter the posterior (Plotting.py filters
    `alltemps==1`), so any per-chain statistic must be computed on those alone -- otherwise a
    chain is penalised for having spent time hot. Because the median is robust this only bites
    once a chain is hot for >50% of phase 2, which the t1chains=nchains/2 ladder makes the
    EXPECTED case (~50% hot on average). Measured on Riehen-2 both, cell 18_20: the raw medians
    report 11/16 chains in basin with a 25.0 logL spread, where the identical non-PT run gives
    16/16 and 3.0 -- a pure artifact.

    STRICT NO-OP when tempering is off: BayHunter fills the temperature array with 1.0
    (mcmcOptimizer._init_parallel_tempering else-branch), so the mask keeps every sample. Also a
    no-op when temps is None (older runs predate the temperatures file).
    Mirrors run_bayhunter_cell._chain_median_at_t1, which does this from the file path.
    """
    # NB do NOT cast to float64: BayHunter stores these as float32, and for an even-length trace
    # the median is the mean of the two middle samples, so a widening cast shifts it by ~1e-6 and
    # breaks exact parity with the pre-PT values.
    likes = np.asarray(likes)
    if temps is None:
        return np.nanmedian(likes)
    temps = np.asarray(temps)
    if temps.shape != likes.shape:
        return np.nanmedian(likes)
    m = temps == 1.0
    if not m.any():
        return -np.inf          # never sampled cold -> a genuine outlier, not a silent nan
    return np.nanmedian(likes[m])


def chain_medians_t1(diag):
    """Per-chain phase-2 median log-likelihood, temperature-aware (see median_at_t1)."""
    temps = diag.get("temps_p2") or [None] * len(diag["likes_p2"])
    return np.array([median_at_t1(l, t) for l, t in zip(diag["likes_p2"], temps)])


def n_layers_trace(models_p2):
    """AzAniso model vector: 4 params/layer (vs, z, psi2amp, psi2azi), nan-padded to max size."""
    return np.array([int(np.sum(np.isfinite(np.asarray(m, float))) // 4) for m in models_p2])


def _chain_prof_ensemble(models_p2, vpvs_p2, depth, mantle, subsample=400, seed=0):
    """(nsub, ndepth) Vs(z) for a subsample of this chain's phase-2 (main) posterior models."""
    from BayHunter import Model
    n = len(models_p2)
    idx = (np.arange(n) if n <= subsample
           else np.random.default_rng(seed).choice(n, subsample, replace=False))
    prof = []
    for i in idx:
        try:
            out = Model.get_interpmodel(models_p2[i], depth, vpvs=float(vpvs_p2[i]), mantle=mantle)
            vs = out[1] if isinstance(out, (tuple, list)) else out
            prof.append(np.asarray(vs, float))
        except Exception:
            continue
    return np.array(prof) if prof else np.full((0, len(depth)), np.nan)


def chain_vs_profile(models_p2, vpvs_p2, depth, mantle, subsample=250, seed=0):
    """Median Vs(z) from a subsample of this chain's phase-2 (main) posterior models."""
    prof = _chain_prof_ensemble(models_p2, vpvs_p2, depth, mantle, subsample, seed)
    return np.nanmedian(prof, axis=0) if len(prof) else np.full(len(depth), np.nan)


def chain_vs_summary(models_p2, vpvs_p2, depth, mantle, subsample=400, seed=0):
    """Per-chain (p16, p50, p84) Vs(z) -- the compact within-chain posterior width + median
    that noisepy.vs_reliability needs (a few KB/chain, so raw traces can be dropped after)."""
    prof = _chain_prof_ensemble(models_p2, vpvs_p2, depth, mantle, subsample, seed)
    if not len(prof):
        return np.full((3, len(depth)), np.nan)
    return np.nanpercentile(prof, [16, 50, 84], axis=0)


def build_diagnostics(datapath, depth, mantle):
    """Assemble per-chain diagnostic traces/profiles into a dict for plot_diagnostics()."""
    d = dict(chain_idx=[], likes_p1=[], likes_p2=[], temps_p2=[], nlayers_p2=[], vs_profile=[],
             vs_p16=[], vs_p84=[], depth=depth)
    for cidx in find_chains(datapath):
        l2 = _load(datapath, cidx, "p2likes")
        if l2 is None:
            continue
        models_p2 = _load(datapath, cidx, "p2models")
        vpvs_p2 = _load(datapath, cidx, "p2vpvs")
        d["chain_idx"].append(int(cidx))
        d["likes_p1"].append(_load(datapath, cidx, "p1likes"))
        d["likes_p2"].append(l2)
        # per-sample temperature (all 1.0 without tempering; absent in pre-PT runs) -- needed so
        # per-chain likelihood statistics count only T=1 samples. See median_at_t1.
        d["temps_p2"].append(_load(datapath, cidx, "p2temperatures"))
        d["nlayers_p2"].append(n_layers_trace(models_p2) if models_p2 is not None else None)
        if models_p2 is not None:
            p16, p50, p84 = chain_vs_summary(models_p2, vpvs_p2, depth, mantle)
        else:
            p16 = p50 = p84 = np.full(len(depth), np.nan)
        d["vs_profile"].append(p50)          # median (back-compat with plot_diagnostics)
        d["vs_p16"].append(p16)
        d["vs_p84"].append(p84)
    return d


def convergence_summary(diag):
    """Scalar convergence metrics.

    chain_disagree     : max over depth of the inter-chain std of each chain's median Vs(z)
                         [km/s] -- a Gelman-Rubin-style proxy on the quantity that actually
                         matters (the inverted profile), not just the likelihood.
    frac_chains_ok      : fraction of chains whose post-burnin median log-likelihood is within
                         DELTA_LOGL (absolute) of the best chain -- i.e. sampling the same
                         basin. Temperature-aware (T=1 samples only, see median_at_t1).
                         NB this used to reuse BayHunter's own |1 - med/best| <= 0.05 rule; that
                         is RELATIVE, so its tolerance in real log units is 0.05*|best| and its
                         strictness depends entirely on the likelihood scale (across runs of
                         identical construction here, best spans -34..+97 -> 0.11..4.84 logL).
                         A likelihood RATIO is the meaningful quantity; a relative deviation is
                         not. Recomputable for any run from the saved chain_loglike_med.
    burnin_delta_frac   : |mean(last 10%% of burn-in) - mean(first 10%% of main phase)| relative
                         to the main-phase std, averaged over chains -- near 0 means the chain
                         had already stabilized by the end of burn-in (no visible jump into the
                         main phase).
    """
    vsprof = np.array(diag["vs_profile"])
    chain_disagree = float(np.nanmax(np.nanstd(vsprof, axis=0))) if len(vsprof) else np.nan
    meds = chain_medians_t1(diag) if len(diag["likes_p2"]) else np.array([])
    best = np.nanmax(meds) if len(meds) else np.nan
    frac_ok = float(np.mean((best - meds) <= DELTA_LOGL)) if np.isfinite(best) else np.nan
    deltas = []
    for l1, l2 in zip(diag["likes_p1"], diag["likes_p2"]):
        if l1 is None or len(l1) < 10 or len(l2) < 10:
            continue
        tail = np.mean(l1[-max(1, len(l1) // 10):])
        head = np.mean(l2[:max(1, len(l2) // 10)])
        s = np.std(l2)
        if s > 0:
            deltas.append(abs(tail - head) / s)
    burnin_delta_frac = float(np.mean(deltas)) if deltas else np.nan
    return dict(chain_disagree=chain_disagree, frac_chains_ok=frac_ok,
               burnin_delta_frac=burnin_delta_frac, n_chains=len(diag["chain_idx"]))


def plot_diagnostics(diag, path, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10")
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))

    # (0,0) log-likelihood trace, burn-in (dim) -> main (bold), vertical divider
    ax = axs[0, 0]
    div = None
    for i, cidx in enumerate(diag["chain_idx"]):
        c = cmap(i % 10)
        l1, l2 = diag["likes_p1"][i], diag["likes_p2"][i]
        if l1 is not None:
            ax.plot(np.arange(len(l1)), l1, color=c, lw=0.5, alpha=0.35)
            div = len(l1)
        x2 = np.arange(len(l1) if l1 is not None else 0,
                       (len(l1) if l1 is not None else 0) + len(l2))
        ax.plot(x2, l2, color=c, lw=0.7, alpha=0.85, label=f"c{cidx}")
    if div is not None:
        ax.axvline(div, color="k", ls="--", lw=1)
    ax.set(xlabel="iteration (thinned)", ylabel="log-likelihood",
           title="burn-in (dim) -> main phase (bold) trace, all chains")
    ax.legend(fontsize=6, ncol=2)

    # (0,1) trans-D layer-count trace, main phase
    ax = axs[0, 1]
    for i, cidx in enumerate(diag["chain_idx"]):
        nl = diag["nlayers_p2"][i]
        if nl is None:
            continue
        ax.plot(nl, color=cmap(i % 10), lw=0.6, alpha=0.7)
    ax.set(xlabel="iteration (thinned, main phase)", ylabel="n layers",
           title="trans-D model dimension trace (main phase)")

    # (1,0) post-burnin log-likelihood distribution per chain
    ax = axs[1, 0]
    for i, cidx in enumerate(diag["chain_idx"]):
        ax.hist(diag["likes_p2"][i], bins=40, histtype="step", color=cmap(i % 10), density=True)
    ax.set(xlabel="log-likelihood", ylabel="density",
           title="post-burnin log-likelihood distribution, per chain")

    # (1,1) inter-chain Vs(z) agreement
    ax = axs[1, 1]
    dep = diag["depth"]
    for i, cidx in enumerate(diag["chain_idx"]):
        ax.plot(diag["vs_profile"][i], dep, color=cmap(i % 10), lw=1.2, alpha=0.8, label=f"c{cidx}")
    ax.invert_yaxis()
    summ = convergence_summary(diag)
    ax.set(xlabel="Vs [km/s]", ylabel="depth [km]",
           title=f"per-chain median Vs(z) -- max inter-chain std {summ['chain_disagree']:.3f} km/s")
    ax.legend(fontsize=6, ncol=2)

    if title:
        fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return summ
