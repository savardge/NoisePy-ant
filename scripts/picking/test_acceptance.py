#!/usr/bin/env python
"""Detailed-balance tests for BayHunter's trans-dimensional acceptance ratio.

Run with the bayhunter env:
    /opt/anaconda3/envs/bayhunter/bin/python test_acceptance.py            # all tiers
    /opt/anaconda3/envs/bayhunter/bin/python test_acceptance.py --tier 1   # just the no-op

WHY THIS EXISTS
---------------
`SingleChain.get_acceptance_probability` adds a layer-count term `D` for trans-dimensional
birth/death when anisotropy is on (`len(self.anisomods) > 0`, SingleChain.py:93-98), citing
"E10 in Bodin et al. 2016". Two things are wrong with it as written, and one thing is unclear:

  1. `D` is a RAW RATIO added into a LOG-space acceptance (`alpha = log(A) + B + C/T + D`).
  2. The death branch is off by one: E11 gives k/(k-1); the code has (k+1)/k. `k` is the PRE-move
     layer count (get_acceptance_probability is called at :804, before accept at :808).
  3. Unclear -- and the reason for TIER 2: Bodin 2016 proposes interface positions on a DISCRETE
     grid (q(z'|m) = 1/(N-k), eq. E3), and its k/(k+1) survives from that prior ratio. BayHunter
     draws z CONTINUOUSLY (`_model_layerbirth:339`, uniform(zmin, zmax)) -- the Bodin 2012
     parameterization -- and vanilla BayHunter has NO `D` term at all. So it is not clear from
     reading whether `D` belongs here in any form.

Reading cannot settle (3); this test can.

VERDICT AS OF 2026-07-16 -- THE FIX IS NOT A ONE-LINER. DO NOT PATCH `D` AND CALL IT DONE.
-------------------------------------------------------------------------------------------
Measured (400k main iters, theta=1.5, layers=(1,6), azimuthal anisotropy ON; TVD = total
variation distance of the sampled k-distribution from the uniform prior; control floor 0.0109
over 3 seeds, PASS threshold 0.0326):

    isotropic control (D dead)  TVD 0.005-0.011   PASS  <- harness is sound; reference sampler
    (a) current  raw ratio      TVD 0.187         FAIL  biased LOW  k (0.271 -> 0.070)
    (b) E10/E11  logged         TVD 0.048         FAIL  biased low  k (0.200 -> 0.142)
    (c) no D                    TVD 0.142         FAIL  biased HIGH k (0.092 -> 0.238)
    (d) derived log((k+1)/(k+1-l))                TVD 0.433  FAIL  biased HIGH k -- sign wrong

NONE of them recovers the prior. So the defect is NOT confined to the `D` term, and no choice of
`D` alone repairs it. Two structural causes, both measured:

  * ASYMMETRIC ELIGIBILITY. _model_layerbirth:350 always inserts an ISOTROPIC cell (psi2amp = 0);
    _model_layerdeath:365 may only remove one of the (k-l) ISOTROPIC cells. The reverse of a birth
    is therefore a 1-of-(k+1-l) choice, not the 1-of-(k+1) that Bodin 2012's cancellation assumes
    -- which is why vanilla (isotropic, no D) is exact but the anisotropic path is not.
  * THE ANISOTROPY MOVES ARE THEMSELVES UNBALANCED. aniso_birth/aniso_death take the plain
    likelihood-ratio branch (SingleChain.py:670), so under a flat likelihood alpha = 0 and they
    are ALWAYS accepted: l performs an unweighted random walk instead of sampling its prior.
    Measured l/k = 0.50 at EVERY k (0.481, 0.500, 0.507, 0.488, 0.497, 0.521 for k = 2..7), i.e.
    the configuration space grows with k, and with D = 0 the sampler targets P(k) ~ 1.2^k rather
    than the uniform p(k) the `layers` prior states. Bodin 2016 E12-E17 does reduce the
    anisotropy move to a bare likelihood ratio -- but only because THEIR prior carries the
    matching C(k,l) combinatorial term. Whether this fork's prior does is exactly the open
    question, and it cannot be answered from the code.

CONSEQUENCE. Every anisotropic/radial run's layer-count AND anisotropic-occupancy posterior is
biased, and gamma is spike-and-slab, so occupancy is precisely the quantity of interest. Isotropic
runs are untouched -- proven by TIER 1, not assumed.

WHAT IS NEEDED. A derivation of the anisotropic trans-dimensional acceptance for THIS fork's
parameterization (continuous z, Gaussian Vs birth proposal, isotropic-only layer death, psi2amp ~
U(0, 0.1) spike-and-slab), i.e. a decision about what prior over (k, l, config) the fork intends.
That is an author-level question, not a code-reading one. This test is the oracle that will settle
it: add the candidate to `_D` (and, if the aniso moves need one, to the 'aniso' branch of
`_patch_acceptance`) and require TVD within the control floor.

THE TIERS
---------
TIER 1 -- isotropic no-op. With `anisomods = []` the `D` branch is dead (D = 0 = log 1), so any
    change to `D` MUST leave isotropic runs bitwise identical. Compares the live module against
    the pre-audit baseline (`SingleChain.py.pre-audit.bak`) over randomized inputs. This is the
    regression check that protects every accepted isotropic production result.

TIER 2 -- reciprocity. Detailed balance requires the k <-> k+1 pair to be exactly reversible:
    D_birth(k) + D_death(k+1) == 0. Pure arithmetic, no sampling.

TIER 3 -- PRIOR RECOVERY (the oracle that decides the fix). Set the likelihood CONSTANT. The
    posterior then equals the prior by construction, so the sampled distribution over the layer
    count k must reproduce the prior over k. With thickmin=0, no lvz/hvz and no mohoest, every
    proposal inside the box is valid, so the prior over k is UNIFORM and the test is a chi-square
    against a flat histogram. Whichever `D` variant recovers the prior is the correct one:

      (a) current   : D = k/(k+1) raw            (birth), (k+1)/k raw    (death)
      (b) E10/E11   : D = log(k/(k+1))           (birth), log(k/(k-1))   (death)
      (c) none      : D = 0                      -- vanilla/Bodin 2012, the continuous-z form

    This is a statement about the sampler alone: no data, no forward model, no seismology.
"""
import argparse
import importlib.machinery
import importlib.util
import os
import sys

import numpy as np

BH = "/opt/anaconda3/envs/bayhunter/lib/python3.10/site-packages/BayHunter"
BAK = os.path.join(BH, "SingleChain.py.pre-audit.bak")


# ----------------------------------------------------------------------------- tier 1
def _load_baseline():
    """Import the pre-audit SingleChain.py under a private name, as the regression oracle."""
    if not os.path.exists(BAK):
        return None
    # SourceFileLoader is required: the baseline is a .bak, and spec_from_file_location returns
    # None for an unrecognised suffix.
    loader = importlib.machinery.SourceFileLoader("_baseline_singlechain", BAK)
    spec = importlib.util.spec_from_loader("_baseline_singlechain", loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_baseline_singlechain"] = mod
    loader.exec_module(mod)
    return mod


class _Stub:
    """Minimal duck-type carrying exactly the attributes get_acceptance_probability reads."""

    def __init__(self, rng, anisomods):
        self.anisomods = anisomods
        # the continuous-zeta refactor gates D on initparams['azimuthal_anisotropy'] instead of
        # len(anisomods); the stub carries both so it can drive the pre- and post-refactor code
        self.initparams = {"azimuthal_anisotropy": bool(anisomods)}
        self.radial = False
        self.fixedvelmodel = False
        self.propdist = np.array([0.05, 0.3, rng.uniform(0.01, 0.5), 0.005, 0.01])
        self.dv = rng.uniform(0.5, 4.0)
        self.dvs2 = rng.uniform(0.0, 0.5)
        self.temperature = rng.choice([1.0, 1.37, 2.0, 7.3])
        self.currentlikelihood = rng.uniform(-200, 200)
        k = rng.integers(2, 25)
        self.currentmodel = np.zeros(4 * k)

        class _T:
            pass
        self.targets = _T()
        self.targets.proposallikelihood = self.currentlikelihood + rng.uniform(-50, 50)


def tier1_isotropic_noop(ndraw=100_000, seed=0):
    """Isotropic acceptance must be BITWISE identical to the pre-audit baseline."""
    from BayHunter.SingleChain import SingleChain as Live
    base_mod = _load_baseline()
    print(f"\nTIER 1 -- isotropic no-op ({ndraw:,} randomized draws, bitwise)")
    if base_mod is None:
        print(f"  SKIP: baseline not found at {BAK}")
        return None
    Base = base_mod.SingleChain

    rng = np.random.default_rng(seed)
    bad = 0
    for i in range(ndraw):
        s = _Stub(rng, anisomods=[])                       # ISOTROPIC: D branch dead
        modify = rng.choice(["birth", "death", "vsmod"])
        a = Live.get_acceptance_probability(s, modify)
        b = Base.get_acceptance_probability(s, modify)
        if not (a == b or (np.isnan(a) and np.isnan(b))):  # bitwise, not allclose
            bad += 1
            if bad <= 3:
                print(f"  MISMATCH [{modify}] live={a!r} base={b!r}")
    ok = bad == 0
    print(f"  {'PASS' if ok else 'FAIL'}: {ndraw - bad:,}/{ndraw:,} bitwise identical")
    return ok


# ----------------------------------------------------------------------------- tier 2
def _D(variant, k, move, l=0):
    """The layer-count term added to the log acceptance. k, l are PRE-move counts.

    k = total cells, l = anisotropic cells (psi2amp != 0).

    (d) is derived from BayHunter's ACTUAL moves rather than from a paper:
        birth (_model_layerbirth:350) always inserts an ISOTROPIC cell (psi2amp = 0);
        death (_model_layerdeath:365) selects uniformly among the (k-l) ISOTROPIC cells only.
    So the reverse of a birth at (k,l) is a death at (k+1,l) that must pick the new cell out of
    (k+1-l) candidates -- probability 1/(k+1-l), NOT the 1/(k+1) that Bodin 2012's cancellation
    assumes. The prior's labelling ratio is (k+1) (that is what cancels 1/(k+1) in the l=0 case,
    which is why vanilla BayHunter needs no D at all), leaving

        D_birth(k,l) = log((k+1) / (k+1-l))        D_death(k,l) = log((k-l) / k)

    Reciprocity holds: D_birth(k,l) + D_death(k+1,l) = 0. At l = 0 both are log(1) = 0, so it is a
    strict no-op for isotropic runs -- consistent with the isotropic control recovering the prior
    with no D term.
    """
    if variant == "a":                                     # current code, raw ratio
        return k / (k + 1) if move == "birth" else (k + 1) / k
    if variant == "b":                                     # Bodin 2016 E10/E11, logged
        return np.log(k / (k + 1)) if move == "birth" else np.log(k / (k - 1))
    if variant == "d":                                     # derived for BayHunter's own moves
        if move == "birth":
            return np.log((k + 1) / (k + 1 - l)) if (k + 1 - l) > 0 else 0.0
        return np.log((k - l) / k) if (k - l) > 0 else 0.0
    return 0.0                                             # (c) vanilla / Bodin 2012


def tier2_reciprocity():
    """D_birth(k) + D_death(k+1) must be 0: the k->k+1 birth and its reverse death must cancel."""
    print("\nTIER 2 -- reciprocity  D_birth(k,l) + D_death(k+1,l) == 0")
    res = {}
    for v, name in [("a", "current     "), ("b", "E10/E11     "), ("c", "no D        "),
                    ("d", "derived(k,l)")]:
        r = [_D(v, k, "birth", l) + _D(v, k + 1, "death", l)
             for k in range(2, 31) for l in range(0, k)]
        ok = np.allclose(r, 0.0, atol=1e-12)
        res[v] = ok
        print(f"  {name}: max|sum| = {np.max(np.abs(r)):.3f}   {'PASS' if ok else 'FAIL'}")
    print("  (reciprocity is NECESSARY, not sufficient -- b, c and d all pass it and only one")
    print("   of them recovers the prior. Tier 3 is what discriminates.)")
    return res


# ----------------------------------------------------------------------------- tier 3
def _patch_acceptance(variant):
    """Swap get_acceptance_probability's D term for `variant`, leaving A/B/C untouched."""
    from BayHunter.SingleChain import SingleChain
    if not hasattr(SingleChain, "_orig_get_acceptance_probability"):
        SingleChain._orig_get_acceptance_probability = SingleChain.get_acceptance_probability

    def get_acceptance_probability(self, modify):
        if (modify in ["vsmod", "zvmod", "noise", "vpvs"] or "aniso" in modify
                or self.fixedvelmodel):
            alpha = self.targets.proposallikelihood - self.currentlikelihood
            alpha *= 1. / self.temperature
        elif modify in ("birth", "death"):
            theta = self.propdist[2]
            B = self.dvs2 / (2. * np.square(theta))
            C = self.targets.proposallikelihood - self.currentlikelihood
            k = int(len(self.currentmodel) / 4)
            # l = anisotropic cells, PRE-move. Model layout is concat(vs, z, psi2amp, psi2azi),
            # each length k (Models.split_modelparams), so psi2amp is the third block.
            l = int(np.count_nonzero(self.currentmodel[2 * k:3 * k]))
            D = _D(variant, k, modify, l) if len(self.anisomods) > 0 else 0
            if modify == "birth":
                A = (theta * np.sqrt(2 * np.pi)) / self.dv
                alpha = np.log(A) + B + C * 1. / self.temperature + D
            else:
                A = self.dv / (theta * np.sqrt(2 * np.pi))
                alpha = np.log(A) - B + C * 1. / self.temperature + D
        return alpha

    SingleChain.get_acceptance_probability = get_acceptance_probability


def _flat_likelihood_chain(nmain, seed, layers=(1, 6), aniso="azimuthal", theta=1.5,
                           collect_gamma=False):
    """One chain against a CONSTANT likelihood -> it must sample the PRIOR.

    aniso="azimuthal" activates the spike-and-slab `D` branch while keeping the prior over k
    UNIFORM: psi2amp ~ U(0, 0.1) (_model_azianiso_birth) and _validmodel checks exactly [0, 0.1],
    so every proposal is valid, P(valid | k) = 1, and the analytic uniform is the right reference.

    aniso="radial" runs the CONTINUOUS-gamma sampler (post CONTINUOUS_ZETA_PLAN.md). Here the
    analytic uniform is NOT the reference: _validmodel enforces Vsh = Vs*(1+gamma) in
    [vsmin, vsmax], which couples gamma to Vs and makes P(valid | k) fall with k. tier_radial
    therefore compares against a REJECTION SAMPLE of the same constrained prior instead.
    Pass collect_gamma=True to also get the per-layer gamma samples and the chain object.

    aniso=None is the ISOTROPIC control: `D` is dead, making it the reference Bodin-2012 sampler.

    theta = propdist[2], the birth Vs proposal width. The production default (0.15) is far
    narrower than the Vs prior (dv = 3.0), so birth acceptance ~ A*e^B with A = theta*sqrt(2pi)/dv
    is small and k mixes too slowly to equilibrate in any affordable run (measured: control drift
    0.054 at theta=0.15 vs 0.014 at theta=1.5 over 400k iters). Widening theta approaches Bodin's
    own remark that a Vs proposal equal to the prior gives qv2*dVs = 1. This changes MIXING only,
    never the stationary distribution -- which is the thing under test.
    """
    from BayHunter import Targets, MCMC_Optimizer

    T = np.linspace(0.5, 5.0, 12)
    tgt = Targets.RayleighDispersionGroup(T, np.ones_like(T), yerr=np.full(T.size, 0.01) ** 2)
    tgt.moddata.plugin.set_modelparams(mode=1)
    target = Targets.JointTarget(targets=[tgt])

    # THE test condition: likelihood is constant, so posterior == prior exactly. This also skips
    # the forward model entirely, which is what makes 1e6 iterations cheap.
    def evaluate(**kwargs):
        target.proposallikelihood = 0.0
        target.proposalmisfits = np.array([1.0, 1.0])
    target.evaluate = evaluate

    priors = {"vpvs": 1.73, "layers": layers, "vs": (1.0, 4.0), "z": (0.0, 60.0),
              "mohoest": None, "mantle": None, "rfnoise_corr": 0.9, "swdnoise_corr": 0.0,
              "rfnoise_sigma": (1e-5, 0.1), "swdnoise_sigma": (1e-4, 0.5),
              "swdnoise_sigma_c1": (1e-5, 0.02), "swdnoise_sigma_c2": (1e-5, 0.02),
              "triangular_zprop": False, "radial": (-0.35, 0.35)}
    ip = {"nchains": 1, "iter_burnin": max(2000, nmain // 10), "iter_main": nmain,
          "propdist": (0.05, 2.0, theta, 0.005, 0.01),
          "propfixed": (1, 1, 1, 1, 1),      # FIX the proposals: adaptation would itself perturb
          "acceptance": (40, 48),            #   detailed balance and confound the test
          "thickmin": 0.0, "relative_thickmin": False,   # -> every in-box proposal is valid,
          "lvz": None, "hvz": None,                      #    so the prior over k is UNIFORM
          "rcond": 1e-5, "station": "t", "savepath": f"/tmp/bal_{seed}/", "maxmodels": 100000,
          "parallel_tempering": False, "t1chains": 1, "maxtemp": 2.0,
          "azimuthal_anisotropy": (aniso == "azimuthal"),
          "radial_anisotropy": (aniso == "radial"),
          "radial_overdisperse": False}
    opt = MCMC_Optimizer(target, initparams=ip, priors=priors, random_seed=seed)
    ch = opt.chains[0]

    # Record k at EVERY iteration, straight off currentmodel. Do NOT reconstruct it from
    # chainmodels weighted by diff(chainiter): under a flat likelihood the acceptance rate is very
    # high, which trips the append-skip at SingleChain.py:811-815 (an accepted model updates
    # currentmodel but is NOT appended, so chainiter loses the transition and the weights are
    # silently wrong). That bug would corrupt exactly the statistic under test. Reading
    # currentmodel per iteration is immune to it and IS the stationary trace by definition.
    ktrace = np.empty(ch.iter_phase2, dtype=np.int16)
    gammas = [] if collect_gamma else None
    while ch.iiter < ch.iter_phase2:
        ch.iterate()
        if ch.iiter > 0:
            k = len(ch.currentmodel) // 4
            ktrace[ch.iiter - 1] = k
            # every 20th iteration, snapshot all layers' gamma (the psi2amp block)
            if collect_gamma and ch.iiter % 20 == 0:
                gammas.append(np.array(ch.currentmodel[2 * k:3 * k]))
    if collect_gamma:
        return ktrace, (np.concatenate(gammas) if gammas else np.array([])), ch
    return ktrace


def _khist(kt, lo, hi):
    edges = np.arange(lo, hi + 2) - 0.5
    half = kt.size // 2
    p2, _ = np.histogram(kt[half:], bins=edges)              # 2nd half of the main phase
    p1, _ = np.histogram(kt[:half], bins=edges)
    f2 = p2 / max(1, p2.sum())
    f1 = p1 / max(1, p1.sum())
    tvd = 0.5 * np.abs(f2 - np.full(f2.size, 1.0 / f2.size)).sum()   # distance from uniform
    drift = 0.5 * np.abs(f1 - f2).sum()                              # half-vs-half: converged?
    return f2, tvd, drift


def tier3_prior_recovery(nmain, seed, layers=(1, 6), nctrl=3):
    """Constant likelihood => the sampled k must reproduce the (uniform) prior over k.

    Verdict is by TOTAL VARIATION DISTANCE from uniform, CALIBRATED against the isotropic control
    -- NOT by a chi-square p-value. A chi-square assumes iid samples; MCMC output is strongly
    autocorrelated, so its effective sample size is orders of magnitude below the raw count and
    the p-value is meaninglessly small even for a perfect sampler (measured: the reference
    Bodin-2012 control gives p ~ 1e-18 while sitting within 0.5% of uniform). The control IS the
    reference sampler, so whatever TVD it shows is this test's noise floor; a variant is judged
    against that floor, not against zero.
    """
    lo, hi = layers[0] + 1, layers[1] + 1                    # k = h.size; prior is on h.size-1
    print(f"\nTIER 3 -- prior recovery (flat likelihood, {nmain:,} main iters, k={lo}..{hi})")
    print("  Flat likelihood => posterior == prior. Azimuthal anisotropy ON activates the D")
    print("  branch while keeping the prior over k uniform. Verdict = TVD vs the control floor.")

    # --- calibrate the noise floor on the ISOTROPIC control (D dead => reference Bodin-2012)
    _patch_acceptance("c")
    ctrl = []
    for s in range(nctrl):
        f, tvd, drift = _khist(_flat_likelihood_chain(nmain, seed + s, layers, aniso=None), lo, hi)
        ctrl.append(tvd)
        print(f"  {'ISO control':<12} seed {seed+s}  TVD={tvd:.4f}  drift={drift:.3f}   "
              + " ".join(f"{x:.3f}" for x in f))
    floor = max(ctrl)
    thresh = max(3.0 * floor, 0.02)
    print(f"  -> control TVD floor = {floor:.4f} over {nctrl} seeds; PASS threshold = "
          f"{thresh:.4f} (3x floor)")

    out = {"floor": floor, "thresh": thresh}
    print(f"\n  {'variant':<12} {'TVD':>8} {'drift':>7}  verdict")
    for v, name in [("a", "current"), ("b", "E10/E11"), ("c", "no D"), ("d", "derived(k,l)")]:
        _patch_acceptance(v)
        f, tvd, drift = _khist(
            _flat_likelihood_chain(nmain, seed, layers, aniso="azimuthal"), lo, hi)
        ok = tvd <= thresh and drift < 0.05
        out[v] = (tvd, drift, ok)
        note = ("PASS (recovers prior)" if ok else
                "FAIL (not converged)" if drift >= 0.05 else "FAIL (prior distorted)")
        print(f"  {name:<12} {tvd:8.4f} {drift:7.3f}  {note}")
        print(f"               k dist: " + " ".join(f"{x:.3f}" for x in f)
              + f"   (uniform = {1.0/(hi-lo+1):.3f})")
    return out


def _rejection_prior(ch, layers, nkeep, rng):
    """Rejection sample of the radial prior AS IMPLEMENTED: k ~ U, (vs, z, gamma) ~ U(box),
    keep iff ch._validmodel accepts. This is the exact stationary distribution the flat-likelihood
    sampler must reproduce -- including the Vsh = Vs*(1+gamma) truncation, which makes both the
    k-marginal non-uniform (P(valid|k) falls with k) and the gamma-marginal non-flat. Using the
    chain's own _validmodel means the reference tracks the code, not our reading of it.
    """
    lo, hi = layers[0] + 1, layers[1] + 1
    vsmin, vsmax = ch.priors["vs"]
    zmin, zmax = ch.priors["z"]
    glo, ghi = ch.priors.get("radial", (-0.35, 0.35))
    ks, gs = [], []
    while len(ks) < nkeep:
        k = int(rng.integers(lo, hi + 1))
        vs = rng.uniform(vsmin, vsmax, k)
        z = np.sort(rng.uniform(zmin, zmax, k))
        g = rng.uniform(glo, ghi, k)
        model = np.concatenate((vs, z, g, np.zeros(k)))
        if ch._validmodel(model):
            ks.append(k)
            gs.append(g)
    return np.array(ks), np.concatenate(gs)


def tier_radial(nmain, seed, layers=(1, 6)):
    """Continuous-gamma radial sampler vs a rejection sample of its own prior.

    Flat likelihood => the sampler's stationary distribution IS the prior. Two marginals checked:
    k (layer count; inherits vanilla isotropic birth/death, so this doubles as the regression that
    removing the D term changed nothing) and gamma (aggregated over layers; must match the
    Vs-truncated prior, NOT a flat one). Verdict = TVD vs the rejection reference, floored by the
    TVD between two independent MCMC seeds (the sampler's own noise at this run length).
    """
    # This tier tests the LIVE deployed code -- undo any tier3 monkeypatch, whose old
    # len(anisomods)>0 gate would wrongly re-enable D under continuous radial (anisomods is
    # ['aniso_ampmod'], non-empty).
    from BayHunter.SingleChain import SingleChain
    if hasattr(SingleChain, "_orig_get_acceptance_probability"):
        SingleChain.get_acceptance_probability = SingleChain._orig_get_acceptance_probability

    lo, hi = layers[0] + 1, layers[1] + 1
    kedges = np.arange(lo, hi + 2) - 0.5
    print(f"\nTIER RADIAL -- continuous-gamma prior recovery ({nmain:,} main iters, k={lo}..{hi})")

    runs = []
    for s in (seed, seed + 1):
        kt, gam, ch = _flat_likelihood_chain(nmain, s, layers, aniso="radial",
                                             collect_gamma=True)
        half = kt.size // 2
        runs.append((kt[half:], gam[gam.size // 2:], ch))

    rng = np.random.default_rng(seed)
    kref, gref = _rejection_prior(runs[0][2], layers, nkeep=200_000, rng=rng)

    glo, ghi = runs[0][2].priors.get("radial", (-0.35, 0.35))
    gedges = np.linspace(glo, ghi, 21)

    def h(x, edges):
        c, _ = np.histogram(x, bins=edges)
        return c / max(1, c.sum())

    hk = [h(r[0], kedges) for r in runs]
    hg = [h(r[1], gedges) for r in runs]
    hkr, hgr = h(kref, kedges), h(gref, gedges)

    floor_k = 0.5 * np.abs(hk[0] - hk[1]).sum()      # seed-to-seed = the sampler's own noise
    floor_g = 0.5 * np.abs(hg[0] - hg[1]).sum()
    out = {}
    for name, hs, hr, floor in (("k", hk, hkr, floor_k), ("gamma", hg, hgr, floor_g)):
        tvd = max(0.5 * np.abs(hs[i] - hr).sum() for i in range(2))   # worst of the two seeds
        thresh = max(2.0 * floor, 0.02)
        ok = tvd <= thresh
        out[name] = (tvd, floor, ok)
        print(f"  {name:<6} TVD vs rejection prior = {tvd:.4f}   seed-to-seed floor = {floor:.4f}"
              f"   threshold = {thresh:.4f}   {'PASS' if ok else 'FAIL'}")
    print(f"  k dist    MCMC: " + " ".join(f"{x:.3f}" for x in hk[0]))
    print(f"  k dist    prior: " + " ".join(f"{x:.3f}" for x in hkr)
          + "   (non-uniform BY DESIGN: Vsh truncation lowers P(valid|k) as k grows)")
    print(f"  gamma     MCMC p16/p50/p84: {np.percentile(runs[0][1], [16, 50, 84]).round(3)}")
    print(f"  gamma     prior p16/p50/p84: {np.percentile(gref, [16, 50, 84]).round(3)}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", type=int, choices=[1, 2, 3, 4], default=None,
                    help="4 = continuous-gamma radial prior recovery (tier_radial)")
    ap.add_argument("--ndraw", type=int, default=100_000)
    ap.add_argument("--nmain", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if a.tier in (None, 1):
        tier1_isotropic_noop(a.ndraw, a.seed)
    if a.tier in (None, 2):
        tier2_reciprocity()
    if a.tier in (None, 3):
        tier3_prior_recovery(a.nmain, a.seed)
    if a.tier in (None, 4):
        tier_radial(a.nmain, a.seed)


if __name__ == "__main__":
    main()
