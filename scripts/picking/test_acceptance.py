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
def _D(variant, k, move):
    if variant == "a":                                     # current code, raw ratio
        return k / (k + 1) if move == "birth" else (k + 1) / k
    if variant == "b":                                     # Bodin 2016 E10/E11, logged
        return np.log(k / (k + 1)) if move == "birth" else np.log(k / (k - 1))
    return 0.0                                             # (c) vanilla / Bodin 2012


def tier2_reciprocity():
    """D_birth(k) + D_death(k+1) must be 0: the k->k+1 birth and its reverse death must cancel."""
    print("\nTIER 2 -- reciprocity  D_birth(k) + D_death(k+1) == 0")
    res = {}
    for v, name in [("a", "current  "), ("b", "E10/E11  "), ("c", "no D     ")]:
        r = [_D(v, k, "birth") + _D(v, k + 1, "death") for k in range(2, 31)]
        ok = np.allclose(r, 0.0, atol=1e-12)
        res[v] = ok
        print(f"  {name}: max|sum| = {np.max(np.abs(r)):.3f}   {'PASS' if ok else 'FAIL'}")
    return res


# ----------------------------------------------------------------------------- tier 3
def _patch_acceptance(variant):
    """Swap get_acceptance_probability's D term for `variant`, leaving A/B/C untouched."""
    from BayHunter.SingleChain import SingleChain

    def get_acceptance_probability(self, modify):
        if (modify in ["vsmod", "zvmod", "noise", "vpvs"] or "aniso" in modify
                or self.fixedvelmodel):
            alpha = self.targets.proposallikelihood - self.currentlikelihood
            alpha *= 1. / self.temperature
        elif modify == "birth":
            theta = self.propdist[2]
            A = (theta * np.sqrt(2 * np.pi)) / self.dv
            B = self.dvs2 / (2. * np.square(theta))
            C = self.targets.proposallikelihood - self.currentlikelihood
            k = len(self.currentmodel) / 4
            D = _D(variant, k, "birth") if len(self.anisomods) > 0 else 0
            alpha = np.log(A) + B + C * 1. / self.temperature + D
        elif modify == "death":
            theta = self.propdist[2]
            A = self.dv / (theta * np.sqrt(2 * np.pi))
            B = self.dvs2 / (2. * np.square(theta))
            C = self.targets.proposallikelihood - self.currentlikelihood
            k = len(self.currentmodel) / 4
            D = _D(variant, k, "death") if len(self.anisomods) > 0 else 0
            alpha = np.log(A) - B + C * 1. / self.temperature + D
        return alpha

    SingleChain.get_acceptance_probability = get_acceptance_probability


def _flat_likelihood_chain(nmain, seed, layers=(1, 10), radial=True):
    """One chain against a CONSTANT likelihood -> it must sample the PRIOR."""
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
          "propdist": (0.05, 0.3, 0.15, 0.005, 0.01),
          "propfixed": (1, 1, 1, 1, 1),      # FIX the proposals: adaptation would itself perturb
          "acceptance": (40, 48),            #   detailed balance and confound the test
          "thickmin": 0.0, "relative_thickmin": False,   # -> every in-box proposal is valid,
          "lvz": None, "hvz": None,                      #    so the prior over k is UNIFORM
          "rcond": 1e-5, "station": "t", "savepath": f"/tmp/bal_{seed}/", "maxmodels": 100000,
          "parallel_tempering": False, "t1chains": 1, "maxtemp": 2.0,
          "azimuthal_anisotropy": False, "radial_anisotropy": radial,
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
    while ch.iiter < ch.iter_phase2:
        ch.iterate()
        if ch.iiter > 0:
            ktrace[ch.iiter - 1] = len(ch.currentmodel) // 4
    return ktrace


def tier3_prior_recovery(nmain, seed, layers=(1, 10)):
    """Constant likelihood => sampled k must reproduce the (uniform) prior over k."""
    from scipy import stats
    print(f"\nTIER 3 -- prior recovery (flat likelihood, anisotropy ON, {nmain:,} main iters)")
    print("  posterior == prior by construction; prior over k is UNIFORM here")
    lo, hi = layers[0] + 1, layers[1] + 1                    # k = h.size; prior is on h.size-1
    edges = np.arange(lo, hi + 2) - 0.5
    print(f"  {'variant':<12} {'chi2':>9} {'p':>9}  {'drift':>7}  verdict")
    out = {}
    # The isotropic control runs first: D is dead for every variant there, so it IS the reference
    # Bodin-2012 sampler. If the control does not recover the prior, the TEST is wrong, not the
    # code -- report that rather than reading anything into the variants.
    for v, name, radial in [("c", "ISO control", False), ("a", "current", True),
                            ("b", "E10/E11", True), ("c", "no D", True)]:
        _patch_acceptance(v)
        kt = _flat_likelihood_chain(nmain, seed, layers=layers, radial=radial)
        half = kt.size // 2
        obs, _ = np.histogram(kt[half:], bins=edges)         # 2nd half of main phase only
        exp = np.full(obs.size, obs.sum() / obs.size)
        chi2, p = stats.chisquare(obs, exp)
        # convergence guard: if the k-distribution differs between the two halves of the main
        # phase, the chain has not equilibrated and the chi-square is meaningless either way
        o1, _ = np.histogram(kt[:half], bins=edges)
        drift = 0.5 * np.abs(o1 / max(1, o1.sum()) - obs / max(1, obs.sum())).sum()
        ok = p > 0.01 and drift < 0.05
        out[v if radial else "control"] = (chi2, p, drift, ok)
        note = "PASS (recovers prior)" if ok else (
            "FAIL (not converged)" if drift >= 0.05 else "FAIL (prior distorted)")
        print(f"  {name:<12} {chi2:9.1f} {p:9.3g}  {drift:7.3f}  {note}")
        print(f"               k dist: " + " ".join(f"{f:.3f}" for f in obs / obs.sum()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", type=int, choices=[1, 2, 3], default=None)
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


if __name__ == "__main__":
    main()
