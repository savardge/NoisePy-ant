# Trans-dimensional acceptance and parallel tempering — audit and fix

An audit of the rj-McMC sampler used for the 1D Vs inversion (`BayHunter_Aniso`, driven by
`run_bayhunter_cell.py`), benchmarked against an independent reference implementation. Two defects
were found. Both are silent — they produce plausible numbers, not errors. **The parallel-tempering
(PT) defects are fixed. The trans-dimensional acceptance defect is NOT fixed: it is worse than it
first appeared, and no one-line correction repairs it — see §3.** The single most important
statement for a reader of existing results: **isotropic runs are unaffected, and that is proven
(§2), not assumed; anisotropic (radial γ) layer-count and occupancy posteriors are biased.**

Reference implementation: Jan Dettmer's Fortran/MPI trans-D rj-McMC,
`~/Codes/receiver_rjmcmc_varpar_sourceinv_joint` — an independent code of the same class, used here
only as a cross-check on the PT design. Literature: Bodin et al. (2012, *JGR* 117, B02301) and
Bodin et al. (2016, *GJI* 206, 605–629, doi:10.1093/gji/ggw124, App. E).

| file | role | changed? |
|---|---|---|
| `BayHunter_Aniso/src/SingleChain.py:676-702` | trans-D acceptance ratio (`A`,`B`,`C`,`D`) | **no** — see §3 |
| `BayHunter_Aniso/src/mcmcOptimizer.py:184-237` | temperature ladder + swap | no — swap verified correct (§4) |
| `BayHunter_Aniso/deploy.sh`, `check_deploy.py` | close the copy-install trap | **new** (§5) |
| `noisepy/pt_defaults.py` | ladder defaults, single source of truth | **new** (§4) |
| `scripts/picking/run_bayhunter_cell.py` | provenance, T=1 traces, ladder wiring | **yes** (§4–6) |
| `scripts/picking/well_vs_qc.py` | chain-kept rule, T=1 masking, figure stamps | **yes** (§6) |
| `scripts/picking/chaincount_well_study.py` | chain-kept rule | **yes** (§6) |
| `noisepy/vs_reliability.py` | `confidence` semantics under PT | **yes** (§6) |
| `scripts/picking/test_acceptance.py` | the detailed-balance oracle | **new** (§2, §3) |

---

## 1. The acceptance ratio

For a trans-dimensional birth (k → k+1 layers) the code computes, in log space
(`SingleChain.get_acceptance_probability`, `SingleChain.py:676-702`):

```
alpha = log(A) + B + C/T + D        A = (theta*sqrt(2pi))/dv     B = dvs2/(2*theta^2)
```
with `C` the log-likelihood ratio, `T` the temperature, `dv = vsmax - vsmin` (`:42`), `theta` the
Gaussian Vs birth-proposal width, and `dvs2 = (v'_{k+1} - v_i)^2`. Death mirrors it with
`A = dv/(theta*sqrt(2pi))` and `-B`. `D` is added only when `len(self.anisomods) > 0` (`:93-98`),
i.e. for azimuthal or radial runs; it is `0` for isotropic ones.

Bodin et al. (2016) App. E gives, for their algorithm:

| eq. | move | acceptance |
|---|---|---|
| E10 | isotropic birth, k→k+1 | `min[1, (1/(q_v2·ΔVs)) · k/(k+1) · L'/L]` |
| E11 | isotropic death, k→k−1 | `min[1, (q_v2·ΔVs) · k/(k−1) · L'/L]` |
| E17 | anisotropic birth/death | `min[1, L'/L]` — **likelihood ratio only** |

**`A` and `B` are correct.** They are exactly E10/E11's `1/(q_v2·ΔVs)` and `q_v2·ΔVs`, verified
numerically to 1e-16 (`log(1/(q_v2·ΔVs)) = -0.482258848457483` vs `log(A)+B =
-0.48225884845748324` at θ=0.1, dv=3.0, dvs2=0.04).

**`D` is wrong in two ways.** It is added as a **raw ratio into a log-space acceptance** — birth
`D = k/(k+1)` (`:683`), death `D = (k+1)/k` (`:696`) — where it must be `log(...)`. And the death
branch is **off by one**: `k` is the pre-move count (`get_acceptance_probability` is called at
`:804`, before `accept_as_currentmodel` at `:808`), so E11 requires `k/(k−1)`, not `(k+1)/k`.
Detailed balance requires the birth/death pair to be exactly reversible:

| `D_birth(k) + D_death(k+1)` | value |
|---|---|
| Bodin E10/E11 | **0.000** ∀k (as required) |
| code as shipped | **+2.000** ∀k |

**Two corrections to our own earlier audit**, recorded because they were wrong and a reviewer will
re-derive them:
- We hypothesised the count should be `n_isotropic`, not total `k`. **Wrong**: in E7×E9 the
  `(k+1−l)` and `(k−l+1)` factors cancel, so the anisotropic count `l` drops out of E10 entirely.
- We hypothesised `D` belonged on the *anisotropy* birth/death move instead. **Wrong**: E17 shows
  that move reduces to the bare likelihood ratio, which is exactly what `SingleChain.py:670`
  already does. That branch is correct as written.

**Effect size, stated honestly.** `D` is inflated by ≈+1.0 in log-alpha on both branches (D≈0.9
and 1.1 where log D≈∓0.1). Since `u = log(uniform) ≤ 0`, moves already at `alpha > 0` are accepted
either way, so **e¹≈2.7× is an upper bound at the acceptance boundary, not the effect size.**

## 2. Isotropic no-op — proof

`D` is gated on `len(self.anisomods) > 0`, and `anisomods = []` for isotropic runs
(`SingleChain.py:93-98`), so `D = 0 = log 1` and the branch is dead. This is asserted, not assumed:
`test_acceptance.py --tier 1` calls `get_acceptance_probability` for birth/death/vsmod over
**100,000 randomized draws** (propdist, dv, dvs2, model size, likelihoods, temperature) against the
pre-audit baseline imported from `SingleChain.py.pre-audit.bak`, and requires **bitwise** equality.

> **PASS: 100,000/100,000 bitwise identical** (2026-07-16)

This is stronger evidence than any single-seed full run, and it is why every accepted isotropic
production result stands. (A full-run bitwise check is not possible anyway: `SingleChain.py:811`
draws from the **global** `np.random`, not `self.rstate`.)

## 3. Detailed balance — why the fix is NOT a one-liner

**The oracle.** Set the likelihood **constant**. The posterior then equals the prior by
construction, so the sampled distribution over the layer count `k` must reproduce the prior over
`k`. This needs no paper and no data — it is a statement about the sampler alone
(`test_acceptance.py --tier 3`).

Three design points that a reviewer should check, because each was a trap:
- **Verdict is total-variation distance (TVD) from uniform, calibrated on an isotropic control —
  not a chi-square.** A chi-square assumes iid samples; MCMC output is strongly autocorrelated, so
  its effective sample size is orders of magnitude below the raw count. Measured: the reference
  Bodin-2012 control reports **p ~ 1e-18** while sitting within **0.5%** of uniform. The control is
  the reference sampler, so its TVD is the test's noise floor.
- **`k` is read off `currentmodel` per iteration**, never reconstructed from `chainmodels` weighted
  by `diff(chainiter)`: a flat likelihood drives acceptance high, which trips the append-skip at
  `SingleChain.py:811-815` (an accepted model updates `currentmodel` but is not appended, so
  `chainiter` loses the transition) and silently corrupts the very statistic under test.
- **Azimuthal, not radial.** Radial's `_validmodel` requires `Vsh = Vs(1+γ) ∈ [vsmin, vsmax]`,
  coupling γ to Vs, so `P(valid|k)` falls with k and the prior over k stops being uniform — nothing
  clean to test against. Azimuthal draws `psi2amp ~ U(0, 0.1)` (`_model_azianiso_birth`) and
  `_validmodel` checks exactly `[0, 0.1]`, so `P(valid|k) = 1`. `D` is the same code path either way.
- θ = 1.5 (production default is 0.15) affects **mixing only, never the stationary distribution**.
  At 0.15 the control cannot equilibrate at all (half-vs-half drift 0.054 vs 0.014 at 400k iters).

**Result** (400k main iterations, layers=(1,6), azimuthal anisotropy ON, 2026-07-16):

| variant | TVD from uniform | verdict |
|---|---|---|
| isotropic control (`D` dead) | **0.005 – 0.011** (3 seeds) | **PASS** — harness is sound |
| (a) current, raw ratio | 0.187 | FAIL — biased **low** k (0.271→0.070) |
| (b) Bodin E10/E11, logged | 0.048 | FAIL — biased low k; best of four, still 4.4× floor |
| (c) no `D` (vanilla / Bodin 2012) | 0.142 | FAIL — biased **high** k (0.092→0.238) |
| (d) derived `log((k+1)/(k+1−l))` | 0.433 | FAIL — sign wrong |

Control floor 0.0109; PASS threshold 0.0326 (3× floor). **None of them recovers the prior**, so the
defect is not confined to `D` and no choice of `D` alone repairs it. Two structural causes, both
measured:

1. **Asymmetric eligibility.** `_model_layerbirth:350` always inserts an **isotropic** cell
   (`psi2amp = 0`); `_model_layerdeath:365` may only remove one of the `(k−l)` **isotropic** cells.
   The reverse of a birth is therefore a 1-of-`(k+1−l)` choice, not the 1-of-`(k+1)` that Bodin
   2012's cancellation assumes. This is exactly why vanilla BayHunter (isotropic, no `D`) is exact
   while the anisotropic path is not.
2. **The anisotropy moves are themselves unbalanced.** `aniso_birth`/`aniso_death` take the plain
   likelihood-ratio branch (`SingleChain.py:670`), so under a flat likelihood `alpha = 0` and they
   are **always accepted**: `l` performs an unweighted random walk rather than sampling its prior.
   Measured **l/k = 0.50 at every k** (0.481, 0.500, 0.507, 0.488, 0.497, 0.521 for k = 2..7), i.e.
   the configuration space grows with k, and with `D = 0` the sampler targets **P(k) ∝ ~1.2^k**
   instead of the uniform p(k) the `layers` prior states. E17 legitimately reduces that move to a
   bare likelihood ratio — but only because Bodin's prior carries the matching `C(k,l)`
   combinatorial term. **Whether this fork's prior does is the open question, and it is not
   answerable from the code.**

**Why nothing was shipped.** A `np.log()` patch converts a known-wrong term into a
differently-wrong term while *looking* fixed — worse for an audit than the status quo. Resolving
this requires deciding what prior over `(k, l, config)` the fork intends: an author-level question.
The oracle above is what will settle it — add the candidate to `test_acceptance._D` and require TVD
within the control floor.

**Consequence for existing results.** Every anisotropic/radial run's layer-count **and**
γ-occupancy posterior is biased. γ is spike-and-slab, so occupancy is precisely the quantity of
interest. **The synthetic gates inherit the same bug**, so this does not surface as a null-case
failure: the recorded `null P(γ≠0) = 0.09` was measured with a broken sampler and is **not** a
target to reproduce.

## 4. Parallel tempering

**Checked and correct — a negative result, recorded so it is not re-audited.** The swap criterion
(`mcmcOptimizer.py:228`) is
`alpha = (likes[j]-likes[i]) * (1/temps[i] - 1/temps[j])`, i.e. the textbook Geyer (1991)
`(β_i − β_j)(logL_j − logL_i)`, matching the Dettmer reference exactly. Temperature multiplies
**only** the likelihood ratio (`:674`, `:687`, `:700`) and never the prior — the classic PT bug, and
it is absent. Temperatures are stored per sample and correctly aligned with the weight-expanded
model traces, so `Plotting.py:271`'s `alltemps == 1` posterior filter is sample-exact. **The
posterior itself was always built correctly.**

**The defect is the ladder.** Defaults were `t1chains = nchains//2`, `maxtemp = 2.0`, putting the
hottest chain at **β = 0.5**: it merely *halves* every log-likelihood difference. A 25-logL barrier
remains 12.5 logL (e^−12.5 ≈ 4e-6) — uncrossable. So the ladder melted nothing while discarding
every sample not at T=1 (only `t1chains/nchains` are kept). All cost, no benefit; PT could only
ever look neutral-to-worse.

**The prior verdict was wrong, and the reason matters.** "31% swap acceptance, squarely in the
healthy 20–30% band, so the ladder was never the problem" optimised ladder **spacing** and never
checked ladder **range**. Swap acceptance is necessary, not sufficient: 31% only says the rungs are
well spaced *within [1, 2]*. A **high** rate is in fact a symptom of a ladder so tight that adjacent
rungs are nearly the same distribution, so swapping is free and buys no exploration. Measured in
this fork (synthetic Rayleigh group curve, 12 chains, 15k burn-in + 15k main):

| ladder | swap acceptance | T=1 sample fraction |
|---|---|---|
| `t1chains=6, maxtemp=2` (old default) | **44.5%** | **0.50** |
| `t1chains=3, maxtemp=50` (Dettmer-style) | 19.0% | 0.25 |

**Fix** (`noisepy/pt_defaults.py`, a single source of truth — the defaults were duplicated in
`well_vs_qc.py` and `run_bayhunter_cell.py` and could diverge silently, while six other cfg builders
rely on the runner's copy): `PT_T1CHAINS = 3`, `PT_MAXTEMP = 50.0`. Matches Dettmer's
`NPTCHAINS1=3, dTlog=1.35` → T = 1,1,1,1.35,…,36.6,**49.5**, hottest β = 0.02, turning that same
25-logL barrier into **0.51 logL**. Only three chains are pinned cold because chains at equal
temperature **never swap with each other** (`_swap_temperatures` skips equal-T pairs), so
`t1chains = nchains//2` was 8 redundant cold chains while only 8 carried the whole ladder.

**Correct knob order: set the RANGE from the barrier height, then add chains until the SPACING
gives ~20–30% acceptance.** Tuning spacing by shrinking range is how you land on a ladder that is
perfectly spaced and useless.

**The diagnostic is round-trip rate, not swap acceptance** — `pt_round_trips` in the npz, computed
from the saved `pt_temp_history` (`pt_defaults.round_trips`). A round trip (T=1 → T_max → T=1) is
the thing that actually transports a state from the hot end, where barriers are crossable, down to
T=1, where the posterior is collected. Swap acceptance cannot distinguish a good ladder from a
too-cold one. Want ≳1 round trip per chain per run.

**Negative result from the verification run, stated rather than buried**: at 6 chains the new
ladder gives 7.7% swap acceptance and only 2 round trips across 6 chains — the ladder
`[1, 1, 1, 3.68, 13.57, 50]` is too coarse because 6 chains cannot span [1, 50]. `maxtemp=50`
requires enough chains to keep the spacing; at low chain counts, raise `--n-chains` or lower
`--maxtemp` deliberately, and read `pt_round_trips` before believing any PT run.

## 5. Provenance, and the copy-install trap

`BayHunter_Aniso` is pip-installed **non-editable — as a copy**. `import BayHunter` resolves to
site-packages, not to `src/`. **Editing `src/*.py` is therefore silently inert**: the inversion
runs, produces plausible numbers, and uses the old code, with no error anywhere. This is the most
dangerous failure mode in the repo. Three layers, in order prevent → prevent → detect:

1. **`deploy.sh`** copies tracked `src/*.py` into site-packages, refuses on a dirty tree (which
   would deploy code matching no commit) unless `--force`, prints the git HEAD and a per-module
   sha256, and stamps `.deploy_info.json`. It deliberately avoids `pip install .`, which re-runs
   the fragile f2py/gfortran build of `surfdisp96_ext`.
2. **`check_deploy.py`** resolves BayHunter *by importing it* — so it checks what Python would
   actually load, not a hardcoded path — and exits non-zero on drift or a dirty tree. Verified both
   ways: clean → rc=0; `src/Models.py` edited but not deployed → rc=1 naming the stale module.
3. **The npz records both shas**: `bayhunter_git_sha` (the repo) **and** `singlechain_sha256` (the
   module that was actually imported). Recording both is the point — they diverge exactly when
   `src/` was edited but not redeployed, and the git sha alone would happily certify a run that
   used stale code. The git sha comes from `deploy.sh`'s stamp because the deployed tree lives
   outside the checkout and cannot be asked (`git -C site-packages rev-parse HEAD` simply fails).

Given a result npz you can now answer *what code, and what tempering, produced this?* Keys:
`pt_enabled`, `pt_t1chains`, `pt_maxtemp`, `pt_ladder` (the **realised** ladder — see below),
`pt_swaps_accepted/total`, `pt_temp_history`, `pt_round_trips`, `outlier_delta`,
`singlechain_sha256`, `bayhunter_git_sha`, `bayhunter_dirty`, `bayhunter_deployed_utc`,
`driver_git_sha`.

The **realised** ladder is recorded, not just the two knobs: `_create_temperature_ladder`
(`mcmcOptimizer.py:184-198`) silently clamps `t1chains > nchains` to all-ones with only a logger
warning, so PT can be a complete no-op while `pt_enabled = 1`. Only the ladder itself shows that.

> **Consumers must treat a MISSING key as "legacy run, unknown" — never as "PT off."** A pre-2026-07
> npz may well have been a PT run.

## 6. The chain-kept rule, and `confidence` under PT

The relative rule `|1 − median/best| ≤ dev` was retired everywhere in favour of an absolute
Δ-logL cut (`vs_reliability.DELTA_LOGL = 5.0`, ≈150× likelihood ratio): `dev`'s tolerance in real
log units is `dev·|best|`, so it drifts with the likelihood scale (across runs of identical
construction `best` spans −34..+97 ⇒ 0.11..4.84 logL) and explodes near `best ≈ 0` (Böttstein
group+phase, `best = −5.4`: a chain 0.4 log units away — plainly the same basin — scores
`dev = 0.075` and is cut; `dev=0.05` kept 1/16 chains, an artifact). **A likelihood ratio is the
meaningful quantity; a relative deviation is not.**

It **survived in two consumers**, now fixed: `well_vs_qc._chain_kept` and
`chaincount_well_study.chain_stats`. `_chain_kept` is the worse of the two — it claimed to mirror
"what the ensemble ACTUALLY contains", but had not since `_use_abs_outlier_cut` landed, because the
posterior is built with the absolute cut. Both now read the run's own `outlier_delta`.
`--legacy-dev-rule` reproduces pre-2026-07 figures, and **every figure states the rule used**
(`"12/16 chains kept (ΔlogL ≤ 5)"`), so a regenerated panel is distinguishable from an old one on
sight.

Under PT, `chain_like_p2` is **temperature-mixed**: a T>1 sample comes from a flattened posterior
and fits worse *by construction*, so drift statistics and trace plots were reading healthy tempering
as wander. `chain_temps_p2` is now saved, thinned on the **same indices** as `chain_like_p2`
(independently-computed index sets would mislabel which samples were hot — worse than saving
nothing), and `well_vs_qc` masks the drift statistic to T=1 and labels the panel.

`vs_reliability.confidence`: `n_kept` already counts chains that were **ever** at T=1, which is the
right definition — temperatures swap, so with `t1chains=3` it is not 3 fixed chains that stay cold;
over a full run nearly every chain visits T=1. This falls out of `chain_medians_t1` returning `-inf`
only for a never-cold chain. So the `n_kept >= 8` gate is **not** a function of `t1chains` and PT
does not silently downgrade cells. The all-`-inf` case (no chain ever cold) previously fell through
to "keep everything" and is now an explicit zero — it is a broken run, not a confident cell.

## 7. Limitations

- **§3 is unresolved.** The anisotropic trans-D acceptance does not sample its own prior, and the
  fix is left unshipped pending a decision on the intended prior over `(k, l, config)`. Every
  radial γ layer-count/occupancy result before that fix carries this.
- **The synthetic gates cannot detect §3**, since they run the same sampler. `null P(γ≠0) = 0.09`
  is a calibration measured with a broken acceptance ratio — re-baseline it after any fix rather
  than treating it as a target.
- **§3's prior-recovery test is run on the azimuthal path**, not radial, because radial's
  `_validmodel` destroys the uniform-k reference (§3). `D` is the same code path, but the radial
  move set (3 anisomods vs 4) is not exercised.
- **`t1chains=3` costs ~2.7× fewer posterior samples per unit wall-clock** than the old 8/16 (≈19%
  of samples at T=1 vs ≈50%). Raise `--n-chains` or `--iter-main`. This is a real price for working
  mixing, not a free improvement.
- **Round-trip rate is measured on synthetics, not on real cells.** The 6-chain verification run
  shows `maxtemp=50` is too coarse at low chain counts (§4). No production PT run has been made
  with the new ladder.
- **`SingleChain.py:811` uses the global `np.random`**, not `self.rstate`, so full runs are not
  bitwise reproducible; §2 works around this rather than fixing it. The same append-skip also
  drops accepted models from `chainiter` without recording the transition — a weighting bias
  independent of everything above, and temperature-dependent under PT (hot chains accept more).
- **`_model_azianiso_ampchange:404` hardcodes `normal(0, 0.01)`** regardless of the radial prior
  width (`well_vs_qc.py:196` passes (−0.35, +0.35), i.e. 0.70 wide — 70 proposal σ across the
  prior), so γ *amplitudes* explore slowly. Out of scope here; check before trusting γ magnitudes.
  Note the two radial-prior defaults disagree: `well_vs_qc.py:196` uses (−0.35, 0.35) while
  `run_bayhunter_cell.py:267` falls back to (−0.15, 0.25) when the key is absent.
- **No CI.** Verification is the manual protocol below.

## Reproduce

```bash
BH=~/Codes/BayHunter_Aniso
NP=~/Codes/NoisePy-ant
PY=/opt/anaconda3/envs/bayhunter/bin/python

# 0. the deployed copy MUST match src/ -- otherwise everything below tests the wrong code
cd $BH && ./deploy.sh && $PY check_deploy.py          # expect rc=0

# 1. isotropic no-op: bitwise vs SingleChain.py.pre-audit.bak   -> PASS 100000/100000
cd $NP/scripts/picking
OMP_NUM_THREADS=1 $PY test_acceptance.py --tier 1 --ndraw 100000

# 2. reciprocity D_birth(k,l) + D_death(k+1,l) == 0  -> current FAILS at +2.000
OMP_NUM_THREADS=1 $PY test_acceptance.py --tier 2

# 3. prior recovery (the oracle; ~15 min). Control PASSes, all four variants FAIL -- see §3
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES \
  $PY test_acceptance.py --tier 3 --nmain 400000

# 4. synthetic gates (inherit §3 -- read as consistency checks, not proof)
PYTHONPATH=$NP /opt/anaconda3/envs/bayesbay_dev/bin/python synth_radial_gate.py \
  --case null --measure both --template-npz <a real cell npz> --outdir /tmp/gate_null \
  --bayhunter-python $PY --bayhunter-runner run_bayhunter_cell.py
# repeat for --case vti (expect gamma ~ +0.10 in-band) and --case leak (expect a loud failure)

# 5. PT end-to-end: provenance + ladder + round trips, on and off
#    (t1chains/maxtemp unset in the cfg -> noisepy/pt_defaults.py)
python - <<'EOF'
import numpy as np
r = np.load("<out>.npz", allow_pickle=True)
print("PT:", int(r["pt_enabled"]), "ladder:", np.round(r["pt_ladder"], 2))
print("round trips:", r["pt_round_trips"], " swaps:",
      int(r["pt_swaps_accepted"]), "/", int(r["pt_swaps_total"]))
print("code:", str(r["bayhunter_git_sha"])[:12], str(r["singlechain_sha256"])[:16])
EOF
```

**Gate before any production rerun**: `check_deploy.py` rc=0 **and** Tier 1 bitwise PASS. Tier 3
currently fails for every candidate by design (§3) — it is the open item, not a regression.
