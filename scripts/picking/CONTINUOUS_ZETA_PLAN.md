# Continuous per-layer radial anisotropy — refactor plan

Pivot the radial-anisotropy inversion from a **spike-and-slab** parameterization (each layer is
isotropic *or* anisotropic, toggled by trans-dimensional `aniso_birth`/`aniso_death` moves) to a
**continuous per-layer** one (every layer carries a signed γ that varies like velocity; the only
trans-D move is ordinary layer birth/death). This matches the directly-comparable published study
(Esteve & Gosselin, Vienna Basin — same pull-apart-basin geology, Bayesian trans-D, Rayleigh+Love)
and the standard practice it cites (Tomar 2016, Mordret 2015, Wu 2024). Crucially it **dissolves the
trans-D acceptance blocker**: the broken combinatorial acceptance exists *only* for the spike-and-slab
moves, and removing them reduces the trans-D acceptance to the vanilla isotropic layer birth/death
that BayHunter already implements correctly (proven bitwise by `test_acceptance.py` Tier 1). What
remains is small: draw γ from its prior on birth, perturb it continuously with a properly-scaled
step, and read anisotropy significance from the γ(z) posterior rather than a spike occupancy.

Supersedes the anisotropy-sampling sections of `RADIAL_ANISOTROPY_PLAN.md` and closes the open item
in `RADIAL_ACCEPTANCE_AUDIT.md` §3 for the radial path. Isotropic runs are unaffected throughout.

| file | change |
|---|---|
| `BayHunter_Aniso/src/SingleChain.py` | move set, birth γ-from-prior, γ perturbation, init, D-term gate |
| `BayHunter_Aniso/src/Targets.py` | none (forward already Love→Vsh, Rayleigh→Vsv, `:483`) |
| `NoisePy-ant/scripts/picking/run_bayhunter_cell.py` | γ output = sign significance, prior default |
| `NoisePy-ant/scripts/picking/well_vs_qc.py` | γ strip panel: P(γ>0)/band-excludes-0 |
| `NoisePy-ant/scripts/picking/test_acceptance.py` | new Tier: continuous-radial prior recovery |
| `NoisePy-ant/scripts/picking/synth_radial_gate.py` | re-run + re-baseline (no code change) |

## 1. Why this is correct, and why it is small

The audit (`RADIAL_ACCEPTANCE_AUDIT.md` §3) proved that **no choice of the spike-and-slab `D` term
recovers the prior**, because two spike-and-slab-specific moves are unbalanced: layer birth inserts
an isotropic cell while death removes only isotropic cells (asymmetric eligibility), and the
`aniso_birth`/`aniso_death` moves carry no combinatorial factor. Both defects live entirely in the
spike-and-slab machinery.

Continuous per-layer γ removes that machinery. A born layer draws γ **from its prior** (a
birth-from-prior move), for which the proposal ratio `1/p(γ)` and the prior ratio `p(γ)` cancel to
**1** — so the layer birth/death acceptance is exactly the vanilla isotropic `log(A) + B + C`, with
no anisotropy term at all. That path is already proven correct: `test_acceptance.py` Tier 1 shows it
bitwise-identical to upstream over 10⁵ draws, and Tier 3's isotropic control recovers the uniform
prior over k at TVD 0.005–0.011. γ then varies within-dimension via a single continuous perturbation
move (a standard Metropolis step, detailed-balanced by construction). **Correctness is inherited, not
re-derived** — the new Tier only confirms it.

## 2. Parameterization

Keep the fork's `(Vsv, γ)` layer parameters, γ = (Vsh − Vsv)/Vsv, Vsh = Vsv·(1+γ), Love forwarding on
Vsh and Rayleigh on Vsv (`Targets.py:474,483`) — no forward-model change. Note this differs
**definitionally** from Esteve's ζ = (Vsh − Vsv)/V_VOIGT with V_VOIGT = √((2Vsv² + Vsh²)/3): same two
degrees of freedom, different normalization. Provide a post-hoc γ→ζ conversion in the output so
numbers are directly comparable to that paper; do not change the sampler's parameterization.

## 3. Fork changes (`BayHunter_Aniso/src/SingleChain.py`) — all gated on radial

**3.1 Move set (`:95-96`).** For radial: `anisomods = ['aniso_ampmod']` — the continuous γ
perturbation only. Drop `aniso_birth`, `aniso_death`. (Azimuthal branch `:93-94` untouched — see 3.7.)

**3.2 Layer birth (`_model_layerbirth:350`).** When radial, the new cell's γ is drawn
`uniform(glo, ghi)` from the radial prior instead of set to 0. Death (`:365`) already drops the
cell's γ with it. This is the birth-from-prior move that keeps the acceptance vanilla-isotropic.
(For azimuthal, keep γ=0 — unchanged.)

**3.3 Acceptance (`get_acceptance_probability:676-702`).** Gate the `D` term on
`self.initparams['azimuthal_anisotropy']` instead of `len(self.anisomods) > 0`. Consequences:
isotropic → `D=0` (unchanged, Tier 1 stays bitwise); **radial → no `D`** (correct, birth-from-prior);
azimuthal → unchanged (still its current broken `D`, but out of scope, 3.7). Birth/death for radial
is now exactly `log(A) + B + C`.

**3.4 γ perturbation (`_model_azianiso_ampchange:396-408`).** Two fixes:
- pick **any** layer, not only `psi2amp != 0` (every layer now carries γ);
- replace the hardcoded `normal(0, 0.01)` with a step scaled to the prior width. `0.01` across a
  0.4–0.7-wide prior is ~40–70 σ, so γ never mixes. Initialize to ≈`(ghi−glo)/10` and adapt it via
  the existing acceptance-rate machinery (extend `propfixed`/`propdist` to cover the move, or a
  dedicated adaptive step). Correctness is independent of the step (any symmetric step is
  detailed-balanced); this is purely mixing, but it is the difference between exploring γ and not.

**3.5 Init (`draw_initmodel:172-186`).** When radial, initialize **every** layer's γ ~
`uniform(glo, ghi)` (natural overdispersion — each chain starts from an independent γ draw). Remove
the odd-chain `radial_overdisperse` special-case (`:179-185`): it existed only because all-zero
starts could not reach the anisotropic mode through `aniso_birth`, a problem that no longer exists.

**3.6 Forward + validity.** No change. `Targets.evaluate:460-483` already dispatches correctly.
`_validmodel`'s radial branch (Vsh = Vs(1+γ) ∈ [vsmin, vsmax]) stays — it truncates the effective γ
prior by Vs, which is physical and intended (see Limitations).

**3.7 Azimuthal path — explicitly out of scope.** The `azimuthal_anisotropy` branch keeps its current
spike-and-slab moves and its (still-broken, per the audit) `D` term, unchanged. It is not used in
this project. Document at the top of the file that azimuthal anisotropy is **unvalidated** and must
not be used for results until given the same continuous treatment or a proper derivation.

## 4. Driver / output (`NoisePy-ant`)

**4.1 Sign significance instead of spike occupancy (`run_bayhunter_cell.py:422-425`).**
`gamma_frac_nonzero` is meaningless under continuous γ (never exactly 0). Replace with
`gamma_p_positive` (posterior fraction with γ>0, per depth) and keep `gamma_p16/median/p84`.
Significance of anisotropy at a depth = whether the 68/95% band excludes 0 — exactly how Esteve reads
it. Add the γ→ζ (Voigt) conversion here so the npz carries both.

**4.2 γ strip panel (`well_vs_qc.py`, the radial γ panel).** Replace the `P(γ≠0)` (spike) readout
with `P(γ>0)` and a visual marker where the band excludes 0. Update the docstring note that warns γ
is spike-and-slab (no longer true).

**4.3 Prior default consistency (audit #6).** `well_vs_qc.py:196` uses (−0.35, 0.35);
`run_bayhunter_cell.py:267` falls back to (−0.15, 0.25). Make them agree (recommend −0.35..0.35) so a
run that omits the key does not silently get a narrower prior.

## 5. Noise model (audit #3 — shared with the group/phase goal)

Tighten `swdnoise_sigma` (`run_bayhunter_cell.py:248`, currently (1e-4, 0.5)) to bracket the measured
per-curve σ, so the free hierarchical noise can no longer inflate to absorb the Love-vs-Rayleigh
mismatch. That mismatch is the signal that should *drive* γ; today it is hidden behind chi_eff≈1.
Esteve propagates the tomographic uncertainties directly as data errors — the same intent. Run
tightened vs free and record the difference; `noise_post` already saves the realised σ per target.

## 6. Verification

- **Tier 1 (regression, must pass first).** `test_acceptance.py --tier 1`: removing `D` from the
  radial/isotropic path must leave isotropic acceptance **bitwise identical** to the pre-audit
  baseline (D was already 0 there). Non-negotiable gate.
- **Tier 3 continuous-radial (the new oracle).** Add a mode that runs the continuous-γ radial sampler
  under a flat likelihood and checks two marginals recover their priors: (a) k → uniform (inherited
  from the isotropic control, so it should pass by construction), (b) γ per layer → its
  (Vs-truncated) prior. Verdict by TVD vs the isotropic control floor, as in the existing Tier 3.
- **Synthetic gates.** Re-run `synth_radial_gate.py` null/vti/neg/leak on the refactored sampler and
  **re-baseline** — the recorded `null P(γ≠0)=0.09` was measured with the broken sampler and is void.
  New reads: null → γ(z) posterior centred on 0, bands include 0 (P(γ>0)≈0.5); vti/neg → recovered
  where Love kernels are sensitive, band excludes 0; **leak → the identifiability test** (isotropic
  truth, Love built from the Rayleigh forward): γ must stay indistinguishable from 0, or the graben
  negative-γ is absorbed bias, not anisotropy.
- **End-to-end.** A radial well run producing provenance + a γ(z) posterior with sign significance,
  same harness as the PT verification.

Gate before any production radial rerun: `check_deploy.py` rc=0 + Tier 1 bitwise + Tier 3
continuous-radial within floor + the `leak` gate behaving.

## 7. Deploy & order

Fork edits are silently inert until copied to site-packages — every step ends with
`./deploy.sh && python check_deploy.py`. Land in this order, one commit each:

1. **Fork refactor** (3.1–3.5) + Tier 1 re-check (must stay bitwise) → deploy.
2. **Tier 3 continuous-radial** oracle; confirm k and γ recover their priors.
3. **Output/plotting** (4.1–4.3) — sign significance, γ→ζ, prior consistency.
4. **Noise tightening** (5) as a cfg option; tightened-vs-free comparison.
5. **Synthetic gates** re-run + re-baseline; only then interpret γ.

Steps 1–2 are the critical path and are self-contained in the fork + test. Nothing here needs the
author-level derivation that the spike-and-slab route required — that requirement is retired.

## 8. Limitations (carry into any writeup)

- **Two-isotropic-forwards approximation** (unchanged): Love on Vsh, Rayleigh on Vsv, both via
  isotropic surf96. Ignores Rayleigh's Vsh sensitivity, η, and Vp anisotropy — standard first-order
  practice (as in Esteve), but it sets a floor on how well γ can reconcile the two.
- **Vs-truncated γ prior**: `_validmodel` caps Vsh∈[vsmin,vsmax], so the effective γ prior narrows
  where Vs is near its bounds (positive γ suppressed at basement, negative in slow cells). Physical
  and intended; report γ significance against this truncated prior, not a flat one.
- **No per-layer anisotropy model-selection**: continuous γ cannot say "P(this layer is
  anisotropic)". By design, and matching Esteve — significance is read from the γ(z) posterior
  excluding 0. This is the *better* readout for the graben sign question anyway.
- **γ vs Love-slow-bias identifiability** is not solved by the refactor; it is *tested* by the `leak`
  gate (§6). The negative graben γ is not interpretable until that gate says the sampler can tell a
  genuine γ from absorbed bias.
- **Azimuthal anisotropy remains unvalidated** (§3.7) — do not use it for results.
