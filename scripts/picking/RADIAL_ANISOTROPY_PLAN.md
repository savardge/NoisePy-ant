# Radial-anisotropy inversion workflow — implementation plan

Goal: invert **radial anisotropy γ(z) = (Vsh−Vsv)/Vsv jointly with isotropic Vs (=Vsv) and a
relaxed (searched) Vp/Vs** in BayHunter, with **independent Rayleigh-only and Love-only runs as
data-QC screens** to establish that any resolved γ is real and not a data artifact.
Written 2026-07-13 from the joint Rayleigh+Love session; see memory `radial-anisotropy-plan`
for the condensed cross-session record.

## 1. Why (findings that motivate this)

- **Joint isotropic Rayleigh+Love inversion works and is production-tested** (WAVEDEF machinery,
  wavesets fund/love/fundot/fundlove/fundotlove; wells + full grids both nets, 2026-07-13,
  outputs `Projects/{net}/tomo/vs_inversion/{wells_joint,grid_joint_{physical,combined}}/`).
- **Love/Rayleigh contradiction is sharply spatial**: W-Riehen URG graben cells (Basel-1,
  Otterbach-2) fit Love-only beautifully (χL 0.5–0.8) but joint χL explodes to 5.3–5.8 while
  χF stays ~0.3 — the joint satisfies Rayleigh and abandons Love. East-basement cells
  (Riehen-1/2) and Aargau (Böttstein, Riniken) are consistent (joint χL 0.8–1.0). The spatial
  χ_love map (grid_joint qc_map.png) lights up exactly the graben/flexure.
- **Free (searched) single Vp/Vs does NOT reconcile Love with Rayleigh** but IS demanded by the
  Rayleigh data in the graben: posteriors peak at 2.1–2.6 (fund+ot at Otterbach-2 RAILS the 2.8
  prior ceiling — rerun with wider prior), and freeing it fixes most of the W-Riehen *overtone*
  tension (χO 3.15→2.14 Otterbach-2; 1.40→0.82 Basel-1). So: keep Vp/Vs relaxed in production.
  Love-only Vp/Vs posterior ≈ prior (SH has no Vp sensitivity) — never interpret it.
- **The deep-graben Love cell-curves are partly CONTAMINATED**: observed Love group rises 3×
  steeper (1.25→1.94 km/s over T 1.5–2.3 s at Basel-1) than any smooth isotropic model can
  produce (Love-only best fit rises only +0.24); the short-T end (T 1.4–1.6, U≈1.25) hugs the
  Rayleigh FUNDAMENTAL (1.14) = R-fund leakage onto TT; the long-T plateau (~1.85) forces an
  apparent −35% deep ζ. Ray coverage/res_diag at the cell are normal → data, not tomography.
- **Sign/magnitude sanity (Vienna Basin analog, Esteve et al. AJES26_08 under review)**: nodal
  profile, R+L 0.8–5 s, trans-D (V_Voigt, ζ) per layer, TWO-ISOTROPIC-FORWARDS approximation
  (Rayleigh from Vsv model, Love from Vsh model). They find bi-layered ζ: slightly NEGATIVE
  0–1.5 km (vertical cracks), strongly POSITIVE deeper (horizontal bedding). Flat-lying graben
  fill should give POSITIVE deep ζ — our apparent large NEGATIVE deep γ is the wrong sign and
  too big ⇒ artifact until Love is re-QC'd. Expected real signal: up to tens of % positive in
  Neogene sediments, small/negative shallow.
- **Why keep separate R-only/L-only runs at all** (screening, not estimation): in the joint
  likelihood a genuine ζ(z) and a coherent Love-data bias are mathematically indistinguishable —
  a joint γ inversion will absorb contaminated Love into a beautifully converged, WRONG γ(z).
  The separate runs preserve per-wave information that exposes pathology (unfittable Love-only
  curve, implausible magnitude/sign). They are data-QC canaries run BEFORE the estimator; the
  joint spike-and-slab inversion is the sole estimator and significance test afterwards.

## 2. Data prerequisites (do these first)

**UPDATED 2026-07-13 (later session): the pick-level re-QC is now IMPLEMENTED and CALIBRATED**
in the unified picking system — read `UNIFIED_PICKING_SUMMARY.md` (same dir) before touching
anything here. It supersedes this section's original hand-rolled cuts. Key deltas vs the
original plan:
- The R-fund-on-TT leakage veto is the `rf_leak` gate (`|dU_rayfund| ≤ 0.15` vs the SAME
  pair's G_LR0 curve), plus `ot_leak` (vs G_LR1) and `mode_overlap` — computed in-picker,
  thresholded at merge by `qc_unified_picks.py`, every kill counted in
  `qc_rejection_budget.txt`.
- **The env_ratio ≥ 1.5 gate is WRONG at pick level and stays OFF** (Aargau calibration:
  provably-good Love phase picks have median env_ratio 0.88; the absolute gate killed 74% of
  them). Velocity-coincidence vetoes are the real contamination tests; env_ratio is a
  report-only column.
- Mutual suppression (`xmode_amp ≤ 0.6`) gates the Rayleigh OVERTONE only — never the
  fundamental (spurious 28–42% kills at T > 1.5 s on short paths).
- **CWT-scale dedupe** (group AND phase): without it, counts are inflated ×1.35–2.28 above
  2 s and aggregated std deflated exactly in the long-T band that feeds deep structure. Use
  `T_scale` (not nominal_period) as the period label for phase picks.
- Love overtone stays OFF by default (single-component SH: overtone phase reads a mixed
  transform; overtone-labeled TT ridges are dropped, never relabeled fundamental).

Steps:
1. **Full-network unified runs: DONE both nets (2026-07-14)** — Aargau 17,350 pairs
   (2.86M rows; QC'd survivors: R-fund 641k group / 306k phase, R-ot 235k / 107k, L-fund
   450k / 212k) and Riehen (graben rf-leak kill W 0.29 vs E 0.17 — the diagnosed
   contamination is spatially confirmed).
   **⚠ AUDIT FINDING (2026-07-14, Aargau full network): the rf_leak veto has a long-T
   CENSORING BIAS that must be fixed before the γ(z) inversion.** Kill fraction rises
   monotonically with period — 0.10 (T<1 s), 0.15 (1.5–2 s), 0.28 (2–2.5 s), 0.34
   (2.5–3 s), **0.40 (3–4 s)** — a coherent network-wide long-T band, i.e. genuine
   Love/Rayleigh group-velocity convergence, not leakage. Consequences: (a) up to 40% of
   the deepest-sensitivity Love picks are deleted; (b) the surviving Love set is censored
   AGAINST Love≈Rayleigh, i.e. pre-selected to differ from Rayleigh — feeding it to a γ(z)
   inversion biases toward spurious nonzero anisotropy at depth. FIX before rebuilding the
   Love tomography: make rf_leak (and ot_leak, same logic) **conditional on reference-curve
   separation** — veto only where |ref_love_group(T) − ref_rayfund_group(T)| > 2×leak_tol
   (coincidence is only diagnostic where the curves are separated) — or restrict the veto
   to T ≤ ~2 s (the diagnosed contamination band). Re-run qc_unified_picks after.
   **Re-verification of the separation-conditioned fix (2026-07-14): works as designed in
   Aargau** (veto active ≤1.5 s, tapers to 2 s, OFF above — long-T survivors restored:
   38.6k Love-fund group picks at 3–4 s vs near-total culling before) **but UNDER-VETOES in
   Riehen at 1–2 s**: the NETWORK references mix the slow-west/fast-east provinces (Love ref
   flat ~2.03 = east-dominated; sep drops to 0.28 at 1.2 s, 0.24 at 1.5 s < 0.30 threshold),
   so the veto shuts off across the 1.4–1.6 s band where the Basel-1 graben contamination
   was diagnosed (W/E kill contrast fell 0.29/0.17 → 0.15/0.10, residual = T<1 s only).
   Locally the west separation is ~0.3–0.4 (graben R-fund ~1.14 vs genuine Love ~1.45).
   REFINEMENT NEEDED: make the T-cap a FLOOR, not only a fallback —
   `(T <= leak_tmax) | (isfinite(sep) & (sep > factor*tol))` in `_leak_diagnostic` — i.e.
   always diagnostic at short T (leakage physics is short-T), reference-separation extends
   the veto beyond the cap only where curves truly separate. One line; re-run QC both nets.
   **DONE + verified 2026-07-14.** T-floor applied (`_leak_diagnostic` in qc_unified_picks.py,
   leak_tmax=2.0), QC regenerated both nets. Fire-fraction now: Aargau 0.10/0.12/0.15 over
   0.2–2 s then **0.00 above 2 s** (long-T censoring gone); Riehen 0.19/0.22/0.19 over 0.2–2 s
   then 0.00 above (**graben veto restored: 1.4–1.6 s band W 0.25 vs E 0.21, veto active and
   W>E as designed**), 0.00 above 2 s. This is the FINAL leak-veto behavior. QC'd survivors
   (group | phase): Aargau L-fund 535k | 259k, R-fund 653k | 306k, R-ot 232k | 111k; Riehen
   L-fund 212k | 108k, R-fund 379k | 187k, R-ot 106k | 69k. **Prereq #1 (pick QC) COMPLETE
   both nets — next is the Love/Rayleigh tomography rebuild from `picks_unified_QCd.csv`.**
2. **Rebuild LOVE group tomography** from `picks_unified_QCd.csv` (love/fundamental,
   `group_ok==1`) — swtomotv, same grids (Riehen 500 m LC 1.5, Aargau 1 km LC 3.0). The
   aggregation gains real per-(pair,T) std again via the dedupe.
   RECOMMENDED (decide by A/B): rebuild the **Rayleigh** maps from unified picks too — the
   scale-dedupe finding means the existing production maps' long-T counts/std were
   inflated/deflated (validator cross-check: 100% velocity agreement, 85.8% pick overlap, so
   changes will be second-order in v but first-order in σ).
   Optional extension: PHASE tomography (τ=dist/c straight-ray numerically fine in swtomotv,
   semantics assume group — treat as a fork; phase valid to 1λ reaches ~2× deeper).
3. **Cell-curve screen** (unchanged): curvature sanity via the Love-only inversion χ +
   visual — pick-level gates can't see cell-level curve shape.
4. Wider Vp/Vs prior available: `run_bayhunter_cell.py` accepts `cfg["vpvs"]=[lo,hi]`
   (implemented 2026-07-13; `well_vs_qc.py --vpvs-range lo,hi`). Use [1.5, 3.5] in the graben
   (the 2.8 ceiling clipped the fund+ot posterior at Otterbach-2).

> **STATUS 2026-07-14: §2 COMPLETE** (unified tomography rebuilt from QC'd picks into
> `Projects/{net}/tomo/swtomotv-output-uni/production/{fund,overtone,love}` — one root, all
> three waves; export = `export_unified_tomo_picks.py`, group bounds re-applied there).
> **§3 IMPLEMENTED + DEPLOYED** (edited files also copied into the bayhunter env's
> site-packages — the fork is pip-installed as a copy; `.pre-radial.bak` backups kept).
> **Synthetic gates 1+2 PASSED** (+10% γ recovered as +0.122 in-band / +0.03 out, all-wave
> χ≈0.9; γ=0 null stays dead, P(γ≠0)=0.09; radial-OFF control reproduces χ_love≈5.9 ⇒ the
> real graben signature ≈ 10–15% equivalent γ). Remaining: §5 gates 3–5 (real-data screens,
> Aargau null cells, well arbitration) → §6 run matrix.

## 3. BayHunter refactor (fork: ~/Codes/BayHunter_Aniso) — reuse the AzAniso machinery

Model vector is 4 NaN-padded per-layer blocks `[vs | z_vnoi | psi2amp | psi2azi]`
(`src/Models.py:18-24` split_modelparams). The azimuthal fork already solved per-layer
anisotropy trans-D (birth/death/proposals/storage). Plan: **reinterpret the psi2amp block as
signed γ under a new flag**; psi2azi goes dead. Zero layout/storage changes.

| where | change |
|---|---|
| `src/SingleChain.py:81` | new `initparams["radial_anisotropy"]` flag → `anisomods = ['aniso_birth','aniso_death','aniso_ampmod']` (drop `aniso_dirmod`). Existing birth/death semantics (death only of γ==0 cells, line ~334) = spike-and-slab prior: layers isotropic unless data demand γ≠0. **P(γ≠0|data) from the single chain is the significance test** — no cross-run comparisons. |
| `src/SingleChain.py:463` | replace hardcoded psi2amp bound [0, 0.1] with signed `priors["radial"]=(-0.15,+0.25)` under the flag (γ must allow negative: shallow cracks, and the manuscript's shallow layer). |
| `src/Targets.py:460-470` (`JointTarget.evaluate`) | THE physics edit (~10 lines): under the flag skip c1/c2; pass `vs*(1+γ_layers)` to targets whose `ref` startswith 'l' (ldispgr/ldispph), plain `vs` (=Vsv) to 'r' targets. vp/rho stay tied to Vsv (SH insensitive to Vp). No extra forward cost — each target already runs its own calc_synth. |
| `_validmodel` | LVZ/HVZ contrast cap stays on vs(=Vsv), NOTE it is ABSOLUTE km/s not fractional (lvz=hvz=0.5 ⇒ |ΔVs|≤0.5). Optionally add an adjacent-layer |Δγ| cap, same pattern. |

Parameterization: sample **vs ≡ Vsv** + per-layer γ; report Voigt Vs and ζ=(Vsh−Vsv)/V_Voigt in
post-processing (V_Voigt=sqrt((2Vsv²+Vsh²)/3), the Esteve/Tomar/Mordret convention). Rayleigh
forward literally unchanged. Two-isotropic-forwards approximation = community standard (ignores
η, Vp-anisotropy; fine at these periods and comparable with the literature).

## 4. Our-code changes (NoisePy-ant)

- `scripts/picking/run_bayhunter_cell.py`: pass `radial_anisotropy` + `priors["radial"]` from
  cfg; split γ from posterior models via `split_modelparams` (rides free in stored vectors);
  save `gamma_p16/50/84(z)` + `gamma_ens` profiles like vs; disba posterior-predictive uses
  `vs*(1+γ)` for 'love' band (and `{w}_phase` keys — group+phase joint via `curves_phase` and
  `wmeta` already exist from the phase step-2 work; radial rides through the same dispatch).
  **With the unified picker, feed PHASE targets too**: Love-fund phase is validated (+π/4,
  resid −0.02) and `Targets.LoveDispersionPhase` exists — Rayleigh+Love group+phase jointly is
  the strongest γ(z) configuration (phase reaches deeper; use `T_scale` period labels and
  `phase_ok==1` picks). Requires phase cell curves (from phase tomography, or per-cell
  aggregation of QC'd phase picks as an interim).
- `noisepy/vs_inversion.py`: result schema additions (gamma bands), `data_misfit` unchanged
  (per-wave keys); plotting: γ(z) panel + ζ conversion.
- `well_vs_qc.py` / `grid_vs_inversion.py`: `--radial` flag + `--radial-prior lo,hi`; new
  waveset key(s) e.g. `fundotlove_r` (radial on); volume npz gains `gamma_*` fields;
  postprocess ζ(z) maps + P(γ≠0) map.
- Keep per-wave χ decomposition in the joint radial runs (already saved): the internal
  consistency argument = joint-γ fits both waves with sane γ, while the SAME model with γ
  forced 0 cannot.

## 5. Validation gates (in order; do not skip)

1. **Synthetic VTI recovery**: build Vsv(z)+γ(z) (e.g. γ=0 above 1 km, +10% below; also a
   negative-shallow case), forward R0/R1 with disba on the Vsv model and L0 on the Vsh model,
   realistic σ; invert. Pass = γ(z) recovered where Love kernels live (~0.2–2.5 km),
   prior-dominated below (this empirically answers the resolution question).
2. **Isotropic null synthetic**: γ=0 truth → posterior must keep γ dead (spike-and-slab
   parsimony check; guards against γ absorbing noise).
3. **Aargau null cells** (Böttstein, Riniken — isotropically consistent, joint χL≈0.6–1.0):
   radial run should return γ≈0 with high P(γ=0). If it doesn't, the prior/proposals leak.
4. **Screens as canaries (per target cell, BEFORE the estimator)**: R-only and L-only isotropic
   inversions on the re-QC'd curves. Love-only must now fit smoothly (χL<1, no over-steep
   residual structure) and the implied discrepancy must have plausible magnitude (|ζ|≲15–20%)
   and sign (positive at depth in bedded fill). Fail → back to picks, do NOT run the estimator.
5. **Well arbitration**: Otterbach-2/Basel-1 (Michel2016 log incl. its in-situ Vs and
   Vp/Vs≈1.9–2.5 curves), Aargau Nagra logs. The freed-Vp/Vs Rayleigh Vs already tracks
   Michel2016 (~2.4 at depth); the radial model's Vsv must keep doing so, with Vsh explaining
   Love — not Vs drifting to split the difference.
6. **Posterior hygiene**: convergence via existing bh_diagnostics (chain_disagree,
   frac_chains_ok); per-wave χ table; Vp/Vs posterior interpretable ONLY when Rayleigh waves in
   the target set (Love-only ⇒ prior).

## 6. Run matrix

1. Wells first (6 in-hull cells), wavesets: `love`, `fund`, `fundot` (screens) +
   `fundotlove` radial-ON with vpvs [1.5,3.5] (estimator) + `fundotlove` radial-OFF (control),
   criteria physical+combined. Heavy ensembles + save_ensemble (wsoverlap + γ panels).
2. Graben transect / full Riehen grid radial-ON (fundotlove_r), Aargau grid as the null network.
3. Products: ζ(z) depth maps + P(γ≠0) maps + the χ_love map before/after (contamination
   removed + anisotropy modeled should kill the graben χ_love anomaly).

## 7. Environments & key paths (unchanged from the joint-Love work)

- BayHunter subprocess: `/opt/anaconda3/envs/bayhunter/bin/python` (fork mp workarounds in the
  runner); driver env `bayesbay_dev`; PYTHONPATH=~/Codes/NoisePy-ant everywhere.
- Rayleigh production maps: `Projects/{net}/tomo/swtomotv-output/production/{fund,overtone}`;
  Love: `swtomotv-output-love-{500m,1km}/production/love` (to be rebuilt after re-QC).
- Love phase references: `Projects/{net}/vsg_modesep/ref_love_phase.txt` +
  `ref_love_overtone_phase.txt` (both nets, built by `vsg_love_reference.py` from the TT VSG
  stack; the overtone ref is required by the unified picker's Love labeling even with the
  Love-overtone toggle off).
- Unified picking system: `noisepy/unified_picking.py`, `scripts/picking/dispersion_unified.py`,
  `qc_unified_picks.py`, `validate_unified_picks.py` — full spec, gate calibration lessons, and
  run commands in `UNIFIED_PICKING_SUMMARY.md` (same dir). Aargau 800-pair calibration output:
  `Projects/aargau/dispersion_unified/{picks_unified_QCd.csv, qc_rejection_budget.txt}`.
- Session findings: memory notes `vs-inversion-module`, `vs-inversion-findings`,
  `v4-love-picking-audit`, `swtomotv-bridge`, `unified-picker`, `radial-anisotropy-plan`.
