# 1D Vs depth inversion from mode-separated group-velocity maps

Trans-dimensional Bayesian inversion of a tomography **cell's** effective fundamental +
1st-overtone Rayleigh **group-velocity** dispersion curves (from the swtomotv production maps)
for a 1D Vs(z) profile, with **two engines compared** and a physically-motivated prior.

- Library: [`noisepy/vs_inversion.py`](../../noisepy/vs_inversion.py)
- Driver: [`run_vs_inversion.py`](run_vs_inversion.py)  ·  BayHunter subprocess runner:
  [`run_bayhunter_cell.py`](run_bayhunter_cell.py)

## What it does

1. **Cell curves** — `load_cell_curves(production_root, ix, iy)` collects, across all period
   maps, the cell's `U(T)` for `fund` and `overtone` with 1σ = `unc_s·U²` (slowness→velocity).
2. **Forward** — disba `GroupDispersion`, fundamental = mode 0, 1st overtone = mode 1;
   `vp = 1.73·vs`, `rho = 0.32·vp + 0.77` (Brocher).
3. **Prior / constraint** — layered Vs that **allows LVZ and HVZ** (no monotonic-increase
   requirement) but **caps the adjacent-layer velocity contrast at 50%** (`MAX_ADJ_FRAC=0.5`).
4. **Two engines**, same data, same constraint:
   - **bayesbay** (trans-D Voronoi1D + disba) — runs in-process (`bayesbay_dev` env). Constraint
     = hard prior support: a constraint-respecting initializer (bounded random walk) + rejection
     in the forward when `|ΔVs/Vs| > 0.5` or the mode is undefined.
   - **BayHunter** (surf96 forward, AzAniso fork, isotropic) — runs as a subprocess in its own
     env. Constraint = **native `initparams lvz = hvz = 0.5`** (deeper layer within (1±0.5)× the
     one above). Overtone = a second `RayleighDispersionGroup` target with
     `.moddata.plugin.set_modelparams(mode=2)`.
5. **Post-processing / QC** — `plot_inversion` (posterior Vs(z) median + 68/95% bands +
   dispersion fit) and `compare_engines` (overlaid Vs(z), fits, and a runtime / posterior-layers
   / data-χ table).

## Environments (two, by necessity)

bayesbay needs numpy ≥ 2; BayHunter's `numpy.distutils` Fortran build needs numpy < 1.26 — so
they live in **separate conda envs**, and the driver (bayesbay env) shells out to the BayHunter
env. This mirrors the cluster split.

- `bayesbay_dev`: bayesbay 0.3.7, disba, pysurf96, swtomotv (`pip install -e`).
- `bayhunter`: **created for this** — `python=3.10 numpy=1.24 scipy matplotlib configobj cython
  pyzmq`, then `setuptools<60` (numpy.distutils needs `distutils.msvccompiler`, removed in
  setuptools 82), then `pip install . --no-build-isolation` of `~/Codes/BayHunter_Aniso`
  (with `/opt/homebrew/bin` on PATH for gfortran), plus `pip install disba`. The fork's
  `setup.py` rfmini C++ extension is commented out (RF-only, fragile on macOS; SW path never
  imports it).

## Reproduce (Riehen best-covered cell, deep MCMC, both engines)

```bash
PYTHONPATH=~/Codes/Noisepy-ant /opt/anaconda3/envs/bayesbay_dev/bin/python run_vs_inversion.py \
  --production ~/Codes/extract_higher_modes/Projects/riehen/tomo/swtomotv-output/production \
  --config    ~/Codes/extract_higher_modes/Projects/riehen/tomo/riehen_swtomotv.yaml \
  --ix 13 --iy 15 --outdir <out> --engines bayesbay,bayhunter \
  --n-chains 8 --iterations 300000 --burnin 100000 --bh-iter-burnin 80000 --bh-iter-main 40000 \
  --bayhunter-python /opt/anaconda3/envs/bayhunter/bin/python \
  --bayhunter-runner run_bayhunter_cell.py
```

Outputs in `<out>`: `bayesbay_result.npz`, `bayhunter_result.npz` (shared schema),
`{engine}_inversion.png`, `engine_comparison.png`. `--reuse` skips an engine whose result npz
exists. Aargau's best cell is (8,10).

## macOS gotchas baked into the runner (why BayHunter now runs in seconds)

BayHunter targets Linux clusters; three macOS-specific fixes were required (all handled
automatically by the runner / `run_bayhunter` env):

1. **`fork` before importing BayHunter** — its `MCMC_Optimizer.__init__` creates a
   `multiprocessing.Manager`; macOS' default `spawn` deadlocks it. `run_bayhunter_cell.py` sets
   `mp.set_start_method("fork")` at the very top, before the BayHunter import.
2. **Serial chains** — `mp_inversion`'s Process/shared-array layer crawls on macOS; the chains
   are independent (no parallel tempering), so the runner iterates them serially
   (`chain.iterate()` … `finalize()`), reproducing the same on-disk output (~0.3 ms/iter).
3. **disba for the predicted-fit band** — surf96's group-velocity root-finder can **hang**
   (infinite Fortran loop) on some LVZ posterior models; the fit band is computed with disba
   (iteration-limited, ~1% agreement) instead. The MCMC itself still uses surf96.
   Plus BLAS pinned to 1 thread (`VECLIB/OMP/OPENBLAS_NUM_THREADS=1`) + objc fork safety.

## Notes

- vpvs is fixed at 1.73 in both engines for a clean comparison (BayHunter can also search it).
- **Data noise is HIERARCHICAL with a WIDE prior in both engines** (bayesbay `noise_std=(0.01,0.5)`,
  BayHunter `swdnoise_sigma=(1e-4,0.5)`). The tomographic formal σ (~0.04–0.16 km/s) is a *formal*
  posterior error that underestimates true uncertainty (1-D theory error, mode-ID, tomographic
  regularization). Feeding it in tight makes the trans-D **over-fit** (bayesbay went to ~11 layers,
  wiggly); letting the data infer the noise from a wide prior gives parsimonious, smooth models
  (~7 layers) that agree between engines. This is the single most important tuning choice.
- **Convergence**: `run_bayesbay` reports `chain_disagree` = worst-depth between-chain std of the
  median Vs(z). Small (≲0.1 km/s) ⇒ converged; large ⇒ increase `n_iterations`/`n_chains` or the
  posterior is genuinely multi-modal (group-only inversion of shallow structure is non-unique).
- The fundamental usually fits well; a **short-period (<1 s) overtone tension** is common —
  the flat short-period overtone from tomography isn't always reproducible by the 1-D model that
  fits the fundamental (thinner short-period overtone coverage, and its formal σ likely
  underestimates its true error), and shows up as a higher χ. Consider restricting the overtone
  to T ≳ 1 s if it dominates the misfit.
- Cluster runs: use BayHunter's native `mp_inversion` (parallel) and its own plotting; this
  runner's serial path is the macOS workaround.

## Per-cell period-reliability trimming (`noisepy/period_resolution.py`)

Tomography is Tarantola–Valette, so the stored slowness σ is *prior-bounded* and stays low even
where data is sparse — hiding poor reliability toward the array edge at long periods. Symptom:
edge cells (e.g. Aargau **Böttstein**) invert to implausible shallow-fast / deep-LVZ Vs because a
decreasing, unreliable long-period branch is over-fit; interior cells (**Riniken**) are fine.

`period_resolution.trim_reliable(cell, net, criterion, params)` trims each cell's curve to its
well-resolved periods before inversion, with three selectable criteria:
- **C `tomographic`** — keep T where `res_diag ≥ R_frac·max(res_diag)` (relative, since res_diag
  is low everywhere; default `R_frac=0.5`).
- **B `physical`** — keep T where `d_edge ≥ alpha·λ(T)` (λ=c_phase·T from `ref_*_phase.txt`;
  `d_edge` = distance to the station convex hull; default `alpha=0.5`) AND kernel depth
  `z_eff(T) ≤ beta·depth_max` (disba `GroupSensitivity`). This is spatially adaptive — aggressive
  at the edge, inactive in the centre.
- **A `combined`** — intersection of B and C.

Wired into `well_vs_qc.py --criterion {none,tomographic,physical,combined,all}` (produces
`well_<name>_<waveset>_criteria.png`, a Vs-posterior panel per criterion vs the well log).
Diagnostics: `period_resolution_qc.py` → `reliability_<wells>.png` (res_diag, d_edge/λ, z_eff,
keep-matrix per criterion) and `max_reliable_period_{A,B,C}.png` (bull's-eye map). Finding:
trimming removes the Böttstein artifact (smooth monotonic Vs, χ unchanged/improved) and leaves
interior Riniken essentially unchanged; the **physical** criterion is the cleanest discriminator.

## Third engine: Dinver (Geopsy NA) via the SWinvert workflow

`--engines dinver` runs Geopsy's neighbourhood-algorithm inversion the way Vantassel & Cox
(2021, *SWinvert*, GJI 224:1141) prescribe, and returns it in the shared result schema so
`compare_engines` / `data_misfit` / the grid assembler treat it like the other two.

- Library: `noisepy/dinver_target.py` (units + `.target` writer) · `vs_inversion.dinver_config`
  / `run_dinver` · runner: [`run_dinver_cell.py`](run_dinver_cell.py) · grid:
  `grid_vs_inversion.py --engine dinver` · well QC: [`dinver_well_compare.py`](dinver_well_compare.py)
- Env: `das-ambient-noise` + swprepost (`~/Codes/swprepost` on `PYTHONPATH`, or `pip install -e`)
  + the local Geopsy build (`~/Codes/geopsy-install`, `dinver.app/Contents/MacOS/dinver`,
  never `-clear-plugins`). Cluster: `module load GCC/11.3.0 OpenMPI/4.1.4 geopsy/3.4.2`
  (headless-clean, plain `dinver` on PATH → `--dinver-bin`).

**What SWinvert dictates and how it maps here** (all flags default to the paper's values):

| paper | here |
|---|---|
| resample the target in log-wavelength, 20–30 pts | `--dinver-n-resample 30` (`ModalTarget.easy_resample`, wavelength, log) |
| COV floor 0.05–0.10 where σ is unquantified | `--dinver-min-cov 0.05` on top of the tomographic σ |
| several parameterizations: LN {3,4,5,7} + LR {3.0,2.0,1.5,1.2} | `--dinver-lns / --dinver-lrs`, one `.param` each (swprepost `from_ln`/`from_lr`) |
| layer sizing from fundamental **phase** λ: hmin=λmin/3, dmax=λmax/df, df=2 | per cell, from `fund_phase` (fallback group U·T, warned) — validated period ranges applied first |
| Vp linked to the Vs layering, ν free, **no fixed Poisson** | `from_parameter_and_link`, `--dinver-vp 0.8,8.0`; ρ fixed `--dinver-rho 2000`; ν **`--dinver-pr 0.2,0.35`** (crustal — the paper's 0.2–0.5 is a soil range, see below) |
| Ns0 ≥ Nr, It·Ns ≥ 50 000, Nr ≈ 100, ≥ 3 trials | `--dinver-ns0 10000 --dinver-ns 50000 --dinver-nr 100 --dinver-ntrials 3` |
| reject a parameterization only if misfit high **and** edge-of-range with inconsistent Vs | `_reject()` in the runner, both criteria, logged per param |
| report the 100 lowest-misfit models per accepted param + σ_ln,Vs | pooled ensemble → `vs_median/p16/p84/p025/p975`, `sigma_ln_vs`, `ens_vs`, `ens_param` |

**Deliberate deviation from the other two engines:** Dinver's Vp/ν are free (paper's strong
advice), the others pin `VPVS=1.73`. `compare_engines` labels Dinver's `n_layers` as fixed
per parameterization for the same reason.

**Three target sets** are selectable with `--dinver-measure group|phase|joint` (the joint one is
group and phase as separate ModalCurves in a single `.target`); Dinver reads the slowness tag
per curve. **This is why the writer is ours** — swprepost hard-codes `Phase`.

**Group-slowness grid padding — read before changing `dinver_target.py`.** Dinver evaluates the
forward only at the target's own frequencies and gets group slowness by finite-differencing the
phase curve *across that grid* (`Dispersion::setGroupSlowness`), invalidating the two end
points. Measured on a synthetic: 14-point grid → up to 3.7 % group error vs disba, and a
group-only target lost its two extreme periods and had its misfit multiplied by
`1+nData−nValues` (= ×3). Fix: each Group curve carries dense `<valid>false</valid>` padding
points (40/decade, ±15 % beyond the data band). They enter the forward grid but not the misfit.
With padding: 0.06 % max vs disba, all data points fitted, misfits comparable to phase.
`group_pad_per_decade=0` disables it — only for experiments.

**Cost (M4 Max, Basel-1 group, 8 params × 3 trials × 60 000 models):** 250–480 s per run,
**943 s wall at 12 concurrent ≈ 2.4 core-h per cell**. NA cost is super-linear in model
count, so do not extrapolate from small `-ns`. Reports are ~560 MB each and are deleted after
the best-Nr models are cached to `<run>.report.gm.txt` (`--dinver-keep-reports` to keep);
resume is keyed on that cache. `--dinver-parallel N` runs the 24 jobs N at a time (single-cell);
under `grid_vs_inversion.py` it is 1 and the pool supplies the parallelism. Always `-j 1`.

**Prior binding.** The npz carries `bind_vs_frac` (layers within 2 % of vs_min/vs_max → widen)
and `bind_depth_frac` (deepest interface at dmax_param — the many-layer LR sets park their last
interface where the data stop resolving; informational, not a bound to widen). Basel-1 group:
0.056 / 0.32.

Per-network priors (from the blanket prod3_k3 maps, validated period ranges): riehen
`--depth-max 6.5 --vs-min 0.5 --vs-max 4.2`; aargau `12.0 / 0.5 / 5.0`; hautesorne
`14.5 / 0.5 / 4.8`. Note the repo's historical `vs_max=3.6` sits *below* the Vs implied by the
observed phase velocities in all three networks.

**Resampling domain — the second thing to know before touching `dinver_target.py`.** SWinvert's
log-wavelength resample (swprepost `easy_resample(domain="wavelength")`) interpolates V against
λ = V·T and assumes λ is monotone in T. It is not for group curves through an Airy minimum
(Riehen Love group: 3 reversals) nor for wiggly phase curves (Riehen fund_phase: 3 reversals).
On such curves the interpolation is garbage — Love group came out with *negative* frequencies,
fund_phase with up to 0.46 km/s (2–3 σ) of spurious velocity — and this silently poisoned every
phase/joint Dinver run made before 2026-08-18 (archived under `_v0_prefix/`). `to_modal_target`
now uses log-wavelength only when λ(T) is strictly monotone and falls back to log-frequency
(the paper's second choice) otherwise, warns, and refuses to write a target that is not finite,
positive and ordered. Post-fix all well curves are within 0.05 km/s of raw.

**Well-cell results** (Riehen Basel-1 (23,47) / Otterbach-2 (26,43), blanket prod3_k3 inputs,
v1 period ranges, ν 0.2–0.35): the full 12-combo waveset matrix (R0g, R0p, L0g, L0p, R0g+R0p,
L0g+L0p, R0g+L0g, R0p+L0p, all-four-fundamental, R0g+R1g, R0p+R1p, R0+R1 both) for both engines
lives in `tests/dinver_swinvert/` — `well_comparison.csv` from `dinver_well_compare.py`, and the
per-combo panel figures `well_matrix_<well>.png` from `dinver_well_matrix_figure.py`. Read the
CSV rather than numbers copied here. Headlines (2026-08-18): (i) R0g alone is engine-independent
(χ 0.4, both) and cannot see the basement; (ii) R0g+R1g is the only combo that fits its data at
χ≈1 *and* recovers the basement step, and only Dinver does it — BayHunter fits R0g at 0.4 and
leaves R1g at χ≈2.5 at both wells (its noise hyper-parameter absorbs the overtone misfit);
(iii) every phase-containing combo has χ ≥ 2 on the phase curve in both engines, and group+phase
of the same wave cannot be co-fitted (χ 3–5) — a measurement inconsistency, see the phase-pick
diagnosis in the session notes (cycle skips + r < 2λ near-field); (iv) Love-only is pathological
(L0g pins vs_max / collapses; L0p wants a faster half-space than Rayleigh does).

**Poisson prior at crustal scale.** With the paper's ν∈[0.2,0.5] the phase-only Basel-1 run
drove Vp/Vs to a median 3.4 (35 % of layers > 4) at Vs 1.4 km/s — soil values in rock — and
pinned the half-space at vs_max; ν∈[0.2,0.35] (Vp/Vs 1.63–2.08) left the group fit unchanged
(χ 1.07/0.98 vs 1.04/0.97) and dropped Vs-bound binding 5.6 % → 0.3 %. (Those phase runs
predate the resample fix above, so their χ values are not quotable; the Vp/Vs behaviour is.)
**The code default is therefore ν 0.2–0.35**: the paper images the upper <100 m in soils, we
image the upper ~5 km in rock, and the prior has to follow the material.
`--dinver-pr 0.2,0.5` reproduces the paper's setting when wanted.

**Per-run diagnostics — `dinver_diagnostics.py`.** One figure per (well, combo) in
`tests/dinver_swinvert/<well>/diagnostics/<combo>.png`, following Vantassel & Cox: (a) Fig 7 min
misfit per parameterization with one marker per trial (spread across seeds = convergence proxy)
and each trial's 100-kept range; (b) Fig 6b lowest-misfit model per parameterization, rejected
greyed; (c) Fig 10c the 100 lowest-misfit per accepted parameterization; (d) Fig 10d σ_ln,Vs for
best/10-best/100-best; (e) pooled Vs(z) density; (f) Fig 4 NA convergence — best misfit vs model
index per trial, from `<run>.report.bestcurve.txt` (`gpdcreport -best-curve`, a few hundred
bytes, now written by the runner before the .report is deleted; runs before 2026-08-19 have none
and the panel shows trial-spread proxies instead); (h) interface-depth histogram; (i) Vp/Vs of
pooled layers; (j) half-space Vs against vs_max; (k) misfit histogram per parameterization;
(g) Fig 8 each target curve ±1σ with the resampled target, best model per parameterization and
the pooled median, with the disba χ. Read (g)+(a)+(k) together: a high misfit floor shared by all
parameterizations with a flat (a) is data inconsistency, not a search failure.

## Chain rejection (BayHunter arms): the absolute Δlog L cut

Every per-cell BayHunter posterior is built from a SUBSET of the chains that ran. Two things
were changed from stock BayHunter; both live in `run_bayhunter_cell.py` +
`noisepy/vs_reliability.py`, and both are visible per cell in the result npz.

**1. The criterion is an absolute log-likelihood gap, not BayHunter's relative `dev`.**
Stock `get_outliers` scores each chain as `1 - median/best` (best > 0) or `1 - best/median`
(best < 0) and cuts at `dev`. The tolerance in real logL units is then `dev * |best|`, so
identical runs get wildly different strictness purely from the likelihood SCALE — across runs
of identical construction here `best` spans −34…+97, making `dev = 0.05` mean anywhere from
0.11 to 4.84 logL, and near `best ≈ 0` it collapses (one run discarded a chain 0.4 logL away
and kept 1 of 16). It also raises `UnboundLocalError` at `best == 0` exactly.
`_use_abs_outlier_cut()` replaces it with

```
keep chain j  ⟺  best_median_logL − median_logL_j  ≤  DELTA_LOGL      (= 5.0)
```

scale-free, and the same 5 logL everywhere. `outliers.dat` records what was dropped and by
how much; `outlier_delta` is stamped in the npz.

**2. Each chain's median is taken over its T=1 samples only** (`_chain_median_at_t1`).
Under parallel tempering a chain's stored log-likelihoods mix every temperature it held
(temperatures SWAP between chains), and a sample drawn at T > 1 comes from a FLATTENED
posterior, so it fits worse by construction. Taking the median over the raw array penalises a
chain for having been hot and drops it as an "outlier". Observed: a PT run kept 11/16 chains
where the identical non-PT run kept 16/16. Stock BayHunter's `get_outliers`
(`Plotting.py:130-133`) has the same flaw. **Strict no-op when PT is off** — BayHunter fills
`sharedtemperatures` with 1.0, so the mask keeps every sample.

**What "kept" means differs with PT, and this is the part to watch.** A chain that never
reached T=1 has no T=1 samples, scores −inf, and is dropped by construction — not because it
fit badly. So with PT on, the eligible pool is not `nchains` but the number of chains that
ever ran cold (`t1chains` pinned + whatever the ladder transported down). The honest
denominator is that eligible pool, which is why `kept_chains_maps.py` always plots the
absolute count, the fraction of the eligible pool, and the eligible pool itself.
`vs_reliability.confidence()` gates on `n_kept ≥ 8 / ≥ 3` — an ABSOLUTE count — so a PT run
with `t1chains = 3` can never exceed `n_kept = 3`-ish and is structurally capped at "medium".

**Downstream.** The kept mask feeds `vs_reliability.assess()`: only kept chains enter the
per-depth Gelman–Rubin-like `rho(z)`, which with the wavelength depth floor gives the
`reliable_mask` / `z_reliable_min..max` window and the cell `confidence`
(`high` / `medium` / `low`) recorded in every npz.

**Status of PT in this project (audited 2026-08-24).** `--parallel-tempering` existed only in
`well_vs_qc.py`; `grid_vs_inversion.py` had no flag for it, so **every production arm on every
network ran with `pt_enabled = 0`** — 3036 npz carrying provenance all say PT off, and no
PT-enabled cell exists on scratch or in the project tree. The flag is now exposed on the grid
driver too (`--parallel-tempering / --t1chains / --maxtemp`, defaults from
`noisepy.pt_defaults`). Treat a npz with NO `pt_enabled` key as "legacy, unknown", never as
"PT off".

### Measured PT on/off contrast (hautesorne, 6 cells, 2026-08-24) — and a gate miscalibration

`grid_vs_inversion.py` run twice on the same cells, identical in everything but
`--parallel-tempering` (t1chains=3, maxtemp=50, 24 chains, 200k+200k):

| cell | PT off: cold/kept/conf | PT on: cold/kept/conf | models off→on | z_rel_max off→on | logL spread off→on |
|---|---|---|---|---|---|
| 10_20 | 24/1 low | 7/6 marginal | 20000→9978 | 4.10→4.10 | 50.0→34.8 |
| 15_35 | 24/23 high | 9/9 **low** | 19987→6611 | 6.00→1.75 | 27.6→**1.9** |
| 20_30 | 24/14 high | 8/6 marginal | 19992→9468 | 6.00→3.15 | 64.3→13.4 |
| 25_25 | 24/14 high | 9/5 **low** | 19992→9286 | 6.00→6.00 | 40.6→8.6 |
| 30_40 | 24/22 high | 9/7 **low** | 19998→8624 | 6.00→2.85 | 29.5→28.4 |
| 35_25 | 24/20 high | 17/6 marginal | 20000→9438 | 6.00→6.00 | 14.9→18.8 |

Net: **5 of 6 cells downgraded, 1 upgraded** (the pathological 10_20).

**PT does what it is supposed to do to the likelihoods** — the between-chain logL spread
collapses (15_35: 27.6 → 1.9; the cold chains land in one basin) and the pathological cell
10_20 goes from a posterior resting on ONE chain to six. **But the downstream gates then
penalise it**, and 4 of 5 cells were downgraded:

1. **Only 7–9 of 24 chains ever reach T=1** in five of six cells (4–16 round trips per run).
   The exception is decisive: 35_25 made **98** round trips and got **17 of 24** cold. So the
   cold pool is set by how well the ladder TRANSPORTS, not by `t1chains`, and at these
   settings it usually transports poorly. The docstring on
   `vs_reliability.confidence()` asserts that "over a full run nearly every chain visits T=1",
   and concludes the `n_kept >= 8` gate "is therefore NOT a function of t1chains, and PT does
   not silently downgrade cells". **Measured, that is false** at these settings: the eligible
   pool is ~1/3 of the chains, so `n_kept >= 8` needs 8 of ~9 eligible chains.
2. **`frac_kept` uses the wrong denominator for PT.** `assess()` computes
   `frac_kept = n_kept / len(loglike_med)`, i.e. over ALL LAUNCHED chains — including those
   that never went cold and were therefore excluded *by construction*, not on fit. With ~9
   cold chains of 24 it is structurally capped near 0.38, so `frac_kept >= 0.5` ("high") is
   unreachable no matter how well the cold chains agree. 15_35 is the clean demonstration:
   9 of 9 eligible chains kept, logL spread 1.9 — and it is labelled `low`.
3. **`reln_frac` drops** because the shorter T=1 record shortens the chain-resolved band
   (15_35 z_rel_max 6.00 → 1.75 km, 30_40 6.00 → 2.85), which fails the `>= 0.5` gate too.
4. **Cost**: ~2–3x fewer posterior models and ~3x the wall time per cell (6.5 → 21 min),
   because the runner drives all chains in lockstep.

**Proposed fix (NOT applied — it changes confidence labels and is the user's call):** score
`frac_kept` against the ELIGIBLE pool (chains with finite T=1 medians), and make the `n_kept`
gate relative to that pool rather than an absolute 8. Until then, read `confidence` on any PT
run as pessimistic, and use `kept_chains_maps.py`'s middle panel (kept ÷ eligible), which
already uses the right denominator. No production volume is affected — every production arm
ran PT off.

**Runs can fail silently, and a short `param_labels` is the symptom.** A dinver run that
aborts (rc != 0) is logged by the runner and skipped; the driver captures runner stdout and
prints it only if the whole cell fails, and the work dir is deleted once the cell succeeds —
so the reason used to vanish. Measured: aargau LVZ cell 20_27 lost `lr1.2` outright (both
trials aborted, leaving 18 kB / 397 kB truncated reports in `$TMPDIR`), producing a
7-parameterization pool biased toward simpler models with nothing in the npz to say why.
The npz now carries `failed_runs` (`<label>_t<trial>:rc<N>`) and `n_failed_runs`; check them
before trusting a cell whose `param_labels` is short. Failures concentrate in the
most-complex parameterization (`lr1.2`, 14 layers) under `--dinver-vs-rev` at deep `dmax`.

## Per-cell well diagnostics — `well_cell_diagnostics.py`

One figure per cell within `--radius-km` (default 0.5) of a deep well (`well_vs_qc.WELLS`):
Vs(z) posterior density of the top-100 models + interface-probability strip (from
`iface_depths` when saved, else from steps in the ensemble profiles); each observed curve with
its posterior-predictive band and χ; per-model misfit distributions per curve and combined;
plus per engine: BayHunter's hierarchical noise-σ posterior (`noise_post`), Dinver's
per-parameterization misfit histogram. Engine-agnostic on the shared schema; works for any
vs_prod3 arm on riehen/aargau/hautesorne and for Dinver arms alike.

**It needs per-model content the production grids do not carry** (Dinver ran `--dinver-lean`;
BayHunter grids never saved ensembles), so near-well cells are re-run into `<arm>_wellcells`
dirs: Dinver without `--dinver-lean`, BayHunter with the new `--save-ensemble` grid flag
(identical priors/budget otherwise; see `~/scratch/vs_dinver/wc_*.sbatch` on bamboo).

**LVZs (velocity reversals) in the Dinver layering.** Every Dinver run up to 2026-08-28 used
the SWinvert notebook convention `par_rev=False`: Vs constrained NON-DECREASING with depth
(swprepost writes `lastParamCondition=true` per layer). BayHunter does not share this — its
`lvz=hvz=maxfrac` permits reversals — and the Basel-1 sonic log contains one (~2.4→1.6 km/s at
0.5–0.8 km), so the constraint is an engine asymmetry to flag in any comparison.
`--dinver-vs-rev` (both drivers → cfg `vs_rev`) drops the condition for Vs and the linked Vp.
Measured at the two Riehen well cells (R0gR1g, standard 8 params, otherwise identical):
`--dinver-vs-rev` improves the data fit at both wells (Basel-1 χ 1.07/0.98 → 0.99/0.86;
Otterbach-2 0.95/0.95 → 0.86/0.70) and removes the systematic slow bias vs the log
(−0.24 → +0.01 km/s at Otterbach-2) with basement depth slightly better (z_bas_err
0.99 → 0.77 / 0.71 → 0.65 km) — but rmse vs the log is unchanged (0.39/0.40), σ_ln,Vs
roughly doubles at 1 km (0.05 → 0.10 / 0.17), runtime doubles (the reversal space slows the
NA), and the LVZ the log actually has (2.4 → 1.6 km/s at 0.5–0.8 km) is NOT recovered — the
group data cannot see it, so the extra freedom mostly buys fit and width, not the true
structure. At the (22,44) demo (10 params incl. LN10/15, top-1000 kept): reversals HALVE the best
misfit (0.773 → 0.401, won by ln10) and the top-100 de-collapses (lr2:71, ln10:26 vs lr2:100)
— but the many-layer fans turn oscillatory (ln15 spanning 0.7–4 km/s at most depths),
σ_ln@1 km rises 0.08 → 0.21, and the fit gain does not translate into log agreement at the
scored well cells. Reversal freedom + many layers = noise-fitting; exactly the
overparameterization SWinvert's rejection criteria exist to catch, but misfit-based rejection
cannot flag models that fit BETTER. Keep `par_rev=False` as the default; use the flag where a
log shows a reversal or when comparing against BayHunter LVZ behaviour, and read σ_ln
accordingly.

## Period-band trimming per measure and wave (GVL-1 sweep, 2026-08-28/29)

`tests/test_2026-08-28_gvl1_R0g_tmax` and `.../test_2026-08-28_gvl1_tmax_arms`: 4 cells at
hautesorne GVL-1, ladder {control, 6.0, 4.83, 3.9 s} on each of the four single-wave arms,
52 runs. `vs_prod3` ran `--criterion none`, i.e. the uniform DECISIONS_v1 band and no per-cell
trim at all, so nothing in production removed a far-field long-period branch.

**Cheap predictor, no inversion needed.** Look at the observed curve's LONG-PERIOD branch (the
one after the Airy minimum — the global argmax is the near-surface peak and tells you nothing).
If it turns over, that arm has the artifact:

| arm | branch peak | fall to Tmax | Tmax verdict |
|---|---|---|---|
| R0g fund group | 4.83–5.12 s | −0.37…−0.42 km/s | **trim to ~4.83 s** |
| L0g love group | 5.12 s | −0.42…−0.51 km/s | trim needed but NOT sufficient — see below |
| R0p fund phase | 9.12 s (last) | 0.00 | **do not trim** |
| L0p love phase | 8.61 s (last) | 0.00 | **do not trim** |

Group turns over, phase does not. That is the driver's own `--alpha 0.5` (group, ~2.5 λ) vs
`--phase-alpha 0.2` (phase, ~1 λ) showing up in the data, and it independently reproduces the
Tmax ≈ 4.83 s that the R0g posterior sweep found.

- **R0g**: dVs(3.5→6 km) −0.66 → −0.12 km/s, χ 1.59 → 1.50, z_reliable 8.0 → 5.2 km. The deep
  LVZ is the untrimmed >5 s far-field band; adopt Tmax ≈ 4.83 s.
- **R0p / L0p**: clean null. No LVZ at any Tmax, no structural change, only 2–3 km of lost
  depth. This is the control that rules out "trimming just deletes inconvenient data".
- **L0g**: **Tmax alone does nothing** — flat ~2.26 km/s two-layer half-space, χ ≈ 3 at every
  Tmax. See `LOVE_GROUP_BAND_PLAN.md`: the Love group failure is a *short*-period mode
  contamination (U ≥ c at 1.2–2.0 s), and it needs the mode gate AND the Tmax cut together.

Practical consequence: a single network-wide `T_valid_max` per (net, measure, wave) in
`param_files/period_ranges_DECISIONS_v1.csv` is the wrong shape for this. The cut belongs on
**group only**, and Love group additionally needs its low end handled.
