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
