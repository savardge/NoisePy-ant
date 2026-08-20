# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research pipeline for ambient-noise surface-wave tomography: raw miniseed/SAC → cross-correlations
→ FTAN dispersion picks → group-velocity maps → 1D/3D Vs models. Derived from the old NoisePy, but
re-arranged and heavily extended (mode separation, phase velocity, Love, Bayesian Vs inversion).

It is **not** a packaged application: there is no test suite, no linter config, no CI. Scripts are
run by hand or via SLURM; correctness is checked with diagnostic figures and the `compare_*` /
`validate_*` / `*_test.py` analysis scripts in `scripts/picking` (those are experiments, not unit
tests). Do not invent a build/lint/test workflow that doesn't exist.

## Running anything

```bash
export PYTHONPATH=/Users/genevievesavard/Codes/NoisePy-ant
```

`noisepy` is imported as a top-level package but is normally **not** pip-installed — `PYTHONPATH`
pointing at the repo root is the mechanism (`setup.py` exists but is vestigial). Scripts are run
from their own directory (`scripts/picking`, etc.) because they import sibling scripts as modules
(e.g. `grid_vs_inversion.py` imports `profile_vs_inversion`).

### Conda environments (four, deliberately separate)

| env | used for | why separate |
|---|---|---|
| `das-ambient-noise` | everything by default: picking, tomography, figures. Has pycwt, findpeaks, h5py, obspy, swtomotv (editable) | — |
| `bayesbay_dev` | `run_vs_inversion.py` / `grid_vs_inversion.py` drivers | bayesbay needs numpy ≥ 2 |
| `bayhunter` | `run_bayhunter_cell.py` only, launched as a **subprocess** by the drivers above | BayHunter's `numpy.distutils` Fortran build needs numpy < 1.26 |
| base anaconda | `lateral_structure_map.py`, `ffscan_styled_maps.py` fault/geology overlays | only env with geopandas + shapely |

On the cluster the split is the same (`bayhunter_aniso` on bamboo). Scripts that cross the boundary
take `--bayhunter-python /opt/anaconda3/envs/bayhunter/bin/python`.

The third Vs engine, **Dinver** (Geopsy NA, SWinvert workflow — `--engines dinver`,
`grid_vs_inversion.py --engine dinver`), runs in `das-ambient-noise` and needs
`~/Codes/swprepost` on `PYTHONPATH` (not pip-installed anywhere) plus the Geopsy binaries
(`~/Codes/geopsy-install/bin/dinver.app/Contents/MacOS/dinver` locally; `module load
GCC/11.3.0 OpenMPI/4.1.4 geopsy/3.4.2` on bamboo/yggdrasil). See `scripts/picking/VS_INVERSION.md`
"Third engine" — in particular the group-slowness grid-padding note before touching
`noisepy/dinver_target.py`.

## Data lives outside the repo

The repo holds **code and YAML only**. All inputs and outputs live in a per-network project tree,
locally `~/Codes/extract_higher_modes/Projects/<net>/` (`riehen`, `aargau`, `hautesorne`), and on
BeeGFS scratch on the cluster. YAML files in `param_files/` carry absolute paths into that tree —
which is why `param_files/cluster/*.yaml` are machine-generated (`cluster/make_cluster_configs.py`)
rather than hand-edited, and why `cluster/preflight.sh` hard-checks that no `/Users` or `/Volumes`
path survived the rewrite.

## Architecture

```
noisepy/           library — algorithms, no CLI
scripts/<stage>/   drivers — argparse + YAML, one file per step
param_files/       per-network YAML (the only place parameters live)
cluster/           SLURM wrappers for the two heavy campaigns
ant_matlab/        legacy MATLAB inversion (superseded by swtomotv, kept for reference)
vg_maps/           WIP pure-Python group-velocity map inversion (Tarantola–Valette)
```

The pipeline, stage by stage:

1. **`scripts/raw2stack`** — `S0B_to_ASDF.py` (miniseed → pyasdf H5) → `S1_fft_cc_MPI*.py`
   (cross-correlation, MPI) → `S2_stacking.py` (linear/pws/robust stacks + substacks). Core code in
   `noisepy/cross_correlation.py`, `preprocess_h5.py`, `stacking.py`. Output: one `.h5` per station
   pair with 9-component tensor, under `STACK_*/<src>/<src>_<rec>.h5`.
2. **`scripts/postprocess_stacks`** — `extract_ncts.py` (H5 → numpy gathers), `beamform.py`,
   `phaseshift_dispersion.py` (per-virtual-source VSG panels, the input to the reference-curve
   step), `fj_dispersion.py` (frequency-Bessel alternative).
3. **`scripts/picking`** — the bulk of the repo. FTAN + dispersion picking. See below.
4. **Tomography** — `export_tomo_picks.py` / `export_unified_tomo_picks.py` → the external
   `swtomotv` package (Tarantola–Valette), driven here by `run_production.py`. `ant_matlab` is the
   older route.
5. **Vs depth inversion** — `noisepy/vs_inversion.py` + `run_vs_inversion.py` (single cell, two
   engines compared) and `grid_vs_inversion.py` (every cell → 3-D volume).

### `scripts/picking` — read the docs first

This directory has ~140 scripts and its own markdown documentation. **Read these before touching
anything in it**; they record the decisions and the reasons, which are not recoverable from the code:

| doc | covers |
|---|---|
| `README.md` | script inventory, V1→V6 evolution, the mode-separated workflow step by step |
| `UNIFIED_PICKING_SUMMARY.md` | the current unified picker: 8 pick types, CSV schema, QC columns |
| `PARAMETERS.md` | production group-velocity maps — every tomography parameter and its rationale |
| `VS_INVERSION.md` | Bayesian Vs inversion, the two engines, macOS gotchas, noise-prior tuning |
| `VSG_REFERENCE_METHODOLOGY.md` | how the data-derived reference curves are built |
| `MODE_OSCULATION_NOTES.md`, `RADIAL_ANISOTROPY_PLAN.md`, `CONTINUOUS_ZETA_PLAN.md` | in-progress method notes |

Generations of pickers coexist on purpose (`dispersion_curves.py` → `_V2` → `_V5` →
`_V6_modesep.py` → `dispersion_unified.py`). Older versions are kept because published/cluster runs
used them — **do not delete or "consolidate" them**. New work goes through
`noisepy/unified_picking.py` + `dispersion_unified.py`, orchestrated per network by
`run_pipeline.py --stage pick|qc|export|all --config param_files/pipeline_<net>.yaml`.

## Conventions that matter

- **Config precedence is CLI flag > YAML > built-in default**, implemented in
  `scripts/picking/modesep_config.py` (`load_config`, `apply_overrides`). Parameters belong in
  `param_files/*.yaml`, never hard-coded in a driver. `run_pipeline.py` maps YAML keys to
  downstream CLI flags via explicit `QC_FLAGS` / `EXPORT_FLAGS` dicts — extend those when adding a
  knob.
- **Batch scripts are skip-if-exists and resumable at per-item granularity.** A job killed by a
  walltime limit is fixed by resubmitting the identical command. Preserve this when editing a batch
  driver; write to a per-PID temp name and `os.replace` into place so a killed task never leaves a
  partial file.
- **Old CLI forms are kept working.** `dispersion_batch_modesep.py` and `dispersion_unified.py`
  still honour the legacy positional `stack_root out_root [nproc]` + `DISP_NET` / `DISP_LIMIT` /
  `DISP_REF_DIR` env invocation when `--config` is absent, because `dispersion.slurm` uses it.
- **Pin BLAS to one thread** (`OMP_NUM_THREADS=1` etc.) in anything using `multiprocessing.Pool` or
  SLURM tasks — each worker already owns its core, and oversubscription is the usual slowdown.
- **QC is applied at merge time, not in the picker.** The picker emits discriminator columns
  (`snr_nbG`, `xmode_amp`, `ot_flag`, `env_ratio`, …); `qc_unified_picks.py` thresholds them. Keep
  new quality metrics on that side of the line so thresholds can be re-swept without re-picking.
- Output directories are labelled by settings (`qc_<label>/`) with a `qc_current` symlink
  repointed by `run_pipeline.py`. Downstream scripts read `qc_current`.

## Physics constraints encoded in the code

Getting these wrong produces plausible-looking but wrong maps, so they are worth knowing:

- Rayleigh mode separation uses the Nayak & Thurber (2020) ±π/2 component synthesis (`G_LR0` /
  `G_LR1`) — it needs two orthogonal components, so **Love (single-component SH) has no analogue**
  and separates only by FTAN ridge topology against reference curves.
- Phase measurement is wave-agnostic: the +π/4 stationary-phase shift applies to Love (TT) exactly
  as to ZZ/RR. The ±π/2 / +3π/4 offsets exist only for the RZ/ZR cross components.
- **Phase velocities of modes cannot cross; group velocities can** (near Airy minima). "Overtone
  slower than fundamental" is therefore a flag, never an automatic rejection — the decisive test is
  mutual suppression in the other mode's image (`xmode_amp`).
- `DISP_VMIN` / `vmin` is a **grid floor, not a quality gate**. Gate on SNR.
- The tomographic formal σ underestimates true uncertainty; both Vs engines run with a wide
  hierarchical noise prior. Feeding the formal σ in tight makes the trans-D inversion overfit.

## Cluster campaigns

Two independent SLURM setups, each self-documented:

- `cluster/` — ts-PWS stack build + jackknife + finalize on yggdrasil. `env.sh` is the single source
  of paths; run `sh preflight.sh` before submitting; stages selected with
  `--export=ALL,STAGE=build|jackknife|finalize` (the `ALL` matters).
- `cluster/bamboo_vs/` — the per-cell Vs inversion campaign (3 networks × 6 configs). `env_bamboo.sh`
  holds paths and the (net, cfg) → flags tables; `./smoke_test.sh` before `./submit_all.sh <net>`.

Shards are strided (`files[I::N]`), not contiguous, so expensive pairs spread evenly across tasks.
