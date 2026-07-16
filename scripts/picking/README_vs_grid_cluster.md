# Cluster Vs-grid inversion (far-field picks, production convergence)

Runs the two full Vs(z) grids at the **production convergence standard
`--n-chains 10 --iter-burnin 200000 --iter-main 100000`** — too heavy for the
macOS box (chains iterate serially there: ~30 min/cell, ~5-6 days for both
grids), fast on the cluster where the array fans cells out across nodes.

The vg tomography (swtomotv) is already done and is NOT redone here — only the
BayHunter Vs depth inversion.

## What runs

`vs_grid.slurm` is a SLURM **array** job. Each array task `i` of `N_SHARDS`
inverts the disjoint cell slice `tasks[i::N_SHARDS]` via
`grid_vs_inversion.py --shard i/N_SHARDS`, using the validated per-cell runner
`run_bayhunter_cell.py` (serial chains, BLAS pinned to 1 thread). Within a task,
`--n-workers = cpus-per-task` cells run in parallel. All shards write uniquely
named files into the shared `<outdir>/cells/`; a final `--assemble-only` pass
stitches them into `volume_fund.npz` / `volume_fundot.npz`.

Why array-shard and not BayHunter `mp_inversion` (parallel chains within a
cell): `mp_inversion` uses a `multiprocessing.Manager` + shared arrays that the
project memory documents as deadlock/crawl-prone (it's why the runner iterates
chains serially). Array-sharding parallelises over **cells** instead of chains
and reuses the known-good runner untouched, for the same wall-time win. If you
do want intra-cell `mp_inversion`, see the last section.

## Steps

1. **Stage data to the cluster.** The `Projects/<net>/tomo/` tree must exist on
   the cluster with, per network:
   - `swtomotv-output-*-farfield*/production/` (the vg maps — cheap to rsync)
   - `<net>_swtomotv_*_farfield*.yaml`
   - the reliability inputs `period_resolution` reads (station `xstat/ystat`
     cache under the production cache dir; `ref_*_phase.txt` under `vsg_modesep`)
   rsync example (run from macOS):
   ```
   rsync -av Projects/riehen/tomo/swtomotv-output-200m-farfield \
             Projects/riehen/tomo/riehen_swtomotv_200m_farfield.yaml \
             Projects/riehen/tomo/vsg_modesep \
             <cluster>:.../Projects/riehen/tomo/
   ```
   (same for aargau with `-500m-farfield-v2` + `aargau_swtomotv_500m_farfield_v2.yaml`)

2. **Edit `vs_grid.slurm`** — every `# >>> EDIT` line: cluster `REPO`, `PROJ`,
   the two conda-env pythons, the module-load/activate block, and `NET`.
   Confirm the `bayhunter` env exists on the cluster (py3.10, numpy1.24,
   BayHunter_Aniso installed, disba, gfortran) — same recipe as VS_INVERSION.md.

3. **Size the array.** `#SBATCH --array=0-(N_SHARDS-1)` and `N_SHARDS` must
   match. Runs per shard ≈ total_runs / N_SHARDS; wall ≈
   (runs_per_shard / cpus-per-task) × ~30 min. Targets that fit an 11 h limit:
   - Riehen: 770 cells × 2 = **1540 runs**. N_SHARDS=30, cpus=8 → ~51 runs/shard
     → ~3.2 h/shard.
   - Aargau: ~865 cells × 2 ≈ **1730 runs**. Same 30×8 → ~3.6 h/shard.
   Add `%` concurrency if the queue is busy, e.g. `--array=0-29%15`.

4. **Submit, once per network** (set `NET` inside the file, or override):
   ```
   mkdir -p outslurm
   sbatch vs_grid.slurm                      # NET=riehen (default in file)
   # edit NET=aargau (or: sbatch --export=ALL,NET=aargau vs_grid.slurm)
   ```
   Resumable: a re-submitted array skips cells whose `cells/cell_*.npz` exist.

5. **Assemble when all shards finish** (cheap, login node or a 1-cpu job):
   ```
   $DRIVER_PY grid_vs_inversion.py --production $PROD --config $CONFIG \
       --outdir $OUTDIR --net $NET --criterion physical --assemble-only \
       --bayhunter-python $BH_PY --bayhunter-runner run_bayhunter_cell.py
   ```
   → `volume_fund.npz`, `volume_fundot.npz` in `$OUTDIR`.

6. **Pull `$OUTDIR` back to macOS**, then locally:
   ```
   python grid_vs_postprocess.py --net <net> --griddir <outdir>
   python smooth_maps.py         --net <net> --griddir <outdir>
   ```
   and the before/after LVZ comparison (`lvz_before_after.py` in
   Projects/azimuthal_source_bias/) picks up the new `farfield` volumes.

## Sanity check the convergence actually improved

Each cell writes `<out_npz basename>_diagnostics.png` + a `convergence` dict
(chain_disagree, frac_chains_ok, burnin_delta_frac) via `bh_diagnostics`. Spot
-check a few deep/edge cells: `chain_disagree` should be well below the ~0.8-1.0
seen with the under-converged 4×80k settings.

## Alternative: intra-cell mp_inversion (parallel chains)

If you prefer BayHunter's native parallel chains instead of array-sharding over
cells: set `cpus-per-task = n-chains`, keep `--n-workers 1`, and switch the chain
loop in `run_bayhunter_cell.py` (the `for k, chain in enumerate(opt.chains)`
block) to `opt.mp_inversion(nthreads=<n-chains>, ...)` guarded by an env flag so
macOS keeps the serial path. Untested on this cluster — validate on one cell
(compare its posterior to a serial-chain run of the same cell) before a full
grid. The array-shard path above is the recommended default.
