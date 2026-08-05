# ts-PWS production chain on yggdrasil

Moves the pipeline that was running on the laptop to Slurm. Produces, per network, the
production pick tables: **tf-PWS stacks + substack-jackknife sigma**.

Everything is **skip-if-exists** at per-pair granularity, so any job that hits a time limit
is fixed by resubmitting the identical command — nothing is recomputed and nothing is
corrupted. Output files are written to a per-PID temp name and `os.replace`d into place, so
a killed task never leaves a partial `.h5`.

## Files

| File | Purpose |
|---|---|
| `env.sh` | The only place paths live. Edit the three variables at the top. |
| `make_cluster_configs.py` | Regenerates the six YAMLs from the laptop originals (path-only rewrite, so picking parameters stay bit-identical). |
| `preflight.sh` | Readiness check — Slurm limits, code, env, paths, inputs, per-pair cost. |
| `riehen.sbatch` `aargau.sbatch` `hautesorne.sbatch` | One per network; three stages selected by `STAGE`. |
| `../param_files/cluster/*.yaml` | Generated configs (do not hand-edit). |

## What to copy to the cluster

- `NoisePy-ant/` (code + `param_files/`, including `param_files/cluster/`) → `$NOISEPY`
- Nothing else is required. The substacks are already on BeeGFS, and every output is
  rebuilt on the cluster.
- **Optional shortcut:** copying the finished laptop trees skips work already done —
  `Projects/riehen/stacks_tspws` (1.4 GB, complete) and the two
  `substack_jackknife_k2` trees (75 MB + 118 MB) → `$PROJ/<net>/`. Skip-if-exists then
  makes those stages near-instant. Rebuilding them on the cluster is also fine.

## Step 1 — set paths

Edit the top of `env.sh`:

```sh
NOISEPY=/srv/beegfs/scratch/shares/cdff/savardg/NoisePy-ant
PROJ=/srv/beegfs/scratch/shares/cdff/savardg/extract_higher_modes/Projects
CONDA_ENV=das-ambient-noise
```

If you change them, regenerate the configs so the YAML paths agree — `preflight.sh`
verifies that they do:

```bash
python make_cluster_configs.py --noisepy <NOISEPY> --proj <PROJ>
```

The generator also sets `picker.nproc` (default 32). The picker reads its process count
from the YAML, **not** from `SLURM_CPUS_PER_TASK`, so this must match the
`--cpus-per-task` you submit `finalize` with or the extra cores sit idle. Change both
together: `--nproc 16` here and `--cpus-per-task=16` there.

## Step 2 — check everything is ready

```bash
cd $NOISEPY/cluster && mkdir -p logs && sh preflight.sh
```

It exits non-zero if anything blocking is wrong, and prints six sections:

1. **Slurm limits** — your association/QOS CPU ceiling, `MaxSubmit`, `MaxArraySize`, and
   the partition time limits. This answers how wide the arrays may be.
2. **Code + environment** — every script it will call, and the imports
   (`numpy scipy h5py pycwt pandas matplotlib yaml`) plus a headless-matplotlib check.
3. **Output tree writable** and its free space.
4. **Configs agree with `env.sh`** — including a hard check that no `/Users` or `/Volumes`
   laptop path survived.
5. **Input substacks** — pair count per network and how many stacks already exist.
6. **Per-pair cost probe** — times a real `tf_pws` on three pairs per network so you can
   re-derive the array widths below on actual hardware.

**Sizing from the probe:** `tasks x cpus-per-task ≈ pairs x s_per_pair / target_seconds`.
Then set the `%N` throttle so `N x cpus-per-task` stays under your CPU ceiling.

## Step 3 — submit (per network)

```bash
mkdir -p logs
B=$(sbatch --parsable --array=0-49%25 riehen.sbatch)
J=$(sbatch --parsable --array=0-49%25 --export=ALL,STAGE=jackknife --dependency=afterok:$B riehen.sbatch)
sbatch --dependency=afterok:$J --export=ALL,STAGE=finalize \
       --partition=shared-cpu --time=12:00:00 --cpus-per-task=32 --mem=64G riehen.sbatch
```

Same for `aargau.sbatch` and `hautesorne.sbatch`.

**Walltimes are deliberately generous (12 h, shared-cpu's ceiling).** Over-requesting only
costs backfill priority; under-requesting ends tasks as TIMEOUT and drops every `afterok`
dependent, which is far more expensive to recover from.

`--export=ALL,STAGE=...` matters: without `ALL` the job loses your environment.

### Starting sizes (verify against the probe)

| Network | Pairs | Build array | Build walltime | Notes |
|---|---|---|---|---|
| riehen | 19,503 | `0-49%25`, 8 cpus | 2 h | Already built locally — copy the tree and this is a no-op. |
| aargau | 20,376 | `0-49%25`, 8 cpus | 4 h | ~2x riehen's cost per pair (longer substack blocks). |
| hautesorne | ~232,600 | `0-199%50`, 8 cpus | 12 h | Cost per pair **unmeasured** — see the warning below. |

> **Haute-Sorne is the one real unknown.** Its stacks are `STACK_coh_60s_substacks`, and if
> 60 s substacks means far more windows per pair than Riehen's ~4500 s ones, the cost per
> pair rises in proportion. Section 6 of the preflight prints the actual windows/pair —
> check it before submitting the full array, and scale the width or `--pre-block` up if the
> number is large. `--pre-block K` averages K windows into one PWS element and cuts the
> transform count by K.

## Stages

| STAGE | What runs | Parallelism |
|---|---|---|
| `build` (default) | `build_tspws_stacks.py --shard I/N` → `$PROJ/<net>/stacks_tspws` | array, strided shards |
| `jackknife` | `substack_jackknife.py --k 2 --shard I/N` → `$PROJ/<net>/substack_jackknife_k2` | array, strided shards |
| `finalize` | `run_pipeline.py --stage all` (pick → QC → export), then `attach_substack_sigma.py`, then the exported-pick and sigma-census figures | single job, `--cpus-per-task` |

Shards are **strided** (`files[I::N]`), not contiguous, so expensive pairs spread evenly
instead of piling into one task.

## Checking progress and results

```bash
squeue -u $USER
grep -h "done:" logs/riehen_*.out | tail            # per-task build tallies
find $PROJ/riehen/stacks_tspws -name '*.h5' | wc -l # against the preflight pair count
```

After `finalize`, the deliverables per network are:

- `$PROJ/<net>/tomo/1_velocity_maps/inputs_tspws/picks_{fund,overtone,love}_uni.csv`
  — `std` holds the jackknife sigma, with `sigma_src` recording `jackknife` vs fallback
- `.../picks_hist2d.png` — exported picks, linear period axis
- `$PROJ/<net>/substack_jackknife_k2/sigma_census.png` — the uncertainty census

Confirm the sigma actually attached (coverage was ~95-96% on the laptop):

```bash
python -c "import pandas as pd,sys; d=pd.read_csv(sys.argv[1]); \
print(d.sigma_src.value_counts()); print('median sigma', d['std'].median())" \
  $PROJ/riehen/tomo/1_velocity_maps/inputs_tspws/picks_fund_uni.csv
```

## Gotchas carried over from the laptop run

- **Killing a build does not kill its workers.** `pkill -f build_tspws_stacks` misses the
  pool workers, whose command line reads `multiprocessing.spawn`. Under Slurm use
  `scancel` on the job id, which cleans up the whole step.
- **`u_bin` / group-scale dedupe are OFF** in these configs, deliberately.
- **`DISP_VMIN=0.2` is a grid floor, not a quality gate** — QC re-truncates, and the
  configs set `vbounds_fund: "0.2,5.0"` to match. Gate on `snr_nbG`.
- **Haute-Sorne Love overtone stays off** (`love_overtone: false`).
