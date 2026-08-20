# bamboo_vs — per-cell Vs inversion campaign (vs_prod3)

Slurm job arrays on bamboo (login1.bamboo.hpc.unige.ch) inverting every well-covered
tomography cell of the `tspws_{group,phase}_blanket_*_prod3_k3` velocity maps for a 1D Vs
profile with the BayHunter_Aniso fork. Three networks (riehen, aargau, hautesorne) × six
configs:

| CFG | targets | maps |
|---|---|---|
| `R0g` | Rayleigh fund group | group run |
| `R0p` | Rayleigh fund phase (phase-only) | phase run |
| `L0g` | Love fund group | group run |
| `L0p` | Love fund phase (phase-only) | phase run |
| `RLg_radial` | R+L fund group, continuous-ζ radial anisotropy | group run |
| `RLg_iso` | R+L fund group, isotropic | group run |

Fixed campaign choices (2026-08-13): 24 chains, 500k burn-in + 500k main per chain; free
(hierarchical, unbounded) noise; period bands from
`inputs/period_ranges_DECISIONS_v1.csv` (uniform across cells, `--period-ranges`); per-period
cell participation = the production map `mask` (what the styled maps display); no
`--criterion` trim, `--phase-tmin 0` (the CSV alone governs the phase band).

## Layout

- `env_bamboo.sh` — every path + the conda activation + the (net, cfg) → flags tables.
- `vs_prod3.sbatch` — one array task = one shard of cells for one (NET, CFG); `STAGE=assemble`
  stitches `cells/` into `volume_<waveset>.npz`.
- `submit_all.sh` — full campaign (or one net / one (net,cfg)); each array gets a dependent
  assemble job (`afterany`, so partial timeouts still stitch what finished). Writes a
  manifest TSV under `$SCRATCH_VS/runs/`.
- `smoke_test.sh` — 1 cell × 6 configs, 3 chains × 7k iters, debug-cpu, output to
  `$SCRATCH_VS/smoke/` (never the production tree).

Data + outputs live on BeeGFS scratch: `/srv/beegfs/scratch/users/s/savardg/vs_prod3/`
(`inputs/`, `runs/<net>/<cfg>/{cells,work,volume_*.npz}`, `logs/`, `smoke/`).

## Sizing

24 cpus/task, one cell at a time, all 24 chains forked concurrently (`--use-mp
--mp-nthreads 24`): fewer threads than chains would run chains in sequential waves — same
core-hours, longer per-cell wall, more `--cell-timeout` exposure. 32 shards per (net,cfg),
throttled `%8`. Shard arrays are resume-safe (skip-if-exists), so a timed-out array is just
resubmitted with the same command.

## Order of operations

1. `./smoke_test.sh` → check `$SCRATCH_VS/smoke/riehen/*/cells/*.npz` (phase configs carry
   `obsT_*_phase` only; `RLg_radial` carries `gamma_*`/`zeta_*`; `RLg_iso` has them NaN).
2. `./submit_all.sh riehen`, sanity-check ETA + a few cells, then
   `./submit_all.sh aargau` and `./submit_all.sh hautesorne`.
3. Volumes appear as `runs/<net>/<cfg>/volume_<waveset>.npz` when the assemble jobs fire.

The conda env (`bayhunter_aniso`, py3.10/numpy1.24 + BayHunter_Aniso fork + disba +
swtomotv) was built by `~/vs_prod3_setup/build_env.sbatch`; verify with
`python ~/BayHunter_Aniso/check_deploy.py` and the `LoveDispersionGroup` import.

## Dinver (SWinvert) arm — `dinver_vs.sbatch`, `submit_dinver.sh`, `smoke_dinver.sh`

Sibling of the vs_prod3 campaign, same sharding/resume semantics, outputs under
`~/scratch/vs_dinver/{runs/<net>/<cfg>/cells,logs}` (inputs symlinked to vs_prod3's — no copy).

```
./smoke_dinver.sh riehen R0gR1g      # 1 cell, ns=500, debug-cpu, writes to smoke/ only
./submit_dinver.sh riehen R0gR1g     # 32 shards x 24 cpus, %8 throttle, + dependent assemble
```

- Engine: `module load GCC/11.3.0 OpenMPI/4.1.4 geopsy/3.4.2` (`activate_dinver`), `dinver -j 1`,
  runner in `bayhunter_aniso` (numpy/disba + `pip install swprepost`, done on a compute node).
- One task = 24 cells running concurrently, each walking 8 params x 3 trials x 60 000 models
  (~2.7 core-h/cell locally). Riehen R0gR1g = 4355 cells ≈ 12 000 core-h; a 32-shard array of
  ~136 cells at 24-wide is ~1 day of wall per shard.
- **Strict-minimum outputs**: `--dinver-lean` (percentile-only npz, ~50 KB/cell, no ensembles),
  reports deleted after the best-100 extraction, per-cell work dir removed on completion.
  Transient disk per running cell ≤ 1 report (~560 MB) → ≤ ~13 GB per task.
- Sizing: `--dinver-size-phase-root $PHASE_PROD` sizes the SWinvert layering from the
  fundamental PHASE λ without inverting phase (group U·T would put dmax ~30 % too shallow).
- Priors per network are in the sbatch (`DMAX`/`VSMAX`); ν 0.2–0.35 is the runner default.
- **First submission post-mortem (2026-08-19).** dinver streams a ~560 MB `.report` per run;
  with 192 cells writing to BeeGFS a 3-layer run took 1120–1320 s (vs 325–355 s locally), every
  cell hit the 6 h `--cell-timeout`, and the driver deleted the work dir on timeout — 1150
  core-h for 3 cells. Measured A/B on one node, same run: `/tmp` 509 s vs BeeGFS 642 s
  (uncontended). Fixes now in `dinver_vs.sbatch`: `--dinver-report-dir $TMPDIR` (transient
  report on node-local disk, only the ~100 KB caches touch scratch), `--work-tag cell` + keep
  the work dir on timeout/error (per-(param,trial) resume across submissions — so **no booster
  arrays on this arm**), `CELL_TIMEOUT` 24 h as a soft bound. Expect ~520 s/run, ~3.5 h/cell,
  ~23 h per 24-cpu shard for Riehen. A second cause, `overtone: resampled target is invalid`,
  was the log-λ resample producing unordered frequencies on steep group curves even when λ(T)
  is monotone — `to_modal_target` now validates and falls back to log-frequency; verified on
  all 8 601 curves of the 4 355 Riehen cells.
