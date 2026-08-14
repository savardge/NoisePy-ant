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
