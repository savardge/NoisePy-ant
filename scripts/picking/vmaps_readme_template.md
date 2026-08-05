# @@NET@@ — surface-wave velocity maps

Workflow layout (2026-08-05). Each numbered directory is one stage; a run directory keeps
its original `tspws_<measure>_<Cd>_dx<dx><tag>` name so provenance stays in the name.
`RUN_LINEAGE.md` (generated, do not hand-edit) lists every run with parameters read from
its own npz files.

## Layout

| dir | contents |
|---|---|
| `0_inputs/exported_picks_tspws/` | exported pick tables (ts-PWS stacks + substack-jackknife sigma) + `stations_all.csv` — the common source for all workflows |
| `0_inputs/culled_picks_vbounds/` | track-vbounds culled trees `k2/ k3/` (+ `k_none..k5`, sweep figures). **k3 feeds the current production.** `k_none` is symlinks to the exported picks |
| `0_inputs/culled_picks_hist2d/` | p75/p80/p90 histogram-cell culls — REJECTED experiment, kept for the record |
| `0_inputs/configs/` | dataset YAMLs (bounds, DEM, tecto layers) used by styling/analysis |
| `1_production/` | **current maps: `*_prod3_k3`** (+ `_prod3_k2` comparison). Corrected Fresnel, vbounds k-cull, NO output masking |
| `2_superseded/` | every earlier series, each with the reason it was superseded (below) |
| `3_diagnostics/` | investigation figures (riehen: graben-west diagnosis) |
| `production` (symlink) | compatibility shim → `1_production`, so older commands and cluster rsync targets still resolve |
| `tests/ evidence/ _archive/` | pre-existing families from the 2026-07-16 reorg (riehen/aargau only); each carries its own provenance README — untouched by the 2026-08-05 reorg |

Cross-network diagnostics (se-wall sweeps, cd_scale census) live at `Projects/_se_wall_sweep/`.

## Provenance

Every run family carries a `PROVENANCE.md` answering: how the pairs were picked (and
where the picks are), how they were QC'ed and with which thresholds, how the inversion
was configured, and how the maps were post-processed:
`0_inputs/exported_picks_tspws/PROVENANCE.md` (shared picking/QC/export/sigma chain,
with the authoritative QC parameter record in `QC_provenance/`),
`1_production/PROVENANCE.md`, and one per `2_superseded/<family>/`.

## Which run to use

**The Cd recommendation differs by measure.** From the 2026-08-05 quantitative comparison
(3 nets × 2 k × 3 Cd × 2 measures × 3 waves × all periods = 3891 map-level records;
`Projects/_inversion_comparison/`, tool `compare_inversion_choices.py`):

| measure | use | why |
|---|---|---|
| **group** | `scaled` | best geology correlation in **78%** of paired period-cases (median Δeta² = +0.011 vs blanket); amplitude, feature size and period continuity all within ~4% of the alternatives — Cd barely matters for group |
| **phase** | `blanket` | blanket wins on **83%** of paired period-cases (median Δeta² = +0.032). scaled/measured **inflate amplitude 51%** (6.4→9.7% pooled), **shrink features 21%** (3.78→2.98 km) and are **~2× rougher** period-to-period — the signature of amplified noise, driven by the group-derived proxy sigma (bug 8) |

So: **`1_production/tspws_group_scaled_dx@@DX@@_prod3_k3`** for group,
**`1_production/tspws_phase_blanket_dx@@DX@@_prod3_k3`** for phase.

Phase continuity, median per-cell roughness [% of cell mean V], and the fraction of cells
exceeding 5% roughness — the sharpest single discriminator, and independent of any geology
assumption:

| phase wave | blanket | measured | scaled |
|---|---|---|---|
| fund | **0.82** | 1.42 | 1.73 |
| love | **0.89** | 1.62 | 1.93 |
| overtone | **2.01** (0.1% of cells >5%) | 3.67 (9.4%) | 3.58 (7.2%) |

**Cull k.** k=3 gives ~5% more amplitude, ~3% smaller features, ~5% lower eta² and 5–9%
rougher dispersion curves than k=2 — an order of magnitude smaller than the Cd effect on
phase. k=3 is kept for the 2–3 s structural band; k=2 is defensible if period continuity
matters more than amplitude.

**Geology signal peaks at 1–2 s** (eta² 0.18–0.29 for fund, vs 0.03–0.05 below 1 s). Riehen
group fund, 1–2 s, blanket: Mesozoic carbonate **+11.4%**, Tertiary molasse **−11.2%**,
Quaternary −3.9% — a 22-point spread in the geologically expected direction. Haute-Sorne
shows the INVERTED contrast (carbonate ~0%, molasse +2.2 to +3.8%) in **all six** Cd/measure
combinations, so it is a property of the data, not of an inversion choice (see
`3_diagnostics/` where topographic and fold-damage explanations were tested and rejected).

### Caveats on how these recommendations were reached

- **eta² is scored against SURFACE geology (GK500) while the waves sample the top ~1 km.**
  It is a proxy for "does the map see real structure", not ground truth. A map could
  legitimately disagree with surface lithology at depth — which is exactly the Haute-Sorne
  case. This is why period continuity was weighted alongside it.
- **Amplitude and feature size cannot separate real small-scale structure from noise on
  their own.** The phase scaled/measured verdict rests on the CONJUNCTION of higher
  amplitude + smaller features + lower eta² + worse continuity; no single one of those
  would be sufficient.
- **The geology classes are a keyword aggregation** of `LEG_GEOL` into
  Quaternary / Tertiary molasse / Mesozoic carbonate / crystalline
  (`velocity_by_geology.COVER`). The tectonic levels do NOT work for this: `LEG_TEC_2`
  classifies by structural domain and puts the whole Delémont basin inside "Jura interne".
- **The paired period-cases are not independent samples.** Adjacent CWT rungs are
  correlated, so "78%" and "83%" describe consistency across the band, not an n=180
  significance test.
- **Continuity assumes the true per-cell dispersion curve is smooth.** A genuine sharp
  change with period would be penalised as roughness.
- **The decorrelation length is mask-normalised but not independent of the prior**: LC sets
  a floor on achievable feature size, so it measures "size given this prior", not intrinsic
  resolution. All runs compared here share the same LC scheme, so the comparison is fair
  even though the absolute value is not an independent resolution estimate.
- All metrics use the RELATIVE anomaly `100·(V−med)/med`, so they are blind to a
  period-wide DC offset — deliberate (the dispersion trend would otherwise dominate) but it
  means a systematically fast/slow map is not penalised here.

### Supporting evidence in `Projects/_inversion_comparison/`

| artefact | what it holds |
|---|---|
| `per_period_metrics.csv` | 3891 rows: amplitude, 1/e feature size, geology eta², per-class anomalies, var_red, N — one row per net × k × Cd × measure × wave × period |
| `continuity_metrics.csv` + `cell_roughness.npz` | per-run roughness summary and the full per-CELL roughness field |
| `per_period_<net>_<k>.png` | the three metrics vs period, Cd overlaid, 1–2 s band shaded (6 figures) |
| `k3_minus_k2.png` | the cull effect isolated, all nets |
| `continuity.png`, `geology_1to2s.png` | continuity summary; 1–2 s per-class anomalies |
| `pick_vs_cell_hist2d/` | **36 figures**, `<net>_<measure>_<wave>_{k2,k3}.png`: input-pick vs map-cell period-velocity distributions, one row per Cd, each period column normalised to sum 1. The direct view of what the inversion did to the data — the phase scaled/measured runs show a tail the picks do not have, and every run shows the map's low-velocity floor (Riehen group fund: picks reach ~0.4 km/s, cells stop near 1.0) |

A caveat specific to those last figures: column-normalising each period makes SHAPES
comparable but discards the count difference (~77k picks vs ~4.7k cells/period), so they
say nothing about how well-determined any given period is — read `N` from
`per_period_metrics.csv` or the run's `production_<wave>.csv` for that.

## Bug & issue ledger (why each older series was superseded)

Every issue below was found and characterised 2026-08-03 → 2026-08-05; details in the
session's memory notes and in the docstrings of the scripts named.

| # | issue | affected | status |
|---|---|---|---|
| 1 | **Fresnel half-width √2 too large** — `lc_for()` used `sqrt(lam*L/2)` (full-wavelength detour = 2nd Fresnel zone); Yoshizawa & Kennett 2002 eq 24 gives `sqrt(lam*L)/2`. LC therefore √2 too wide wherever the floor didn't bind; `--lc-max 8` cap saturated 9–39% of periods | prod1, `_lccov`, `_lcinfl`, `_lcfix1`, `_histfilt` | fixed in `run_production.py` 2026-08-03; corrected runs = prod2+, showing ~15–20% less apparent structure at unchanged var_red (the surplus was prior-enabled) |
| 2 | **Pick-histogram veil** (`--vplaus dens:0.02`) masked output cells outside the pick-histogram support. Removed 111–144 N-Aargau cells/period — 100% one real unit (Plate-forme mésozoïque épivarisque), flagged purely for being fast; bit hardest where picks are densest | `_lccov`, `_lcinfl`, `_histfilt`, prod2 | disabled 2026-08-04 (`--vplaus off` hardcoded in sbatch); veiled values preserved in `vel_hidden`, merged back by current `styled_maps.py` |
| 3 | **Resolution rank-cut** (`--res-drop-q 0.25`) discarded the worst-resolved QUARTER of cells by rank; res_diag is speckly → scattered interior holes, ~33% of area lost (371–455 cells/period, Aargau phase) | everything before prod3 | set to 0 in prod3 (2026-08-05); `styled_maps.py` also closes interior speckle from `vel_full` when restyling old runs |
| 4 | **hist2d p75 pick cull REJECTED** — removes 56% of the Riehen graben signal (crossing-path slope −0.84→−0.37), deletes period rungs (love 34→24), crashed 2 scaled-Cd overtone inversions. Only helpful below ~1.5 s | `_histfilt` runs, `0_inputs/culled_picks_hist2d` | rejected; use vbounds k=3 (keeps 99% of the signal at 2.3% pick cost) |
| 5 | **Hillshade relief inverted** — `LightSource.hillshade` assumes row 0 = image top; with `origin='lower'` the light came from the SW and valleys read as ridges | all styled maps made before 2026-08-03 | fixed (`dy` negated in `swtomotv/products/figures.py`); everything restyled since |
| 6 | **Colormaps** — `jet_r` replaced by `inferno`; anomaly maps use RdBu with **red = slow, blue = fast** (user convention) | old figures | fixed 2026-08-03 |
| 7 | **scaled-Cd fixed-point runaway** — on thin data (N ≪ ncell) the χ² rescale diverges (cd_scale → 1e-23) and kills the Cholesky. Detect by NON-CONVERGENCE / var_red→1, never by magnitude (a σ×[0.2,5] clamp would wrongly flag 4.9% of maps, mostly legitimate phase-sigma recalibrations) | any thin wave under `--cd-mode scaled`; crashed twice in `_histfilt` | OPEN — guard designed & tested (dof-normalised χ², damped update, convergence flag, σ×[0.02,50] backstop), not yet implemented |
| 8 | **Phase sigma is a group proxy** — QUANTIFIED 2026-08-05: phase under scaled/measured inflates amplitude 51%, shrinks features 21%, halves period-continuity and loses 37% of the geology correlation vs blanket (see "Which run to use"). Root cause — phase tables carry group-derived jackknife sigma ("absolute scale NOT calibrated for phase" per their meta); under scaled/measured Cd, phase runs have negative var_red on ~11% (HS) to ~30% (Riehen) of periods | all `phase_scaled` / `phase_measured` runs | OPEN (sigma uncalibrated); MITIGATED — use `phase_blanket` |
| 9 | **`vel_full` NaN at longest periods** — e.g. Aargau phase T=6.09 s: 64–94 interior cells have no inversion value at all (not a masking issue) | longest 1–2 periods, phase mostly | OPEN |
| 10 | **Context, not bugs**: prior width se=0.025 is AT its stability ceiling (fraction-outside-physical-band criterion); resolution is ~19 effective dof, invariant from 1089 to 9216 cells → dx=@@DX@@ km kept for comparability, not resolution; two silent-transpose traps (`vec_to_map` returns (nx,ny); `xy2ll` returns (lat,lon)) are documented in the analysis scripts | — | — |

## Cluster note

The cluster tree (`yggdrasil:~/extract_higher_modes/...`) keeps the OLD flat layout —
`production/tspws_*`. The local `production` symlink → `1_production` keeps old-style
rsync destinations working for CURRENT tags; superseded tags must be pulled into
`2_superseded/<family>/` explicitly. Cluster-only leftovers not mirrored here: the
cancelled partial `_vbk2/_vbk3` runs (superseded by prod2/prod3).

## Regenerating things

```
# lineage (this folder's RUN_LINEAGE.md)
python NoisePy-ant/scripts/picking/write_run_lineage.py --projects .../Projects
# styled maps for any run family
NoisePy-ant/scripts/picking/restyle_runs.sh @@NET@@ @@DX@@ 1_production
# html browser for the current runs
python NoisePy-ant/scripts/picking/make_map_index.py --production-root @@V@@/1_production --net @@NET@@
# geology-grouped velocity distributions
python NoisePy-ant/scripts/picking/velocity_by_geology.py --net @@NET@@ --level cover
# the Cd/k comparison behind "Which run to use" (all nets at once)
python NoisePy-ant/scripts/picking/compare_inversion_choices.py --all
# input-pick vs map-cell 2D distributions (--k k2 for the other cull)
python NoisePy-ant/scripts/picking/pick_vs_cell_hist2d.py --all --k k3
```
