# `scripts/picking` — FTAN & dispersion-curve picking

Scripts that turn stacked cross-correlations (ASDF `.h5` from `scripts/raw2stack` S0–S2) into
group- and phase-velocity dispersion picks. Two generations coexist:

- **Classic** single-component group-velocity picking, `dispersion_curves.py` → `V5`, merged and
  histogrammed with `step1_merge_picks.py` / `step2_pick_histograms.py` (see §5).
- **Mode-separated (V6)** workflow that additionally recovers the **first higher mode** via the
  Nayak & Thurber (2020) ±π/2 component synthesis, with a data-derived VSG reference, a consensus
  validator, and per-network YAML config (see §3). This is the current recommended workflow.

## 1. Environment

```bash
export PYTHONPATH=/path/to/NoisePy-ant          # noisepy is imported as a package
# the picking scripts need: numpy scipy pandas matplotlib obspy pycwt findpeaks (+ pyasdf OR h5py)
python <script>.py ...
```

On the development machine that is `/opt/anaconda3/envs/das-ambient-noise/bin/python` (has
pycwt + findpeaks + h5py; the V6 scripts fall back to h5py when pyasdf is absent). **Exception:**
`lateral_structure_map.py`'s fault analysis needs `geopandas`+`shapely`, available only in the base
anaconda env — it degrades gracefully (maps only) elsewhere.

## 2. Configuration (mode-separated workflow)

The V6 workflow is driven by one YAML per network — template
[`param_files/modesep_params.yaml`](../../param_files/modesep_params.yaml), loaded by
[`modesep_config.py`](modesep_config.py). **Precedence: explicit CLI flag > YAML > built-in
default.** Sections: `network` (name, code), `paths` (vsg_dir, stack_root, project_dir, derived
dispersion_dir/ref_dir, optional model_reference_csv & faults_shapefile), `vsg_modesep`,
`pick_windows` (per-branch, **network-tuned**), `batch`, `analysis`. All products are written under
`paths.project_dir` with fixed filenames.

`dispersion_batch_modesep.py` also honours a **legacy** invocation (positional
`stack_root out_root [nproc]` + `DISP_NET`/`DISP_REF_DIR`/`DISP_LIMIT` env vars) when `--config` is
absent, so `dispersion.slurm` keeps working. `dispersion_curves_V6_modesep.py` accepts an optional
`--config` that overrides only its reference-curve dir and output root.

## 3. Recommended workflow (mode separation, Rayleigh fundamental + 1st higher mode)

Upstream (once per network): `scripts/postprocess_stacks/phaseshift_dispersion.py` produces the
per-virtual-source VSG panels under `paths.vsg_dir`. Then, all with
`--config param_files/modesep_params.yaml`:

| # | command | output |
|---|---------|--------|
| 1 | `python vsg_modesep.py --config …` (and `--sign -1` as a control) | network-stacked mode-separated images `vsg_modesep_stacks_sign±1.npz` (+figure) in `ref_dir` |
| 2 | `python pick_reference_ridges.py --config …` | `ref_{fundamental,overtone}_phase.txt`, `picked_reference_curves.csv`, diagnostic PNGs |
| 3 | `python dispersion_batch_modesep.py --config …` | full-network per-pair `*_dispersion_all.csv` + `*_glr_images.npz`; runs `validate_modes.py` inline → `*_modes_validated.csv` |
| 4 | `python network_station_qc.py --config …` | `station_qc.csv/png` — per-station coupling / orientation-polarity flags |
| 5 | `python network_consistency_analysis.py --config …` | `network_consistency.png`, `network_anisotropy.png`, `final_network_stats.txt` |
| 6 | `python network_pick_maps.py --config …` | `network_pick_histograms.png`, `map_overtone_rays.png`, `overtone_ray_counts.csv` |
| 7 | *(optional)* `/opt/anaconda3/bin/python lateral_structure_map.py --config …` | `lateral_structure.png`, `lateral_structure_stats.txt` — Vg west/east domains + higher-mode vs fault crossing |

Steps 1–2 (the VSG reference-curve construction) are documented in full in
[`VSG_REFERENCE_METHODOLOGY.md`](VSG_REFERENCE_METHODOLOGY.md). Single-pair spot checks with
quicklook plots: `python dispersion_curves_V6_modesep.py <pair.h5> [--config …]`.

## 4. Script inventory

### Mode-separated (V6) workflow & config
| script | role | key inputs → outputs |
|--------|------|----------------------|
| `modesep_config.py` | shared YAML loader + helpers (`load_config`, `apply_overrides`, `ref_curve_paths`, `vsg_station_coords`) | imported by the scripts below |
| `vsg_modesep.py` | G_LR0/G_LR1 synthesis (Nayak & Thurber 2020) + Park phase-shift beamform + network stack | VSG per-source npz → `vsg_modesep_stacks_sign±1.npz` |
| `pick_reference_ridges.py` | windowed-argmax reference-curve picking + dominance/roughness/run QC | stack npz → `ref_*_phase.txt`, `picked_reference_curves.csv` |
| `dispersion_curves_V6_modesep.py` | per-pair V5 + mode separation (tf-PWS G_LR0/G_LR1), group + 2πN phase picks, quicklooks | one pair `.h5` → `*_dispersion_all.csv` (+images) |
| `dispersion_batch_modesep.py` | lean full-network batch of the validated V6 config (pws/sym only), inline validation, resume | stack root → `dispersion_V6/**` |
| `validate_modes.py` | consensus mode validator: confirms G_LR0 vs ZZ/RR/all4 argmax, G_LR1 separation + mutual suppression | `*_dispersion_all.csv` → `*_modes_validated.csv` (+`*_modeQA.png`) |
| `network_station_qc.py` | per-station LSQ effect terms + robust-z flags (coupling / orientation-polarity) | validated CSVs → `station_qc.csv/png` |
| `network_consistency_analysis.py` | picks vs VSG reference (group & phase residuals) + 2ψ azimuthal anisotropy | validated CSVs + refs → `network_consistency/anisotropy.png`, `final_network_stats.txt` |
| `network_pick_maps.py` | 2-D pick histograms + higher-mode ray map | validated CSVs + coords → `network_pick_histograms.png`, `map_overtone_rays.png` |
| `lateral_structure_map.py` | Vg west/east domain maps + higher-mode vs fault-crossing test (optional geopandas) | validated CSVs + faults → `lateral_structure.png/.txt` |

### Classic versioned pickers (single-file, per pair)
| script | adds |
|--------|------|
| `dispersion_curves.py` | original topology-based FTAN group-velocity picker |
| `dispersion_curves_V2.py` | production baseline: argmax + topology, findpeaks (driven by `dispersion.slurm`) |
| `dispersion_curves_V3.py` | per-station output tree; ZZ/RR/RZ/ZR components |
| `dispersion_curves_V4_love.py` | Love components (TT/RT/TR/ZT/TZ); robust + auto_covariance stacks |
| `dispersion_curves_V5.py` | Rayleigh+Love unified: Shapiro/Levshin period corrections, Bensen phase velocity + 2πN reference resolution, segment-aware picking |

### Merge / statistics / QC
| script | role |
|--------|------|
| `step1_merge_picks.py` | merge per-pair CSVs into one table; SNR + wavelength-distance filters |
| `step2_pick_histograms.py` | 2-D pick histogram (T × Vg), mean±std per period |
| `QC_filter_GMM.py` | Gaussian-mixture outlier rejection per period |
| `QC_filter_paths_dbscan.py` | (empty stub) |

### Diagnostics / references / theory
| script | role |
|--------|------|
| `reference_model.py` | layered Vs model → disba forward Rayleigh/Love fundamental+overtone reference curves |
| `synthetic_phase_calibration.py` | synthetic single-mode Green's function: validate group velocity, calibrate Morlet phase offset |
| `compare_picks_image.py` | argmax vs topology vs Viterbi + phase-2πN overlays on one pair |
| `compare_ftan_methods.py` | CWT vs narrowband-Gaussian FTAN images |
| `compare_norm_picking.py` | per-period vs global FTAN normalization (osculation visibility) |
| `show_unnormalized_ftan.py`, `show_shapiro_correction.py` | global-normalized FTAN & Shapiro period-correction diagnostics |
| `batch_compare_pairs.py` | run a comparison on the N largest-offset pairs |
| `dispersion.slurm` | SLURM array template (runs `dispersion_curves_V2.py` per pair) |

## 5. Classic workflow (single-component group velocity)

1. `dispersion_curves_V2.py` (or `V5` for phase velocity + Love) per pair — use `dispersion.slurm`
   on a cluster.
2. `step1_merge_picks.py` → one merged CSV.
3. `step2_pick_histograms.py` (and `QC_filter_GMM.py`) → per-period distributions for tomography.

## 6. Further reading

- [`VSG_REFERENCE_METHODOLOGY.md`](VSG_REFERENCE_METHODOLOGY.md) — the mode-separated reference-curve
  method, with a Riehen worked example.
- [`MODE_OSCULATION_NOTES.md`](MODE_OSCULATION_NOTES.md) — handling modal energy transfer
  (osculation / "kissing" curves) in FTAN picking (Bhaumik & Cox 2026).
- `noisepy/dispersion.py` — the algorithms (CWT FTAN, `phase_corrected_components`, `tf_pws`,
  `phase_from_group`, 2πN resolution).
