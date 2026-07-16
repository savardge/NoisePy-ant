# Production group-velocity maps from V6 picks — repeatable workflow & parameter choices

End-to-end, from mode-separated V6 picks to masked production group-velocity maps, via the
`swtomotv` Tarantola–Valette package. Every parameter is explicit and defaulted here so a run
is reproducible and auditable. Three scripts (this directory):

| step | script | what |
|------|--------|------|
| 1. export | `export_tomo_picks.py` | V6 `modes_validated.csv` → swtomotv pick/station CSVs (per wave), excluding QC-flagged stations |
| 2. produce | `run_production.py` | per-period TV inversion + resolution-masked maps, transparent damping |
| (alt) | `swtomotv sweep`/`produce` | the package's own coherence-synthesis pipeline (heavier, legacy-coupled) |

Run everything with `PYTHONPATH=<NoisePy-ant> /opt/anaconda3/envs/das-ambient-noise/bin/python`
(swtomotv installed editable in that env, with disba + pyproj).

## Reproduce (both networks, both modes)

```bash
export PYTHONPATH=/Users/genevievesavard/Codes/Noisepy-ant
PY=/opt/anaconda3/envs/das-ambient-noise/bin/python
cd /Users/genevievesavard/Codes/Noisepy-ant/scripts/picking
RIE=/Users/genevievesavard/Codes/extract_higher_modes/Projects/riehen/tomo/riehen_swtomotv.yaml
AAR=/Users/genevievesavard/Codes/extract_higher_modes/Projects/aargau/tomo/aargau_swtomotv.yaml

# 1. export clean picks (drops pairs touching any station flagged in station_qc.csv)
$PY export_tomo_picks.py --config ../../param_files/modesep_params.yaml  --exclude-flagged
$PY export_tomo_picks.py --config ../../param_files/modesep_aargau.yaml  --exclude-flagged

# 2. production maps (LC per network; se fixed; see choices below)
$PY run_production.py --config $RIE --wave fund     --lc 1.5 --se 0.025
$PY run_production.py --config $RIE --wave overtone --lc 1.5 --se 0.025
$PY run_production.py --config $AAR --wave fund     --lc 3.0 --se 0.025
$PY run_production.py --config $AAR --wave overtone --lc 3.0 --se 0.025
```

Outputs per run: `{output_root}/production/{wave}/` → `map_T{T}.npz` (per period),
`figures/map_T{T}.png` (**one standalone figure per period** — station triangles, colorbar,
title with N/σ_eff/LC/var_red/χ²), `production_{wave}.csv` (audit table),
`discrepancy_{wave}.png` (χ² vs σ_eff audit), `maps_{wave}.png` (contact-sheet panel/index).

## Parameter choices and why

| parameter | value | rationale |
|-----------|-------|-----------|
| **flagged stations** | excluded (`--exclude-flagged`) | Removing pairs touching QC-flagged stations left interior structure unchanged and *improved* var_red; the all−noflag difference localized on the flagged sites (leverage, not signal). Riehen −14% fund / −21% overtone picks (11 stations); Aargau −2/−4% (2 stations: AA.3007136/37). |
| **pick error (Cd)** | blanket `(rel_err·τ)²`, `rel_err=0.10` (`--rel-err`) | Single-stack V6 picks have no repeatability ensemble (std=0), so the legacy blanket 10% relative travel-time error is used — a fixed error for every pick, as agreed. Only rescales absolute χ²; the map *pattern* is set by the relative weighting, unchanged by this knob. |
| **prior** | homogeneous slowness `1/v_moy` | Standard TV: `v_moy` = mean pick group velocity at each period (per-period cache). No external reference model enters the inversion. |
| **correlation length LC** | Riehen 1.5 km, Aargau 3.0 km (`--lc`; default `max(1, 3·dx)`) | ≈ 3 grid cells ≈ the array's short-path resolution floor. Larger LC = smoother/more stable; smaller = finer but noisier. Fixed per network (not swept) for a legible, comparable product. |
| **prior slowness std σ_eff** | 0.025 s/km fixed (`--se`) | **No data-driven optimum exists**: these maps are structure/theory-error dominated, not noise dominated — χ²_red stays >1 at all damping (short/mid periods) and only long periods dip below 1 (overfitting). Both the L-curve and the discrepancy principle therefore *rail* to a grid edge. So σ_eff is a documented analyst choice; 0.025 s/km (~4% of the ~0.6 s/km slowness) balances stability against resolution. The per-period χ²(σ_eff) curves are saved (`discrepancy_*.png`) so the trade-off is visible and `--se` is trivially revised. |
| **inversion variant** | one-step TV (no Liu–Yao reweighting) | Matches the swtomotv sweep/parity default; two-step reweighting is for outlier-heavy legacy Love sets. |
| **coverage mask** | legacy ray-count: >`min_density`(3) rays each >`thres_dist`(10 m) | swtomotv parity invariant (do not change — see swtomotv/AGENTS.md). |
| **resolution mask** | drop the worst `--res-drop-q` (0.25) quantile of covered cells by resolution-diagonal | Self-scaling with damping/period. **This removes the implausible E/W edge high-velocity blobs** — those cells had resolution 0.08 vs 0.14 map-wide and ~16 vs ~48 ray-km/cell (edge-ray smearing, not structure). `map_T*.npz` also stores `vel_full` (coverage-only) so any threshold can be re-applied without re-inverting. |

## Reading the outputs

- `production_{wave}.csv`: per period → `N` (rays), `se_eff`, `LC`, `var_red`, `restit_post` (%),
  `chi2_red`, `cells_shown`/`cells_covered`. Trust periods with many rays and `cells_shown`
  comparable to `cells_covered`; long periods thin out.
- `map_T{T}.npz` keys: `vel` (masked, nx×ny), `vel_full` (coverage-only), `mask`, `res_diag`,
  `unc_s` (posterior slowness std), `se`, `LC`, `chi2_red`, `var_red`, `N`, `coverage`.
- Interpret only inside the station hull; the resolution mask already hides the poorly-constrained rim.

## Known caveats / next steps

- Absolute velocities depend on σ_eff/LC (regularization); the **pattern** (slow-W/fast-E Riehen
  flexure; fast-N/slow-S Aargau) is stable across reasonable choices — that is the robust product.
- No checkerboard/resolution-length test yet: the resolution-diagonal mask bounds *where* the model
  is data-driven, but a synthetic recovery test would quantify the recoverable wavelength per period.
- For the package's coherence-based per-period σ_eff/LC selection instead of a fixed choice, run
  `swtomotv sweep` then `produce` (heavier; writes legacy-schema .mat trees). Our fixed-σ_eff driver
  was chosen for transparency given the railing diagnostic above.
- Anisotropy: use `swtomotv aniso` (joint solve) — never fit 2ψ to isotropic residuals (biased low;
  swtomotv/AGENTS.md). Group ≠ phase anisotropy coefficients.
