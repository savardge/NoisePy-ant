# Unified dispersion picking + QC — handoff summary

Written 2026-07-13. Self-contained reference for the unified surface-wave picking system built in
`~/Codes/NoisePy-ant` and validated on the Aargau network. Companion to
`RADIAL_ANISOTROPY_PLAN.md` (this system implements its §2 data-prerequisite cuts at pick level).

---

## 1. What it is

One driver produces, per station pair, **all eight dispersion pick types in a single CSV**:

|  | group velocity | phase velocity |
|---|---|---|
| Rayleigh fundamental | ✅ (G_LR0) | ✅ |
| Rayleigh 1st overtone | ✅ (G_LR1) | ✅ |
| Love fundamental | ✅ (TT) | ✅ |
| Love 1st overtone | toggle, default OFF | toggle, default OFF |

Previously this capability was split across `dispersion_curves_V6_modesep.py` (Rayleigh both modes
+ Love fundamental group only; Love phase disabled), `dispersion_batch_love.py` (Love group only,
overtone unlabeled), and `phase_batch.py` (Rayleigh phase only).

**Components** (all in `~/Codes/NoisePy-ant` unless noted):

| file | role |
|---|---|
| `noisepy/unified_picking.py` | reusable module: `pick_all_modes()` + all algorithm config (`Config` class) |
| `scripts/picking/dispersion_unified.py` | network batch driver (mp.Pool, resumable, `--overwrite`) |
| `scripts/picking/qc_unified_picks.py` | merge-time QC gates + rejection budget + before/after figure |
| `scripts/picking/validate_unified_picks.py` | 8-panel pick-distribution figure + phase sign check |
| `scripts/picking/vsg_love_reference.py` | Love fundamental + overtone phase reference curves from the TT VSG stack |

---

## 2. Theoretical foundation

**Phase measurement is identical for Love and Rayleigh.** The far-field 2-D surface-wave Green's
function is ~ e^{-i(kr - π/4)} for both wave types — the −π/4 is the stationary-phase (Hankel)
term, not a polarization effect. So Love (TT) phase uses the exact same
`dispersion.measure_corrections_and_phase` chain and the same **+π/4** shift as ZZ/RR (TT is an
auto-component; the ±π/4 / +3π/4 offsets exist only for the RZ/ZR cross components which carry an
e^{±iπ/2} factor). The library is wave-agnostic: wave type enters only via the scalar `phase_shift`
and the reference callable `c_ref`.
**Empirically validated on Aargau**: Love-fundamental measured phase vs the data-derived Love
reference has median residual −0.02 km/s (n≈19k), symmetric at zero, tighter than the Rayleigh
control (+0.09) — a wrong sign would show a half-fringe offset. The old code caveat
"Love sign less settled" is resolved: **+π/4 is correct**.

**Mode separation is fundamentally different per wave and cannot be symmetric.**
- Rayleigh: G_LR0 (fundamental, retrograde) / G_LR1 (1st higher, prograde) are synthesized from
  ZZ/RR/RZ/ZR via the ±π/2 phase corrections of Nayak & Thurber (2020) eqs 3/4
  (`dispersion.phase_corrected_components`) and combined with a t-f phase-weighted stack
  (`dispersion.tf_pws`, Ventosa 2017). Needs two orthogonal components 90° apart (elliptical Z–R
  coupling).
- Love is single-component SH — **no orthogonal partner exists, so no G_LR analog is possible even
  in principle.** Love fundamental/overtone separate only in the image/array domain: per-pair FTAN
  ridge topology on TT, labeled against data-derived reference curves. Consequently Love overtone
  phase would be read off the *mixed* TT wavelet transform — reliable only where the modes are
  temporally separated. This, plus weak evidence for a credible Love overtone on these short
  paths, is why `Config.PICK_LOVE_OVERTONE = False` by default (re-enable: env `DISP_LOVE_OT=1`).
  When off, overtone-labeled TT ridges are **dropped, never relabeled fundamental** (relabeling
  would contaminate the fundamental stream with fast ridges).

**Mode-ordering physics used by the QC** (important asymmetry): phase velocities of modes cannot
cross (c₁ > c₀ strictly at a given frequency), but **group velocities CAN cross** near Airy-phase
minima. Therefore "overtone slower than fundamental" is a *flag*, never an automatic rejection;
the physically decisive test is **mutual suppression** (a genuine mode is weak in the other G_LR
stack's image at its (T, U) — Nayak & Thurber's core discriminating physics).

---

## 3. Per-pair picking pipeline (`pick_all_modes`)

Input: `Allstack_pws` H5 stacks (NoisePy-ant format), components ZZ/RR/RZ/ZR + TT + cross-terms
TZ/ZT/RT/TR (Love SNR context). All picks on the **sym** lag fold.

1. **Rayleigh**: fold → `phase_corrected_components` → `tf_pws` → G_LR0/G_LR1 traces →
   CWT (`compute_cwt`) → FTAN image (`disp_image_from_cwt`) for BOTH modes first (two-pass, so
   every pick can be checked against the other mode's image) → picks by `extract_dispersion`
   (argmax, segment-aware) AND `extract_curves_topology` → COI removal →
   `measure_corrections_and_phase` (per-mode reference: G_LR0 → fundamental ref, G_LR1 → overtone
   ref; `joint='unwrap'` 2πN branch tracking).
2. **Love**: fold TT → CWT/FTAN → both pick methods → `label_love_ridges`: each ridge point is
   assigned fundamental/overtone by nearest **group** reference (phase refs converted via
   `phase_ref_to_group_ref` = poly-smoothed Bensen dispersion relation, clamped to [0.5c, 1.02c]
   with 0.92c fallback), with the hard physical guard that an overtone must be **faster than the
   fundamental reference by ≥0.25 km/s** → per labeled stream, same
   `measure_corrections_and_phase` with the Love refs.
3. QC discriminator columns computed in-picker (see §5) — thresholds are applied later, at merge.

Config lives in `unified_picking.Config` (grid Tmin 0.2 / dT 0.1 / v 0.5–4.5 / dvel 0.01,
vave 3.0 → Tmax = dist/3; topology min_score 0.6; group gate 1λ; phase gate 1λ, no tau_max cap —
the step-2 validated relaxation; `use_period='nominal'` keeps group & phase on one period axis,
with `T_scale`/`scale_j` carried for later dedupe/relabeling).

## 4. Reference curves (all data-derived, 2-col text `period[s] c[km/s]`)

In `Projects/<net>/vsg_modesep/` (extract_higher_modes repo):

| file | source |
|---|---|
| `ref_fundamental_phase.txt` | Rayleigh fund — picked from mode-separated VSG phase-shift stacks (`pick_reference_ridges.py`) |
| `ref_overtone_phase.txt` | Rayleigh overtone — same, G_LR1 image |
| `ref_love_phase.txt` | Love fund — network TT VSG slant-stack, argmax + dominance/roughness/min-run QC (`vsg_love_reference.py`) |
| `ref_love_overtone_phase.txt` | Love overtone — second faster ridge in the SAME TT stack; **separation gate** rejects picks with c_ot − c_fund < 0.3 km/s (window-floor artifacts). Aargau: only T 1.01–1.43 s survives — the one band with a distinct Love overtone ridge. |

## 5. Unified CSV schema (v2, 31 columns)

`nominal_period, T_centroid, T_inst, group_velocity, phase_velocity, N_ambiguity, U_from_phase,
score, snr_nbG, snr_bb, ratio_d_lambda, azimuth, backazimuth, distance, lag, component, wave_type,
mode, stack_method, pick_method, snr_bb_other, snr_nbG_other, env_ratio, ot_flag, mode_overlap,
xmode_amp, dU_rayfund, dU_rayot, dUdT_local, T_scale, scale_j`

Key QC columns (computed in-picker, thresholded at merge — the design philosophy throughout):

| column | meaning |
|---|---|
| `xmode_amp` | **mutual suppression**: per-period-max-normalized amplitude of the OTHER G_LR image at this pick's (T, U). Genuine mode ≤ 0.6 (validate_modes CONTRAST_MAX). In-memory port of the production validator's anchor test — no disk images needed. |
| `ot_flag` | Rayleigh overtone: `sep` / `unresolved` (U1−U0 ≤ sep_req = max(0.30, U0²T/dist), osculation guard) / `slow` (U1 < U0−0.1, leakage candidate). Love overtone: `sep`/`overlap`. |
| `dU_rayfund`, `dU_rayot` | Love pick velocity minus the **same pair's** G_LR0 / G_LR1 argmax curve at the nearest period (±0.15 s). \|dU_rayfund\| ≤ 0.15 = the R-fundamental-on-TT leakage fingerprint (the Basel/Riehen graben diagnosis). |
| `mode_overlap` | Love: 1 where the fund/overtone group references are within 0.15 km/s (modes unresolvable → labels ambiguous). |
| `env_ratio` | TT narrowband SNR / max(cross-term TZ/ZT/RT/TR SNR) at the pick period. **Column only — not an absolute discriminator** (see §7). |
| `dUdT_local` | rolling \|dU/dT\| along the argmax curve — mode-mixing/steepness indicator, report-only (real geological steps exist on mixed-geology paths). |
| `T_scale`, `scale_j` | true Fourier period and index of the CWT scale the phase was read on. Picks sharing scale_j are the SAME measurement — required for dedupe. |

## 6. QC script (`qc_unified_picks.py`) — gates and outputs

Reads the `<pair>_unified.csv` tree; a row can survive as group but lose phase (or vice versa):
output columns `group_ok` / `phase_ok` (int). **Every gate's kill count is reported per
(wave, mode, measure)** in `qc_rejection_budget.txt` — nothing dropped silently. All thresholds
CLI-tunable; disable any gate with `--disable name1,name2`.

Ordered defaults:

| gate | applies to | default |
|---|---|---|
| snr | all | snr_nbG ≥ 5 |
| vbounds | all | fundamental 0.5–5.0, overtone 1.5–5.0 km/s (group and phase separately; relaxed from 3.6/4.5 so long-period PHASE picks are not clipped — the GROUP-side 3.6/4.5 bounds are re-applied downstream by `export_unified_tomo_picks.py`, which consumes group picks only) |
| farfield | group only | ratio_d_lambda ≥ 2.0 (deep-LVZ near-field finding; phase keeps the 1λ picker gate — phase is valid to ~1λ) |
| suppression | **Rayleigh OVERTONE only** | xmode_amp ≤ 0.6 |
| ot_res | Rayleigh overtone | drop ot_flag ∈ {slow, unresolved} |
| love_env | Love | **OFF by default** (--env-min 0); see §7 |
| love_overlap | Love overtone | mode_overlap == 0 |
| rf_leak | Love | drop \|dU_rayfund\| ≤ 0.15 (R-fund leakage veto), **only where diagnostic**: fires only where the network reference Love/R-fund group curves are separated by > 2×tol (0.30 km/s); T ≤ 2 s fallback where refs unavailable. See lesson 6 — unconditional, this gate censors converging physics at long T. |
| ot_leak | Love | drop \|dU_rayot\| ≤ 0.15 (R-overtone leakage veto), same separation-conditioned logic |
| phase_phys | phase | require c > U (2πN branch physicality) |
| station | all | optional --station-qc csv; --drop-flagged to kill |
| scale_dedupe | phase | one pick per (pair, component, lag, mode, scale_j); keeps the pick nearest T_scale, argmax preferred |
| group_scale_dedupe | group | same single-scale principle for group picks; key adds pick_method and a **0.2 km/s velocity bin** so distinct branches at one scale (different wave packets) are never collapsed |

Outputs: `picks_unified_QCd.csv`, `qc_rejection_budget.txt`, `qc_before_after.png`
(4 rows (wave,mode) × 4 cols group-before/after, phase-before/after, LogNorm 2D histograms with
colorbars, phase refs overlaid).

## 7. Gate-calibration lessons (Aargau 800 pairs — do NOT re-tighten without new evidence)

1. **env_ratio is not an absolute discriminator.** Love phase picks sitting ON the reference
   (n≈23k, provably good) have median env_ratio 0.88; an absolute ≥1.5 gate killed 74% of them.
   Cross-term SNR comparable to TT is normal on short paths. Gate defaults to OFF; the
   velocity-coincidence vetoes (rf_leak/ot_leak) are the real contamination tests.
2. **Never apply mutual suppression to the fundamental.** At T > 1.5 s on short paths the stacks
   cannot separate the modes, so the *dominant* fundamental fails xmode spuriously (28–42% of
   G_LR0 picks). The production validator only ever used suppression as the overtone anchor test;
   the fundamental is confirmed by consensus/other gates.
3. **The Love-fundamental histogram "gap" (T≈1.1–1.4 s, U>2.2) is the overtone-labeling window,
   not a plotting artifact.** The overtone group reference exists only there; inside it, fast TT
   ridges are labeled overtone (and dropped by the toggle); at T=1.0 (ref undefined) identical
   fast picks stay fundamental. Expected footprint, not a bug.
4. **CWT-scale duplication affects group picks too, not just phase.** The FTAN image interpolates
   log-spaced scales (ΔT/T ≈ 5.9% for dj=1/12) onto the 0.1 s nominal grid, so above ~2 s adjacent
   nominal-period picks are the same measurement (Aargau: ×1.35 at 2–3 s, ×1.85 at 3–4 s, ×2.28 at
   4–6 s; within-scale U spread 0.02 km/s). Without dedupe this inflates counts and deflates std at
   exactly the long-T band feeding deep structure. After group_scale_dedupe: ×1.07–1.14 residual =
   same-scale picks >0.2 km/s apart = genuinely distinct packets, deliberately kept.
5. **After-QC striping in phase panels at long T is honest**, not over-aggressive QC: it is the
   discrete independent-measurement structure (~one per CWT scale, ~0.3 s apart, vs 0.1 s bins).
   The smooth "before" panels are cosmetically smooth *because of* duplicated measurements.
   For tomography, prefer `T_scale` as the phase pick's period label.
6. **⚠ CENSORING BIAS in unconditional leak vetoes (independent audit, 2026-07-14, full Aargau).**
   The rf_leak kill fraction rose monotonically with period — 0.10 (T<1 s) → 0.40 (3–4 s) —
   because Love-fund and Rayleigh-fund group velocities genuinely CONVERGE at long T; that is
   physics, not leakage. Unconditional, the veto (a) deletes up to 40% of the deepest-sensitivity
   Love picks and (b) pre-selects the surviving Love set to DIFFER from Rayleigh — feeding it to a
   γ(z)/radial-anisotropy inversion manufactures spurious nonzero anisotropy at depth. **Fixed**:
   rf_leak/ot_leak now fire only where the network reference group curves are separated by
   > `--leak-sep-factor` (2.0) × `--leak-tol` (i.e. 0.30 km/s); fallback `--leak-tmax` (2 s) T-cap
   where refs are unavailable. Verified diagnostic bands: Aargau veto ON T ≲ 2 s / OFF beyond;
   Riehen ON only T ≲ 1.2 s (its refs converge earlier — thick sediments). **Riehen caveat**: the
   Basel graben contamination was diagnosed at T 1.4–1.6 s, where Riehen's *network-mean* refs
   already converge — for graben-focused re-QC consider `--leak-sep-factor 1.5` or the
   `--leak-tmax` variant, since network-average references blur the strong W/E split.

## 8. Validation results (Aargau, 800 pairs, `Projects/aargau/dispersion_unified/`)

- **Love phase sign**: median residual vs data-derived ref −0.02 km/s pre-QC, −0.019 (MAD 0.109)
  post-QC → +π/4 confirmed, gates unbiased.
- **Rayleigh overtone leakage cloud** (broad sub-fundamental picks down to 0.5–0.7 km/s at all T)
  **eliminated** by suppression + ot_res; after = clean fast branch 2–4 km/s.
- **Cross-validation vs the production validator** (`dispersion_V6/*_modes_validated.csv`):
  100% velocity agreement on shared picks; **~70–85%** of production ot_use==1 picks survive the
  new QC (85.8% with a period-tolerant join, 73.6% with an exact network-wide join — the figure is
  dedupe/join-method-sensitive; the losses are the stricter 2λ far-field and snr≥5-vs-3 gates).
- **Final QC'd survivors**: Rayleigh fund 29.2k group / 13.9k phase; Rayleigh overtone 10.9k / 5.0k;
  Love fund 20.0k / 9.4k; Love overtone 0 (toggle off).

## 9. How to run

```bash
# env: needs pycwt + findpeaks (+ pandas); pyasdf optional (h5py fallback). Stacks on /Volumes/T7blue.
export PYTHONPATH=~/Codes/NoisePy-ant
PY=/opt/anaconda3/envs/das-ambient-noise/bin/python

# (once per network) Love references from the TT VSG stack (add --from-stack to reuse an existing stack)
$PY vsg_love_reference.py --config ../../param_files/modesep_<net>.yaml

# batch picking (env toggles: DISP_LOVE_OT=1 re-enables Love overtone; DISP_STACK, DISP_OVERWRITE)
$PY dispersion_unified.py --config ../../param_files/modesep_<net>.yaml \
    --out <project>/dispersion_unified --nproc 10 [--limit N] [--overwrite]

# QC gates + budget + figures
$PY qc_unified_picks.py --dir <project>/dispersion_unified --ref-dir <project>/vsg_modesep

# optional: 8-panel distributions + sign check
$PY validate_unified_picks.py --dir <project>/dispersion_unified --ref-dir <project>/vsg_modesep
```

Config YAMLs: `param_files/modesep_aargau.yaml` (done), `modesep_params.yaml` (Riehen). Output CSVs
land in `<out>/<src>/<pair>_unified.csv`; schema v1 (no xmode_amp) is incompatible — rerun with
`--overwrite`.

## 10. Full-network runs (DONE 2026-07-14) + next steps

**Both networks picked and QC'd in full** (~45 min picking + QC on 10 cores each):

| | Aargau (17,350 pairs, 2.86M rows) | Riehen (19,503 pairs, 1.72M rows) |
|---|---|---|
| Rayleigh fund | 641k group / 306k phase | 375k / 182k |
| Rayleigh overtone | 235k / 107k | 108k / 67k |
| Love fund | 450k / 212k | 194k / 96k |

Outputs per network: `Projects/<net>/dispersion_unified/{<src>/<pair>_unified.csv,
picks_unified_QCd.csv, qc_rejection_budget.txt, qc_before_after.png}`. Cross-network diagnostics
(`compare_unified_networks.py`): `Projects/unified_diagnostics/{unified_compare_distributions.png,
unified_compare_ray_maps.png, unified_compare_stats.txt}` — side-by-side post-QC distributions and
ray-path coverage maps (16-18k pairs with surviving picks per type per network; whole-array
azimuthal coverage, no dead sectors).

**rf_leak is the dominant Love gate on both networks** (Aargau 371k kills, Riehen 120k) — with the
env gate off, the velocity-coincidence veto inherits the full contamination load.
**Graben contamination spatially CONFIRMED at network scale** (the Basel-1/Otterbach-2 cell-level
diagnosis): median per-pair Love rf-leak coincidence fraction (|dU_rayfund| <= 0.15) is **0.286
WEST of the Rhine Valley Flexure (URG graben) vs 0.174 EAST (basement)** — the R-fund-on-TT
leakage concentrates in the graben, exactly where the radial-anisotropy plan expected it.

Next steps:
- Graben Love re-QC for the radial-anisotropy inversion now has its input: filter the Riehen QC'd
  table (`group_ok`/`phase_ok`) and rebuild the Love cell curves; the West-side curves should
  steepen less once the rf-leaked picks are gone.
- Love overtone stays off unless a credible use case appears (flip `DISP_LOVE_OT=1`; overlap +
  leak vetoes then gate it).
- `dUdT_local` (curve steepness) is report-only at pick level; cell-curve-level steepness
  screening lives in the radial-anisotropy workflow (Love-only inversion χ).
- For tomography exports: filter `group_ok==1` / `phase_ok==1`, and use `T_scale` (not
  nominal_period) as the period label for phase picks.
