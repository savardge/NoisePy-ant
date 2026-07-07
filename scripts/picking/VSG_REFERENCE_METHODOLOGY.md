# Mode-separated phase-velocity reference curves — methodology

Data-derived fundamental and first-overtone Rayleigh **phase-velocity** dispersion curves,
picked from network-stacked, mode-separated frequency–velocity images. The curves are measured
directly from the data; **no forward/earth model is used** at any stage. They become the
reference against which the per-pair V6 workflow resolves the 2πN phase-velocity ambiguity
(`REFERENCE_CURVES` in `dispersion_curves_V6_modesep.py` / `REFS` in `dispersion_batch_modesep.py`).

Scripts (both take `--config param_files/modesep_params.yaml`):

| script | role |
|--------|------|
| `vsg_modesep.py` | Steps 1–2: G_LR0/G_LR1 synthesis + Park phase-shift beamforming + network stack → `vsg_modesep_stacks_sign{+-1}.npz` (+ figure), written to `paths.ref_dir` |
| `pick_reference_ridges.py` | Step 3: windowed argmax + QC → `picked_reference_curves.csv`, `ref_fundamental_phase.txt`, `ref_overtone_phase.txt`, two diagnostic PNGs |

## 1. Input data

**Source:** per-virtual-source dispersion panels produced by the array phase-shift
(vertical-source-gather / "VSG") analysis of the ambient-noise cross-correlations
(`scripts/postprocess_stacks/phaseshift_dispersion.py`), under `paths.vsg_dir`. Each per-source
file `<comp>/sources/<NET>.<src>.npz` (comp ∈ ZZ, RR, RZ, ZR) contains, for that virtual source:

- `sym` — symmetric (folded, causal-side) cross-correlation traces, one per receiver
  `(n_receivers, n_lag)`, sampling `dt`;
- `x` — source–receiver offsets [km]; `rx_codes` — receiver station codes;
- `f`, `vel` — the frequency and velocity axes of the analysis grid; source/receiver coords.

Because this is the same primitive data the reference VSG analysis consumed, the mode-separated
images live on exactly the same `(f, vel)` grid and are directly comparable to it. The analysis
NFFT is recovered from the stored `f` grid (`nfft = round(1/((f[1]-f[0])·dt))`); override with
`vsg_modesep.nfft` if needed.

## 2. Mode separation and network stacking (`vsg_modesep.py`)

Isolate a single Rayleigh mode per image so a simple amplitude-maximum pick yields an
unambiguous single-mode curve.

**Step 1 — phase-corrected synthesis (per source–receiver pair).** Following Nayak & Thurber
(2020, *GJI* 222, 1590; eqs 3–4), the four radial–vertical components are combined with ±90°
phase shifts to constructively stack one mode and suppress the other:

```
G_LR0 = ZZ + RR + IFFT( RZ·e^{+iπ/2} + ZR·e^{-iπ/2} )     (fundamental,  retrograde)
G_LR1 = ZZ + RR + IFFT(-RZ·e^{+iπ/2} - ZR·e^{-iπ/2})      (1st overtone, prograde)
```

- Only receivers present in **all four** components are used (`vsg_modesep.min_common_receivers`);
  the four traces are aligned on that common set before combination.
- Component convention (verified against the NoisePy-ant E-N-Z→R-T-Z rotation): first letter =
  source component, second = receiver component, so `RZ` = radial-at-source/vertical-at-receiver
  = the paper's `G_RZ`, `ZR` = `G_ZR`. The `+π/2 / -π/2` assignment is the paper sign
  (`vsg_modesep.sign: 1`); flipping it swaps the mode assignments (control — see §5).
- This is a **linear** stack of the four components (eqs 3/4 exactly). The per-pair V6 workflow
  uses a time–frequency phase-weighted stack instead; here the linear form is used so the
  operation is transparent and matches the published equations.

**Step 2 — phase-velocity imaging + network stack.** Each synthesized trace set is transformed by
the **phase-only Park phase-shift** (the same transform as the reference analysis):

```
E(c, f) = Σ_j  ( S_j(f)/|S_j(f)| ) · e^{+i·2πf·x_j / c}
```

with `S_j` the receiver spectrum, `x_j` the offset. Unit phasors make the image a pure
phase-coherence measure (amplitude-independent). Each image is normalized per frequency column
(peak = 1), then `|E|` is **stacked linearly across virtual sources**. `ZZ` is processed
identically as a control. Output: `vsg_modesep_stacks_sign{+-1}.npz` with images `ZZ`, `G_LR0`,
`G_LR1` on the common `(f, vel)` grid.

## 3. Ridge picking (`pick_reference_ridges.py`)

For each mode-isolated image the curve is the **amplitude maximum vs frequency** — plain
per-frequency `argmax` inside a fixed rectangular search window (`pick_windows` in the YAML).
**No Viterbi tracking and no smoothing of the picked curve.** Windows bound the velocity range
each mode occupies and exclude the low-frequency fundamental/overtone merge and the near-DC edge;
they are the only manual input, set by inspection of the stacked figure (drawn as dotted
rectangles in the diagnostic PNGs). Three objective QC filters, none of which smooths the
retained picks:

1. **dominance** = (in-window peak) / (full-column maximum) ∈ [0, 1]; = 1 when the targeted mode
   is the strongest arrival at that frequency. Drop picks below `dom_min` (default 0.60). Written
   to the CSV so a reviewer can re-threshold.
2. **roughness** = rolling mean of |Δc| between consecutive argmax picks (`rough_win` samples). On
   a coherent ridge picks move ~1–2 velocity-grid steps; where the ridge dissolves the argmax
   jumps many steps and roughness rises an order of magnitude. Drop picks above `rough_max`. This
   objectively terminates each curve where its ridge ends — the automated replacement for a manual
   high-frequency clip.
3. **minimum run length**: contiguous runs shorter than `min_run` samples are dropped (gaps ≤ 2
   bridged), removing isolated spurious columns while keeping every substantial clean segment.

**Per-network tuning.** The windows and `rough_max` are genuinely network-specific — read them off
the stacked `sign+1` figure before picking. If a branch returns 0 picks, its window/roughness gate
is the first thing to check (see the Riehen overtone note in the worked example below).

## 4. Output

`picked_reference_curves.csv`: `wave, branch, from_image, frequency [Hz], period [s],
c_phase [km/s], dominance`. The 2-column `ref_fundamental_phase.txt` / `ref_overtone_phase.txt`
(period [s], c [km/s], ascending period) carry the same picks in the format expected by
`noisepy.dispersion.load_reference_curve` for the per-pair 2πN resolution.

## 5. Validation and controls

- **Independent benchmark.** These images come from array phase-shift beamforming; the per-pair V6
  workflow they validate uses single-pair CWT-FTAN + polarity stacking — independent method,
  domain and error budget.
- **Mode isolation is quantified.** The script prints the mean column-normalized amplitude along
  each branch (when a `model_reference_csv` is configured): `G_LR0` high on the fundamental / low on
  the overtone, `G_LR1` the reverse, raw `ZZ` intermediate (both modes present).
- **Sign control.** Re-running with `--sign -1` swaps the branch assignments exactly, confirming
  the ±π/2 convention.

## 6. Limitations

- The **overtone** is reliable only over the band where its ridge is coherent (the roughness gate
  cuts it automatically); it does not reach the shortest periods.
- The **fundamental** long-period end rests on few, lower-amplitude columns near the array
  resolution limit; the short-to-mid-period core is robust.
- These are **network-average** curves. Real lateral heterogeneity is averaged out by design;
  single-path measurements scatter around them (that scatter is itself signal — see
  `lateral_structure_map.py`).

## 7. Reproduce

```bash
# env with numpy/scipy/pandas/matplotlib, e.g. das-ambient-noise; PYTHONPATH=<repo root>
python vsg_modesep.py --config ../../param_files/modesep_params.yaml            # Steps 1-2
python vsg_modesep.py --config ../../param_files/modesep_params.yaml --sign -1  # sign control
python pick_reference_ridges.py --config ../../param_files/modesep_params.yaml  # Step 3
```

---

## Worked example — Riehen network (RI)

Basel-area urban node array, 87 virtual sources, grid 876 freqs (0.176–5 Hz) × 551 vels
(0.5–6 km/s), NFFT 4536.

- **Mode isolation** (mean column-normalized amplitude): `G_LR0` 0.998 (fund) / 0.704 (overtone);
  `G_LR1` 0.734 / 1.000; raw `ZZ` 0.967 / 0.792. Sign flip swaps them exactly.
- **Reference curves:** fundamental 225 picks, 0.52–3.94 s, 1.51–2.80 km/s (window unchanged from
  Aargau); overtone 101 picks, 1.02–3.30 s, 3.29–4.51 km/s. The overtone window was retuned to the
  faster/lower-frequency branch (f 0.24–1.05 Hz, v 3.0–4.7) and its `rough_max` relaxed 0.030 →
  **0.060 km/s** — the broad high-velocity peak jitters more than Aargau's; at 0.030 it returned 0
  picks. The Aargau values live in the `pick_windows` comments of `param_files/modesep_params.yaml`.
