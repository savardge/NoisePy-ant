# Handling mode osculation in FTAN picking — advisory note

Based on Bhaumik & Cox (2026, *Eng. Geol.* 369, 108834), "Predominant-mode inversion of
surface waves." Advisory only — no code changes were made when this note was written.

## Context

Goal: pick group and phase velocity dispersion curves more reliably when modes
osculate ("kiss"). The paper's core result: at sites where the shear-velocity contrast
exceeds ~2, modal energy transfers **smoothly** from the fundamental (R0) to the first
higher mode (R1) at low frequencies, around an **osculation point (OP)**. Because real
dispersion data (merged MASW + MAM, or ambient-noise cross-correlations) have limited
spatial resolution, this transition looks like a single continuous curve, and is routinely
mis-picked as pure fundamental — causing **overestimated deep Vs and mislocated bedrock**
(their Fig. 2c–d vs e–f).

Their solution is on the **forward/inversion** side: instead of mode-indexing by eye,
define the **predominant mode** at each frequency as the mode carrying the **maximum
normalized vertical surface amplitude** (Eq. 7: `k_pred(ω) = argmax_m |φ̂_z,m(0; k_m, ω)|`),
and invert against that curve. The data inherently records the predominant mode, so no
manual mode numbering or source–receiver geometry is needed.

This is a near-surface geotechnical paper (MASW/MAM, metres, Hz), whereas NoisePy-ant is
basin/crustal-scale ANT (km layers, periods to ~12 s in `reference_model.py`). **The
physics is scale-invariant** — osculation from strong impedance contrast, and
predominant = max surface amplitude, apply identically to your dispersion images.

Targets both group and phase velocity. Ellipticity is treated as a *secondary* cross-check
only (see caveat in §5).

---

## 1. The key reframing for the FTAN workflow

The amplitude-following ridge — `extract_dispersion_viterbi` / `extract_dispersion`
(argmax) on the `|CWT|²` image in `noisepy/dispersion.py` — *already* tracks the
energy-dominant branch. For vertical-component Rayleigh that is, by definition, the
predominant mode. **So the picking machinery is philosophically aligned with the paper
already.** The two failure modes to guard against are:

1. **Hiding the amplitude information** by over-normalizing the image (per-period
   normalization), so you can no longer see *which* branch is predominant.
2. **Mislabeling the result** — calling the whole continuous curve "fundamental" when it
   is physically R0-below-OP then R1-above-OP. This is an *interpretation/inversion*
   error, not a picking error.

A crucial distinction: **osculation is kissing, not crossing.** Because the branches come
very close at the OP, *continuity is your friend* — a smooth amplitude ridge carries you
correctly from R0 onto R1 through the kiss. The mistake is purely interpretive. (Contrast
with a true mode *crossing*, where continuity would mis-track.)

---

## 2. See the energy transfer: image normalization

This is the single most actionable change and it costs nothing — it is just a config
choice already supported.

- `disp_image_from_cwt(..., norm='per_period')` makes every period row equally bright.
  Great for ridge tracking, but it **erases the amplitude-transfer signature** of
  osculation — you literally cannot see R0 dimming and R1 brightening.
- `norm='global'` (or `'none'`) preserves relative amplitude across periods, so the
  R0→R1 energy hand-off becomes visible as the bright ridge migrating to the higher-
  velocity branch as period grows. This is exactly the paper's Fig. 1/3 picture.

**Recommendation:** when inspecting for osculation, view the **global-normalized** image.
`scripts/picking/show_unnormalized_ftan.py` already contrasts global vs per-period (Esteve
2025 / Shirzad 2025). Use it as the osculation diagnostic. Keep per-period only for the
final continuity-tracking pass.

---

## 3. Group velocity under osculation

1. **Pick on the global-normalized image** so the ridge follows true energy, not
   per-period-rescaled energy. Otherwise a per-period argmax can lock onto the *persistent
   but weaker* R0 continuation past the OP instead of the now-predominant R1.
2. **Keep amplitude (emission) dominant over smoothness in the Viterbi.** In
   `extract_dispersion_viterbi`, the energy ridge should win; because the branches kiss,
   the `max_step` hard cap will not be tripped at the OP (the velocity jump is small by
   definition), so a moderate `smooth_weight` and `max_step` track the predominant curve
   smoothly across it. Do **not** raise `smooth_weight` so high that the ridge resists the
   small upward velocity step at the OP.
3. **Watch `SHORT_PRIORITY` / short-period anchoring.** Anchoring to strong short-period
   (high-frequency) peaks and following by continuity is correct *as long as the branches
   kiss* — continuity then lands you on R1 above the OP automatically. The anchoring only
   becomes dangerous if the branches separate appreciably (more crossing-like); in that
   regime, lower the anchoring weight near the OP.
4. **Use topology candidates as the osculation flag** (see §6): where
   `extract_curves_topology` returns two comparable-persistence peaks per period over a
   band, you are in the transition zone.

Net: the group ridge is mostly fine *if* you pick on the global image; the real work is
flagging the transition band and labeling, not changing the tracker.

---

## 4. Phase velocity under osculation — the reference curve is everything

The phase pipeline (`phase_velocity`, `resolve_phase_curve` joint Viterbi,
`measure_corrections_and_phase`) resolves the 2πN ambiguity by minimizing
`Σ|c(N) − c_ref|/c_ref + w·Σ|Δc|/c` against a **reference curve** `c_ref`.

**The danger:** if `c_ref` is the *fundamental-only* (mode-0) disba curve, then at and
beyond the OP the true predominant data lives on R1, and resolving 2πN against an R0
reference can snap onto the wrong N branch → wrong phase velocity exactly where it matters.

**Recommendation:** build `c_ref` (the `REFERENCE_CURVES` in
`scripts/picking/dispersion_curves_V5.py`) as the **predominant-mode** curve — i.e. R0
below the OP, switching to R1 above it (§5). This guides `resolve_phase_curve` onto the
correct branch through the transition. Everything downstream (`U_from_phase` consistency
via `group_from_phase`, Bensen eq. 7) then checks against the right branch too.

Also: the phase-velocity *image* (`phase_velocity_image` / `phase_image_from_cwt`) shows
2πN fringes; the **brightest** crest is the predominant branch. Picking among fringes by
brightness is the phase-domain analogue of the predominant-mode rule — consistent with how
the phase image is already built from the group ridge.

---

## 5. Building a predominant-mode reference curve with disba

This is the one genuinely new computation, and it is feasible with the existing
`disba` 0.7.0 — but with an important normalization caveat.

**What works:** `disba.EigenFunction(th, vp, vs, rho)(T, mode, wave='rayleigh')` returns
`(depth, ur, uz, tz, tr, period, mode)`. The surface vertical eigenfunction is `uz[0]` —
exactly `φ̂_z,m(0)` in Eq. 7.

**The caveat (verified):** disba normalizes Rayleigh eigenfunctions to `uz(0)=1` for
*every* mode (CPS convention). So you **cannot** compare raw surface `uz` across modes —
they are all 1. You must **renormalize each mode to a consistent energy normalization**
before comparing surface amplitude. Practical recipe per period T, per mode m:

1. Get the depth eigenfunctions `(depth, ur, uz)` from disba.
2. Compute the Rayleigh energy integral `I_m = ∫ ρ(z)[u_r(z)² + u_z(z)²] dz`
   (trapezoid over `depth`; this is the quantity the paper's orthonormalization
   `Lᵀ A R = Λ^{1/2}` fixes).
3. The comparable surface amplitude is `A_m(T) = |uz(0)| / sqrt(I_m) = 1/sqrt(I_m)`
   (since disba's `uz(0)=1`).
4. **Predominant mode** `m*(T) = argmax_m A_m(T)`. The predominant phase/group curve is
   the phase/group velocity of `m*` at each T (per-mode velocities already extracted in
   `scripts/picking/reference_model.py`).

**This must be validated** — the exact normalization is the subtle part. Cross-check
options: (a) against the paper's Fig. 1/3 modal-amplitude maps for a comparable 2-layer
model; (b) against Geopsy `gpdc` (which can output mode amplitude / medium response);
(c) disba exposes `srfker96`/`swegn96` (CPS energy-kernel programs) — the energy integrals
may be obtainable directly rather than by hand-integration. Treat the hand-integrated
recipe as a first cut to be confirmed before trusting the OP location quantitatively.

**Output:** extend `reference_curves()` to also return `(wave, kind, 'predominant')` and a
per-period mode-index array. The OP is then simply where that mode-index switches 0→1.
Love (TT) is analogous using the Love horizontal eigenfunction `uu(0)`.

---

## 6. Detecting and flagging the osculation point

Combine three independent signals; agreement raises confidence.

1. **Image-based (primary, data-driven):** at each period, take the top-2 amplitude peaks
   (`extract_curves_topology` already returns multiple candidates with persistence
   scores). Where the two strongest peaks have *comparable* amplitude/persistence over a
   contiguous period band, that band is the transition zone. Record an amplitude-ratio
   metric per pick.
2. **Theory-based:** the OP is where the predominant-mode index (§5) switches 0→1 for the
   model. Gives an expected OP period/frequency to compare against the data.
3. **Ellipticity (secondary cross-check only):** the radial surface eigenfunction `ur(0)`
   flips sign (prograde→retrograde) near the OP (paper Fig. 13; observed in the model via
   disba). There is an HV pipeline and 3-component data (RR/RZ/ZR), so this is an available
   comparison. **Caveat:** the HV/ellipticity here does not match HVSR well (and the HV
   pipeline had a global-argmax bug), so use ellipticity only as a soft corroboration of
   the OP frequency, never as the primary picker.

**Suggested artifact (if later implemented):** a per-pick `mode_ambiguity` /
`transition_flag` column in the V5 CSV, plus an OP marker on the quicklook — mirroring how
the paper flags transition frequencies.

---

## 7. Don't force the fit at the transition — for inversion

The paper's other practical lesson (their Fig. 10, Model k): near the OP the data are
*effective-mode* and match no single theoretical branch (relative misfit `m_dr > ~5%`).
Forcing those points onto one branch biases the velocity profile. They **exclude** the
transition-band points and re-invert, which tightens and de-biases the result.

Downstream of picking: **flag transition-band picks (§6) so they can be down-weighted or
excluded** when building an inversion target, rather than trusting them. And if/when you
invert, the misfit should be against the **predominant-mode** forward curve (§5), not the
fundamental — that is the paper's central message.

---

## 8. Summary of recommendations

| Topic | Recommendation | Where it touches |
|---|---|---|
| Diagnose osculation | View **global-normalized** FTAN image, not per-period | `show_unnormalized_ftan.py`, `disp_image_from_cwt(norm=...)` |
| Group pick | Pick on global image; keep amplitude dominant; don't over-smooth across OP | `extract_dispersion_viterbi` |
| Phase pick | Make `c_ref` the **predominant-mode** curve so 2πN resolves the right branch | `REFERENCE_CURVES`, `resolve_phase_curve` |
| Reference model | Add predominant-mode selection via disba `EigenFunction` + **energy renormalization** (`uz(0)=1` caveat) | `reference_model.py` |
| OP detection | Top-2 comparable peaks (data) + mode-index switch (theory); ellipticity as soft check only | `extract_curves_topology`, `EigenFunction`/`Ellipticity` |
| Inversion | Flag/exclude transition-band picks; invert against predominant mode | downstream |

**Biggest, cheapest win:** switch the osculation diagnostics to the global-normalized
image (§2) and rebuild the phase reference as a predominant-mode curve (§4–5). Those two
address the bulk of the mis-pick risk before any new tooling.

**Caveats to keep front of mind:** (1) the energy-renormalization formula in §5 needs
independent validation before the OP location is quantitatively trusted; (2) ellipticity is
corroboration only given the HV/HVSR mismatch; (3) this is all on the picking/reference
side — realizing the full benefit requires a predominant-mode *inversion* target (§7),
which is out of scope here.

## References
- Bhaumik & Cox (2026), *Eng. Geol.* 369, 108834 — the subject paper.
- `disba` 0.7.0: `EigenFunction`, `Ellipticity`, `srfker96`, `swegn96` (eigenfunctions &
  energy kernels available locally).
- Background on mode kissing: Boaga et al. (2013), Gao et al. (2016), Kausel et al. (2015);
  Foti et al. (2018) on smooth transitions from limited resolution.
