# Viterbi / Dynamic-Programming Ridge Tracking on 2D Images
## Literature context for `phaseshift_dispersion.py` and `noisepy/dispersion.py`

This document records the peer-reviewed literature on Viterbi and dynamic-programming (DP)
ridge tracking in 2D spectral images, compiled via a systematic multi-agent literature search
with adversarial claim verification (June 2026). It places the Viterbi ridge tracker
implemented in this repository in the context of the broader literature.

---

## 1. The algorithm — canonical formulation

The implementation in this repository follows the standard Viterbi / minimum-cost-path
formulation for discrete 2D image ridge tracking:

```
E(x₁, x₂, …, xₙ) = Σᵢ c(xᵢ) + Σᵢ d(xᵢ₋₁, xᵢ)
```

where:
- **i** indexes sequential image columns (here: frequency bins)
- **xᵢ** is the discrete state at column i (here: velocity bin)
- **c(xᵢ)** is the **emission cost** — negative locally-contrast-enhanced amplitude, so
  bright cells are preferred
- **d(xᵢ₋₁, xᵢ)** is the **transition cost** — penalises inter-column velocity jumps
  proportionally to |Δv|, with a hard cap (1×10⁹) for jumps exceeding `max_step` km/s
- The Bellman recursion propagates left-to-right (low-f to high-f); backtracking recovers
  the globally optimal continuous path

**Source:** Ungru, A. & Jiang, X. (2017). *Dynamic programming for biomedical image
segmentation — a survey.* PMC open access.
DOI/URL: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5338725/
Confidence: **high** (verified 3-0 by adversarial agents).

### Implementation details (this repository)

Before running the DP, each image column undergoes **per-column contrast enhancement**:
```
A_enhanced[:, j] = max(A[:, j] - percentile(A[:, j], 10), 0)
```
followed by global normalisation to [0, 1]. This removes the per-column noise floor so
that uniformly bright columns (noise plateau) contribute near-zero emission signal and the
path bridges them by continuity alone, rather than anchoring on a spurious plateau.

---

## 2. Confirmed analogous methods in other fields

### 2.1 Medical imaging — coronary vessel centreline extraction

**Metz, C.T., Schaap, M., van Walsum, T., van der Giessen, A.G., Weustink, A.C.,
Mollet, N.R., Krestin, G.P. & Niessen, W.J. (2009).** Coronary centerline extraction from
CT coronary angiography images using a minimum cost path approach. *Medical Physics*, 36(12).
PMID: 20095269. URL: https://pubmed.ncbi.nlm.nih.gov/20095269/

- Extracts 3D vessel centrelines from CT volumes using a minimum-cost-path (Dijkstra-style
  graph DP) algorithm
- **Emission cost:** hybrid of vesselness enhancement measure + image intensity (two
  variants tested)
- Only 1–2 user clicks required as boundary conditions
- Validated on 252 coronaries: 88% success rate, 0.64 mm mean centreline error
- Technically Dijkstra on a 3D graph rather than Viterbi on a strict 2D lattice, but the
  minimum-cost-path formulation is directly analogous
- Confidence: **high** (verified 3-0)

### 2.2 Signal processing — IF ridge tracking on 2D time-frequency distributions

**Khan, N.A., Mohammadi, M. & Djurovic, I. (2019).** A modified Viterbi algorithm-based IF
estimation algorithm for adaptive directional time-frequency distributions.
*Circuits, Systems and Signal Processing*, 38(5), 2227–2244.
DOI: https://link.springer.com/article/10.1007/s00034-018-0960-z

- Applies a **modified Viterbi algorithm** to instantaneous frequency (IF) ridge tracking
  on 2D time-frequency distributions
- Targets multi-component scenarios where signal components intersect in the TF domain or
  suffer from low TF resolution — directly analogous to mode interference in dispersion images
- States are IF bins at each time column; transitions penalise inter-frame frequency jumps
- Uses adaptive directional TF distributions as the 2D cost surface to better resolve close
  or crossing components
- Confidence: **medium** (problem scope verified 3-0; exact DP formulation details not
  fully confirmable from abstract alone — full text recommended)

### 2.3 Canonical DP formulation survey

**Ungru, A. & Jiang, X. (2017).** [Survey on DP-based biomedical image segmentation].
*PMC open access.* URL: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5338725/

- Confirms the E = Σc + Σd formulation as the standard DP/Viterbi approach for 2D image
  ridge/curve extraction across biomedical fields
- Covers applications including vessel tracking, contour detection, and path optimisation
  in medical images
- Confidence: **high** (verified 3-0)

---

## 3. Geophysics-specific context

### 3.1 Surface-wave dispersion picking

The dominant automated picking methods in the surface-wave literature do **not** use
Viterbi or DP:

- **Levshin, A.L. & Ritzwoller, M.H. (2001).** Automated detection, extraction, and
  measurement of regional surface waves. *Pure and Applied Geophysics*, 158(8), 1531–1545.
  DOI: https://link.springer.com/article/10.1007/PL00001233
  → Uses FTAN with greedy per-period local-maximum scan and heuristic continuity rules.
  No 2D dispersion image formed; no global path optimisation.

- **Bensen, G.D., Ritzwoller, M.H., Barmin, M.P., Levshin, A.L., Lin, F., Moschetti, M.P.,
  Shapiro, N.M. & Yang, Y. (2007).** Processing seismic ambient noise data to obtain reliable
  broad-band surface wave dispersion measurements. *Geophysical Journal International*,
  169(3), 1239–1260. DOI: https://academic.oup.com/gji/article/169/3/1239/626431
  → Extends Levshin & Ritzwoller FTAN to ambient noise. Same greedy local-maximum tracker.

Recent methods (2020–2024) use **deep learning** (U-Net CNNs, residual networks) on
dispersion images. No confirmed peer-reviewed paper applying Viterbi/DP to surface-wave
phase-velocity dispersion images (f-v panels) was found in this search.

**The Viterbi ridge tracker in this repository applied to phase-shift stacked f-v images
appears to be novel in the surface-wave dispersion picking literature.**

---

## 4. Refuted candidates — papers often cited as DP/Viterbi that are not

The following papers were investigated and confirmed to **not** use Viterbi or dynamic
programming. They should not be cited as DP/Viterbi analogues.

### 4.1 Melody extraction — Salamon & Gomez (2012)

**Salamon, J. & Gomez, E. (2012).** Melody extraction from polyphonic music signals using
pitch contour characteristics. *IEEE Transactions on Audio, Speech, and Language Processing*,
20(6), 1759–1770.
URL: https://www.semanticscholar.org/paper/Melody-Extraction-From-Polyphonic-Music-Signals-Salamon-G%C3%B3mez/7db166db77f884533dfe1448ff7a15ad8b153b84

- Groups pitch candidates into contours using **auditory streaming heuristics**, not DP:
  hard continuity thresholds (max 27.5625 cents/ms pitch change; max 100 ms gap)
- Melody contour selected by scoring characteristic measures (salience, voicing, octave)
- The open-source Essentia MELODIA pipeline confirms zero Viterbi/DP step
- Confidence of refutation: **high** (verified 3-0)

### 4.2 Seismic semblance velocity picking — Beveridge et al. (2002)

**Beveridge, J.R., Ross, J., Whitley, D. & Fish, R. (2002).** Automated velocity analysis
using a heuristic search. *Machine Vision and Applications*, 14(1), 23–34.
DOI: https://link.springer.com/article/10.1007/s001380100068

- Uses **bit-string hill climbing with random restarts** (heuristic combinatorial local
  search) to select a polyline through detected semblance peaks
- Scoring: summed velocity + smoothness + proximity to local median — heuristic objectives,
  not probabilistic emission/transition costs
- Full-text search confirms zero hits for "Viterbi", "shortest path", "Dijkstra", "trellis",
  "Bellman"
- Confidence of refutation: **high** (verified 3-0)

### 4.3 Geophysics velocity surface picking — Decker & Fomel (2022)

**Decker, L. & Fomel, S. (2022).** Automatic velocity picking with a variational approach.
*Geophysics*, 87(3), U69–U80.
DOI: https://library.seg.org/doi/full/10.1190/geo2021-0336.1

- Minimises a **variational cost functional** (nonlinear elliptic PDE) via continuation,
  discretised and solved with L-BFGS (gradient-based continuous optimisation)
- No discrete trellis, no Bellman recursion — fundamentally different approach from DP
- Uses a continuation strategy over progressively less-smoothed semblance scans to avoid
  local minima (analogous problem to DP's global optimality guarantee, but different method)
- Confidence of refutation: **high** (verified 3-0)

---

## 5. Open questions from the literature search

1. **Depalle, Garcia & Rodet (1993, ICASSP)** on HMM partial tracking in spectral analysis
   for additive synthesis — commonly cited as an early Viterbi application to spectrogram
   ridges but adversarial verification was inconclusive (1-2 vote). Full text access needed
   to confirm whether Viterbi decoding was actually used on a 2D spectral lattice.

2. **Khan et al. (2019)** — exact DP formulation (states, emission cost, transition cost,
   hard constraints) not verifiable from abstract alone. Specifically: how is the adaptive
   directional TF distribution incorporated into the emission cost, and are hard continuity
   caps used?

3. **Published comparisons** between Viterbi/DP ridge tracking and variational PDE methods
   (Decker & Fomel 2022) or heuristic local-search methods (Beveridge et al. 2002) on the
   same dispersion or semblance panels — none found.

4. **Ultrasonics / guided wave dispersion** — multiple papers in this space were identified
   but none confirmed to use Viterbi/DP specifically. Synchrosqueezed wavelet transforms and
   clustering methods appear to dominate (not DP).

---

## 6. Suggested methods section text

For use in papers describing the ridge tracker in this repository:

> Ridge tracking on the stacked phase-shift dispersion image follows the canonical
> dynamic-programming formulation of Ungru & Jiang (2017), applied here to surface-wave
> phase-velocity images: at each frequency column, velocity bins are discrete states;
> the emission cost is the negative locally-contrast-enhanced amplitude (per-column
> 10th-percentile noise floor subtracted, then globally normalised); and the transition
> cost penalises inter-column velocity jumps proportionally to |Δv| with a hard continuity
> cap. This is the dispersion-image analogue of the Viterbi IF ridge tracker of Khan et al.
> (2019) and the minimum-cost-path centreline extractor of Metz et al. (2009).
> To our knowledge, no prior published method has applied this formulation directly to
> ambient-noise phase-velocity f–v images.

---

## 7. Search metadata

- **Search conducted:** 2 June 2026
- **Method:** Multi-agent deep-research harness (5 parallel search angles, 24 sources
  fetched, 35 claims extracted, 25 verified by 3-vote adversarial process, 11 refuted)
- **Search angles covered:** (1) speech/audio spectrogram ridge tracking; (2) ultrasonics /
  guided wave / Lamb wave dispersion; (3) medical imaging ridge/vessel extraction; (4) signal
  processing TF ridge algorithms; (5) seismic velocity analysis (NMO semblance, surface waves)
- **Limitation:** corpus-bounded search — exhaustive coverage of all geophysics and
  near-surface journals not guaranteed. Papers behind paywalls with no accessible abstract
  could not be verified.

---

*File generated by Claude Code from adversarially-verified multi-agent web research.*
*All claims marked high-confidence verified 3-0 by independent adversarial agents.*
*Claims marked medium-confidence partially verified; full-text access recommended.*
