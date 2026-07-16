# Stacking choices and velocity picking on the geophone networks — findings

Two related studies, one conclusion: **on Aargau/Riehen, stacking choices move SNR a lot and
velocity-pick accuracy not at all.** §1–6 = regime-clustered restacking; §7 = tf-PWS vs plain PWS.

---

# Part 1 — Regime-clustered CCF restacking

**Question (2026-07-14):** can dispersion picking on the 3C geophone nets (Aargau AA, Riehen RI)
benefit from clustering CCF time-window substacks into noise regimes and restacking per regime —
as done for DAS in `masw-das` / `das-ambient-noise` — exploiting the 9-component redundancy?

**Answer: no, not for velocity picking.** Per-pair clusters do carry real physics (Love/TT SNR
+6.5%, verified against a random control), but **dispersion accuracy is unchanged at chance level**,
so there is nothing to gain for picking. Stacking is not the lever for these datasets; pick-level QC
is (see `UNIFIED_PICKING_SUMMARY.md`). Do not redo without a materially different premise (§6).

---

## 1. Enabling fact (useful beyond this study)

The per-window CCF substacks **already exist** in the stack files — no re-correlation is ever needed
for window-level experiments:

- `STACK_CH{AA,RI}_normZ/<NET>.<src>/<pair>.h5` → `AuxiliaryData/T<epoch>` groups
- Aargau **204 windows/pair** (median 4.0 h spacing, Dec 2020); Riehen **221** (1.2 h, Sept 2022)
- each T-group holds the **full unrotated 9C ENZ tensor** (EE,EN,EZ,NE,NN,NZ,ZE,ZN,ZZ) + attrs
  (`dist`, `dt`, `azi`, `baz`, `ngood`)
- recipe: subset-sum T-groups → `noisepy.stacking.rotation` (ENZ order above → RTZ
  ZR,ZT,ZZ,RR,RT,RZ,TR,TT,TZ). Production stacks in ENZ then rotates; keep that order (PWS is
  non-linear, so rotate-then-stack ≠ stack-then-rotate).
- caveat: Riehen pairs average only ~41 *usable* windows after feature extraction → 18/60 pairs met
  the ≥4×min_win bar; Aargau 60/60.

## 2. Scripts (all in `NoisePy-ant/scripts/picking/`)

| script | role |
|---|---|
| `regime_restack_pilot.py` | pass A: per-(pair,window) 9C physical features → `window_features.npz` (**reused by the others — keep**); plus the pooled-clustering + cross-pair-consensus variant (§5) |
| `per_pair_regime_stack.py` | **the correct design**: clusters each pair independently, stacks each cluster + all-window, measures per wave type, includes the random control |
| `regime_select_restack.py` + `regime_pilot_score.py` | per-pair median-split-on-ellipticity variant (all/top50/bot50) (§5) |

Features per (pair, window): RTZ energy partition, per-window `env_ratio` (TT vs cross-terms),
**Z–R 90° elliptical coherence** (the exact quantity the G_LR mode synthesis relies on),
causal/acausal asymmetry (directivity), log energy. Outputs in `Projects/<net>/regime_pilot/`.

## 3. ⚠ Method lesson — the winner's curse (this reversed the conclusion twice)

"Best cluster beats the all-window stack" is a **post-hoc max over k clusters** and is *not a test*.
A **size-matched random partition** (each pair's window labels shuffled into clusters of identical
sizes) carries both the winner's curse *and* the fewer-windows SNR handicap, and none of the
physics. Only **real vs random** is meaningful:

| Aargau `ridge_dev` | vs all-window stack | vs **random partition** |
|---|---|---|
| ZZ | 83 % of pairs "win" | **47 %** (chance) |
| G_LR0 | 69 % | **52 %** (chance) |
| TT | 77 % | **49 %** (chance) |

Every impressive number evaporated. Related trap: the specialization chance level is **(k−1)/k per
pair, not 50 %** — for the observed k-mix it was 52 % (Riehen) / 60 % (Aargau).

## 4. Results — per-pair clustering (the correct design)

Each pair clustered independently (own k∈2..4 by silhouette; median silhouette ≈ 0.20), each
cluster stacked + an all-window stack through one linear code path, measured per wave type against
the external network VSG group reference. **Best-real vs size-matched RANDOM (50 % = no information):**

| metric | Riehen | Aargau | verdict |
|---|---|---|---|
| **TT (Love) `snr_med`** | **67 %** (12/18), **+6.4 %** | **70 %** (42/60), **+6.6 %** | **REAL, both nets** |
| G_LR0 `snr_med` | 78 % (14/18), +5.1 % | 48 %, −1.5 % | mixed (Riehen only) |
| ZZ `snr_med` | 50 % | 47 % | nothing |
| **`ridge_dev` (all comps)** | 33–62 % | 47–52 % | **nothing anywhere** |
| `n_pick` | 22–33 % | 25–40 % | nothing (random ≥ real) |
| specialization (best cluster R vs L differs) | 47 % (chance 52 %) | 54 % (chance 60 %) | **none** |

**Interpretation.** Per-pair clusters genuinely isolate windows with more SH/Love energy (+6.5 % TT
SNR, consistent across both networks, above the random control — presumably source-azimuth/type
variation). But: (a) the *same* cluster tends to win for both Rayleigh and Love ⇒ the regimes are
"good vs bad coherence", **not** "Rayleigh-illuminating vs Love-illuminating"; and (b) **the SNR
gain does not propagate to dispersion accuracy** — `ridge_dev` vs the external reference is at
chance everywhere. The all-window stack already has ample SNR and the ridge simply does not move.
Production `Allstack_pws` also already applies soft continuous coherence weighting, i.e. adaptive
window de-emphasis at stack time — hard selection is partly redundant with it.

## 5. Earlier variants (negative, superseded, but instructive)

- **Pooled clustering + cross-pair epoch consensus** (the `masw-das` design — valid for DAS where
  channels share a fibre): silhouette 0.11–0.156; consensus agreement **0.41** (random 0.25; AFL
  used 0.7); collapses to a single dominant label; no diurnal structure. **Why it cannot work here:**
  window quality varies **15× WITHIN a pair** (ellipticity p10 0.01 → p90 0.15) but **incoherently
  across the array** (epoch-median only 0.05–0.07 network-wide) ⇒ regimes are *pair-local*;
  cross-pair consensus is simply the wrong assumption for geophone networks.
- **Per-pair median split on ellipticity** (all / top50 / bot50; top and bot have equal window
  counts): the all-window stack won on both networks against the external VSG reference MAD —
  Riehen 0.274 (all) vs 0.281 (top) vs **0.257 (bot — the *worst* half was better)**; Aargau 0.099
  (all, best) vs 0.103 vs 0.106. Effects are small (2–9 %) and **inconsistent in sign**. Halving the
  stack (~1.4× SNR loss) costs more than any coherence gain.
  - *One isolated positive:* Riehen graben Love-contamination ordered exactly as physics predicts —
    rf-coincidence (|dU_rayfund| ≤ 0.15, T 1.3–1.7 s, W-graben pairs) **top50 0.257 < all 0.298 <
    bot50 0.345**. Ellipticity selection does suppress R-fund-on-TT leakage — it just never reaches
    the dispersion measurement.

## 6. Conclusion & when to revisit

**Stacking is not the lever for these datasets — pick-level QC is.** Revisit only with a materially
different premise:
- many more windows per pair (so halving the stack is cheap),
- a network with discrete strong sources (the AFL DAS case, where regimes were genuinely discrete),
- selection applied **on top of pws** rather than compared against linear,
- or a **TT/Love-specific** need where +6.5 % SNR matters for its own sake (the one live thread).

---

# Part 2 — §7. tf-PWS vs plain PWS vs linear (`stack_method_test.py`)

**Question:** does tf-PWS (wavelet-domain, Ventosa et al. 2017) actually beat plain PWS
(time-domain, Schimmel & Paulssen 1997) or linear — *for velocity picking*?

Tested at the **two independent levels** where the choice enters, each isolating the other, scored
only against the **external VSG reference** (`phase_absdev` = median |c − c_ref| is primary; it
includes bias, unlike `phase_mad` which is scatter-only). Paired design (same pairs, every method),
n = 55 pairs/network (40–46 contribute a phase measurement), **binomial p-values in the report**.
Self-consistency check passed: the shared configuration (window=pws + synth=tf-PWS) returns
identical numbers in both experiments.

### EXPERIMENT A — time-window stacking (the production `Allstack_pws` choice), baseline = pws

| vs pws | Aargau | Riehen |
|---|---|---|
| **`phase_absdev`** tf-PWS | 54 %, −2.5 %, **p=0.66 n.s.** | 60 %, −3.1 %, **p=0.28 n.s.** |
| **`phase_absdev`** linear | 52 %, −4.4 %, **p=0.88 n.s.** | 64 %, −7.3 %, **p=0.09 n.s.** |
| `snr_med` tf-PWS | 78 %, **+111 %**, **p<0.001 SIG** | 91 %, **+79 %**, **p<0.001 SIG** |

⇒ **tf-PWS roughly doubles SNR — and buys exactly nothing in velocity accuracy** (every accuracy
comparison n.s. in both networks). It is also very expensive (a CWT per window per component: ~200×9
per pair; 10 pairs exhausted a 10-minute budget). **Keep production `Allstack_pws`.**

### EXPERIMENT B — G_LR mode-synthesis stack (`unified_picking.Config.GLR_STACK`), baseline = tf-PWS

| vs tf-PWS | Aargau | Riehen |
|---|---|---|
| `phase_absdev` linear | 55 %, −2.2 %, p=0.64 n.s. | 50 %, −0.1 %, p=1.00 n.s. |
| `phase_absdev` pws | 45 %, +10 %, p=0.64 n.s. | 65 %, −6.7 %, p=0.08 n.s. |
| **`n_pick` linear** | 27 %, **−14 %**, **p=0.001 SIG WORSE** | 31 %, **−6.7 %**, **p=0.006 SIG WORSE** |
| **`ridge_dev` pws** | 32 %, **+28 %**, **p=0.028 SIG WORSE** | 57 %, −1.5 %, p=0.45 n.s. |

⇒ No method wins on phase accuracy. But **linear yields significantly FEWER usable picks in both
networks**, and pws is significantly worse on group accuracy in Aargau. **Keep `GLR_STACK='tfpws'`**
— same accuracy, more usable picks, and it matches Nayak & Thurber's published real-data choice.

### The cross-cutting result

Three independent lines of evidence in this study — regime clusters (+6.5 % Love SNR), quality
selection, and tf-PWS (+80–110 % SNR) — **all improve SNR and none improve velocity accuracy.**
That is a property of these datasets: the stacks already have ample SNR, so what limits pick
accuracy lies elsewhere (mode mixing, path heterogeneity, reference quality, the 2πN branch), and
**more SNR does not move the ridge**. Corollary for future work: *stop optimizing the stack; optimize
the pick and its QC* (`UNIFIED_PICKING_SUMMARY.md`). Note `n_pick` (usable band/coverage) is the one
place a stacking choice does matter, and there tf-PWS wins for the synthesis step.
