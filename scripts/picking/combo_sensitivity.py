#!/usr/bin/env python3
"""Depth-sensitivity kernels per waveset combination, and SENSITIVITY-MASKED comparisons.

The raw combo comparison (waveset_combo_summary.py) averaged |dVs| over the full 0-6 km grid.
That conflates two different things: genuine data conflict, and each posterior reverting to
its prior at depths its own data never constrained. A Love-fundamental-only run has little
sensitivity below ~lambda/3 of its longest period; disagreement with Rayleigh THERE is not
evidence of anything.

For each (well, combo) this script:
  1. rebuilds the combo's target set (same curves + v1 period trims the runs used, read from
     the run's own bayhunter_result.npz `waves` + observed periods),
  2. computes finite-difference 1D sensitivity kernels on a COMMON reference model per well
     (the full-combo BayHunter posterior median), layered at DZ km:
     K_j = sum over targets,periods of |d pred / d Vs_j| / sigma. A common model makes the
     kernels comparable across combos (same model, different DATA sets) and sidesteps a real
     failure mode: a collapsed near-uniform posterior (Otterbach-2 L0g) supports no Love
     fundamental at all, so kernels on that run's own model are undefined.
  3. defines the sensitive depth range: layers with K above KFRAC of that combo's max K
     AND above z95 (the depth where cumulative sensitivity reaches 95% -- the pointwise cut
     alone is too permissive because kernels decay slowly), plus z90 for reference,
  4. recomputes the pairwise combo |dVs| matrices restricted to MUTUALLY sensitive depths,
  5. re-evaluates the Love-vs-Rayleigh depth split within the mutual mask.

Outputs (beside the well dirs): combo_sensitivity.csv, per-well combo_kernels.png,
combo_disagree_masked.csv.
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/genevievesavard/Codes/NoisePy-ant")
from noisepy import vs_inversion as vi                     # noqa: E402

EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
COMBOS = ["R0g", "R0gR1g", "R0gR0p", "L0g", "L0gL0p", "R0gL0g", "R0pL0p", "R0gL0gR0pL0p"]
COL = dict(zip(COMBOS, plt.cm.tab10(np.linspace(0, 0.9, len(COMBOS)))))
DZ = 0.25          # kernel layer thickness [km]
ZMAX = 6.0
KFRAC = 0.10       # (legacy summed-kernel cut; kept for the figure reference line)
UNION_FRAC = 0.30  # a layer is "sensitive" if ANY period's own-normalized kernel >= this.
                   # The summed kernel is biased shallow: the CWT ladder is geometric, so
                   # short periods outnumber long ones ~3:1 and dominate the sum, and a
                   # pointwise cut on the sum compares deep layers to the shallow peak.
                   # Per-period normalization asks the right question: does ANY datum
                   # constrain this depth? (Riehen Love: 1.6 km summed -> ~2.4 km union.)
DV = 0.03          # +-3% Vs perturbation for the finite difference


def layered_model(depth, vs_median, zmax=None):
    """Posterior median resampled onto the DZ kernel layers (last layer = half-space)."""
    edges = np.arange(0.0, (zmax or ZMAX) + DZ, DZ)
    mids = 0.5 * (edges[:-1] + edges[1:])
    vs = np.interp(mids, depth, vs_median)
    th = np.full(len(mids), DZ)
    th[-1] = 100.0
    return th, vs, edges


def kernels_for_run(npz, ref_depth, ref_vs, zmax=None):
    """(edges, K[nlayer]) sigma-weighted sensitivity of this run's target set, on the
    common per-well reference model."""
    z = np.load(npz, allow_pickle=True)
    th, vs, edges = layered_model(ref_depth, ref_vs, zmax)
    K = np.zeros(len(vs))
    rows = []                                   # per-(target,period) kernels for the union mask
    for w in [str(x) for x in np.atleast_1d(z["waves"])]:
        T = z[f"obsT_{w}"]
        sig = z[f"obssig_{w}"] if f"obssig_{w}" in z else np.ones(len(T))
        disba_wave, mode, meas = vi.curve_def(w)
        Kp = np.zeros((len(T), len(vs)))
        for j in range(len(vs)):
            # TWO-SIDED difference. One-sided from the unperturbed model fails on a real
            # edge case: a near-uniform posterior median (e.g. an L0g run that collapsed to
            # ~constant Vs) supports NO Love fundamental -- disba returns all-NaN for the
            # base model, while either perturbation restores the waveguide.
            vp_ = vs.copy(); vp_[j] *= (1 + DV)
            vm_ = vs.copy(); vm_[j] *= (1 - DV)
            up = vi.dispersion_velocity(th, vp_, T, mode, measure=meas, disba_wave=disba_wave)
            dn = vi.dispersion_velocity(th, vm_, T, mode, measure=meas, disba_wave=disba_wave)
            d = np.abs(up - dn) / (2 * vs[j] * DV)          # |dU/dVs_j|, km/s per km/s
            ok = np.isfinite(d) & np.isfinite(sig) & (sig > 0)
            K[j] += float(np.sum(d[ok] / sig[ok]))
            Kp[:, j] = np.where(np.isfinite(d), d, 0.0)
        rows.append(Kp)
    Kp = np.vstack(rows) if rows else np.zeros((0, len(vs)))
    mx = Kp.max(axis=1, keepdims=True)
    un = np.zeros(len(vs))
    good = (mx[:, 0] > 0)
    if good.any():
        un = (Kp[good] / mx[good]).max(axis=0)   # union: best own-normalized kernel per layer
    return edges, K, un


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="test_2026-08-06_waveset_combos")
    a = ap.parse_args()
    wells = sorted(glob.glob(f"{EHM}/*/tomo/2_vs_depth_inversion/tests/{a.tag}/*_cell_*"))

    import pandas as pd
    sens_rows, dis_rows = [], []
    for wdir in wells:
        net = wdir.split("/Projects/")[1].split("/")[0]
        well = os.path.basename(wdir).rsplit("_cell_", 1)[0]
        refnpz = os.path.join(wdir, "R0gL0gR0pL0p", "bayhunter_result.npz")
        if not os.path.exists(refnpz):
            print("  %s: no full-combo reference run -- skipped" % well)
            continue
        ref = np.load(refnpz, allow_pickle=True)
        ref_depth, ref_vs = ref["depth"], ref["vs_median"]
        # Kernel box = the POSTERIOR's own box. Interpolating the reference model past the
        # inversion's depth_max flat-extends the half-space, so kernels computed there would
        # describe a model the run never sampled. If the union mask then reaches the bottom
        # layer, the box itself was too shallow for the data -- reported, not extrapolated.
        zmax = float(ref_depth.max())
        edges = None
        K, mask, prof = {}, {}, {}
        for combo in COMBOS:
            npz = os.path.join(wdir, combo, "bayhunter_result.npz")
            if not os.path.exists(npz):
                continue
            edges, k, un = kernels_for_run(npz, ref_depth, ref_vs, zmax)
            z = np.load(npz, allow_pickle=True)
            if not np.isfinite(k).any() or k.max() <= 0:
                # an all-zero kernel would make the mask vacuously all-True; exclude instead
                print(f"  {well:<13}{combo:<14} KERNEL FAILED (forward undefined) -- excluded")
                sens_rows.append(dict(net=net, well=well, combo=combo, z90_km=np.nan,
                                      z_sens_min=np.nan, z_sens_max=np.nan, n_sens_layers=0))
                continue
            K[combo] = k
            mids = 0.5 * (edges[:-1] + edges[1:])
            prof[combo] = np.interp(mids, z["depth"], z["vs_median"])
            c = np.cumsum(k) / k.sum()
            z90 = float(mids[np.searchsorted(c, 0.90)])
            # UNION criterion (see UNION_FRAC comment); half-space layer excluded -- a kernel
            # peaking there mostly senses below the box, not the box bottom
            mask[combo] = un >= UNION_FRAC
            mask[combo][-1] = False
            zsen = mids[mask[combo]]
            sens_rows.append(dict(net=net, well=well, combo=combo,
                                  z90_km=round(z90, 2),
                                  z_sens_min=round(float(zsen.min()), 2),
                                  z_sens_max=round(float(zsen.max()), 2),
                                  n_sens_layers=int(mask[combo].sum())))
            print(f"  {well:<13}{combo:<14} z90={z90:4.1f} km  sensitive "
                  f"{zsen.min():.1f}-{zsen.max():.1f} km")
        if not K:
            continue
        mids = 0.5 * (edges[:-1] + edges[1:])

        # masked pairwise disagreement
        have = [c for c in COMBOS if c in K]
        for i, ci in enumerate(have):
            for cj in have[i + 1:]:
                m = mask[ci] & mask[cj]
                full = float(np.mean(np.abs(prof[ci] - prof[cj])))
                inmask = float(np.mean(np.abs(prof[ci] - prof[cj])[m])) if m.any() else np.nan
                dis_rows.append(dict(net=net, well=well, a=ci, b=cj,
                                     dvs_full=round(full, 3),
                                     dvs_masked=round(inmask, 3),
                                     mutual_z_max=round(float(mids[m].max()), 2) if m.any() else np.nan,
                                     n_mutual=int(m.sum())))

        # figure: kernels + masked-vs-full L0g/R0g difference
        fig, axs = plt.subplots(1, 3, figsize=(15, 6.2))
        for combo in have:
            axs[0].plot(K[combo] / K[combo].max(), mids, "-", color=COL[combo], lw=1.9,
                        label=combo)
        axs[0].axvline(KFRAC, color="0.5", ls="--", lw=1, label=f"mask cut ({KFRAC:g})")
        axs[0].set_ylim(ZMAX, 0); axs[0].grid(alpha=0.3)
        axs[0].set_xlabel("K / max(K)"); axs[0].set_ylabel("depth [km]")
        axs[0].set_title("sigma-weighted sensitivity per combo\n(finite-diff on the common full-combo median)",
                         fontsize=10)
        axs[0].legend(fontsize=7.5)
        for combo in have:
            v = np.where(mask[combo], prof[combo], np.nan)
            axs[1].plot(prof[combo], mids, "-", color=COL[combo], lw=0.9, alpha=0.35)
            axs[1].plot(v, mids, "-", color=COL[combo], lw=2.2)
        axs[1].set_ylim(zmax, 0); axs[1].grid(alpha=0.3)
        axs[1].set_xlabel("Vs [km/s]")
        axs[1].set_title("posterior medians\nbold = within own sensitivity, faint = prior-dominated",
                         fontsize=10)
        if "R0g" in K and "L0g" in K:
            m = mask["R0g"] & mask["L0g"]
            d = prof["L0g"] - prof["R0g"]
            axs[2].plot(d, mids, "-", color="0.6", lw=1.2, label="full grid")
            axs[2].plot(np.where(m, d, np.nan), mids, "-", color="crimson", lw=2.4,
                        label="mutual sensitivity")
            axs[2].axvline(0, color="k", lw=1)
            axs[2].set_ylim(zmax, 0); axs[2].grid(alpha=0.3)
            axs[2].set_xlabel("Vs(L0g) - Vs(R0g) [km/s]")
            axs[2].set_title("Love-only minus Rayleigh-only\n(only the red part is evidence)",
                             fontsize=10)
            axs[2].legend(fontsize=8)
        fig.suptitle(f"{net} — {well}: depth sensitivity by input combination",
                     fontsize=12.5, fontweight="bold")
        fig.tight_layout()
        p = os.path.join(wdir, "combo_kernels.png")
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("wrote", p)

    outdir = os.path.dirname(wells[0])
    pd.DataFrame(sens_rows).to_csv(os.path.join(outdir, "combo_sensitivity.csv"), index=False)
    pd.DataFrame(dis_rows).to_csv(os.path.join(outdir, "combo_disagree_masked.csv"), index=False)
    print("wrote combo_sensitivity.csv + combo_disagree_masked.csv under", outdir)


if __name__ == "__main__":
    main()
