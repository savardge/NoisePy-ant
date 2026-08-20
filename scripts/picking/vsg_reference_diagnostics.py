#!/usr/bin/env python3
"""Full per-combo diagnostics for the VSG-reference inversions.

One figure per config, plus one CSV of every scalar, covering:

  ROW 1  dispersion fit -- observed reference curve with its sigma, the 150-member POSTERIOR
         PREDICTIVE ensemble stored in the npz (`pred_<w>`; no forward modelling needed), and
         the predictive median.
  ROW 2  normalised residual (obs - pred_med)/sigma vs period, per target. THE branch
         diagnostic: a 2*pi*N branch error is a smooth systematic offset or trend, whereas
         picking noise scatters about zero. Reading row 1 alone cannot separate them.
  ROW 3  misfit distributions (per target and total), the running mean of the TOTAL misfit,
         and the noise-sigma posteriors against their prior bounds with the rail fraction.
  ROW 4  chain diagnostics -- per-chain likelihood traces (phase 2), per-chain median logL with
         the outlier cut drawn, and the per-chain median Vs profiles that `chain_disagree`
         is computed from.
  ROW 5  model space -- layer-count posterior, Vs posterior with 16-84/2.5-97.5 bands and
         z_reliable_max, and the ensemble interface probability.

TWO ARRAY LAYOUTS that are easy to misread and are handled explicitly here:
  * `ens_misfit` is (n_models, n_targets+1) flattened -- per-target misfits then the TOTAL.
    Treating it as a flat per-model sequence interleaves the targets and makes any running
    mean meaningless.
  * `noise_post` is (n_models, 4*n_targets); column 1 of each 4-block is sigma (0=corr,
    2,3 unused for surface-wave targets and stored as NaN).

CAVEAT on sigma: for these runs sigma's prior came from the pick SCATTER of each reference
curve, so a high rail fraction means the data disagree with the model by more than the pick
smoothness allows -- it does not by itself mean the curve is wrong.

Usage:
  python vsg_reference_diagnostics.py --net riehen
  python vsg_reference_diagnostics.py --net riehen --tag test_2026-08-16_vsg_reference_vmax5
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
ORDER = ["R0p", "R0pR1p", "R0pL0p", "R0pR1pL0p", "R0pR1pL0pL1p"]
WCOL = {"fund_phase": "tab:blue", "overtone_phase": "tab:orange",
        "love_phase": "tab:green", "love_ot_phase": "tab:red"}


def _keep_mask(chains_kept, n):
    """Boolean keep-mask over chains, accepting either storage form.

    `chains_kept` is stored as a BOOLEAN MASK of length n_chains. Treating it as an array of
    kept-chain INDICES silently mislabels the plot: booleans cast to 0/1, so only chains 0 and
    1 come out "kept" whatever the run actually did. Both forms are handled explicitly here.
    """
    ck = np.atleast_1d(np.asarray(chains_kept))
    if ck.dtype == bool:
        m = np.zeros(n, bool)
        m[: min(n, ck.size)] = ck[: min(n, ck.size)]
        return m
    m = np.zeros(n, bool)
    idx = ck[(ck >= 0) & (ck < n)].astype(int)
    m[idx] = True
    return m


def misfit_table(z):
    """(n_models, n_targets+1): per-target misfits then the total."""
    nm = int(z["n_models"])
    return np.asarray(z["ens_misfit"], float).ravel().reshape(nm, -1)


def sigma_cols(z, n_targets):
    """sigma posterior per target -- column 1 of each 4-wide noise block."""
    npo = np.asarray(z["noise_post"], float)
    return [npo[:, 4 * i + 1] for i in range(n_targets)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", default="riehen")
    ap.add_argument("--tag", default="test_2026-08-16_vsg_reference")
    a = ap.parse_args()
    root = f"{EHM}/{a.net}/tomo/2_vs_depth_inversion/tests/{a.tag}"
    out = os.path.join(root, "diagnostics"); os.makedirs(out, exist_ok=True)

    rows = []
    for cfg in ORDER:
        f = f"{root}/{cfg}/bayhunter_result.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        waves = [str(w) for w in np.atleast_1d(z["waves"])]
        nw = len(waves)
        M = misfit_table(z)
        sig = sigma_cols(z, nw)
        prior = np.atleast_2d(np.asarray(z["noise_sigma_prior"], float))
        rail = np.atleast_1d(np.asarray(z["noise_rail_frac"], float))
        d = z["depth"]; zrel = float(z["z_reliable_max"])

        ncol = max(nw, 3)
        fig, axs = plt.subplots(5, ncol, figsize=(5.0 * ncol, 20), squeeze=False)
        for j in range(ncol):
            for i in range(5):
                axs[i][j].axis("off")

        r = dict(cfg=cfg, waves="+".join(waves), n_models=int(z["n_models"]),
                 runtime_s=round(float(z["runtime_s"]), 0),
                 chain_disagree=round(float(z["chain_disagree"]), 3),
                 chains=f"{int(z['n_chains_kept'])}/{int(z['n_chains_used'])}",
                 frac_ok=round(float(z["frac_chains_ok"]), 3),
                 burnin_delta_frac=round(float(z["burnin_delta_frac"]), 3),
                 confidence=str(z["confidence"]),
                 z_reliable=round(zrel, 2), z_floor=round(float(z["z_floor"]), 2),
                 nlayers_med=int(np.median(z["n_layers_post"])),
                 misfit_total_med=round(float(np.median(M[:, -1])), 4))

        # ---- rows 1 & 2: fit and normalised residual, per target ----
        for k, w in enumerate(waves):
            T = np.asarray(z[f"obsT_{w}"], float)
            obs = np.asarray(z[f"obs_{w}"], float)
            s = np.asarray(z[f"obssig_{w}"], float)
            P = np.atleast_2d(np.asarray(z[f"pred_{w}"], float))
            pmed = np.median(P, axis=0)
            ax = axs[0][k]; ax.axis("on")
            for p in P[:: max(1, len(P) // 60)]:
                ax.plot(T, p, "-", color=WCOL.get(w, "0.5"), lw=0.5, alpha=0.15)
            ax.errorbar(T, obs, yerr=s, fmt="o", ms=3.2, color="k", lw=0.9,
                        capsize=2, label="VSG reference $\\pm\\sigma$")
            ax.plot(T, pmed, "-", color=WCOL.get(w, "0.5"), lw=2.0, label="predictive median")
            ax.set_xscale("log"); ax.set_xlabel("period [s]"); ax.set_ylabel("c [km/s]")
            ax.set_title(f"{cfg} — {w}", fontsize=10); ax.grid(alpha=0.3)
            ax.legend(fontsize=7)

            res = (obs - pmed) / s
            ax = axs[1][k]; ax.axis("on")
            ax.axhline(0, color="k", lw=1)
            for lv, c in ((1, "0.7"), (2, "0.85")):
                ax.axhspan(-lv, lv, color=c, alpha=0.5, zorder=0)
            ax.plot(T, res, "o-", color=WCOL.get(w, "0.5"), lw=1.4, ms=4)
            ax.set_xscale("log"); ax.set_xlabel("period [s]")
            ax.set_ylabel("(obs - pred)/$\\sigma$")
            ax.set_title(f"residual — mean {res.mean():+.2f}, rms {np.sqrt((res**2).mean()):.2f}",
                         fontsize=9.5)
            ax.grid(alpha=0.3)
            r[f"chi_{w}"] = round(float(np.sqrt((res ** 2).mean())), 2)
            r[f"bias_{w}"] = round(float(res.mean()), 2)
            r[f"sigma_med_{w}"] = round(float(np.median(sig[k])), 4)
            r[f"sigma_prior_hi_{w}"] = round(float(prior[k][1]), 4)
            r[f"sigma_rail_{w}"] = round(float(rail[k]), 3) if k < rail.size else np.nan

        # ---- row 3: misfit distributions / running mean / noise sigma ----
        ax = axs[2][0]; ax.axis("on")
        for k, w in enumerate(waves):
            ax.hist(M[:, k], bins=70, histtype="step", lw=1.5,
                    color=WCOL.get(w, "0.5"), label=w)
        ax.hist(M[:, -1], bins=70, histtype="step", lw=1.8, color="k", label="total")
        ax.set_xlabel("misfit"); ax.set_ylabel("count"); ax.legend(fontsize=7)
        ax.set_title("misfit distributions", fontsize=10)

        ax = axs[2][1]; ax.axis("on")
        tot = M[:, -1]
        kk = np.arange(1, tot.size + 1)
        rm = np.cumsum(tot) / kk
        rs = np.sqrt(np.maximum(np.cumsum(tot ** 2) / kk - rm ** 2, 0))
        ax.plot(kk, rm, "-", color="navy", lw=1.3)
        ax.fill_between(kk, rm - rs, rm + rs, color="slateblue", alpha=0.3)
        ax.set_xlabel("posterior sample index"); ax.set_ylabel("total misfit")
        ax.set_title("running cumulative mean $\\pm$ std", fontsize=10); ax.grid(alpha=0.3)

        ax = axs[2][2]; ax.axis("on")
        for k, w in enumerate(waves):
            ax.hist(sig[k], bins=70, histtype="step", lw=1.5, color=WCOL.get(w, "0.5"),
                    label=f"{w} (rail {rail[k]:.0%})" if k < rail.size else w)
            for b in prior[k]:
                ax.axvline(b, color=WCOL.get(w, "0.5"), ls="--", lw=1.0, alpha=0.8)
        ax.set_xlabel("noise $\\sigma$ [km/s]"); ax.set_ylabel("count")
        ax.set_title("noise $\\sigma$ posterior (dashed = prior bounds)", fontsize=10)
        ax.legend(fontsize=7)

        # ---- row 4: chains ----
        ax = axs[3][0]; ax.axis("on")
        L2 = np.asarray(z["chain_like_p2"], float)
        for i in range(L2.shape[0]):
            ax.plot(L2[i], "-", lw=0.6, alpha=0.6)
        ax.set_xlabel("stored step (phase 2)"); ax.set_ylabel("log-likelihood")
        ax.set_title("per-chain likelihood traces", fontsize=10); ax.grid(alpha=0.3)
        lo = np.nanpercentile(L2, 5)
        ax.set_ylim(lo, np.nanmax(L2) + 1)

        ax = axs[3][1]; ax.axis("on")
        med = np.asarray(z["chain_loglike_med"], float)
        keep = _keep_mask(z["chains_kept"], med.size)
        ax.bar(np.arange(med.size)[keep], med[keep], color="tab:green", label="kept")
        ax.bar(np.arange(med.size)[~keep], med[~keep], color="tab:red", label="dropped")
        ax.axhline(np.nanmax(med) - float(z["outlier_delta"]), color="k", ls="--", lw=1.2,
                   label=f"cut = best - {float(z['outlier_delta']):.0f}")
        ax.set_xlabel("chain"); ax.set_ylabel("median logL")
        ax.set_title("per-chain median logL + outlier cut", fontsize=10); ax.legend(fontsize=7)

        ax = axs[3][2]; ax.axis("on")
        cvp = np.asarray(z["chain_vs_profiles"], float)
        cvd = np.asarray(z["chain_vs_depth"], float)
        for i in range(cvp.shape[0]):
            ax.plot(cvp[i], cvd, "-", lw=0.8,
                    color="tab:green" if keep[i] else "tab:red", alpha=0.75)
        ax.set_ylim(d.max(), 0); ax.set_xlabel("Vs [km/s]"); ax.set_ylabel("depth [km]")
        ax.set_title(f"per-chain median Vs (disagree {float(z['chain_disagree']):.3f})",
                     fontsize=10); ax.grid(alpha=0.3)

        # ---- row 5: model space ----
        ax = axs[4][0]; ax.axis("on")
        nl = np.asarray(z["n_layers_post"], int)
        ax.hist(nl, bins=np.arange(nl.min() - 0.5, nl.max() + 1.5), color="0.5")
        ax.set_xlabel("number of layers"); ax.set_ylabel("count")
        ax.set_title(f"layer-count posterior (median {int(np.median(nl))})", fontsize=10)

        ax = axs[4][1]; ax.axis("on")
        ax.fill_betweenx(d, z["vs_p025"], z["vs_p975"], color="tab:blue", alpha=0.18,
                         label="2.5-97.5%")
        ax.fill_betweenx(d, z["vs_p16"], z["vs_p84"], color="tab:blue", alpha=0.35,
                         label="16-84%")
        ax.plot(z["vs_median"], d, "-", color="k", lw=2.0, label="median")
        ax.axhline(zrel, color="crimson", ls="--", lw=1.4, label=f"z_reliable {zrel:.2f} km")
        ax.set_ylim(d.max(), 0); ax.set_xlabel("Vs [km/s]"); ax.set_ylabel("depth [km]")
        ax.set_title("Vs posterior", fontsize=10); ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = axs[4][2]; ax.axis("on")
        ifd = np.asarray(z["iface_depths"], float)
        eb = np.arange(0, d.max() + 0.1, 0.1)
        h, _ = np.histogram(ifd, bins=eb)
        ax.barh(0.5 * (eb[:-1] + eb[1:]), h / max(h.sum(), 1), height=0.09, color="0.45")
        ax.axhline(zrel, color="crimson", ls="--", lw=1.4)
        ax.set_ylim(d.max(), 0); ax.set_xlabel("interface probability")
        ax.set_title("ensemble interface probability", fontsize=10)

        fig.suptitle(f"{a.net} — VSG reference inversion diagnostics — {cfg}   "
                     f"[{'+'.join(waves)}]   {a.tag}",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        p = os.path.join(out, f"diag_{cfg}.png")
        fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
        print("wrote", os.path.basename(p))
        rows.append(r)

    if not rows:
        raise SystemExit(f"no completed runs under {root}")
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(out, "diagnostics_summary.csv"), index=False)
    base = ["cfg", "waves", "chains", "frac_ok", "chain_disagree", "confidence",
            "z_reliable", "nlayers_med", "misfit_total_med"]
    print("\n=== convergence / model ===")
    print(D[base].to_string(index=False))
    for pre, lab in (("chi_", "normalised RMS residual per target"),
                     ("bias_", "MEAN normalised residual (systematic offset -> branch)"),
                     ("sigma_rail_", "noise sigma rail fraction")):
        c = [x for x in D.columns if x.startswith(pre)]
        if c:
            print(f"\n=== {lab} ===")
            print(D[["cfg"] + c].to_string(index=False))
    print(f"\nwrote {out}/diagnostics_summary.csv")


if __name__ == "__main__":
    main()
