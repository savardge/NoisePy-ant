#!/usr/bin/env python3
"""Dinver (SWinvert) vs BayHunter at the Riehen well cells, scored against the Michel (2016)
sonic-derived Vs log and the known crystalline-basement depths.

Companion to riehen_michel_compare.py (same metrics, same reference) for the
`tests/dinver_swinvert/<well>_<ix>_<iy>/` tree written by run_vs_inversion.py --engines dinver:
one npz per Dinver measure (dinver_group / dinver_phase / dinver_joint _result.npz) and, when
present, the BayHunter run(s) in the same folder. Any `*_result.npz` in the well folder is
scored; the label is the file stem.

Metrics (restricted to the log's depth range and zmin..zmax; Dinver has no z_reliable_max, so
--zmax defaults to the log's extent capped at the layering's dmax_param when known):
  rmse   rms(model - log) [km/s]      bias   median(model - log), + = model FAST
  r      Pearson r of the two Vs(z)   z_bas  first depth reaching the log's basement Vs vs
                                              the well's basement depth
Also prints Dinver's own diagnostics: accepted/rejected parameterizations, prior-binding
fraction, sigma_ln,Vs at a few depths, and the disba data misfit per curve.

Usage: python dinver_well_compare.py [--root <tests/dinver_swinvert>] [--zmin 0.3] [--zmax 4]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from noisepy import vs_inversion as vi

EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
MICHEL = ("/Volumes/T7blue/riehen-data/well-data/"
          "Vsmodel_well_Basel1_Otterbach_Michel2016.csv")
ROOT = f"{EHM}/riehen/tomo/2_vs_depth_inversion/tests/dinver_swinvert"
WELLS = {"Basel-1": ((23, 47), 2.426), "Otterbach-2": ((26, 43), 2.650)}


def michel():
    d = pd.read_csv(MICHEL)
    d.columns = [c.strip() for c in d.columns]
    return d["depth"].values / 1000.0, d["Vs"].values / 1000.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--zmin", type=float, default=0.3)
    ap.add_argument("--zmax", type=float, default=None)
    a = ap.parse_args()
    zl, vl = michel()
    rows = []
    # Figure: per well, three panels -- (1) every Dinver combo thin + across-combo consensus,
    # (2) the same for BayHunter, (3) the two consensus curves alone against the log. Consensus =
    # median across combos of each run's median profile; band = 16-84 % across combos. Combos
    # are one colour per ENGINE on purpose: the question is the engine consensus, not which of
    # 12 lines is which -- dinver_well_matrix_figure.py is the per-combo view.
    ECOL = {"dinver": "C0", "bayhunter": "C1"}
    fig, axs = plt.subplots(len(WELLS), 3, figsize=(15, 6.6 * len(WELLS)), squeeze=False)
    for k, (well, ((ix, iy), zbas)) in enumerate(WELLS.items()):
        v_bas = float(np.median(vl[zl >= zbas]))
        wdir = os.path.join(a.root, f"{well}_{ix}_{iy}")
        files = sorted(glob.glob(os.path.join(wdir, "*_result.npz")))
        prof = {"dinver": [], "bayhunter": []}
        best = {"dinver": (np.inf, None, None), "bayhunter": (np.inf, None, None)}
        dref = None
        for ci, f in enumerate(files):
            label = os.path.basename(f).replace("_result.npz", "")
            r = vi.load_result(f)
            d, v = r["depth"], r["vs_median"]
            zcap = zl.max()
            if "z_reliable_max" in r and np.isfinite(float(r["z_reliable_max"])):
                zcap = min(zcap, float(r["z_reliable_max"]))
            elif "dmax_param_km" in r:
                zcap = min(zcap, float(r["dmax_param_km"]))
            zhi = a.zmax if a.zmax else zcap
            m = (d >= a.zmin) & (d <= zhi)
            vref = np.interp(d[m], zl, vl)
            dv = v[m] - vref
            zb = float(d[np.argmax(v >= v_bas)]) if (v >= v_bas).any() else np.nan
            mis = vi.data_misfit(r)
            row = dict(well=well, run=label, engine=r["engine"], z_used=round(zhi, 2),
                       n=int(m.sum()), rmse=round(float(np.sqrt(np.mean(dv ** 2))), 3),
                       bias=round(float(np.median(dv)), 3),
                       r=round(float(np.corrcoef(v[m], vref)[0, 1]), 3),
                       z_bas=round(zb, 2) if np.isfinite(zb) else np.nan,
                       z_bas_err=round(zb - zbas, 2) if np.isfinite(zb) else np.nan,
                       runtime_min=round(float(r.get("runtime_s", np.nan)) / 60, 1),
                       chi=" ".join(f"{w}={c:.2f}" for w, c in mis.items()))
            if r["engine"] == "dinver":
                acc = [str(l) for l, rj in zip(r["param_labels"], r["param_rejected"]) if not rj]
                rej = [str(l) for l, rj in zip(r["param_labels"], r["param_rejected"]) if rj]
                sl = r["sigma_ln_vs"]
                row.update(accepted=",".join(acc), rejected=",".join(rej) or "-",
                           bind=round(float(r["prior_bind_frac"]), 3),
                           dmax_param=round(float(r["dmax_param_km"]), 2),
                           sig_ln_1km=round(float(np.interp(1.0, d, sl)), 3),
                           sig_ln_3km=round(float(np.interp(3.0, d, sl)), 3),
                           sig_ln_5km=round(float(np.interp(5.0, d, sl)), 3))
            rows.append(row)
            eng = r["engine"]
            if eng not in prof:
                continue
            if dref is None:
                dref = d
            prof[eng].append(np.interp(dref, d, v))
            # highlight the best-fitting MULTI-curve combo: a single curve is trivially fitted
            # (R0g: chi 0.4 for both engines) and says nothing about consistency.
            chimax = max(mis.values()) if len(mis) >= 2 else np.inf
            if chimax < best[eng][0]:
                best[eng] = (chimax, label, np.interp(dref, d, v))
        for j, eng in enumerate(("dinver", "bayhunter")):
            ax = axs[k][j]
            P = np.array(prof[eng])
            for pv in P:
                ax.plot(pv, dref, color=ECOL[eng], lw=0.8, alpha=0.35)
            if len(P):
                med = np.median(P, 0); lo, hi = np.percentile(P, (16, 84), axis=0)
                ax.fill_betweenx(dref, lo, hi, color=ECOL[eng], alpha=0.18)
                ax.plot(med, dref, color=ECOL[eng], lw=2.6, label=f"{eng}: median of {len(P)} combos")
                if best[eng][2] is not None:
                    ax.plot(best[eng][2], dref, color="k", lw=1.2, ls="--",
                            label=f"best-fitting multi-curve combo: {best[eng][1].split('_', 1)[1]} (max χ {best[eng][0]:.1f})")
            ax.step(vl, zl, where="post", color="k", lw=2.2, label="Michel (2016) log", zorder=5)
            ax.axhline(zbas, color="0.35", ls="--", lw=1.2)
            ax.set_title(f"{well} — {eng}: all combos (thin), consensus (thick), 16–84 % band", fontsize=9.5)
            ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.3)
        ax = axs[k][2]
        for eng in ("dinver", "bayhunter"):
            P = np.array(prof[eng])
            if not len(P):
                continue
            med = np.median(P, 0); lo, hi = np.percentile(P, (16, 84), axis=0)
            mm = (dref >= a.zmin) & (dref <= min(6.0, zl.max()))
            rm = float(np.sqrt(np.mean((med[mm] - np.interp(dref[mm], zl, vl)) ** 2)))
            ax.fill_betweenx(dref, lo, hi, color=ECOL[eng], alpha=0.15)
            ax.plot(med, dref, color=ECOL[eng], lw=2.6, label=f"{eng} consensus  (rmse vs log {rm:.2f} km/s)")
        ax.step(vl, zl, where="post", color="k", lw=2.2, label="Michel (2016) log", zorder=5)
        ax.axhline(zbas, color="0.35", ls="--", lw=1.2)
        ax.text(0.62, zbas - 0.06, f"basement {zbas:.3f} km", fontsize=8, color="0.35")
        ax.set_title(f"{well} — engine consensus vs log", fontsize=9.5)
        ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.3)
        for ax in axs[k]:
            ax.set_ylim(min(6.5, zl.max()), 0); ax.set_xlim(0.5, 4.2); ax.set_xlabel("Vs [km/s]")
        axs[k][0].set_ylabel("depth [km]")
    fig.suptitle("Riehen wells: Dinver (SWinvert pooled) vs BayHunter — consensus across waveset combos vs Michel log",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(a.root, "well_comparison.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(a.root, "well_comparison.csv"), index=False)
    pd.set_option("display.width", 200)
    for well in WELLS:
        s = D[D.well == well].sort_values("rmse")
        print(f"\n=== {well} vs Michel log ({a.zmin} km .. z_used) ===")
        cols = ["run", "engine", "z_used", "rmse", "bias", "r", "z_bas", "z_bas_err",
                "runtime_min", "chi"]
        print(s[cols].to_string(index=False))
        dd = s[s.engine == "dinver"]
        if len(dd):
            print(dd[["run", "accepted", "rejected", "bind", "dmax_param", "sig_ln_1km",
                      "sig_ln_3km", "sig_ln_5km"]].to_string(index=False))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
