"""Validate the depth-resolved reliability QC on the four well cells, and visualise how each
piece works alone and together -- for fund+overtone (fundot) AND fundamental-only (fund).

Phases:
  extract  -- build a compact <net>_<well>_<waves>_qc.npz per well by running
      noisepy.vs_reliability.assess on per-chain (p16,p50,p84) Vs(z) summaries.
      * new-runner npz (fund runs) already carry chain_vs_p16/p84 -> pure numpy, any env;
      * older fundot npz lack them -> rebuilt from the raw chain traces still on disk
        (needs the bayhunter env for BayHunter.Model).
      Re-running extract re-applies the CURRENT vs_reliability thresholds.

  plot     (bayesbay_dev env; --waves fundot|fund) -- one row per well, four columns:
      1. per-chain median Vs(z) spaghetti, kept (green) vs outlier (grey) -> RAW multimodality
      2. BOTH criterion arms: rho(z) [bottom axis] vs rho_max AND between-chain spread
         [top axis, m/s] vs abs_tol; a depth is resolved if EITHER passes; grey = final
         unreliable; blue dash-dot = wavelength floor -> the DERIVED depth flag, readable
      3. Vs posterior density + median/bands + well log, unreliable hatched -> APPLIED
      4. dispersion fit, all inverted modes in one panel -> the DATA constraint

  compare  -- fund vs fundot side by side per well: posterior medians + reliable intervals
      + rho(z) overlay + dispersion fits.

Usage:
  BH=/opt/anaconda3/envs/bayhunter/bin/python ;  BB=/opt/anaconda3/envs/bayesbay_dev/bin/python
  PYTHONPATH=~/Codes/NoisePy-ant $BH validate_reliability.py --phase extract --waves fundot
  PYTHONPATH=~/Codes/NoisePy-ant $BB validate_reliability.py --phase extract --waves fund
  PYTHONPATH=~/Codes/NoisePy-ant $BB validate_reliability.py --phase plot --waves fundot
  PYTHONPATH=~/Codes/NoisePy-ant $BB validate_reliability.py --phase plot --waves fund
  PYTHONPATH=~/Codes/NoisePy-ant $BB validate_reliability.py --phase compare
"""
import argparse
import os
import pickle

import numpy as np

from noisepy import vs_reliability as vr

PROJ = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
WELLS = [("riehen", "Basel-1", "9_18"), ("riehen", "Otterbach-2", "10_17"),
         ("aargau", "Boettstein", "13_20"), ("aargau", "Riniken", "10_14")]
OUTDIR = f"{PROJ}/_reliability_validation"
VMIN, VMAX, DMAX = 0.3, 3.6, 6.0
CONF_COL = {"high": "tab:green", "marginal": "tab:orange", "low": "tab:red"}
WAVES_LBL = {"fundot": "fund + overtone", "fund": "fundamental only"}


def paths(net, well, cell, waves):
    sub = "chaincount" if waves == "fundot" else "chaincount_fund"
    cc = f"{PROJ}/{net}/tomo/vs_inversion/wells/{sub}"
    return (f"{cc}/well_{well}_nc24_cell_{cell}.npz",
            f"{cc}/work_{well}_nc24/bh_results/data",
            f"{OUTDIR}/{net}_{well}_{waves}_qc.npz")


def _obs_all(r):
    """Concatenated observed (T, U) across whatever modes the npz holds."""
    T, U = [], []
    for w in ("fund", "overtone"):
        if f"obsT_{w}" in r.files:
            T.append(r[f"obsT_{w}"]); U.append(r[f"obs_{w}"])
    return (np.concatenate(T), np.concatenate(U)) if T else (np.array([]), np.array([]))


# --------------------------------------------------------------------------- extract
def extract(waves):
    os.makedirs(OUTDIR, exist_ok=True)
    for net, well, cell in WELLS:
        post, datapath, qc = paths(net, well, cell, waves)
        if not os.path.exists(post):
            print(f"skip {well} ({waves}): no npz yet"); continue
        r = np.load(post, allow_pickle=True)
        dep = r["chain_vs_depth"]
        if "chain_vs_p16" in r.files and len(r["chain_vs_p16"]):
            p16, p50, p84 = r["chain_vs_p16"], r["chain_vs_profiles"], r["chain_vs_p84"]
        elif os.path.isdir(datapath):                         # older npz: rebuild from raw traces
            from noisepy import bh_diagnostics as bd
            mantle = None
            cfgpkl = os.path.join(datapath, "cell_config.pkl")
            if os.path.exists(cfgpkl):
                try:
                    mantle = pickle.load(open(cfgpkl, "rb")).get("mantle")
                except Exception:
                    mantle = None
            diag = bd.build_diagnostics(datapath, dep, mantle)
            p16 = np.array(diag["vs_p16"]); p50 = np.array(diag["vs_profile"])
            p84 = np.array(diag["vs_p84"])
        else:
            print(f"skip {well} ({waves}): no per-chain summaries and no raw traces"); continue
        loglike = r["chain_loglike_med"]
        allT, allU = _obs_all(r)
        rel = vr.assess(dep, p16, p50, p84, loglike, periods=allT, velocities=allU)
        np.savez_compressed(qc, depth=dep, chain_vs_p16=p16, chain_vs_p50=p50, chain_vs_p84=p84,
                            chain_loglike_med=loglike, rho=rel["rho"], rho_smooth=rel["rho_smooth"],
                            between=rel["between"], between_smooth=rel["between_smooth"],
                            resolved=rel["resolved"], reliable=rel["reliable"],
                            z_reliable_min=rel["z_reliable_min"], z_reliable_max=rel["z_reliable_max"],
                            reln_frac=rel["reln_frac"], z_floor=rel["z_floor"], kept=rel["kept"],
                            n_kept=rel["n_kept"], frac_kept=rel["frac_kept"],
                            confidence=rel["confidence"])
        print(f"{net}/{well} [{waves}]: conf={rel['confidence']} n_kept={rel['n_kept']}/"
              f"{len(loglike)} frac={rel['frac_kept']:.2f} reliable={rel['z_reliable_min']:.2f}-"
              f"{rel['z_reliable_max']:.2f} km reln_frac={rel['reln_frac']:.2f} "
              f"(floor {rel['z_floor']:.2f})")


# --------------------------------------------------------------------------- shared plot helpers
def _grey_unreliable(ax, dep, reliable, label=False):
    m = ~reliable; i = 0; lab = label
    while i < len(m):
        if m[i]:
            j = i
            while j < len(m) and m[j]:
                j += 1
            ax.axhspan(dep[i], dep[min(j, len(dep) - 1)], color="0.82", alpha=0.6, zorder=0,
                       label="unreliable" if lab else None)
            lab = False; i = j
        else:
            i += 1


def _hatch_unreliable(ax, dep, reliable):
    m = ~reliable; i = 0
    while i < len(m):
        if m[i]:
            j = i
            while j < len(m) and m[j]:
                j += 1
            ax.axhspan(dep[i], dep[min(j, len(dep) - 1)], facecolor="none", edgecolor="0.35",
                       hatch="//", lw=0.0, alpha=0.9, zorder=5)
            i = j
        else:
            i += 1


def _criteria_panel(ax, q, well, conf):
    """Both criterion arms: rho (bottom axis) + between-chain spread in m/s (top axis)."""
    dep = q["depth"]; reliable = q["reliable"].astype(bool); zfloor = float(q["z_floor"])
    _grey_unreliable(ax, dep, reliable, label=True)
    ax.plot(q["rho_smooth"], dep, "-", color="tab:purple", lw=1.8, label="ρ(z)")
    ax.plot(q["rho"], dep, "-", color="tab:purple", lw=0.5, alpha=0.3)
    ax.axvline(vr.RHO_MAX, color="tab:purple", ls=":", lw=1.3, label=f"ρ_max={vr.RHO_MAX:g}")
    rmax = np.nanmax(q["rho_smooth"][np.isfinite(q["rho_smooth"])])
    ax.set_xlim(0, max(1.6, rmax * 1.08))
    ax.set_ylim(0, DMAX); ax.invert_yaxis()
    ax.set_xlabel("ρ = between / within", color="tab:purple")
    ax.tick_params(axis="x", labelcolor="tab:purple")
    axt = ax.twiny()
    bs = q["between_smooth"] * 1000.0
    axt.plot(bs, dep, "-", color="tab:green", lw=1.6, label="|ΔVs| between chains")
    axt.axvline(vr.ABS_TOL * 1000, color="tab:green", ls=":", lw=1.3,
                label=f"abs_tol={vr.ABS_TOL*1000:.0f} m/s")
    axt.set_xlim(0, max(120.0, np.nanmax(bs) * 1.08))
    axt.set_xlabel("between-chain spread [m/s]", color="tab:green")
    axt.tick_params(axis="x", labelcolor="tab:green")
    if np.isfinite(zfloor) and zfloor < DMAX:
        ax.axhline(zfloor, color="tab:blue", ls="-.", lw=1.2, label=f"λ-floor={zfloor:.2f} km")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = axt.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.2, loc="lower right")
    ax.set_ylabel("depth [km]")
    ax.set_title(f"{well}: resolved if ρ≤{vr.RHO_MAX:g} OR |ΔVs|≤{vr.ABS_TOL*1000:.0f} m/s\n"
                 f"→ {float(q['z_reliable_min']):.2f}–{float(q['z_reliable_max']):.2f} km, "
                 f"confidence = {conf} (reln {float(q['reln_frac']):.2f})", fontsize=9,
                 color=CONF_COL.get(conf, "k"))


def _posterior_panel(ax, r, q, overlay, title, show_legend=True):
    dep = q["depth"]; reliable = q["reliable"].astype(bool)
    ens = r["ens_vs"]; z = r["depth"]
    vsb = np.linspace(VMIN, VMAX, 121)
    dens = np.zeros((len(vsb) - 1, len(z)))
    for k in range(len(z)):
        col = ens[:, k]; col = col[np.isfinite(col)]
        if col.size:
            dens[:, k] = np.histogram(col, bins=vsb, density=True)[0]
    cmax = dens.max(axis=0, keepdims=True)
    dens = dens / np.where(cmax > 0, cmax, 1.0)
    ax.pcolormesh(0.5 * (vsb[1:] + vsb[:-1]), z, dens.T, cmap="hot_r", vmin=0, vmax=1,
                  shading="auto")
    ax.plot(r["vs_median"], z, "c-", lw=1.6, label="posterior median")
    ax.plot(r["vs_p16"], z, "c--", lw=0.7); ax.plot(r["vs_p84"], z, "c--", lw=0.7)
    for v, zc, lab, colc, ls in overlay:
        mm = zc <= DMAX
        ax.plot(v[mm], zc[mm], color=colc, ls=ls, lw=1.6, label=lab)
    _hatch_unreliable(ax, dep, reliable)
    ax.set(xlim=(VMIN, VMAX), ylim=(0, DMAX)); ax.invert_yaxis()
    ax.set_xlabel("Vs [km/s]"); ax.set_ylabel("depth [km]")
    if show_legend:
        ax.legend(fontsize=6, loc="lower left")
    ax.set_title(title, fontsize=9.5)


def _dispersion_panel(ax, r, title):
    colors = {"fund": "tab:blue", "overtone": "tab:red"}
    for w in ("fund", "overtone"):
        if f"obs_{w}" not in r.files:
            continue
        T, U, S = r[f"obsT_{w}"], r[f"obs_{w}"], r[f"obssig_{w}"]
        ax.errorbar(T, U, yerr=S, fmt="o", ms=4, color=colors[w], label=f"{w} obs", zorder=3)
        if f"pred_{w}" in r.files:
            P = r[f"pred_{w}"]; Tp = r[f"predT_{w}"]; o = np.argsort(Tp)
            qq = np.nanpercentile(P, [16, 50, 84], axis=0)
            ax.fill_between(Tp[o], qq[0][o], qq[2][o], color=colors[w], alpha=0.25, zorder=1)
            ax.plot(Tp[o], qq[1][o], "-", color=colors[w], lw=1.3, zorder=2)
    ax.set_xlabel("period [s]"); ax.set_ylabel("group velocity [km/s]")
    ax.legend(fontsize=7.5); ax.set_title(title, fontsize=9.5)


# --------------------------------------------------------------------------- plot (per waveset)
def plot(waves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import well_vs_qc as wq

    rows = [(net, well, cell) for net, well, cell in WELLS
            if os.path.exists(paths(net, well, cell, waves)[2])]
    fig, axes = plt.subplots(len(rows), 4, figsize=(19, 4.4 * len(rows)),
                             constrained_layout=True, squeeze=False)
    for row, (net, well, cell) in enumerate(rows):
        post, _, qcp = paths(net, well, cell, waves)
        r = np.load(post, allow_pickle=True)
        q = np.load(qcp, allow_pickle=True)
        dep = q["depth"]; p50 = q["chain_vs_p50"]; kept = q["kept"]
        reliable = q["reliable"].astype(bool); conf = str(q["confidence"])
        overlay = wq.overlay_curves(net, well)

        ax = axes[row, 0]
        _grey_unreliable(ax, dep, reliable)
        for j in range(len(p50)):
            c, lw, a, z = ("tab:green", 1.3, 0.85, 3) if kept[j] else ("0.6", 0.7, 0.5, 1)
            ax.plot(p50[j], dep, color=c, lw=lw, alpha=a, zorder=z)
        ax.set(xlim=(VMIN, VMAX), ylim=(0, DMAX)); ax.invert_yaxis()
        ax.set_xlabel("Vs [km/s]"); ax.set_ylabel("depth [km]")
        ax.set_title(f"{well}: per-chain medians\n(green=kept {int(kept.sum())}/{len(kept)}, "
                     f"grey line=outlier basin)", fontsize=9.5)

        _criteria_panel(axes[row, 1], q, well, conf)
        _posterior_panel(axes[row, 2], r, q, overlay,
                         f"{well}: posterior (hatched = unreliable)")
        _dispersion_panel(axes[row, 3], r, f"{well}: dispersion fit")

    fig.suptitle(f"Depth-resolved reliability QC -- {WAVES_LBL[waves]} (24 chains, 300k iters, "
                 f"physical trim, ±50% LVZ/HVZ)\ncol1 raw per-chain multimodality → col2 "
                 f"two-arm criterion → col3 posterior with constraint → col4 data fit",
                 fontsize=13)
    out = f"{OUTDIR}/RELIABILITY_wells_{waves}.png"
    fig.savefig(out, dpi=145)
    print("wrote", out)


# --------------------------------------------------------------------------- compare fund vs fundot
def compare():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import well_vs_qc as wq

    rows = [(net, well, cell) for net, well, cell in WELLS
            if all(os.path.exists(paths(net, well, cell, w)[2]) for w in ("fundot", "fund"))]
    fig, axes = plt.subplots(len(rows), 4, figsize=(19, 4.4 * len(rows)),
                             constrained_layout=True, squeeze=False)
    CO = {"fundot": "tab:blue", "fund": "tab:red"}
    for row, (net, well, cell) in enumerate(rows):
        overlay = wq.overlay_curves(net, well)
        R, Q = {}, {}
        for w in ("fundot", "fund"):
            post, _, qcp = paths(net, well, cell, w)
            R[w] = np.load(post, allow_pickle=True); Q[w] = np.load(qcp, allow_pickle=True)

        # col 1+2: posterior density panels, fundot then fund
        for k, w in enumerate(("fundot", "fund")):
            conf = str(Q[w]["confidence"])
            _posterior_panel(axes[row, k], R[w], Q[w], overlay,
                             f"{well} [{WAVES_LBL[w]}]\nconf={conf}, reliable "
                             f"{float(Q[w]['z_reliable_min']):.2f}-"
                             f"{float(Q[w]['z_reliable_max']):.2f} km",
                             show_legend=(k == 0))
            axes[row, k].set_title(axes[row, k].get_title(),
                                   color=CONF_COL.get(conf, "k"), fontsize=9.5)

        # col 3: medians + bands overlaid + rho curves
        ax = axes[row, 2]
        for w in ("fundot", "fund"):
            r, q = R[w], Q[w]
            ax.fill_betweenx(r["depth"], r["vs_p16"], r["vs_p84"], color=CO[w], alpha=0.15)
            ax.plot(r["vs_median"], r["depth"], "-", color=CO[w], lw=2, label=f"{WAVES_LBL[w]}")
            rel = q["reliable"].astype(bool)
            zz = q["depth"][rel]
            if zz.size:                                       # reliable interval marker
                xoff = VMIN + (0.06 if w == "fundot" else 0.16) * (VMAX - VMIN)
                ax.plot([xoff, xoff], [zz.min(), zz.max()], "-", color=CO[w], lw=4, alpha=0.7,
                        solid_capstyle="butt")
        for v, zc, lab, colc, ls in overlay:
            mm = zc <= DMAX
            ax.plot(v[mm], zc[mm], color=colc, ls=ls, lw=1.4, alpha=0.8, label=lab)
        ax.set(xlim=(VMIN, VMAX), ylim=(0, DMAX)); ax.invert_yaxis()
        ax.set_xlabel("Vs [km/s]"); ax.set_ylabel("depth [km]")
        ax.legend(fontsize=6, loc="lower left")
        ax.set_title(f"{well}: median ± 16-84% overlay\n(bars = reliable intervals)", fontsize=9.5)

        # col 4: dispersion fits overlaid (fund fit from both runs; overtone from fundot)
        ax = axes[row, 3]
        r = R["fundot"]
        for wv, c in (("fund", "tab:blue"), ("overtone", "tab:red")):
            if f"obs_{wv}" not in r.files:
                continue
            ax.errorbar(r[f"obsT_{wv}"], r[f"obs_{wv}"], yerr=r[f"obssig_{wv}"], fmt="o", ms=4,
                        color=c, label=f"{wv} obs", zorder=3)
        for w in ("fundot", "fund"):
            rr = R[w]
            for wv in ("fund", "overtone"):
                if f"pred_{wv}" not in rr.files:
                    continue
                P = rr[f"pred_{wv}"]; Tp = rr[f"predT_{wv}"]; o = np.argsort(Tp)
                qq = np.nanpercentile(P, [16, 50, 84], axis=0)
                ax.fill_between(Tp[o], qq[0][o], qq[2][o], color=CO[w], alpha=0.18, zorder=1)
                ax.plot(Tp[o], qq[1][o], "-", color=CO[w], lw=1.3, zorder=2,
                        label=f"{WAVES_LBL[w]} fit" if wv == "fund" else None)
        ax.set_xlabel("period [s]"); ax.set_ylabel("group velocity [km/s]")
        ax.legend(fontsize=6.5); ax.set_title(f"{well}: fits (color = inversion)", fontsize=9.5)

    fig.suptitle("Fundamental-only vs fund+overtone -- posterior, reliability, and fit "
                 "(24 chains, 300k iters, physical trim, ±50% LVZ/HVZ)", fontsize=13)
    out = f"{OUTDIR}/COMPARE_fund_vs_fundot.png"
    fig.savefig(out, dpi=145)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("extract", "plot", "compare"))
    ap.add_argument("--waves", default="fundot", choices=("fundot", "fund"))
    a = ap.parse_args()
    if a.phase == "extract":
        extract(a.waves)
    elif a.phase == "plot":
        plot(a.waves)
    else:
        compare()


if __name__ == "__main__":
    main()
