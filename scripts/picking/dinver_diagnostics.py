"""SWinvert-style diagnostic figure for every Dinver (well, waveset-combo) run.

One PNG per (well, combo) following Vantassel & Cox (2021):
  (a) Fig 7  -- minimum misfit per parameterization, one marker per independent trial (LN filled,
               LR open), the trial spread = convergence proxy; range of each trial's 100 kept
               misfits as a faint bar (flat = converged plateau); rejected params marked R.
  (b) Fig 6b -- lowest-misfit Vs profile per parameterization (rejected greyed) + well log.
  (c) Fig 10c-- the 100 lowest-misfit profiles per ACCEPTED parameterization (all trials).
  (d) Fig 10d-- sigma_ln,Vs vs depth for the three approaches: best-per-param, 10-best, 100-best.
  (e)        -- posterior-style Vs(z) density of the pooled ensemble + median/p16/p84 + log,
               interface-depth histogram alongside.
  (f) Fig 4  -- NA convergence, best misfit vs model index per trial, when the run kept a
               .bestcurve.txt (runs made before 2026-08-18 have none; then the panel says so).
  (g) Fig 8  -- experimental curve (+-1 sigma, resampled target) vs the best model of each
               parameterization and the pooled median, one panel per target curve, x = period.

Reads only what run_dinver_cell.py left on disk: the result npz and the per-run best-100
caches (<label>_t<k>.report.gm.txt). Forward curves via disba with each model's own Vp/rho,
exactly as the runner does.

Usage (from scripts/picking):  python dinver_diagnostics.py [--well Basel-1] [--combo R0gR1g]
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

from noisepy import vs_inversion as vi
from noisepy import dinver_target as dt
from dinver_well_compare import ROOT, WELLS, michel
from run_dinver_cell import _load_models, _profile

LN_RE = re.compile(r"^(ln|lr)([\d.]+)_t(\d+)\.report\.gm\.txt$")


def _param_color(labels):
    """Same colour for same layer count is the paper's convention; here one colour per label,
    LN solid / LR dashed, which is what the reader needs to tell them apart."""
    cmap = plt.get_cmap("tab10")
    return {lab: cmap(i % 10) for i, lab in enumerate(labels)}


def load_runs(workdir):
    """{label: {trial: [(thk, vp, vs, rho, misfit) x <=100 sorted]}}"""
    runs = {}
    for f in sorted(glob.glob(os.path.join(workdir, "reports", "*.report.gm.txt"))):
        m = LN_RE.match(os.path.basename(f))
        if not m:
            continue
        label = m.group(1) + m.group(2); tr = int(m.group(3))
        runs.setdefault(label, {})[tr] = _load_models(None, f[:-len(".gm.txt")], 100)
    return runs


def load_bestcurves(workdir):
    out = {}
    for f in glob.glob(os.path.join(workdir, "reports", "*.report.bestcurve.txt")):
        m = LN_RE.match(os.path.basename(f).replace(".bestcurve.txt", ".gm.txt"))
        if not m:
            continue
        rows = [l.split() for l in open(f) if l.strip() and not l.startswith("#")]
        if rows:
            out.setdefault(m.group(1) + m.group(2), {})[int(m.group(3))] = np.array(rows, float)
    return out


def forward(model, w, T):
    disba_wave, mode, meas = vi.curve_def(w)
    th, vp, vs, rho, _ = model
    thk = th.copy(); thk[-1] = 100.0
    return vi.dispersion_velocity(thk, vs, T, mode, measure=meas, disba_wave=disba_wave,
                                  vp=vp, rho=rho / dt.KM)


def sigma_ln(models, dep):
    prof = np.array([_profile(m[0], m[2], dep) for m in models])
    return np.std(np.log(prof), axis=0, ddof=1) if len(prof) > 1 else np.full(len(dep), np.nan)


def make_figure(npz, workdir, well, zbas, zl, vl, out):
    r = vi.load_result(npz)
    dep = r["depth"]
    labels = [str(x) for x in r["param_labels"]]
    rej = np.asarray(r["param_rejected"], bool)
    nlay = np.asarray(r["param_nlayers"], int)
    runs = load_runs(workdir)
    bcs = load_bestcurves(workdir)
    col = _param_color(labels)
    waves = [str(w) for w in r["waves"]]
    tag = os.path.basename(out).replace(".png", "")

    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 5, figure=fig, height_ratios=[1, 1, 0.9], hspace=0.32, wspace=0.35)
    axa = fig.add_subplot(gs[0, 0]); axb = fig.add_subplot(gs[0, 1]); axc = fig.add_subplot(gs[0, 2], sharey=axb)
    axd = fig.add_subplot(gs[0, 3], sharey=axb); axe = fig.add_subplot(gs[0, 4], sharey=axb)
    axf = fig.add_subplot(gs[1, 0]); axh = fig.add_subplot(gs[1, 1]); axi = fig.add_subplot(gs[1, 2])
    axj = fig.add_subplot(gs[1, 3]); axk = fig.add_subplot(gs[1, 4])
    ncurve = max(1, len(waves))
    axg = [fig.add_subplot(gs[2, i]) for i in range(min(ncurve, 5))]

    # ---- (a) misfit per parameterization / trial ------------------------------------------
    for i, lab in enumerate(labels):
        trials = runs.get(lab, {})
        for tr, models in sorted(trials.items()):
            mis = np.array([m[4] for m in models])
            axa.plot([i, i], [mis.min(), mis.max()], color=col[lab], alpha=0.25, lw=6, solid_capstyle="butt")
            axa.plot(i, mis.min(), marker="o" if lab.startswith("ln") else "s", color=col[lab],
                     mfc=col[lab] if lab.startswith("ln") else "white", ms=7, mew=1.5)
        if rej[i]:
            axa.text(i, axa.get_ylim()[1] if False else np.nanmax([m[4] for t in trials.values() for m in t] or [1]) * 1.05,
                     "R", ha="center", va="bottom", fontsize=11, fontweight="bold")
    axa.set_xticks(range(len(labels))); axa.set_xticklabels(labels, rotation=45)
    axa.set_ylabel("dispersion misfit  m_dc"); axa.set_title("(a) min misfit per param, per trial\n(marker = trial min; bar = its 100 kept)")
    axa.grid(alpha=0.3); axa.axhline(1.0, color="0.5", ls=":", lw=1)

    # ---- (b) best profile per parameterization -------------------------------------------
    axb.step(vl, zl, where="post", color="k", lw=2.2, label="Michel log", zorder=5)
    for i, lab in enumerate(labels):
        allm = sorted((m for t in runs.get(lab, {}).values() for m in t), key=lambda m: m[4])
        if not allm:
            continue
        prof = _profile(allm[0][0], allm[0][2], dep)
        axb.plot(prof, dep, color="0.75" if rej[i] else col[lab], lw=1.6,
                 ls="-" if lab.startswith("ln") else "--", label="%s (%d)%s" % (lab, nlay[i], " R" if rej[i] else ""))
    axb.axhline(zbas, color="0.4", ls="--", lw=1)
    axb.set_ylim(dep.max(), 0); axb.set_xlim(0.4, 4.4)
    axb.set_xlabel("Vs [km/s]"); axb.set_ylabel("depth [km]"); axb.set_title("(b) lowest-misfit model per param")
    axb.legend(fontsize=7, loc="lower left"); axb.grid(alpha=0.3)

    # ---- (c) 100 best per accepted param --------------------------------------------------
    for i, lab in enumerate(labels):
        if rej[i]:
            continue
        allm = sorted((m for t in runs.get(lab, {}).values() for m in t), key=lambda m: m[4])[:100]
        for m in allm:
            axc.plot(_profile(m[0], m[2], dep), dep, color=col[lab], lw=0.4, alpha=0.25)
    axc.step(vl, zl, where="post", color="k", lw=2.2, zorder=5)
    axc.plot(r["vs_median"], dep, color="k", lw=1.2, ls=":", zorder=6, label="pooled median")
    axc.axhline(zbas, color="0.4", ls="--", lw=1)
    axc.set_xlim(0.4, 4.4); axc.set_xlabel("Vs [km/s]"); axc.set_title("(c) 100 lowest-misfit per accepted param")
    axc.legend(fontsize=7, loc="lower left"); axc.grid(alpha=0.3)

    # ---- (d) sigma_ln,Vs three ways --------------------------------------------------------
    acc = [i for i in range(len(labels)) if not rej[i]]
    per = {lab: sorted((m for t in runs.get(lab, {}).values() for m in t), key=lambda m: m[4]) for lab in labels}
    for n, name, c in ((1, "best per param", "C3"), (10, "10-best per param", "C2"), (100, "100-best per param", "C0")):
        pool = [m for i in acc for m in per[labels[i]][:n]]
        if len(pool) > 1:
            axd.plot(sigma_ln(pool, dep), dep, color=c, lw=1.6, label=name)
    axd.axhline(zbas, color="0.4", ls="--", lw=1)
    axd.set_xlim(0, max(0.4, float(np.nanmax(r["sigma_ln_vs"])) * 1.1)); axd.set_xlabel("sigma_ln,Vs")
    axd.set_title("(d) lognormal std of Vs"); axd.legend(fontsize=8); axd.grid(alpha=0.3)

    # ---- (e) Vs(z) density of pooled ensemble --------------------------------------------
    if "ens_vs" not in r:
        # lean (cluster) npz: rebuild the pooled ensemble from the per-run caches when present
        pool = [m for i in acc for m in per[labels[i]][:100]] if runs else []
        ens = np.array([_profile(m[0], m[2], dep) for m in pool]) if pool else np.zeros((0, len(dep)))
        r["ens_misfit"] = np.array([m[4] for m in pool]); r["ens_param"] = np.array(
            [labels[i] for i in acc for _ in per[labels[i]][:100]])
    else:
        ens = np.asarray(r["ens_vs"], float)
    vbins = np.linspace(0.4, 4.4, 81)
    H = np.array([np.histogram(ens[:, k], bins=vbins)[0] for k in range(len(dep))], float)
    H /= np.maximum(H.max(axis=1, keepdims=True), 1)
    axe.pcolormesh(0.5 * (vbins[1:] + vbins[:-1]), dep, H, cmap="Blues", shading="nearest", vmin=0, vmax=1)
    axe.plot(r["vs_median"], dep, "C1", lw=1.5, label="median"); axe.plot(r["vs_p16"], dep, "C1", lw=0.8, ls="--")
    axe.plot(r["vs_p84"], dep, "C1", lw=0.8, ls="--")
    axe.step(vl, zl, where="post", color="k", lw=2, zorder=5, label="log")
    axe.axhline(zbas, color="0.4", ls="--", lw=1)
    axe.set_xlim(0.4, 4.4); axe.set_xlabel("Vs [km/s]"); axe.set_title("(e) pooled-ensemble Vs(z) density (%d models)" % len(ens))
    axe.legend(fontsize=7, loc="lower left")

    # ---- (f) NA convergence -----------------------------------------------------------------
    if bcs:
        for lab, trs in bcs.items():
            for tr, arr in trs.items():
                axf.step(arr[:, 0], arr[:, 1], where="post", color=col.get(lab, "0.5"), lw=1, alpha=0.8)
        axf.set_xscale("log"); axf.set_yscale("log"); axf.set_xlabel("model index"); axf.set_ylabel("best misfit so far")
        axf.set_title("(f) NA convergence per trial (Fig. 4)")
    else:
        # proxy: relative spread of trial minima per param, and 100th/1st misfit ratio
        xs, spread, plateau = [], [], []
        for i, lab in enumerate(labels):
            trs = runs.get(lab, {})
            mins = [min(m[4] for m in t) for t in trs.values() if t]
            if len(mins) > 1:
                xs.append(i); spread.append((max(mins) - min(mins)) / np.mean(mins))
                plateau.append(np.mean([t[-1][4] / t[0][4] - 1 for t in trs.values() if t]))
        axf.bar(np.array(xs) - 0.2, spread, 0.4, color="C0", label="trial-min spread / mean")
        axf.bar(np.array(xs) + 0.2, plateau, 0.4, color="C2", label="(100th - 1st)/1st misfit")
        axf.set_xticks(range(len(labels))); axf.set_xticklabels(labels, rotation=45)
        axf.set_title("(f) convergence proxies (no .bestcurve on this run)"); axf.legend(fontsize=7)
    axf.grid(alpha=0.3)

    # ---- (h) interface-depth histogram, (i) n-layer/param share, (j) Vp/Vs, (k) misfit hist ---
    ifaces = []; vpvs = []; hs = []
    for i in acc:
        for m in per[labels[i]][:100]:
            th = m[0]; ifaces.extend(np.cumsum(th[:-1])); vpvs.extend(m[1] / m[2]); hs.append(m[2][-1])
    axh.hist(ifaces, bins=np.arange(0, dep.max() + 0.1, 0.1), orientation="horizontal", color="C0", alpha=0.7)
    axh.axhline(zbas, color="0.4", ls="--", lw=1); axh.set_ylim(dep.max(), 0)
    axh.set_xlabel("count"); axh.set_ylabel("depth [km]"); axh.set_title("(h) interface depths, pooled"); axh.grid(alpha=0.3)
    axi.hist(vpvs, bins=40, color="C4", alpha=0.8); axi.set_xlabel("Vp/Vs (per layer)"); axi.set_title("(i) Vp/Vs of pooled layers")
    axi.axvline(1.73, color="k", ls=":", lw=1); axi.grid(alpha=0.3)
    axj.hist(hs, bins=30, color="C5", alpha=0.8); axj.set_xlabel("half-space Vs [km/s]"); axj.set_title("(j) half-space Vs, pooled")
    axj.axvline(float(r["vs_bounds"][1]), color="r", ls="--", lw=1, label="vs_max"); axj.legend(fontsize=7); axj.grid(alpha=0.3)
    em = np.asarray(r["ens_misfit"], float); ep = [str(x) for x in r["ens_param"]]
    for lab in labels:
        sel = em[np.array(ep) == lab]
        if sel.size:
            axk.hist(sel, bins=30, histtype="step", color=col[lab], lw=1.3, label=lab)
    axk.set_xlabel("misfit of pooled models"); axk.set_title("(k) misfit distribution per param"); axk.legend(fontsize=7, ncol=2); axk.grid(alpha=0.3)

    # ---- (g) dispersion fits ----------------------------------------------------------------
    for ax, w in zip(axg, waves):
        T, U, S = r["obs"][w]
        ax.errorbar(T, U, yerr=S, fmt="k.", ms=4, lw=0.8, capsize=2, label="observed ±1σ", zorder=5)
        if f"targetT_{w}" in r:
            ax.plot(r[f"targetT_{w}"], r[f"target_{w}"], "kx", ms=4, alpha=0.6, label="target (resampled)")
        for i, lab in enumerate(labels):
            if not per[lab]:
                continue
            ax.plot(T, forward(per[lab][0], w, T), color="0.75" if rej[i] else col[lab],
                    ls="-" if lab.startswith("ln") else "--", lw=1.2)
        pm = np.nanmedian(np.asarray(r["pred"][w][1], float), axis=0) if w in r.get("pred", {}) else None
        if pm is not None:
            ax.plot(T, pm, "k:", lw=1.5, label="pooled median")
        chi = vi.data_misfit(r).get(w, np.nan)
        ax.set_xscale("log"); ax.set_xlabel("period [s]"); ax.set_ylabel("velocity [km/s]")
        from matplotlib.ticker import ScalarFormatter, NullFormatter
        ax.xaxis.set_major_formatter(ScalarFormatter()); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks([t for t in (0.3, 0.5, 1, 2, 3, 5) if T.min() * 0.9 <= t <= T.max() * 1.1])
        ax.set_title("(g) %s   χ=%.2f" % (w, chi)); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)

    fig.suptitle("%s  —  %s   |   accepted %d/%d params, %d pooled models, Vs-bound binding %.2f, "
                 "depth-bound %.2f, ν %s, vs %s"
                 % (well, tag, len(acc), len(labels), len(ens), float(r.get("bind_vs_frac", np.nan)),
                    float(r.get("bind_depth_frac", np.nan)),
                    "%.2f-%.2f" % tuple(np.asarray(r["pr_bounds"], float)),
                    "%.1f-%.1f km/s" % tuple(np.asarray(r["vs_bounds"], float))), y=0.995, fontsize=12)
    fig.savefig(out, dpi=105, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--well", default=None); ap.add_argument("--combo", default=None)
    ap.add_argument("--root", default=ROOT)
    args = ap.parse_args()
    zl, vl = michel()
    for well, ((ix, iy), zbas) in WELLS.items():
        if args.well and well != args.well:
            continue
        wd = os.path.join(args.root, f"{well}_{ix}_{iy}")
        outdir = os.path.join(wd, "diagnostics"); os.makedirs(outdir, exist_ok=True)
        for npz in sorted(glob.glob(os.path.join(wd, "dinver_*_result.npz"))):
            m = re.match(r"dinver_(group|phase|joint)_(\w+)_result\.npz", os.path.basename(npz))
            if not m:
                continue
            meas, tag = m.groups()
            if args.combo and tag != args.combo:
                continue
            workdir = os.path.join(wd, f"combo_{tag}", f"dinver_{meas}_work")
            if not os.path.isdir(workdir):
                print("no workdir for", npz); continue
            out = os.path.join(outdir, f"{tag}.png")
            make_figure(npz, workdir, well, zbas, zl, vl, out)
            print("wrote", out)


if __name__ == "__main__":
    main()
