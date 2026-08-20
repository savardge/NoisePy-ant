"""Dinver (Geopsy neighbourhood algorithm) runner for ONE tomography cell -- the SWinvert
workflow of Vantassel & Cox (2021), executed as a subprocess of noisepy.vs_inversion.run_dinver
(or of grid_vs_inversion.py --engine dinver).

Reads a JSON config, and for the cell's curves:
  1. writes ONE .target (group and/or phase ModalCurves, via noisepy.dinver_target -- swprepost
     cannot emit a Group slowness tag, which is why that module exists);
  2. writes one .param per parameterization: LN in `lns`, LR in `lrs`, Vs layering sized from
     the fundamental PHASE wavelength range (lmin/3 .. lmax/depth_factor), Vp LINKED to the Vs
     layering (the paper strongly advises against a fixed Poisson's ratio), nu free 0.2-0.5,
     density fixed;
  3. runs `dinver -optimization` for every (parameterization, trial) -- Ns0 / Ns / Nr straight
     from the paper's tuning study; skip-if-exists per .report so a walltime kill resumes;
  4. reads the Nr best models of each run with gpdcreport, applies the paper's TWO rejection
     criteria (both required), pools the n_pool lowest-misfit models of each ACCEPTED
     parameterization -- this pooled ensemble IS the uncertainty (inter- + intra-param), the
     thing a single-parameterization Dinver run understates;
  5. forwards every pooled model through disba at the cell's ORIGINAL periods (with Dinver's own
     Vp and rho, nothing smuggled in) so data_misfit() is on the same footing as bayesbay /
     BayHunter, and writes the shared result npz (vs_inversion.load_result schema) atomically.

Units: the pipeline is (s, km/s, km); Geopsy is (Hz, s/m, m). ALL conversion goes through
noisepy.dinver_target (target side) and the two helpers _gm_to_km / km-> m at the .param side.

Usage: python run_dinver_cell.py config.json
"""
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

from noisepy import vs_inversion as vi
from noisepy import dinver_target as dt

DEFAULTS = dict(lean=False,
                lns=(3, 4, 5, 7), lrs=(3.0, 2.0, 1.5, 1.2), ntrials=3, ns0=10_000, ns=50_000,
                nr=100, n_pool=100, depth_factor=2.0, n_resample=30, min_cov=0.05,
                vs_bounds=(0.5, 4.2), vp_bounds=(0.8, 8.0), pr_bounds=(0.2, 0.35), rho=2000.0,
                depth_max=6.5, jobs=1, seed0=1, keep_reports=False, n_parallel=1,
                reject_misfit_ratio=1.5, reject_vs_dev=0.15, bind_tol=0.02)


# --------------------------------------------------------------------- tools
def _find_tool(name, dinver_bin, explicit=None):
    """Locate a Geopsy CLI tool: explicit path > PATH > next to dinver > mac .app three up."""
    if explicit:
        return explicit
    p = shutil.which(name)
    if p:
        return p
    d = os.path.dirname(os.path.abspath(dinver_bin))
    for cand in (os.path.join(d, name), os.path.join(d, "..", "..", "..", name)):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return os.path.normpath(cand)
    raise FileNotFoundError(f"{name}: not on PATH and not found near {dinver_bin}")


def _run(cmd, log=None, timeout=None):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    if log:
        with open(log, "ab") as f:
            f.write(("$ " + " ".join(cmd) + "\n").encode()); f.write(r.stdout or b"")
    return r


# --------------------------------------------------------------------- models
def _param_files(cfg, wmin, wmax, pdir):
    """Write the LN/LR .param set; return [(label, path, n_layers)]. swprepost idiom == the
    SWinvert companion notebook (examples/adv/example_swinvert_workflow.ipynb, cell 12)."""
    import swprepost
    vs_lo, vs_hi = (float(x) * dt.KM for x in cfg["vs_bounds"])     # km/s -> m/s
    vp_lo, vp_hi = (float(x) * dt.KM for x in cfg["vp_bounds"])
    pr_lo, pr_hi = (float(x) for x in cfg["pr_bounds"])
    df = float(cfg["depth_factor"])
    pr = swprepost.Parameter.from_ln(wmin, wmax, 1, pr_lo, pr_hi, False)
    rh = swprepost.Parameter.from_fx(float(cfg["rho"]))
    out = []
    specs = [("ln%d" % ln, lambda ln=ln: swprepost.Parameter.from_ln(
        wmin, wmax, int(ln), vs_lo, vs_hi, False, depth_factor=df)) for ln in cfg["lns"]]
    specs += [("lr%g" % lr, lambda lr=lr: swprepost.Parameter.from_lr(
        wmin, wmax, float(lr), vs_lo, vs_hi, False, depth_factor=df)) for lr in cfg["lrs"]]
    for label, mk in specs:
        vs = mk()
        # Vp: same layering as Vs, its own bounds; Poisson condition ties them per layer.
        vp = swprepost.Parameter.from_parameter_and_link(vp_lo, vp_hi, False, vs, ptype="vs")
        prefix = os.path.join(pdir, label)
        swprepost.Parameterization(vp=vp, pr=pr, vs=vs, rh=rh).to_param(prefix, version="3.4.2")
        out.append((label, prefix + ".param", len(vs.lay_min)))
    return out


def _load_models(gpdcreport, report, nbest, keep_report=False, cache_for=None):
    """(thk_km, vp_kms, vs_kms, rho, misfit) for the nbest lowest-misfit models of a run.

    Reads `<report>.gm.txt` if present, else extracts it with gpdcreport and -- unless
    keep_report -- DELETES the .report. A 60 000-model report is ~560 MB (every model with its
    curves), x24 runs per cell = 13 GB; the best-100 text is ~100 KB and is all we ever use.
    The .gm.txt is what makes a run "done" for skip-if-exists, so a cell resumes correctly
    whether or not its reports were kept."""
    import swprepost
    gm = (cache_for or report) + ".gm.txt"
    bcpath = (cache_for or report) + ".bestcurve.txt"
    if not os.path.exists(gm):
        txt = subprocess.run([gpdcreport, "-best", str(int(nbest)), "-gm", report],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
        tmp = gm + ".part"
        with open(tmp, "wb") as f:
            f.write(txt)
        os.replace(tmp, gm)
        # NA convergence record (Vantassel & Cox Fig. 4): best misfit vs model index. gpdcreport
        # emits only the improvement points, so this is a few hundred bytes -- keep it forever,
        # it is the only trace of convergence once the .report is gone. Best-effort: never let
        # a diagnostics extraction fail the run.
        try:
            bc = subprocess.run([gpdcreport, "-best-curve", report], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, check=True).stdout
            with open(bcpath, "wb") as f:
                f.write(bc)
        except Exception as e:                                          # noqa: BLE001
            print("  (best-curve extraction failed: %s)" % e, flush=True)
        if not keep_report:
            os.remove(report)
    suite = swprepost.GroundModelSuite.from_geopsy(gm, sort=True)
    return [(np.asarray(g.tk, float) / dt.KM, np.asarray(g.vp, float) / dt.KM,
             np.asarray(g.vs, float) / dt.KM, np.asarray(g.rh, float), float(g.misfit))
            for g in suite.gms]


def _profile(th, vs, dep):
    """Step-function Vs(z) on grid `dep` (km). Last layer is the half-space."""
    iface = np.cumsum(th[:-1])
    return vs[np.searchsorted(iface, dep, side="right")]


def _reject(labels, nlay, best, prof_best, ratio, vs_dev, log):
    """Vantassel & Cox (2021) rejection: BOTH (1) min misfit significantly above the others
    AND (2) at the edge of the complexity range with Vs inconsistent with its counterparts.
    Returns (rejected mask, reason strings)."""
    n = len(labels)
    rej = np.zeros(n, bool); why = [""] * n
    if n < 3:
        return rej, why
    for i in range(n):
        others = [j for j in range(n) if j != i]
        med_m = float(np.median([best[j] for j in others]))
        c1 = best[i] > ratio * med_m
        edge = nlay[i] == min(nlay) or nlay[i] == max(nlay)
        med_p = np.median([prof_best[j] for j in others], axis=0)
        dev = float(np.mean(np.abs(np.log(prof_best[i] / med_p))))
        c2 = edge and dev > vs_dev
        why[i] = ("misfit %.3f vs median-others %.3f (%s); nlay %d %s; Vs dev %.2f (%s)"
                  % (best[i], med_m, "HIGH" if c1 else "ok", nlay[i],
                     "EDGE" if edge else "mid", dev, "INCONSISTENT" if dev > vs_dev else "ok"))
        rej[i] = bool(c1 and c2)
        log("  %-6s %s -> %s" % (labels[i], why[i], "REJECT" if rej[i] else "accept"))
    if rej.all():                        # never reject everything; keep the best-fitting one
        rej[int(np.argmin(best))] = False
    return rej, why


# --------------------------------------------------------------------- main
def main(cfgpath):
    cfg = dict(DEFAULTS); cfg.update(json.load(open(cfgpath)))
    t_start = time.time()
    out_npz = cfg["out_npz"]
    workdir = cfg.get("workdir") or os.path.dirname(os.path.abspath(out_npz))
    os.makedirs(workdir, exist_ok=True)
    logpath = os.path.join(workdir, "dinver_cell.log")

    def log(msg):
        print(msg, flush=True)
        with open(logpath, "a") as f:
            f.write(msg + "\n")

    dinver = cfg["dinver_bin"]
    gpdcreport = _find_tool("gpdcreport", dinver, cfg.get("gpdcreport_bin"))
    depth_max = float(cfg["depth_max"])
    dep = np.linspace(0, depth_max, 121)          # same grid as run_bayesbay -> overlays cleanly

    # ---- curves: keys are pipeline curve keys ("fund" = group, "fund_phase" = phase) ---------
    cell = vi.CellData(ix=int(cfg["cell"][0]), iy=int(cfg["cell"][1]),
                       lon=float(cfg["cell"][2]), lat=float(cfg["cell"][3]))
    for key, path in cfg["curves"].items():
        d = np.loadtxt(path, ndmin=2); d = d[d[:, 0].argsort()]
        cell.curves[key] = (d[:, 0], d[:, 1], d[:, 2])
    waves = [k for k in cfg["curves"] if cell.has(k)]
    if not waves:
        raise SystemExit("run_dinver_cell: no non-empty curves")
    log("cell (%d,%d): curves %s" % (cell.ix, cell.iy, ", ".join(
        "%s[%d]" % (w, len(cell.curves[w][0])) for w in waves)))

    # ---- layering wavelength range (metres) -------------------------------------------------
    if cfg.get("wmin_m") and cfg.get("wmax_m"):
        wmin, wmax = float(cfg["wmin_m"]), float(cfg["wmax_m"])
        wsrc = cfg.get("wavelength_source", "config")
    else:
        rayleigh = any(vi.parse_curve_key(w)[0] in ("fund", "overtone") for w in waves)
        prefer, fallback = ("fund_phase", "fund") if rayleigh else ("love_phase", "love")
        wmin, wmax, wsrc = dt.wavelength_range(cell, prefer=prefer, fallback=fallback)
    dmax_param = wmax / float(cfg["depth_factor"])
    log("layering: lambda %.0f-%.0f m from %s -> hmin %.0f m, dmax %.2f km (depth grid to %.1f)"
        % (wmin, wmax, wsrc, wmin / 3, dmax_param / dt.KM, depth_max))

    # ---- .target ------------------------------------------------------------------------------
    entries = dt.cell_target_entries(cell, waves, vi.parse_curve_key,
                                     min_cov=cfg["min_cov"], n_resample=cfg["n_resample"])
    target = os.path.join(workdir, "cell.target")
    dt.write_target(entries, target)
    tgt_curves = {name: dt.curve_from_hz_ms(t.frequency, t.velocity, t.velstd)
                  for t, _, name in entries}
    for name, (T, V, S) in tgt_curves.items():
        log("  target %-16s %2d pts  T %.2f-%.2f s  cov %.3f-%.3f"
            % (name, len(T), T.min(), T.max(), (S / V).min(), (S / V).max()))

    # ---- .param set + inversions ------------------------------------------------------------
    pdir = os.path.join(workdir, "params"); os.makedirs(pdir, exist_ok=True)
    rdir = os.path.join(workdir, "reports"); os.makedirs(rdir, exist_ok=True)
    # report_dir: where dinver STREAMS its ~560 MB .report while running. On the cluster the
    # work dir is BeeGFS and ~200 concurrent writers made a 3-layer run take 20 min (I/O-bound);
    # pointing this at node-local $TMPDIR keeps the transient file off the parallel FS. Only the
    # best-100 cache (~100 KB) is written back to rdir. Unset = beside the caches (local use).
    tdir = cfg.get("report_dir") or rdir
    if tdir != rdir:
        tdir = os.path.join(tdir, "dinver_%d_%d" % (cell.ix, cell.iy)); os.makedirs(tdir, exist_ok=True)
    params = _param_files(cfg, wmin, wmax, pdir)
    ntr = int(cfg["ntrials"]); seed0 = cfg.get("seed0")
    keep = bool(cfg.get("keep_reports", False))
    # NB no `-batch`: the cluster's Geopsy 3.4.2 dinver rejects it in -optimization mode
    # ("bad option '-batch'") although its -h lists it and `-batch -app-version` passes; the
    # local 3.5.2 accepts it. It only suppresses the GUI recent-file history -- not worth a
    # version probe. Everything else on the line is identical across the two builds.

    def one_run(label, ppath, tr):
        """One (parameterization, trial): dinver -> best-nr models. Resumable on .gm.txt."""
        report = os.path.join(rdir, "%s_t%d.report" % (label, tr))     # caches live here
        t0 = time.time()
        if not os.path.exists(report + ".gm.txt") and not os.path.exists(report):
            # the transient .report streams to tdir (node-local when report_dir is set)
            part = os.path.join(tdir, "%s_t%d.report.part" % (label, tr))
            cmd = [dinver, "-j", str(int(cfg["jobs"])), "-i", "DispersionCurve",
                   "-optimization", "-param", ppath, "-target", target,
                   "-ns0", str(int(cfg["ns0"])), "-ns", str(int(cfg["ns"])),
                   "-nr", str(int(cfg["nr"])), "-o", part, "-f"]
            if seed0 is not None:
                cmd += ["-seed", str(int(seed0) + tr)]
            r = _run(cmd, log=os.path.join(workdir, "dinver_runs_%s_t%d.log" % (label, tr)),
                     timeout=cfg.get("run_timeout"))
            if r.returncode != 0 or not os.path.exists(part):
                return label, tr, None, time.time() - t0, r.returncode
            with open(report + ".time", "w") as f:  # keep the inversion time across re-pools
                f.write("%.1f\n" % (time.time() - t0))
            # extract the best-nr cache from the (local) report, then drop the report; the
            # cache is named after `report` in rdir so resume/diagnostics find it as before
            os.replace(part, part[:-len(".part")])
            models = _load_models(gpdcreport, part[:-len(".part")], cfg["nr"], keep_report=keep,
                                  cache_for=report)
            if keep and tdir != rdir:
                os.replace(part[:-len(".part")], report)
            return label, tr, models, time.time() - t0, 0
        models = _load_models(gpdcreport, report, cfg["nr"], keep_report=keep)
        try:
            dtm = float(open(report + ".time").read())
        except Exception:
            dtm = time.time() - t0
        return label, tr, models, dtm, 0

    # NA cost is SUPER-linear in model count (each new sample is placed against the growing
    # Voronoi set): 60 000 models on an 11-layer LR1.2 took ~390 s locally where 6 000 took
    # 9 s. The 8x3 runs are independent, so run them concurrently -- n_parallel=1 under a
    # grid pool (the pool already owns the cores), more for a single-cell run.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    npar = max(1, int(cfg.get("n_parallel", 1)))
    jobs = [(label, ppath, tr) for label, ppath, nlay in params for tr in range(ntr)]
    per_param = {label: [] for label, _, _ in params}   # label -> [(models, runtime)]
    log("running %d (param, trial) inversions, %d at a time" % (len(jobs), npar))
    with ThreadPoolExecutor(max_workers=npar) as ex:
        futs = [ex.submit(one_run, *j) for j in jobs]
        for fut in as_completed(futs):
            label, tr, models, dtm, rc = fut.result()
            if models is None:
                log("  %-6s trial %d: dinver rc=%d -- skipped" % (label, tr, rc))
                continue
            per_param[label].append((models, dtm))
            log("  %-6s trial %d: %d models read, best misfit %.4f  (%.0fs)"
                % (label, tr, len(models), models[0][4] if models else np.nan, dtm))

    # ---- per-parameterization pools + rejection ---------------------------------------------
    labels, nlays, best, pools = [], [], [], []
    for label, ppath, nlay in params:
        allm = [m for models, _ in per_param[label] for m in models]
        if not allm:
            log("  %s: no models at all -- dropped" % label); continue
        allm.sort(key=lambda m: m[4])
        labels.append(label); nlays.append(nlay); best.append(allm[0][4])
        pools.append(allm[:int(cfg["n_pool"])])
    if not labels:
        raise SystemExit("run_dinver_cell: every inversion failed")
    prof_best = [_profile(p[0][0], p[0][2], dep) for p in pools]
    log("rejection (ratio %.2f, Vs dev %.2f):" % (cfg["reject_misfit_ratio"], cfg["reject_vs_dev"]))
    rej, why = _reject(labels, nlays, best, prof_best, float(cfg["reject_misfit_ratio"]),
                       float(cfg["reject_vs_dev"]), log)
    acc = [i for i in range(len(labels)) if not rej[i]]

    # ---- pooled ensemble over ACCEPTED parameterizations --------------------------------------
    ens = [m for i in acc for m in pools[i]]
    prof = np.array([_profile(m[0], m[2], dep) for m in ens])           # (n, ndepth)
    p = np.nanpercentile(prof, (2.5, 16, 50, 84, 97.5), axis=0)
    sigma_ln = np.std(np.log(prof), axis=0, ddof=1) if len(ens) > 1 else np.full(len(dep), np.nan)
    n_layers = np.array([len(m[2]) for m in ens])
    # prior-binding: any layer within bind_tol of a Vs bound, or deepest interface at dmax_param
    # Two SEPARATE binding diagnostics. Velocity binding (a layer within bind_tol of vs_min or
    # vs_max) means the prior is shaping the answer -> widen the bounds. Depth binding (deepest
    # interface at dmax_param = lambda_max/df) is different: in the many-layer LR/LN
    # parameterizations models park their last interface at the allowed maximum simply because
    # the data no longer resolve anything there (Basel-1 group: 0% velocity-bound, 32% depth-
    # bound, all in lr1.2/lr1.5/lr2/ln7 below ~4 km). That is a resolution-reach signal, not a
    # bound to widen -- SWinvert's dmax IS lambda_max/df by construction -- so it is reported
    # but does not trigger the warning.
    tol = float(cfg["bind_tol"]); vlo, vhi = (float(x) for x in cfg["vs_bounds"])
    bind_v, bind_d = [], []
    for m in ens:
        th, vs = m[0], m[2]
        bind_v.append(bool(np.any(vs <= vlo * (1 + tol)) or np.any(vs >= vhi * (1 - tol))))
        bind_d.append(bool(th.size > 1 and np.sum(th[:-1]) >= (dmax_param / dt.KM) * (1 - tol)))
    bind_vs_frac = float(np.mean(bind_v)); bind_depth_frac = float(np.mean(bind_d))
    bind_frac = bind_vs_frac
    log("pooled %d models from %d/%d parameterizations; Vs-bound binding %.3f%s; "
        "deepest interface at dmax %.3f (resolution reach, informational)"
        % (len(ens), len(acc), len(labels), bind_vs_frac,
           "  ** ABOVE 5% -- widen vs_bounds **" if bind_vs_frac > 0.05 else "",
           bind_depth_frac))

    # ---- predicted curves via disba at the ORIGINAL periods (own Vp, own rho) ---------------
    pred = {}
    for w in waves:
        T = cell.curves[w][0]
        disba_wave, mode, meas = vi.curve_def(w)
        rows = []
        for th, vp, vs, rho, _ in ens:
            thk = th.copy(); thk[-1] = 100.0            # half-space sentinel, km
            rows.append(vi.dispersion_velocity(thk, vs, T, mode, measure=meas,
                                               disba_wave=disba_wave, vp=vp, rho=rho / dt.KM))
        pred[w] = np.array(rows)
        m = np.isfinite(pred[w])
        log("  pred %-16s disba finite %.1f%%" % (w, 100 * m.mean()))

    # runtime_s = summed per-run dinver wall time (core-seconds at the concurrency used --
    # the per-run .time files survive a re-pool from cache); runtime_wall_s = this pass.
    runtime = sum(dtm for runs in per_param.values() for _, dtm in runs)
    runtime_wall = time.time() - t_start
    # ---- result npz (shared schema + dinver extras), ATOMIC --------------------------------
    d = dict(engine="dinver", depth=dep, vs_mean=np.nanmean(prof, 0), vs_median=p[2],
             vs_p025=p[0], vs_p16=p[1], vs_p84=p[3], vs_p975=p[4],
             n_models=len(ens), runtime_s=runtime, runtime_wall_s=runtime_wall,
             n_parallel=npar, acceptance=np.nan, chain_disagree=np.nan,
             n_layers_post=n_layers, waves=np.array(waves),
             cell_ixiy=np.array([cell.ix, cell.iy]), cell_lonlat=np.array([cell.lon, cell.lat]),
             sigma_ln_vs=sigma_ln, prior_bind_frac=bind_frac,
             bind_vs_frac=bind_vs_frac, bind_depth_frac=bind_depth_frac,
             param_labels=np.array(labels), param_nlayers=np.array(nlays),
             param_best_misfit=np.array(best), param_rejected=rej,
             param_reject_reason=np.array(why),
             wmin_m=wmin, wmax_m=wmax, wavelength_source=wsrc, dmax_param_km=dmax_param / dt.KM,
             vs_bounds=np.array(cfg["vs_bounds"], float), vp_bounds=np.array(cfg["vp_bounds"], float),
             pr_bounds=np.array(cfg["pr_bounds"], float), rho=float(cfg["rho"]),
             depth_factor=float(cfg["depth_factor"]), ns0=int(cfg["ns0"]), ns=int(cfg["ns"]),
             nr=int(cfg["nr"]), ntrials=ntr, n_pool=int(cfg["n_pool"]),
             )
    if not cfg.get("lean"):
        d.update(ens_misfit=np.array([m[4] for m in ens]),
                 ens_vs=prof.astype(np.float32),
                 ens_param=np.array([labels[i] for i in acc for _ in pools[i]]))
    else:
        d["band_pred"] = True      # pred_<w> rows are (2.5,16,50,84,97.5) percentiles
    for w in waves:
        T, U, S = cell.curves[w]
        d[f"obsT_{w}"], d[f"obs_{w}"], d[f"obssig_{w}"] = T, U, S
        d[f"predT_{w}"] = T
        d[f"pred_{w}"] = (np.nanpercentile(pred[w], (2.5, 16, 50, 84, 97.5), axis=0)
                          if cfg.get("lean") else pred[w])
        tT, tV, tS = tgt_curves[w]
        d[f"targetT_{w}"], d[f"target_{w}"], d[f"targetsig_{w}"] = tT, tV, tS
    tmp = out_npz + ".tmp%d.npz" % os.getpid()
    np.savez_compressed(tmp, **d)
    os.replace(tmp, out_npz)
    log("dinver cell done: %d pooled models, %.0f core-s (%.0f s wall, %d parallel) -> %s"
        % (len(ens), runtime, runtime_wall, npar, out_npz))


if __name__ == "__main__":
    main(sys.argv[1])
