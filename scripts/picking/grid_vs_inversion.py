"""Full-grid 1D Vs inversion: invert EVERY well-covered tomography cell, twice
(fundamental-only and fundamental+overtone), with BayHunter, and assemble a 3-D Vs volume.

Each (cell, waveset) is a BayHunter subprocess (run_bayhunter_cell.py, its own conda env),
run through a bounded thread pool so ~n_workers cells invert concurrently (BLAS pinned to 1
thread each). Per-cell results land in <outdir>/cells/cell_<ix>_<iy>_<waveset>.npz in the shared
result schema (load_result); existing files are skipped so the run resumes. Overtone is
restricted to T >= --overtone-min-t (data-quality cut) for the fundot waveset.

Post-processing (fund vs fundot difference, uncertainty maps, hillshade cross-sections) is
done separately by grid_vs_postprocess.py from the per-cell npz + volume.npz written here.

Example:
  PYTHONPATH=~/Codes/Noisepy-ant /opt/anaconda3/envs/bayesbay_dev/bin/python grid_vs_inversion.py \
    --production .../riehen/tomo/swtomotv-output/production --config .../riehen_swtomotv.yaml \
    --outdir .../riehen/tomo/vs_inversion/grid --n-workers 12 \
    --bayhunter-python /opt/anaconda3/envs/bayhunter/bin/python \
    --bayhunter-runner run_bayhunter_cell.py

Dinver (Geopsy NA, SWinvert workflow) is a second engine behind the same plumbing:
  --engine dinver [--dinver-bin ... --dinver-ns 50000 --dinver-ntrials 3 ...]
Same task list, skip-if-exists, sharding and assembly; the runner is run_dinver_cell.py in
THIS interpreter (needs swprepost + disba, e.g. das-ambient-noise). Keep engines in separate
--outdir trees. Note dinver's per-cell cost is orders of magnitude above BayHunter's at the
SWinvert defaults (~1.2 M models/cell) -- measure on one cell before sizing a grid, and set
--cell-timeout accordingly.
"""
import argparse
import json
import os
import subprocess
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from noisepy import vs_inversion as vi
from noisepy import period_resolution as pr
from profile_vs_inversion import coverage_grid


def invert_one(task):
    """Write curves+config for one (cell, waveset) and run the engine subprocess.

    Engine is cfg_common["_engine"]: "bayhunter" (default, run_bayhunter_cell.py in its own
    conda env) or "dinver" (run_dinver_cell.py, SWinvert workflow). The two differ ONLY in how
    the config is laid out -- BayHunter takes `curves` + `curves_phase` keyed by base wave, the
    Dinver runner one `curves` dict keyed by CURVE KEY ("fund" / "fund_phase") -- everything
    else here (skip-if-exists, timeout, teardown-crash classification, work-dir cleanup) is
    shared, so the SLURM plumbing around this function is engine-agnostic.
    """
    cell, cell_ph, waves, out_npz, workdir, bh_py, runner, cfg_common = task
    if os.path.exists(out_npz):
        return (cell.ix, cell.iy, waves, "skip", 0.0)
    os.makedirs(workdir, exist_ok=True)
    engine = cfg_common.get("_engine", "bayhunter")
    curvefiles, phasefiles = {}, {}
    for w in waves:
        if cell.has(w) and len(cell.curves[w][0]) >= 4:
            T, U, S = cell.curves[w]
            fp = os.path.join(workdir, f"disp_{w}.txt")
            np.savetxt(fp, np.column_stack([T, U, S]), fmt="%.6f")
            curvefiles[w] = fp
        if cell_ph is not None and cell_ph.has(w) and len(cell_ph.curves[w][0]) >= 4:
            T, U, S = cell_ph.curves[w]
            fp = os.path.join(workdir, f"disp_{w}_phase.txt")
            np.savetxt(fp, np.column_stack([T, U, S]), fmt="%.6f")
            phasefiles[w] = fp
    if not curvefiles and not phasefiles:    # love-only cells have no "fund"; only skip if NOTHING
        return (cell.ix, cell.iy, waves, "no-data", 0.0)
    shared = {k: v for k, v in cfg_common.items() if not k.startswith("_")}
    if engine == "dinver":
        dcfg = dict(cfg_common.get("_dinver", {}))
        size_only = bool(dcfg.pop("size_only_phase", False))
        if size_only and not curvefiles:
            return (cell.ix, cell.iy, waves, "no-data", 0.0)   # phase was for sizing only
        # one dict keyed by curve key; the runner tags Group/Phase from the key itself
        curves = dict(curvefiles)
        if not size_only:
            curves.update({f"{w}_phase": p for w, p in phasefiles.items()})
        # SWinvert layering is sized on the fundamental PHASE wavelength; use the phase cell
        # whenever the run has one (joint or --measure phase), else group U*T with a warning.
        from noisepy import dinver_target as dt
        src = vi.CellData(ix=cell.ix, iy=cell.iy)
        src.curves.update(cell.curves)
        if cell_ph is not None:
            src.curves.update({f"{w}_phase": c for w, c in cell_ph.curves.items()})
        rayleigh = any(vi.parse_curve_key(w)[0] in ("fund", "overtone") for w in curves)
        prefer, fallback = ("fund_phase", "fund") if rayleigh else ("love_phase", "love")
        wmin_m, wmax_m, wsrc = dt.wavelength_range(src, prefer=prefer, fallback=fallback)
        cfg = dict(curves=curves, out_npz=out_npz, workdir=workdir,
                   cell=[cell.ix, cell.iy, cell.lon, cell.lat],
                   wmin_m=wmin_m, wmax_m=wmax_m, wavelength_source=wsrc,
                   depth_max=shared["depth_max"], vs_bounds=shared["vs_bounds"], **dcfg)
    else:
        cfg = dict(curves=curvefiles, out_npz=out_npz,
                   savepath=os.path.join(workdir, "bh_results"),
                   cell=[cell.ix, cell.iy, cell.lon, cell.lat], **shared)
        if phasefiles:
            cfg["curves_phase"] = phasefiles     # joint group+phase (runner: keys become *_phase)
    cfgpath = os.path.join(workdir, "config.json")
    with open(cfgpath, "w") as f:
        json.dump(cfg, f)
    env = dict(os.environ, OBJC_DISABLE_INITIALIZE_FORK_SAFETY="YES",
               VECLIB_MAXIMUM_THREADS="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0 = time.time()
    # GRID_VS_DEBUG=1: stream the runner's stdout/stderr live instead of capturing -- for
    # interactive debugging. Default: capture, and print the tail only when the run FAILS.
    sink = None if os.environ.get("GRID_VS_DEBUG") else subprocess.PIPE
    try:
        r = subprocess.run([bh_py, runner, cfgpath], env=env,
                           stdout=sink, stderr=(None if sink is None else subprocess.STDOUT),
                           timeout=cfg_common.get("_timeout"))
        if r.returncode == 0:
            status = "ok" if os.path.exists(out_npz) else "no-out"
        elif os.path.exists(out_npz):
            # Teardown-only failure: the result npz (and figures) are already on disk; the
            # interpreter died on exit (observed intermittently with fork-mp at high iteration
            # counts on the cluster). The RESULT is the npz -- count it ok, but say so.
            print(f"({cell.ix},{cell.iy}) {'+'.join(waves)}: runner rc={r.returncode} AFTER "
                  f"writing the result npz -- counted ok (teardown crash)", flush=True)
            status = "ok"
        else:
            tail = (r.stdout or b"")[-3000:].decode(errors="replace") if sink else ""
            print(f"({cell.ix},{cell.iy}) {'+'.join(waves)}: runner rc={r.returncode}, no "
                  f"result npz. Output tail:\n{tail}", flush=True)
            status = "err"
    except subprocess.TimeoutExpired:
        status = "timeout"
    # the per-cell work dir holds ~90 MB of raw BayHunter chain storage; the result npz has
    # everything downstream needs, so drop the work dir to keep the grid's disk footprint flat.
    # EXCEPT a dinver cell that did not finish: its work dir holds the per-(param, trial)
    # best-100 caches (~100 KB each) that make the runner resume where it stopped -- a cell is
    # 24 runs of ~20 min on the cluster, and deleting them on a timeout threw away ~370 core-h
    # in the first Riehen submission. Only a stale .report.part (<=560 MB) can linger, and
    # the next attempt overwrites it (-f).
    if status == "ok" or engine != "dinver":
        shutil.rmtree(workdir, ignore_errors=True)
    return (cell.ix, cell.iy, waves, status, time.time() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--production", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min-fund", type=int, default=25)
    ap.add_argument("--min-overtone", type=int, default=6)
    ap.add_argument("--min-love", type=int, default=8)
    ap.add_argument("--love-production", default=None,
                    help="Love production root (holds love/map_T*.npz), for love/fundlove/"
                         "fundotlove wavesets. Its dx must match --production.")
    ap.add_argument("--overtone-min-t", type=float, default=1.0)
    ap.add_argument("--depth-max", type=float, default=6.0)
    ap.add_argument("--vs-min", type=float, default=0.3)
    ap.add_argument("--vs-max", type=float, default=3.6)
    ap.add_argument("--n-chains", type=int, default=3)
    ap.add_argument("--iter-burnin", type=int, default=25_000)
    ap.add_argument("--iter-main", type=int, default=12_000)
    ap.add_argument("--maxmodels", type=int, default=20_000)
    ap.add_argument("--pred-nsub", type=int, default=40)
    ap.add_argument("--n-workers", type=int, default=12)
    ap.add_argument("--cell-timeout", type=float, default=900.0)
    ap.add_argument("--wavesets", default="fund,fundot",
                    help="comma list of: fund (R fund), fundot (R fund+overtone), love (Love only), "
                         "fundlove (R fund+Love), fundotlove (R fund+overtone+Love)")
    ap.add_argument("--measure", choices=("group", "phase"), default="group",
                    help="phase = PHASE-ONLY inversion: --production must then be the PHASE "
                         "run's production root; its curves become <wave>_phase targets and no "
                         "group targets are built. Incompatible with --phase-root, which adds "
                         "phase ON TOP of group targets.")
    ap.add_argument("--period-ranges", default=None,
                    help="CSV of validated period ranges (period_validity_table.py: columns "
                         "net,measure,wave,T_valid_min,T_valid_max), applied uniformly to every "
                         "cell. Group rows trim the group curves, phase rows the phase curves; "
                         "rows with both bounds blank impose no restriction. Needs --net when "
                         "the CSV covers several networks.")
    ap.add_argument("--net", default=None, help="riehen|aargau; required for --criterion")
    ap.add_argument("--criterion", default="none",
                    choices=("none", "tomographic", "physical", "combined"),
                    help="per-cell period-reliability trim (period_resolution.trim_reliable)")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--rfrac", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="only first N cells (debug)")
    ap.add_argument("--cells", default=None,
                    help="explicit cell list 'ix_iy,ix_iy,...' (pilot stripes; overrides "
                         "coverage selection order, still intersected with coverage)")
    # ---- 2026-07-17 hybrid-recipe knobs (validated at 6 wells / 2 networks; see the two
    # test_2026-07-16_noise_regime_pair READMEs) ----------------------------------------------
    ap.add_argument("--phase-root", default=None,
                    help="phase tomography production root -> joint group+phase inversion")
    ap.add_argument("--love-phase-root", default=None,
                    help="Love phase production root (with --phase-root)")
    ap.add_argument("--phase-alpha", type=float, default=0.2,
                    help="edge-distance factor (d_edge >= alpha*lambda) for the PHASE curves' "
                         "physical trim. Group keeps --alpha (0.5, ~2.5-lambda far-field "
                         "validity); phase is valid to ~1 lambda (phase_pilot, validated), so "
                         "its factor scales by 1/2.5 -> 0.2. With the group alpha applied to "
                         "phase, hull-edge cells lost ALL long-T phase and their depth reach "
                         "collapsed (Basel-1: floor 1.8 km vs 5.6 km with correct alpha).")
    ap.add_argument("--phase-tmin", type=float, default=2.5,
                    help="fund/Love PHASE envelope cut [s]: the measured cross-well upper "
                         "envelope of the kinematically inconsistent (2piN mis-branch) band; "
                         "overtone phase is never cut")
    ap.add_argument("--noise-regime", choices=["free", "bounded"], default="free",
                    help="bounded = sigma ~ U(0.5*S_min, 3*S_min) per target (recipe)")
    ap.add_argument("--c2-mask", action="store_true",
                    help="apply the per-cell near-field+azimuth GROUP mask "
                         "(noisepy.curve_masks criterion 2) from the pick tables")
    ap.add_argument("--picks-inputs", default=None,
                    help="dir holding picks_{wave}_uni.csv + stations.csv for --c2-mask "
                         "(default: Projects/<net>/tomo/1_velocity_maps/inputs)")
    ap.add_argument("--c2-exempt", default="",
                    help="comma list of waves NOT masked by --c2-mask. Rationale for "
                         "'overtone': its faster U means larger lambda so C2 flags it hardest, "
                         "yet it never exhibited the near-field group misfit empirically "
                         "(clean at 5/6 wells; the 2026-07-17 uniform application deleted it "
                         "at 34%/26% of Riehen/Aargau cells -- grid_pilot DECISIONS.md D9).")
    ap.add_argument("--radial", action="store_true",
                    help="continuous per-layer radial gamma (BayHunter_Aniso fork)")
    ap.add_argument("--radial-prior", default="-0.35,0.35")
    ap.add_argument("--vpvs-range", default=None,
                    help="'lo,hi' -> free Vp/Vs (recipe: 1.5,3.5); default fixed 1.73")
    ap.add_argument("--mode-gate-phase", default=None, metavar="PHASE_ROOT",
                    help="KINEMATIC mode-identification gate: drop (cell, period) samples whose "
                         "GROUP pick has U >= c against the phase maps at PHASE_ROOT. Normal "
                         "dispersion forbids U >= c, and those samples are where the "
                         "'fundamental' has landed on a higher-mode branch. Parameter-free.")
    ap.add_argument("--mode-gate-overtone", default=None, metavar="GROUP_ROOT",
                    help="BRANCH-ORDER gate: also drop samples with U >= U_overtone from the "
                         "overtone maps at GROUP_ROOT. Rigorous for Rayleigh; for Love the only "
                         "reference is the RAYLEIGH overtone, so treat it as a heuristic.")
    ap.add_argument("--mode-gate-margin", type=float, default=1.0,
                    help="drop if U >= margin*reference; 1.0 (default) removes only the "
                         "strictly impossible, <1 also trims near-misses")
    ap.add_argument("--use-mp", action="store_true",
                    help="fork-multiprocess the chains INSIDE each cell's BayHunter subprocess "
                         "(validated: posterior statistically identical to serial). Size "
                         "--n-workers x --mp-nthreads to the CPUs available.")
    ap.add_argument("--mp-nthreads", type=int, default=0,
                    help="worker processes per cell for --use-mp (0 = BayHunter default "
                         "cpu_count; set = --n-chains so all chains run concurrently)")
    ap.add_argument("--engine", choices=("bayhunter", "dinver"), default="bayhunter",
                    help="per-cell engine. dinver = Geopsy NA via the SWinvert workflow "
                         "(run_dinver_cell.py); group/phase/joint follows --measure/--phase-root "
                         "exactly as for BayHunter. Use a SEPARATE --outdir per engine.")
    ap.add_argument("--bayhunter-python", default=None, help="required for --engine bayhunter")
    ap.add_argument("--bayhunter-runner", default=None, help="required for --engine bayhunter")
    # ---- Dinver (SWinvert; same knobs as run_vs_inversion.py) --------------------------------
    ap.add_argument("--dinver-bin", default=vi.DINVER_BIN_DEFAULT,
                    help="dinver executable (cluster: $EBROOTGEOPSY/bin/dinver)")
    ap.add_argument("--dinver-runner", default=None,
                    help="run_dinver_cell.py (default: next to this script)")
    ap.add_argument("--dinver-python", default=None,
                    help="interpreter for the runner (default: this one; needs swprepost+disba)")
    ap.add_argument("--dinver-lns", default="3,4,5,7")
    ap.add_argument("--dinver-lrs", default="3.0,2.0,1.5,1.2")
    ap.add_argument("--dinver-ntrials", type=int, default=3)
    ap.add_argument("--dinver-ns", type=int, default=50_000)
    ap.add_argument("--dinver-nr", type=int, default=100)
    ap.add_argument("--dinver-ns0", type=int, default=10_000)
    ap.add_argument("--dinver-n-pool", type=int, default=100)
    ap.add_argument("--dinver-depth-factor", type=float, default=2.0)
    ap.add_argument("--dinver-rho", type=float, default=2000.0)
    ap.add_argument("--dinver-pr", default="0.2,0.35", help="Poisson range; crustal default, see run_vs_inversion.py")
    ap.add_argument("--dinver-vp", default="0.8,8.0")
    ap.add_argument("--dinver-jobs", type=int, default=1)
    ap.add_argument("--dinver-min-cov", type=float, default=0.05)
    ap.add_argument("--dinver-size-phase-root", default=None, metavar="PHASE_ROOT",
                    help="GROUP-only dinver runs: load this phase production root ONLY to size "
                         "the SWinvert layering from the fundamental PHASE wavelength (lmin/3, "
                         "lmax/df); phase curves are NOT inverted. Without it a group-only run "
                         "sizes from group U*T, ~30%% shallower (Basel-1: 4.7 vs 6.1 km). "
                         "Mutually exclusive with --phase-root.")
    ap.add_argument("--work-tag", choices=("task", "cell"), default="task",
                    help="work-dir layout: 'task' = per Slurm task (race-safe, boosters OK, no "
                         "cross-submission resume); 'cell' = one dir per cell under "
                         "<outdir>/work/ so a dinver cell killed by a timeout or walltime "
                         "resumes from its per-run caches on resubmission. NEVER run two "
                         "arrays on the same outdir with 'cell'.")
    ap.add_argument("--dinver-report-dir", default=None, metavar="DIR",
                    help="node-local dir for dinver's transient ~560 MB .report (e.g. $TMPDIR). "
                         "On BeeGFS with ~200 concurrent cells the runs were I/O-bound (20 min "
                         "for a 3-layer run). Only the ~100 KB best-100 caches go to the work dir.")
    ap.add_argument("--dinver-lean", action="store_true",
                    help="write the per-cell npz with percentiles only: predicted curves as "
                         "(2.5,16,50,84,97.5) percentile rows instead of one row per model, and "
                         "no ens_vs/ens_misfit/ens_param. ~50 KB/cell instead of ~600 KB. "
                         "compare_engines/data_misfit/assemble read it unchanged; the per-model "
                         "diagnostics (dinver_diagnostics.py) do not apply.")
    ap.add_argument("--dinver-n-resample", type=int, default=30)
    ap.add_argument("--shard", default=None,
                    help="'i/N': process only task slice tasks[i::N] (SLURM array fan-out). "
                         "All shards write to the shared cells/ dir; run --assemble-only after.")
    ap.add_argument("--no-assemble", action="store_true",
                    help="skip stitching cells/ into volume_*.npz (use on array workers)")
    ap.add_argument("--assemble-only", action="store_true",
                    help="skip inversion; only stitch existing cells/*.npz into volume_*.npz")
    ap.add_argument("--reverse", action="store_true",
                    help="walk the task list back-to-front. Lets a second ('booster') array "
                         "chew the same outdir from the opposite end while the main array runs "
                         "forward: skip-if-exists keeps them from redoing each other's cells, "
                         "and they only meet in the middle at the very end.")
    args = ap.parse_args()
    celldir = os.path.join(args.outdir, "cells")
    # Work dirs are PER TASK, not shared. Two workers racing on the same cell would otherwise
    # interleave in one scratch dir (config.json, disp_*.txt, bh_results) and corrupt each
    # other's inputs. With separate dirs the loser simply redoes work, and the atomic npz
    # write means whichever finishes last replaces an equivalent file. This is what makes
    # over-provisioning (a booster array on idle short-walltime nodes) safe.
    _tag = (f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}"
            if os.environ.get("SLURM_ARRAY_JOB_ID") else str(os.getpid()))
    # --work-tag cell: ONE work dir per cell, shared across submissions, so a resubmitted
    # array resumes a half-done dinver cell from its caches. The price is the race-safety
    # above: never over-provision (booster) an arm that uses it.
    workroot = os.path.join(args.outdir, "work", _tag if args.work_tag == "task" else "")
    workroot = workroot.rstrip("/")
    os.makedirs(celldir, exist_ok=True)

    want = [w.strip() for w in args.wavesets.split(",") if w.strip()]
    if args.assemble_only:
        assemble_volume(celldir, want, args.outdir)
        return

    wavesets = {"fund": ("fund",), "fundot": ("fund", "overtone"), "love": ("love",),
                "fundlove": ("fund", "love"), "fundotlove": ("fund", "overtone", "love")}
    need_love = any("love" in wavesets[w] for w in want)
    if need_love and not args.love_production:
        # prod3_k3-style runs carry love/ inside the same production tree; older runs kept Love
        # under a separate swtomotv-output-love-<dx> root and must still pass it explicitly.
        args.love_production = args.production
        print(f"--love-production not given; using --production ({args.production})", flush=True)
    if args.measure == "phase" and args.phase_root:
        raise SystemExit("--measure phase is a PHASE-ONLY run (--production IS the phase root); "
                         "--phase-root only makes sense for joint group+phase on top of group")
    pranges = (vi.read_period_ranges(args.period_ranges, net=args.net)
               if args.period_ranges else {})
    # split by measure: group-keyed entries trim the group CellData; phase entries are re-keyed
    # to the plain wave name because the phase CellData (bases_ph / --measure phase) stores its
    # curves under plain keys -- the "phase-ness" is carried by WHICH dict the curve files land
    # in (curves vs curves_phase), not by the curve key (see invert_one / the runner).
    pr_group = {k: v for k, v in pranges.items() if not k.endswith("_phase")}
    pr_phase = {k[: -len("_phase")]: v for k, v in pranges.items() if k.endswith("_phase")}

    covf = coverage_grid(args.production, "fund")
    covo = coverage_grid(args.production, "overtone")
    nx, ny = covf.shape
    # a cell is a candidate if it has enough Rayleigh fundamental OR (for love-only) enough Love;
    # per-waveset, curves the cell lacks are dropped at task build (cell.has) so this over-includes
    # safely. Rayleigh-overtone coverage is NOT required here (fundot drops the wave if absent).
    good = covf >= args.min_fund
    if need_love:
        covl = coverage_grid(args.love_production, "love")
        good = good | (covl >= args.min_love)
    ixs, iys = np.where(good)
    cells_ij = list(zip(ixs.tolist(), iys.tolist()))
    if args.cells:
        wanted = [tuple(int(v) for v in c.split("_")) for c in args.cells.split(",")]
        have = set(cells_ij)
        missing = [c for c in wanted if c not in have]
        if missing:
            print(f"--cells: {len(missing)} not in coverage, dropped: {missing}", flush=True)
        cells_ij = [c for c in wanted if c in have]
    if args.limit:
        cells_ij = cells_ij[: args.limit]
    print(f"{os.path.basename(args.config)}: {len(cells_ij)} cells x {len(want)} wavesets "
          f"= {len(cells_ij)*len(want)} runs, {args.n_workers} workers", flush=True)

    cfg_common = dict(depth_max=args.depth_max, vs_bounds=[args.vs_min, args.vs_max],
                      n_layers=[1, 20], maxfrac=vi.MAX_ADJ_FRAC, nchains=args.n_chains,
                      iter_burnin=args.iter_burnin, iter_main=args.iter_main,
                      maxmodels=args.maxmodels, pred_nsub=args.pred_nsub, skip_pred=False,
                      _timeout=args.cell_timeout)
    if args.noise_regime != "free":
        cfg_common["noise_regime"] = args.noise_regime
    if args.vpvs_range:
        cfg_common["vpvs"] = [float(x) for x in args.vpvs_range.split(",")]
    if args.radial:
        cfg_common["radial_anisotropy"] = True
        cfg_common["radial_prior"] = [float(x) for x in args.radial_prior.split(",")]
    if args.use_mp:
        cfg_common["use_mp"] = True
        cfg_common["mp_nthreads"] = args.mp_nthreads
    # ---- engine selection: interpreter + runner + engine-private config ---------------------
    cfg_common["_engine"] = args.engine
    if args.engine == "dinver":
        import sys
        if args.radial or args.vpvs_range or args.noise_regime != "free":
            raise SystemExit("--engine dinver: --radial / --vpvs-range / --noise-regime are "
                             "BayHunter-only (Dinver has free Poisson's ratio, no zeta, no "
                             "hierarchical noise)")
        eng_py = args.dinver_python or sys.executable
        eng_runner = args.dinver_runner or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "run_dinver_cell.py")
        cfg_common["_dinver"] = dict(
            dinver_bin=args.dinver_bin,
            lns=[int(x) for x in args.dinver_lns.split(",")],
            lrs=[float(x) for x in args.dinver_lrs.split(",")],
            ntrials=args.dinver_ntrials, ns=args.dinver_ns, nr=args.dinver_nr,
            ns0=args.dinver_ns0, n_pool=args.dinver_n_pool,
            depth_factor=args.dinver_depth_factor, n_resample=args.dinver_n_resample,
            min_cov=(args.dinver_min_cov or None),
            vp_bounds=[float(x) for x in args.dinver_vp.split(",")],
            pr_bounds=[float(x) for x in args.dinver_pr.split(",")], rho=args.dinver_rho,
            jobs=args.dinver_jobs, seed0=1, keep_reports=False, lean=bool(args.dinver_lean),
            report_dir=args.dinver_report_dir,
            size_only_phase=bool(args.dinver_size_phase_root))
        if args.dinver_size_phase_root and args.phase_root:
            raise SystemExit("--dinver-size-phase-root and --phase-root are mutually exclusive")
    else:
        if not (args.bayhunter_python and args.bayhunter_runner):
            raise SystemExit("--engine bayhunter needs --bayhunter-python and --bayhunter-runner")
        eng_py, eng_runner = args.bayhunter_python, args.bayhunter_runner

    # build tasks; overtone restricted to T>=overtone-min-t for the fundot waveset, then the
    # per-cell reliability criterion trims each curve to its well-resolved periods.
    trim_params = {"alpha": args.alpha, "R_frac": args.rfrac, "depth_max": args.depth_max}
    if args.criterion != "none" and not args.net:
        raise SystemExit("--criterion needs --net (riehen|aargau)")
    # point the reliability metrics at THIS run's production maps (may be a finer-dx grid);
    # station xstat/ystat are dx-independent so the default cache is reused.
    pr.PROD_ROOT[args.net] = args.production
    # station cache lives beside the production root (<...>/cache/stations_in_grid.csv);
    # period_resolution's legacy fallback was removed in the reorg and RAISES without this.
    _cache = os.path.join(os.path.dirname(args.production.rstrip("/")),
                          "cache", "stations_in_grid.csv")
    if os.path.exists(_cache):
        pr.CACHE_CSV[args.net] = _cache
    load_waves = ("fund", "overtone", "love") if need_love else ("fund", "overtone")
    wave_roots = {"love": args.love_production} if need_love else None
    if need_love:
        pr.PROD_ROOT[(args.net, "love")] = args.love_production
    # ---- pass 1: load all base cells (group and, if requested, phase) with coords -----------
    # --measure phase: --production IS the phase run; its curves go into the PHASE slot
    # (bases_ph -> curves_phase -> phase targets in the runner) and the group slot stays an
    # empty CellData that only carries the cell coords.
    bases, bases_ph = {}, {}
    gate_report = {}
    for (ix, iy) in cells_ij:
        if args.measure == "phase":
            base = vi.CellData(ix=int(ix), iy=int(iy))
            ph = vi.load_cell_curves(args.production, ix, iy, waves=load_waves,
                                     wave_roots=wave_roots)
            if pr_phase:
                ph = vi.restrict_periods(ph, pr_phase)
            vi.attach_cell_coords(ph, args.config)
            bases_ph[(ix, iy)] = ph
        else:
            base = vi.load_cell_curves(args.production, ix, iy, waves=load_waves,
                                       wave_roots=wave_roots)
            if pr_group:
                base = vi.restrict_periods(base, pr_group)
            # mode-identification gate: remove GROUP samples that cannot be the branch they
            # claim to be. Applied AFTER the period-range trim so the band decision stands and
            # this only removes what is physically impossible within it.
            if args.mode_gate_phase or args.mode_gate_overtone:
                base = vi.mode_id_gate(base, ix, iy,
                                       phase_root=args.mode_gate_phase,
                                       overtone_root=args.mode_gate_overtone,
                                       waves=load_waves, wave_roots=wave_roots,
                                       margin=args.mode_gate_margin, report=gate_report)
            size_root = getattr(args, "dinver_size_phase_root", None)
            if args.phase_root or size_root:
                ph_roots = {"love": args.love_phase_root} if (need_love and args.love_phase_root) \
                    else None
                ph = vi.load_cell_curves(args.phase_root or size_root, ix, iy, waves=load_waves,
                                         wave_roots=ph_roots)
                if pr_phase:
                    ph = vi.restrict_periods(ph, pr_phase)
                vi.attach_cell_coords(ph, args.config)
                bases_ph[(ix, iy)] = ph
        vi.attach_cell_coords(base, args.config)
        bases[(ix, iy)] = base
    if gate_report:
        print("mode-identification gate (dropped / judged samples):", flush=True)
        for (w, meas), (bad, tot) in sorted(gate_report.items()):
            print(f"    {w:<10} vs {meas:<9} {bad:>7} / {tot:<7} "
                  f"({100.0 * bad / max(tot, 1):.1f}%)", flush=True)

    # ---- criterion 2: per-(cell, period) group mask from the pick tables, built ONCE --------
    c2_tables = {}
    if args.c2_mask:
        from noisepy import curve_masks as cm
        pin = args.picks_inputs or (f"/Users/genevievesavard/Codes/extract_higher_modes/"
                                    f"Projects/{args.net}/tomo/1_velocity_maps/inputs")
        cells_lonlat = np.array([[bases[ij].lon, bases[ij].lat] for ij in cells_ij])
        if not np.isfinite(cells_lonlat).all():
            # attach_cell_coords leaves NaN when swtomotv cannot be imported (it lives in the
            # bayesbay env, not bayhunter -- run THIS driver with the bayesbay python, the
            # subprocess runner stays bayhunter). NaN coords would make the C2 geometry pass
            # silently find zero crossing pairs and mask EVERY curve to nothing.
            raise SystemExit("--c2-mask: cell coords are NaN (swtomotv not importable in this "
                             "env, or wrong --config yaml). Run grid_vs_inversion with the "
                             "bayesbay env's python; see the module docstring example.")
        exempt = {w.strip() for w in args.c2_exempt.split(",") if w.strip()}
        for w in load_waves:
            if w in exempt:
                print(f"  c2: {w} EXEMPT from the mask (--c2-exempt)", flush=True)
                continue
            pcsv = os.path.join(pin, f"picks_{w}_uni.csv")
            if not os.path.exists(pcsv):
                print(f"  c2: no pick table for {w} ({pcsv}); wave left unmasked", flush=True)
                continue
            print(f"  c2 table: {w} ({len(cells_ij)} cells)", flush=True)
            c2_tables[w] = cm.build_c2_table(
                pcsv, os.path.join(pin, "stations.csv"), cells_lonlat,
                cache=os.path.join(args.outdir, f"c2_table_{w}.npz"))

    def _apply_c2(cell, cellidx):
        import copy
        c = copy.deepcopy(cell)
        for w, tab in c2_tables.items():
            if not c.has(w):
                continue
            T, U, S = c.curves[w]
            k = cm.c2_keep_for_periods(tab, cellidx, T)
            c.curves[w] = (T[k], U[k], S[k])
        return c

    tasks = []
    for ci, (ix, iy) in enumerate(cells_ij):
        base = bases[(ix, iy)]
        if c2_tables:
            base = _apply_c2(base, ci)                    # group mask, waveset-independent
        base_ph = bases_ph.get((ix, iy))
        if base_ph is not None:
            # ORDER MATTERS: reliability criterion FIRST (as in well_vs_qc / the validated well
            # arms), THEN the envelope cut. trim_reliable is curve-dependent (its metrics see
            # the whole curve), so envelope-first shreds the long-T phase it should keep --
            # measured at cell 9_18: 32 raw -> 3 pts envelope-first vs 25 -> ~7 trim-first.
            if args.criterion != "none":
                _saved = pr.PROD_ROOT.get(args.net)
                pr.PROD_ROOT[args.net] = args.phase_root or args.production
                ph_params = dict(trim_params, alpha=args.phase_alpha)   # phase-validity alpha
                base_ph = pr.trim_reliable(base_ph, args.net, args.criterion, ph_params)
                pr.PROD_ROOT[args.net] = _saved
            # phase ENVELOPE cut (fund/love; overtone phase untouched). <=0 disables it (e.g.
            # when --period-ranges alone governs the band).
            if args.phase_tmin > 0:
                base_ph = vi.restrict_periods(base_ph, {"fund": (args.phase_tmin, None),
                                                        "love": (args.phase_tmin, None)})
        for wskey in want:
            cell = base
            cell_ph = base_ph
            if "overtone" in wavesets[wskey] and args.overtone_min_t:
                cell = vi.restrict_periods(base, {"overtone": (args.overtone_min_t, None)})
                if cell_ph is not None:
                    cell_ph = vi.restrict_periods(cell_ph,
                                                  {"overtone": (args.overtone_min_t, None)})
            if args.criterion != "none":
                cell = pr.trim_reliable(cell, args.net, args.criterion, trim_params)
            if cell_ph is not None:
                # only the waveset's waves go into curves_phase
                import copy
                cp = copy.deepcopy(cell_ph)
                cp.curves = {w: cp.curves[w] for w in wavesets[wskey] if cp.has(w)}
                cell_ph_ws = cp
            else:
                cell_ph_ws = None
            out_npz = os.path.join(celldir, f"cell_{ix}_{iy}_{wskey}.npz")
            wd = os.path.join(workroot, f"{ix}_{iy}_{wskey}")
            tasks.append((cell, cell_ph_ws, wavesets[wskey], out_npz, wd,
                          eng_py, eng_runner, cfg_common))

    if args.reverse:
        tasks = tasks[::-1]
        print(f"--reverse: walking {len(tasks)} tasks back-to-front", flush=True)

    # SLURM-array fan-out: this task processes a deterministic disjoint slice. The full task
    # list is identical across shards (cells_ij is np.where order), so tasks[i::N] partitions it.
    if args.shard:
        si, sn = (int(x) for x in args.shard.split("/"))
        tasks = tasks[si::sn]
        print(f"shard {si}/{sn}: {len(tasks)} runs on this task", flush=True)

    t0 = time.time()
    done = {"ok": 0, "skip": 0, "err": 0, "timeout": 0, "no-out": 0, "no-data": 0}
    with ThreadPoolExecutor(max_workers=args.n_workers) as ex:
        futs = [ex.submit(invert_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            ix, iy, waves, status, dt = fut.result()
            done[status] = done.get(status, 0) + 1
            if i % 10 == 0 or status not in ("ok", "skip"):
                el = time.time() - t0
                rate = i / el if el else 0
                eta = (len(tasks) - i) / rate / 60 if rate else 0
                print(f"[{i}/{len(tasks)}] ({ix},{iy}) {'+'.join(waves)} {status} {dt:.0f}s "
                      f"| {dict(done)} | {el/60:.1f}min ETA {eta:.0f}min", flush=True)
    print(f"\nfinished {len(tasks)} runs in {(time.time()-t0)/60:.1f} min: {dict(done)}", flush=True)

    if args.no_assemble or args.shard:
        print("skipping assembly (sharded/--no-assemble); run --assemble-only when all shards "
              "finish", flush=True)
    else:
        assemble_volume(celldir, want, args.outdir)


def assemble_volume(celldir, wavesets, outdir):
    """Stack per-cell npz into per-waveset (ncell, ndepth) arrays with coords + uncertainty."""
    import glob
    for wskey in wavesets:
        files = sorted(glob.glob(os.path.join(celldir, f"cell_*_{wskey}.npz")))
        if not files:
            continue
        dep = None
        ij, lonlat, xy, med, p16, p84, p025, p975, chi_f, chi_o, chi_l, nlay = \
            ([] for _ in range(12))
        # radial anisotropy: gamma(z)=(Vsh-Vsv)/Vsv and the Voigt-referenced zeta are the whole
        # point of a --radial run, so stack them alongside Vs instead of leaving them reachable
        # only per cell. Isotropic runs carry the same keys filled with NaN -- stacking those is
        # harmless and keeps every volume the same shape.
        aniso_keys = ("gamma_median", "gamma_p16", "gamma_p84", "gamma_p_pos",
                      "zeta_median", "zeta_p16", "zeta_p84")
        aniso = {k: [] for k in aniso_keys}
        # per-cell depth reach. Figures MUST mask below z_reliable_max ("the ring fix"): below
        # its data reach the trans-D posterior parsimoniously extends the last constrained
        # velocity downward, painting an unphysical slow ring at depth. Carrying it here keeps
        # the figure scripts independent of the per-cell tree (which stays on scratch).
        zrel = {"z_reliable_min": [], "z_reliable_max": []}
        for f in files:
            try:
                r = vi.load_result(f)
            except Exception:
                continue
            for k in aniso_keys:
                if k in r:
                    aniso[k].append(np.asarray(r[k], float))
            for k in zrel:
                zrel[k].append(float(r[k]) if k in r else np.nan)
            dep = r["depth"]
            ix, iy = (int(v) for v in r.get("cell_ixiy", (-1, -1)))
            lon, lat = (float(v) for v in r.get("cell_lonlat", (np.nan, np.nan)))
            ij.append((ix, iy)); lonlat.append((lon, lat))
            med.append(r["vs_median"]); p16.append(r["vs_p16"]); p84.append(r["vs_p84"])
            p025.append(r["vs_p025"]); p975.append(r["vs_p975"])
            nlay.append(float(np.mean(r.get("n_layers_post", [np.nan]))))
            # A phase-only run stores its curves under "<wave>_phase", so looking up the bare
            # wave name leaves chi_* entirely NaN and silently drops the misfit QC column from
            # every phase volume. A config is group OR phase, never both, so falling back to
            # the _phase key keeps one column per wave whatever the measure.
            mis = vi.data_misfit(r)
            chi_f.append(mis.get("fund", mis.get("fund_phase", np.nan)))
            chi_o.append(mis.get("overtone", mis.get("overtone_phase", np.nan)))
            chi_l.append(mis.get("love", mis.get("love_phase", np.nan)))
        out = os.path.join(outdir, f"volume_{wskey}.npz")
        extra = {k: np.array(v) for k, v in aniso.items() if len(v) == len(ij)}
        extra.update({k: np.array(v) for k, v in zrel.items() if len(v) == len(ij)})
        np.savez_compressed(out, depth=dep, cells=np.array(ij), lonlat=np.array(lonlat),
                            vs_median=np.array(med), vs_p16=np.array(p16), vs_p84=np.array(p84),
                            vs_p025=np.array(p025), vs_p975=np.array(p975),
                            chi_fund=np.array(chi_f), chi_overtone=np.array(chi_o),
                            chi_love=np.array(chi_l),
                            n_layers=np.array(nlay), waveset=wskey, **extra)
        print(f"wrote {out}: {len(ij)} cells"
              + (f" (+{len(extra)} anisotropy arrays)" if extra else ""), flush=True)


if __name__ == "__main__":
    main()
