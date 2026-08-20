"""Driver: 1D Vs depth inversion of one tomography cell's fund+overtone group-velocity
curves, with bayesbay and/or BayHunter, and an engine comparison.

Engines run in different conda envs (bayesbay needs numpy>=2; BayHunter needs numpy<1.26),
so this driver is launched with the bayesbay env python and shells out to the BayHunter env
python for that engine (see --bayhunter-python / --bayhunter-runner).

Example (Riehen best-covered cell, deep MCMC, both engines):
  PYTHONPATH=~/Codes/Noisepy-ant /opt/anaconda3/envs/bayesbay_dev/bin/python run_vs_inversion.py \
    --production ~/Codes/extract_higher_modes/Projects/riehen/tomo/swtomotv-output/production \
    --config    ~/Codes/extract_higher_modes/Projects/riehen/tomo/riehen_swtomotv.yaml \
    --ix 13 --iy 15 --outdir <out> --engines bayesbay,bayhunter \
    --bayhunter-python /opt/anaconda3/envs/bayhunter/bin/python \
    --bayhunter-runner run_bayhunter_cell.py
"""
import argparse
import os

import numpy as np
from noisepy import vs_inversion as vi


# shared with grid_vs_inversion.py; the implementation lives in noisepy.vs_inversion
read_period_ranges = vi.read_period_ranges


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--production", required=True, help="swtomotv-output/production root")
    ap.add_argument("--config", default=None, help="swtomotv dataset YAML (for cell lon/lat)")
    ap.add_argument("--ix", type=int, required=True)
    ap.add_argument("--iy", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--engines", default="bayesbay",
                    help="comma list: bayesbay,bayhunter,dinver")
    ap.add_argument("--depth-max", type=float, default=6.0)
    ap.add_argument("--vs-min", type=float, default=0.3)
    ap.add_argument("--vs-max", type=float, default=3.6)
    ap.add_argument("--maxfrac", type=float, default=vi.MAX_ADJ_FRAC,
                    help="max |dVs/Vs| between adjacent layers (LVZ+HVZ allowed)")
    ap.add_argument("--n-chains", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=300_000, help="bayesbay iters/chain")
    ap.add_argument("--burnin", type=int, default=100_000)
    ap.add_argument("--bayhunter-python", default=None)
    ap.add_argument("--bayhunter-runner", default=None)
    ap.add_argument("--bh-iter-burnin", type=int, default=120_000)
    ap.add_argument("--bh-iter-main", type=int, default=60_000)
    ap.add_argument("--reuse", action="store_true",
                    help="reuse an existing {engine}_result.npz instead of re-running")
    ap.add_argument("--bb-constraint", default="project", choices=("project", "reject"),
                    help="bayesbay constraint handling (project=native-like, reject=hard)")
    ap.add_argument("--overtone-min-t", type=float, default=None,
                    help="DEPRECATED shorthand for --period-ranges; drop overtone periods "
                         "below this (s). Applied after --period-ranges if both are given.")
    ap.add_argument("--production-phase", default=None,
                    help="production root for the PHASE maps, when inverting group+phase "
                         "jointly. Group and phase are recommended from DIFFERENT Cd runs "
                         "(group=scaled, phase=blanket), so this is a separate path, not a "
                         "flag on --production.")
    ap.add_argument("--waves", default="fund,overtone",
                    help="comma list of base waves to load as GROUP curves; may be '' for a "
                         "phase-only inversion (then --waves-phase selects the targets)")
    ap.add_argument("--waves-phase", default=None,
                    help="comma list of base waves to load as PHASE curves from "
                         "--production-phase. Default (unset): mirror --waves when a phase "
                         "root is given. '' = no phase even with a phase root. This is "
                         "independent of --waves so e.g. group-only, phase-only and mixed "
                         "target sets are all expressible.")
    ap.add_argument("--bh-use-mp", action="store_true",
                    help="run BayHunter chains via fork multiprocessing (validated: posterior "
                         "statistically identical to serial, ~3.5x faster)")
    ap.add_argument("--bh-mp-nthreads", type=int, default=0,
                    help="worker processes for --bh-use-mp (0 = BayHunter default cpu_count; "
                         "set lower when several cells run concurrently)")
    ap.add_argument("--radial", action="store_true",
                    help="BayHunter continuous-zeta radial anisotropy: Love targets forward "
                         "on Vsh, Rayleigh on Vsv, gamma(z)=(Vsh-Vsv)/Vsv free per layer. "
                         "Only meaningful when the target set mixes Love and Rayleigh. "
                         "bayesbay has no zeta support -- rejected if it is among --engines.")
    ap.add_argument("--save-ensemble", action="store_true",
                    help="dump the full model ensemble (ens_vs, iface_depths, ens_misfit, "
                         "ens_nlayers) into the npz. Needed for interface-PROBABILITY "
                         "statistics: the default npz keeps only posterior percentiles on a "
                         "depth grid, which have already lost the per-model layer boundaries.")
    ap.add_argument("--radial-prior", default="-0.35,0.35",
                    help="uniform prior bounds for zeta (default from the CZ calibration)")
    ap.add_argument("--period-ranges", default=None,
                    help="CSV of validated period ranges (from period_validity_table.py: "
                         "columns net,measure,wave,T_valid_min,T_valid_max). Rows with both "
                         "bounds blank are treated as 'no restriction'; a wave listed with a "
                         "range is trimmed to it.")
    ap.add_argument("--net", default=None,
                    help="network name, used to select rows from --period-ranges")
    # ---- Dinver (SWinvert workflow, Vantassel & Cox 2021) ---------------------------------
    ap.add_argument("--dinver-bin", default=vi.DINVER_BIN_DEFAULT,
                    help="dinver executable (cluster: $EBROOTGEOPSY/bin/dinver)")
    ap.add_argument("--dinver-runner", default=None,
                    help="path to run_dinver_cell.py (default: next to this script)")
    ap.add_argument("--dinver-python", default=None,
                    help="interpreter for the runner (default: this one; needs swprepost+disba)")
    ap.add_argument("--dinver-measure", choices=("group", "phase", "joint"), default="group",
                    help="which of the loaded curves Dinver inverts: the GROUP keys, the PHASE "
                         "keys, or both as separate ModalCurves in one target (joint)")
    ap.add_argument("--dinver-lns", default="3,4,5,7", help="Layering-by-Number set")
    ap.add_argument("--dinver-lrs", default="3.0,2.0,1.5,1.2", help="Layering-Ratio set")
    ap.add_argument("--dinver-ntrials", type=int, default=3, help="independent NA runs/param")
    ap.add_argument("--dinver-ns", type=int, default=50_000, help="models searched (=It*Ns)")
    ap.add_argument("--dinver-nr", type=int, default=100)
    ap.add_argument("--dinver-ns0", type=int, default=10_000)
    ap.add_argument("--dinver-n-pool", type=int, default=100,
                    help="lowest-misfit models pooled per accepted parameterization")
    ap.add_argument("--dinver-depth-factor", type=float, default=2.0)
    ap.add_argument("--dinver-rho", type=float, default=2000.0, help="fixed density kg/m3")
    ap.add_argument("--dinver-pr", default="0.2,0.35",
                    help="Poisson's ratio range (free). Default is the CRUSTAL range (Vp/Vs "
                         "1.63-2.08); the SWinvert paper's 0.2,0.5 is a near-surface soil "
                         "range and let the NA drive Vp/Vs to 5 in rock (VS_INVERSION.md)")
    ap.add_argument("--dinver-vp", default="0.8,8.0", help="Vp bounds km/s (linked to Vs)")
    ap.add_argument("--dinver-jobs", type=int, default=1, help="dinver -j (internal threads)")
    ap.add_argument("--dinver-parallel", type=int, default=8,
                    help="concurrent dinver runs inside the cell (the 8x3 param/trial runs are "
                         "independent); ~ number of cores")
    ap.add_argument("--dinver-keep-reports", action="store_true",
                    help="keep the ~560 MB .report per run (default: best-nr extracted, deleted)")
    ap.add_argument("--dinver-min-cov", type=float, default=0.05,
                    help="COV floor applied to the target sigma (paper: 0.05-0.10); 0 = none")
    ap.add_argument("--dinver-n-resample", type=int, default=30,
                    help="log-wavelength resample count per curve (0 = keep raw points)")
    args = ap.parse_args()
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    if args.radial and "bayesbay" in engines:
        raise SystemExit("--radial is BayHunter-only (bayesbay has no zeta parameterization); "
                         "use --engines bayhunter")
    os.makedirs(args.outdir, exist_ok=True)

    waves = [w.strip() for w in args.waves.split(",") if w.strip()]
    if args.waves_phase is None:
        waves_phase = waves if args.production_phase else []
    else:
        waves_phase = [w.strip() for w in args.waves_phase.split(",") if w.strip()]
    if waves_phase and not args.production_phase:
        raise SystemExit("--waves-phase given but no --production-phase root to load from")
    if not waves and not waves_phase:
        raise SystemExit("empty --waves and no phase waves: nothing to invert")
    cell = None
    if waves:
        cell = vi.load_cell_curves(args.production, args.ix, args.iy, waves=waves,
                                   measure="group")
    if waves_phase:
        # second pass into the SAME CellData -> keys "fund_phase" etc. alongside the group
        # ones (or a fresh phase-only cell when --waves is empty)
        cell = vi.load_cell_curves(args.production_phase, args.ix, args.iy, waves=waves_phase,
                                   measure="phase", into=cell)

    if args.period_ranges:
        ranges = read_period_ranges(args.period_ranges, args.net)
        if ranges:
            cell = vi.restrict_periods(cell, ranges)
            for k, (lo, hi) in sorted(ranges.items()):
                print("  period range %-14s %s - %s s"
                      % (k, "open" if lo is None else lo, "open" if hi is None else hi))
        else:
            print("  --period-ranges matched no rows (check --net); no restriction applied")
    if args.overtone_min_t:
        cell = vi.restrict_periods(cell, {"overtone": (args.overtone_min_t, None)})
        print(f"overtone restricted to T >= {args.overtone_min_t}s")
    dropped = [k for k, (T, _, _) in cell.curves.items() if len(T) == 0]
    for k in dropped:
        print("  %s: EMPTY after period restriction -- dropped" % k)
        del cell.curves[k]
    if not cell.curves:
        raise SystemExit("no curves survive the period ranges for this cell")
    if args.config:
        vi.attach_cell_coords(cell, args.config)
    print(f"cell (ix,iy)=({cell.ix},{cell.iy}) lon,lat=({cell.lon:.4f},{cell.lat:.4f})")
    for w in cell.curves:
        T, U, S = cell.curves[w]
        print(f"  {w}: {len(T)} periods  T {T.min():.2f}-{T.max():.2f}s  U {U.min():.2f}-{U.max():.2f} km/s")

    results = []
    if "bayesbay" in engines:
        npz = os.path.join(args.outdir, "bayesbay_result.npz")
        if args.reuse and os.path.exists(npz):
            print("\n=== bayesbay (reused) ===")
            r = vi.load_result(npz); r["cell"] = cell
        else:
            print("\n=== bayesbay ===")
            # waves must be the cell's ACTUAL curve keys: the engine default is
            # ("fund","overtone"), which would silently drop love and every phase target
            # right after the driver loaded and printed them.
            r = vi.run_bayesbay(cell, waves=list(cell.curves),
                                depth_max=args.depth_max, vs_bounds=(args.vs_min, args.vs_max),
                                maxfrac=args.maxfrac, n_chains=args.n_chains,
                                n_iterations=args.iterations, burnin=args.burnin, seed=42,
                                constraint=args.bb_constraint)
            vi.save_result(r, npz)
        vi.plot_inversion(r, os.path.join(args.outdir, "bayesbay_inversion.png"),
                          title=f"bayesbay — cell ({cell.ix},{cell.iy})")
        print(f"  {r['n_models']} models, {r['runtime_s']:.0f}s, misfit {vi.data_misfit(r)}")
        results.append(r)

    if "bayhunter" in engines:
        npz = os.path.join(args.outdir, "bayhunter_result.npz")
        if args.reuse and os.path.exists(npz):
            print("\n=== BayHunter (reused) ===")
            r = vi.load_result(npz); r["cell"] = cell
        else:
            print("\n=== BayHunter (subprocess) ===")
            if not (args.bayhunter_python and args.bayhunter_runner):
                raise SystemExit("bayhunter engine needs --bayhunter-python and --bayhunter-runner")
            r = vi.run_bayhunter(cell, npz, args.bayhunter_runner, args.bayhunter_python,
                                 waves=list(cell.curves),
                                 depth_max=args.depth_max, vs_bounds=(args.vs_min, args.vs_max),
                                 maxfrac=args.maxfrac, nchains=args.n_chains,
                                 iter_burnin=args.bh_iter_burnin, iter_main=args.bh_iter_main,
                                 use_mp=args.bh_use_mp, mp_nthreads=args.bh_mp_nthreads,
                                 radial=args.radial, save_ensemble=args.save_ensemble,
                                 radial_prior=[float(x) for x in args.radial_prior.split(",")],
                                 workdir=os.path.join(args.outdir, "bayhunter_work"))
            r["cell"] = cell
        vi.plot_inversion(r, os.path.join(args.outdir, "bayhunter_inversion.png"),
                          title=f"BayHunter — cell ({cell.ix},{cell.iy})")
        print(f"  {r.get('n_models')} models, {r.get('runtime_s', float('nan')):.0f}s, "
              f"misfit {vi.data_misfit(r)}")
        results.append(r)

    if "dinver" in engines:
        # dinver inverts a SUBSET of the loaded keys chosen by --dinver-measure. The other two
        # engines take everything loaded, so a joint driver call can compare e.g. BayHunter on
        # group+phase against Dinver on group only -- label the output by measure so two Dinver
        # runs on the same cell never overwrite each other.
        keys = list(cell.curves)
        if args.dinver_measure == "group":
            keys = [k for k in keys if vi.parse_curve_key(k)[1] == "group"]
        elif args.dinver_measure == "phase":
            keys = [k for k in keys if vi.parse_curve_key(k)[1] == "phase"]
        if not keys:
            raise SystemExit("--dinver-measure %s: no matching curves loaded" % args.dinver_measure)
        npz = os.path.join(args.outdir, f"dinver_{args.dinver_measure}_result.npz")
        if args.reuse and os.path.exists(npz):
            print(f"\n=== Dinver/{args.dinver_measure} (reused) ===")
            r = vi.load_result(npz); r["cell"] = cell
        else:
            print(f"\n=== Dinver/{args.dinver_measure} (subprocess) — targets {keys} ===")
            runner = args.dinver_runner or os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "run_dinver_cell.py")
            r = vi.run_dinver(
                cell, npz, runner, python=args.dinver_python, dinver_bin=args.dinver_bin,
                waves=keys, lns=[int(x) for x in args.dinver_lns.split(",")],
                lrs=[float(x) for x in args.dinver_lrs.split(",")],
                ntrials=args.dinver_ntrials, ns=args.dinver_ns, nr=args.dinver_nr,
                ns0=args.dinver_ns0, n_pool=args.dinver_n_pool,
                depth_factor=args.dinver_depth_factor, n_resample=args.dinver_n_resample,
                min_cov=(args.dinver_min_cov or None), depth_max=args.depth_max,
                vs_bounds=(args.vs_min, args.vs_max),
                vp_bounds=[float(x) for x in args.dinver_vp.split(",")],
                pr_bounds=[float(x) for x in args.dinver_pr.split(",")], rho=args.dinver_rho,
                jobs=args.dinver_jobs, n_parallel=args.dinver_parallel,
                keep_reports=args.dinver_keep_reports, size_from=cell,
                workdir=os.path.join(args.outdir, f"dinver_{args.dinver_measure}_work"))
            r["cell"] = cell
        vi.plot_inversion(r, os.path.join(args.outdir, f"dinver_{args.dinver_measure}_inversion.png"),
                          title=f"Dinver/{args.dinver_measure} — cell ({cell.ix},{cell.iy})")
        acc = [str(l) for l, rj in zip(r.get("param_labels", []), r.get("param_rejected", []))
               if not rj]
        print(f"  {r.get('n_models')} pooled models from {len(acc)} params {acc}, "
              f"{r.get('runtime_s', float('nan')):.0f}s, prior-binding "
              f"{float(r.get('prior_bind_frac', float('nan'))):.3f}, misfit {vi.data_misfit(r)}")
        results.append(r)

    if len(results) >= 2:
        vi.compare_engines(results, os.path.join(args.outdir, "engine_comparison.png"), cell=cell)
        print(f"\nwrote engine_comparison.png under {args.outdir}")
    print("done.")


if __name__ == "__main__":
    main()
