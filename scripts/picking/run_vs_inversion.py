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


def read_period_ranges(path, net=None):
    """{curve_key: (tmin, tmax)} from a period-validity decision CSV.

    Expects columns net, measure, wave and T_valid_min / T_valid_max; blank bounds mean
    "open on that side", and a row with BOTH blank imposes no restriction at all (it is
    skipped, not read as (nan, nan) -- which would silently delete every period).
    """
    import csv
    out = {}
    rows = list(csv.DictReader(open(path)))
    nets = {(r.get("net") or "").strip() for r in rows} - {""}
    if net is None and len(nets) > 1:
        # keys are (wave, measure) only, so rows from a second network would overwrite the
        # first silently and the cell would be trimmed to ANOTHER network's ranges.
        raise SystemExit("read_period_ranges: %s covers %d networks (%s) -- pass --net"
                         % (path, len(nets), ", ".join(sorted(nets))))
    for row in rows:
            if net and (row.get("net") or "").strip() and row["net"].strip() != net:
                continue
            lo_s = (row.get("T_valid_min") or "").strip()
            hi_s = (row.get("T_valid_max") or "").strip()
            if not lo_s and not hi_s:
                continue
            key = vi.curve_key((row.get("wave") or "").strip(),
                               (row.get("measure") or "group").strip())
            out[key] = (float(lo_s) if lo_s else None, float(hi_s) if hi_s else None)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--production", required=True, help="swtomotv-output/production root")
    ap.add_argument("--config", default=None, help="swtomotv dataset YAML (for cell lon/lat)")
    ap.add_argument("--ix", type=int, required=True)
    ap.add_argument("--iy", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--engines", default="bayesbay", help="comma list: bayesbay,bayhunter")
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
                    help="comma list of base waves to load: fund,overtone,love,love_ot")
    ap.add_argument("--period-ranges", default=None,
                    help="CSV of validated period ranges (from period_validity_table.py: "
                         "columns net,measure,wave,T_valid_min,T_valid_max). Rows with both "
                         "bounds blank are treated as 'no restriction'; a wave listed with a "
                         "range is trimmed to it.")
    ap.add_argument("--net", default=None,
                    help="network name, used to select rows from --period-ranges")
    args = ap.parse_args()
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    os.makedirs(args.outdir, exist_ok=True)

    waves = [w.strip() for w in args.waves.split(",") if w.strip()]
    cell = vi.load_cell_curves(args.production, args.ix, args.iy, waves=waves,
                               measure="group")
    if args.production_phase:
        # second pass into the SAME CellData -> keys "fund_phase" etc. alongside the group ones
        cell = vi.load_cell_curves(args.production_phase, args.ix, args.iy, waves=waves,
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
            r = vi.run_bayesbay(cell, depth_max=args.depth_max, vs_bounds=(args.vs_min, args.vs_max),
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
                                 depth_max=args.depth_max, vs_bounds=(args.vs_min, args.vs_max),
                                 maxfrac=args.maxfrac, nchains=args.n_chains,
                                 iter_burnin=args.bh_iter_burnin, iter_main=args.bh_iter_main,
                                 workdir=os.path.join(args.outdir, "bayhunter_work"))
            r["cell"] = cell
        vi.plot_inversion(r, os.path.join(args.outdir, "bayhunter_inversion.png"),
                          title=f"BayHunter — cell ({cell.ix},{cell.iy})")
        print(f"  {r.get('n_models')} models, {r.get('runtime_s', float('nan')):.0f}s, "
              f"misfit {vi.data_misfit(r)}")
        results.append(r)

    if len(results) >= 2:
        vi.compare_engines(results, os.path.join(args.outdir, "engine_comparison.png"), cell=cell)
        print(f"\nwrote engine_comparison.png under {args.outdir}")
    print("done.")


if __name__ == "__main__":
    main()
