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
                    help="drop overtone periods below this (s) as a data-quality test, e.g. 1.0")
    args = ap.parse_args()
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    os.makedirs(args.outdir, exist_ok=True)

    cell = vi.load_cell_curves(args.production, args.ix, args.iy)
    if args.overtone_min_t:
        cell = vi.restrict_periods(cell, {"overtone": (args.overtone_min_t, None)})
        print(f"overtone restricted to T >= {args.overtone_min_t}s")
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
