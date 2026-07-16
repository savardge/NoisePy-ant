"""Vs(z) cross-section along a transect of tomography cells.

Selects a well-covered line of cells through the array centre (W->E = vary ix at fixed iy;
S->N = vary iy at fixed ix), inverts each cell's fund+overtone group-velocity curves for a
1D Vs(z) posterior (bayesbay-project or BayHunter), and assembles a 2-D Vs section
(distance x depth, coloured by posterior-median Vs) plus a per-cell Vs(z) small-multiple.

Example (Riehen West->East through the array middle, bayesbay, overtone T>=1s):
  PYTHONPATH=~/Codes/Noisepy-ant /opt/anaconda3/envs/bayesbay_dev/bin/python profile_vs_inversion.py \
    --production ~/.../riehen/tomo/swtomotv-output/production --config ~/.../riehen_swtomotv.yaml \
    --axis x --outdir <out> --engine bayesbay --overtone-min-t 1.0
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from noisepy import vs_inversion as vi


def coverage_grid(production_root, wave="fund"):
    """(nx,ny) count of valid periods per cell for a wave."""
    import glob
    files = sorted(glob.glob(os.path.join(production_root, wave, "map_T*.npz")))
    cov = None
    for f in files:
        z = np.load(f)
        m = z["mask"].astype(int)
        cov = m if cov is None else cov + m
    return cov


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--production", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--axis", required=True, choices=("x", "y"),
                    help="x = West->East transect (vary ix); y = South->North (vary iy)")
    ap.add_argument("--line-index", type=int, default=None,
                    help="fixed iy (axis=x) or ix (axis=y); default = densest line")
    ap.add_argument("--min-fund", type=int, default=25, help="min fund periods to include a cell")
    ap.add_argument("--min-overtone", type=int, default=6, help="min overtone periods to include")
    ap.add_argument("--step", type=int, default=2, help="take every Nth cell along the line")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--engine", default="bayesbay", choices=("bayesbay", "bayhunter"))
    ap.add_argument("--depth-max", type=float, default=6.0)
    ap.add_argument("--vs-min", type=float, default=0.3)
    ap.add_argument("--vs-max", type=float, default=3.6)
    ap.add_argument("--overtone-min-t", type=float, default=1.0)
    ap.add_argument("--n-chains", type=int, default=6)
    ap.add_argument("--iterations", type=int, default=120_000)
    ap.add_argument("--burnin", type=int, default=45_000)
    ap.add_argument("--bh-iter-burnin", type=int, default=40_000)
    ap.add_argument("--bh-iter-main", type=int, default=20_000)
    ap.add_argument("--bayhunter-python", default=None)
    ap.add_argument("--bayhunter-runner", default=None)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    covf = coverage_grid(args.production, "fund")
    covo = coverage_grid(args.production, "overtone")
    nx, ny = covf.shape
    good = (covf >= args.min_fund) & (covo >= args.min_overtone)

    # choose the transect line = densest row/column, unless given
    if args.axis == "x":                       # vary ix at fixed iy
        li = args.line_index if args.line_index is not None else int(np.argmax(good.sum(axis=0)))
        idxs = [ix for ix in range(nx) if good[ix, li]][:: args.step]
        cells = [(ix, li) for ix in idxs]
        along_label = "x (West -> East) [km]"
    else:                                      # vary iy at fixed ix
        li = args.line_index if args.line_index is not None else int(np.argmax(good.sum(axis=1)))
        idxs = [iy for iy in range(ny) if good[li, iy]][:: args.step]
        cells = [(li, iy) for iy in idxs]
        along_label = "y (South -> North) [km]"
    print(f"{args.axis}-profile at line index {li}: {len(cells)} cells -> {cells}")

    dep = None
    section, along, lonlat, results = [], [], [], []
    for (ix, iy) in cells:
        cell = vi.load_cell_curves(args.production, ix, iy)
        if args.overtone_min_t:
            cell = vi.restrict_periods(cell, {"overtone": (args.overtone_min_t, None)})
        vi.attach_cell_coords(cell, args.config)
        try:
            if args.engine == "bayesbay":
                r = vi.run_bayesbay(cell, depth_max=args.depth_max,
                                    vs_bounds=(args.vs_min, args.vs_max), n_chains=args.n_chains,
                                    n_iterations=args.iterations, burnin=args.burnin,
                                    seed=42, constraint="project")
            else:
                npz = os.path.join(args.outdir, f"bh_{ix}_{iy}.npz")
                r = vi.run_bayhunter(cell, npz, args.bayhunter_runner, args.bayhunter_python,
                                     depth_max=args.depth_max, vs_bounds=(args.vs_min, args.vs_max),
                                     nchains=args.n_chains, iter_burnin=args.bh_iter_burnin,
                                     iter_main=args.bh_iter_main,
                                     workdir=os.path.join(args.outdir, f"bhwork_{ix}_{iy}"))
                r["cell"] = cell
        except Exception as e:
            print(f"  cell ({ix},{iy}) FAILED: {e}")
            continue
        dep = r["depth"]
        section.append(r["vs_median"])
        along.append(cell.x_km if args.axis == "x" else cell.y_km)
        lonlat.append((cell.lon, cell.lat))
        results.append((ix, iy, r))
        mis = vi.data_misfit(r)
        print(f"  ({ix},{iy}) {along[-1]:.1f}km: n_layers "
              f"{np.mean(r.get('n_layers_post',[np.nan])):.1f} chi_fund {mis.get('fund',np.nan):.2f} "
              f"chi_ot {mis.get('overtone',np.nan):.2f}", flush=True)

    section = np.array(section).T               # (ndepth, ncell)
    along = np.array(along)
    np.savez_compressed(os.path.join(args.outdir, "profile_section.npz"),
                        depth=dep, along=along, vs_median=section,
                        cells=np.array([(ix, iy) for ix, iy, _ in results]),
                        lonlat=np.array(lonlat), axis=args.axis)

    # ---- section figure ----
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(along)), 6))
    o = np.argsort(along)
    pc = ax.pcolormesh(along[o], dep, section[:, o], cmap="RdYlBu", shading="auto",
                       vmin=np.nanpercentile(section, 5), vmax=np.nanpercentile(section, 95))
    plt.colorbar(pc, ax=ax, label="posterior-median Vs [km/s]")
    ax.plot(along[o], np.zeros_like(along[o]), "kv", ms=6)
    ax.invert_yaxis()
    ax.set(xlabel=along_label, ylabel="depth [km]",
           title=f"{os.path.basename(args.config).split('_')[0]} {args.axis}-profile "
                 f"({args.engine}, overtone T>={args.overtone_min_t}s) — median Vs section")
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "profile_section.png"), dpi=140)

    # ---- per-cell Vs(z) small multiples ----
    n = len(results)
    if n:
        nc = min(6, n); nr = int(np.ceil(n / nc))
        f2, axs = plt.subplots(nr, nc, figsize=(2.5 * nc, 3.0 * nr), squeeze=False, sharey=True)
        for a, (ix, iy, r) in zip(axs.ravel(), results):
            if "vs_p16" in r:
                a.fill_betweenx(dep, r["vs_p16"], r["vs_p84"], color="0.8")
            a.plot(r["vs_median"], dep, "b-")
            a.invert_yaxis(); a.set_title(f"({ix},{iy})", fontsize=8); a.set_xlim(args.vs_min, args.vs_max)
        for a in axs.ravel()[n:]:
            a.axis("off")
        f2.suptitle(f"{args.axis}-profile per-cell Vs(z)", y=1.0)
        f2.tight_layout(); f2.savefig(os.path.join(args.outdir, "profile_cells.png"), dpi=120)
    print(f"\nwrote profile_section.png + profile_cells.png ({len(results)} cells) under {args.outdir}")


if __name__ == "__main__":
    main()
