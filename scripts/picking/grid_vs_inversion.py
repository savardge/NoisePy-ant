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
    """Write curves+config for one (cell, waveset) and run the BayHunter subprocess."""
    cell, waves, out_npz, workdir, bh_py, runner, cfg_common = task
    if os.path.exists(out_npz):
        return (cell.ix, cell.iy, waves, "skip", 0.0)
    os.makedirs(workdir, exist_ok=True)
    curvefiles = {}
    for w in waves:
        if not cell.has(w):
            continue
        T, U, S = cell.curves[w]
        fp = os.path.join(workdir, f"disp_{w}.txt")
        np.savetxt(fp, np.column_stack([T, U, S]), fmt="%.6f")
        curvefiles[w] = fp
    if not curvefiles:                       # love-only cells have no "fund"; only skip if NOTHING
        return (cell.ix, cell.iy, waves, "no-data", 0.0)
    cfg = dict(curves=curvefiles, out_npz=out_npz,
               savepath=os.path.join(workdir, "bh_results"),
               cell=[cell.ix, cell.iy, cell.lon, cell.lat], **cfg_common)
    cfgpath = os.path.join(workdir, "config.json")
    with open(cfgpath, "w") as f:
        json.dump(cfg, f)
    env = dict(os.environ, OBJC_DISABLE_INITIALIZE_FORK_SAFETY="YES",
               VECLIB_MAXIMUM_THREADS="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0 = time.time()
    try:
        subprocess.run([bh_py, runner, cfgpath], check=True, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=cfg_common.get("_timeout"))
        status = "ok" if os.path.exists(out_npz) else "no-out"
    except subprocess.TimeoutExpired:
        status = "timeout"
    except subprocess.CalledProcessError:
        status = "err"
    # the per-cell work dir holds ~90 MB of raw BayHunter chain storage; the result npz has
    # everything downstream needs, so drop the work dir to keep the grid's disk footprint flat.
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
    ap.add_argument("--net", default=None, help="riehen|aargau; required for --criterion")
    ap.add_argument("--criterion", default="none",
                    choices=("none", "tomographic", "physical", "combined"),
                    help="per-cell period-reliability trim (period_resolution.trim_reliable)")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--rfrac", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="only first N cells (debug)")
    ap.add_argument("--bayhunter-python", required=True)
    ap.add_argument("--bayhunter-runner", required=True)
    ap.add_argument("--shard", default=None,
                    help="'i/N': process only task slice tasks[i::N] (SLURM array fan-out). "
                         "All shards write to the shared cells/ dir; run --assemble-only after.")
    ap.add_argument("--no-assemble", action="store_true",
                    help="skip stitching cells/ into volume_*.npz (use on array workers)")
    ap.add_argument("--assemble-only", action="store_true",
                    help="skip inversion; only stitch existing cells/*.npz into volume_*.npz")
    args = ap.parse_args()
    celldir = os.path.join(args.outdir, "cells")
    workroot = os.path.join(args.outdir, "work")
    os.makedirs(celldir, exist_ok=True)

    want = [w.strip() for w in args.wavesets.split(",") if w.strip()]
    if args.assemble_only:
        assemble_volume(celldir, want, args.outdir)
        return

    wavesets = {"fund": ("fund",), "fundot": ("fund", "overtone"), "love": ("love",),
                "fundlove": ("fund", "love"), "fundotlove": ("fund", "overtone", "love")}
    need_love = any("love" in wavesets[w] for w in want)
    if need_love and not args.love_production:
        raise SystemExit("--love-production is required for love/fundlove/fundotlove wavesets")

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
    if args.limit:
        cells_ij = cells_ij[: args.limit]
    print(f"{os.path.basename(args.config)}: {len(cells_ij)} cells x {len(want)} wavesets "
          f"= {len(cells_ij)*len(want)} runs, {args.n_workers} workers", flush=True)

    cfg_common = dict(depth_max=args.depth_max, vs_bounds=[args.vs_min, args.vs_max],
                      n_layers=[1, 20], maxfrac=vi.MAX_ADJ_FRAC, nchains=args.n_chains,
                      iter_burnin=args.iter_burnin, iter_main=args.iter_main,
                      maxmodels=args.maxmodels, pred_nsub=args.pred_nsub, skip_pred=False,
                      _timeout=args.cell_timeout)

    # build tasks; overtone restricted to T>=overtone-min-t for the fundot waveset, then the
    # per-cell reliability criterion trims each curve to its well-resolved periods.
    trim_params = {"alpha": args.alpha, "R_frac": args.rfrac, "depth_max": args.depth_max}
    if args.criterion != "none" and not args.net:
        raise SystemExit("--criterion needs --net (riehen|aargau)")
    # point the reliability metrics at THIS run's production maps (may be a finer-dx grid);
    # station xstat/ystat are dx-independent so the default cache is reused.
    pr.PROD_ROOT[args.net] = args.production
    load_waves = ("fund", "overtone", "love") if need_love else ("fund", "overtone")
    wave_roots = {"love": args.love_production} if need_love else None
    if need_love:
        pr.PROD_ROOT[(args.net, "love")] = args.love_production
    tasks = []
    for (ix, iy) in cells_ij:
        base = vi.load_cell_curves(args.production, ix, iy, waves=load_waves,
                                   wave_roots=wave_roots)
        vi.attach_cell_coords(base, args.config)
        for wskey in want:
            cell = base
            if "overtone" in wavesets[wskey] and args.overtone_min_t:
                cell = vi.restrict_periods(base, {"overtone": (args.overtone_min_t, None)})
            if args.criterion != "none":
                cell = pr.trim_reliable(cell, args.net, args.criterion, trim_params)
            out_npz = os.path.join(celldir, f"cell_{ix}_{iy}_{wskey}.npz")
            wd = os.path.join(workroot, f"{ix}_{iy}_{wskey}")
            tasks.append((cell, wavesets[wskey], out_npz, wd,
                          args.bayhunter_python, args.bayhunter_runner, cfg_common))

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
        for f in files:
            try:
                r = vi.load_result(f)
            except Exception:
                continue
            dep = r["depth"]
            ix, iy = (int(v) for v in r.get("cell_ixiy", (-1, -1)))
            lon, lat = (float(v) for v in r.get("cell_lonlat", (np.nan, np.nan)))
            ij.append((ix, iy)); lonlat.append((lon, lat))
            med.append(r["vs_median"]); p16.append(r["vs_p16"]); p84.append(r["vs_p84"])
            p025.append(r["vs_p025"]); p975.append(r["vs_p975"])
            nlay.append(float(np.mean(r.get("n_layers_post", [np.nan]))))
            mis = vi.data_misfit(r)
            chi_f.append(mis.get("fund", np.nan)); chi_o.append(mis.get("overtone", np.nan))
            chi_l.append(mis.get("love", np.nan))
        out = os.path.join(outdir, f"volume_{wskey}.npz")
        np.savez_compressed(out, depth=dep, cells=np.array(ij), lonlat=np.array(lonlat),
                            vs_median=np.array(med), vs_p16=np.array(p16), vs_p84=np.array(p84),
                            vs_p025=np.array(p025), vs_p975=np.array(p975),
                            chi_fund=np.array(chi_f), chi_overtone=np.array(chi_o),
                            chi_love=np.array(chi_l),
                            n_layers=np.array(nlay), waveset=wskey)
        print(f"wrote {out}: {len(ij)} cells", flush=True)


if __name__ == "__main__":
    main()
