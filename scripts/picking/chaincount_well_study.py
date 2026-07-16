"""Effect of the number of BayHunter Markov chains on the well-adjacent Vs inversion.

Riehen & Aargau: for the tomography cell closest to each deep borehole, invert the
fundamental + 1st-overtone Rayleigh GROUP-velocity curves with BayHunter, using

  * the PHYSICAL period-trimming criterion (period_resolution, alpha=0.5)   -- fixed
  * the +/-50% LVZ/HVZ adjacent-layer contrast (initparams lvz=hvz=0.5)      -- fixed
  * everything else identical (fund+overtone, iter_burnin/iter_main, Vs bounds, 20 layers)

and sweep ONLY the number of chains: 4 / 8 / 16 / 24.  This isolates the one variable
that governs the outlier-chain removal (a chain is dropped when its post-burnin median
log-likelihood is more than DELTA_LOGL below the best chain's -- the direct analog of
invert_section.py's 2x-best-chi2 filter used in the HVC CHAINCOUNT_effect.png; note this
is an ABSOLUTE log-likelihood difference, having replaced BayHunter's scale-dependent
relative dev=0.05).  The group-only inversion of shallow structure is non-unique, so
independent chains settle in different local modes; more chains = more independent hits on
the good-mode basin => a more robust, stable posterior.

Two phases:
  run  -- invert every (well, nchains) with a UNIQUE out_npz + savepath so each run's raw
          per-chain files persist (needed to show which chains were kept/dropped).
  plot -- one figure per net, one row per well, three columns mirroring CHAINCOUNT_effect.png:
            1. per-chain post-burnin median log-like at each chain count -- green = kept by
               the absolute Delta-logL outlier filter, red = dropped; label = kept/total;
            2. filtered posterior Vs(z) median at 4/8/16/24 chains + 24-chain 16-84% band
               + the well log -- shows the profile STABILISES once enough chains find the
               good mode;
            3. scaling -- kept chains & fraction kept vs total chains.

Run in the bayesbay_dev env (shells out to the bayhunter env for each inversion):
  PYTHONPATH=~/Codes/NoisePy-ant /opt/anaconda3/envs/bayesbay_dev/bin/python \
    chaincount_well_study.py --net riehen --wells Basel-1,Otterbach-2 \
      --bayhunter-python /opt/anaconda3/envs/bayhunter/bin/python \
      --bayhunter-runner run_bayhunter_cell.py
  # ... then --phase plot  (reads whatever runs are on disk)
"""
import argparse
import json
import os
import subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from noisepy import vs_inversion as vi
from noisepy import period_resolution as pr
from noisepy.vs_reliability import DELTA_LOGL

import well_vs_qc as wq   # WELLS, overlay_curves, stations_hull

PROJ = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
NCS = [4, 8, 16, 24]
# "more chains = darker blue"; avoids the well-overlay colors (green/orange/purple/brown)
CCOL = {4: "#9ecae1", 8: "#4292c6", 16: "#08519c", 24: "black"}
WAVES = ("fund", "overtone")
CRIT = "physical"
# Outlier-chain threshold: ABSOLUTE Delta-logL from the best chain (scale-free), shared with
# run_bayhunter_cell._use_abs_outlier_cut and vs_reliability.assess so this sweep, the
# posterior, and the reliability flags all describe the same set of chains. Replaces
# BayHunter's relative dev=0.05, whose tolerance is dev*|best| and so drifts with the
# likelihood scale -- fatal for a study whose whole purpose is comparing chain counts.


# ----------------------------------------------------------------------------- cell selection
def find_cell(net, lon, lat):
    """Nearest inverted tomography cell (ix,iy) + distance km to (lon,lat)."""
    vol = np.load(f"{PROJ}/{net}/tomo/vs_inversion/grid/volume_fundot.npz")
    cells, ll = vol["cells"], vol["lonlat"]
    d = np.hypot((ll[:, 0] - lon) * np.cos(np.deg2rad(lat)), ll[:, 1] - lat) * 111.0
    j = int(np.argmin(d))
    return int(cells[j, 0]), int(cells[j, 1]), float(d[j])


# ----------------------------------------------------------------------------- run one (well, nc)
def run_one(net, well, ix, iy, nc, outdir, args, waves=WAVES):
    name = well[0]
    out_npz = os.path.join(outdir, f"well_{name}_nc{nc:02d}_cell_{ix}_{iy}.npz")
    savepath = os.path.join(outdir, f"work_{name}_nc{nc:02d}", "bh_results")
    if os.path.exists(out_npz) and "chain_loglike_med" in np.load(out_npz).files:
        print(f"  [cache] {os.path.basename(out_npz)}", flush=True)
        return out_npz, savepath
    prod = f"{PROJ}/{net}/tomo/swtomotv-output/production"
    cfgp = f"{PROJ}/{net}/tomo/{net}_swtomotv.yaml"
    cell = vi.load_cell_curves(prod, ix, iy)
    if "overtone" in waves:
        cell = vi.restrict_periods(cell, {"overtone": (args.overtone_min_t, None)})
    vi.attach_cell_coords(cell, cfgp)
    cell = pr.trim_reliable(cell, net, CRIT,
                            {"alpha": args.alpha, "R_frac": 0.5, "depth_max": args.depth_max})
    workdir = os.path.dirname(savepath)
    os.makedirs(workdir, exist_ok=True)
    curves = {}
    for w in waves:
        if not cell.has(w):
            continue
        T, U, S = cell.curves[w]
        fp = os.path.join(workdir, f"disp_{w}.txt")
        np.savetxt(fp, np.column_stack([T, U, S]), fmt="%.6f")
        curves[w] = fp
    cfg = dict(curves=curves, out_npz=out_npz, savepath=savepath,
               cell=[cell.ix, cell.iy, cell.lon, cell.lat], depth_max=args.depth_max,
               vs_bounds=[args.vs_min, args.vs_max], n_layers=[1, 20], maxfrac=vi.MAX_ADJ_FRAC,
               nchains=nc, iter_burnin=args.iter_burnin, iter_main=args.iter_main,
               maxmodels=args.maxmodels, pred_nsub=args.pred_nsub, save_ensemble=True,
               diag_png=os.path.join(workdir, "diagnostics.png"))
    cfgpath = os.path.join(workdir, "config.json")
    json.dump(cfg, open(cfgpath, "w"))
    env = dict(os.environ, OBJC_DISABLE_INITIALIZE_FORK_SAFETY="YES", VECLIB_MAXIMUM_THREADS="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    print(f"  inverting {name} cell ({ix},{iy}) with {nc} chains ...", flush=True)
    subprocess.run([args.bayhunter_python, args.bayhunter_runner, cfgpath], check=True, env=env)
    return out_npz, savepath


# ----------------------------------------------------------------------------- per-chain readout
def chain_stats(r):
    """Per-chain post-burnin median log-like + median Vs(z) + kept-mask, from the result npz.

    Keeps chains within an ABSOLUTE Delta-logL of the best chain -- the same scale-free rule the
    runner builds the posterior with (run_bayhunter_cell._use_abs_outlier_cut) and that
    vs_reliability.assess reports. This used to mirror BayHunter's relative `dev` rule
    (|1 - like/best| <= 0.05), which is not scale-free: its tolerance in real log units is
    dev*|best|, so across runs of identical construction here (best spanning -34..+97) it ranged
    from 0.11 to 4.84 logL, and near best~0 it explodes. A likelihood RATIO is the meaningful
    quantity; a relative deviation is not. This study SWEEPS chain counts, so a rule whose
    strictness drifts with the likelihood scale would confound the very comparison it exists for.

    Uses the run's own `outlier_delta` where recorded, else vs_reliability.DELTA_LOGL.
    Returns (loglike_med[nc], kept[nc], vsprof[nc,ndep], depth[ndep])."""
    meds = np.asarray(r["chain_loglike_med"], float) if "chain_loglike_med" in r.files \
        else np.array([])
    vsprof = np.asarray(r["chain_vs_profiles"], float) if "chain_vs_profiles" in r.files \
        else np.empty((0, 0))
    dep = np.asarray(r["chain_vs_depth"], float) if "chain_vs_depth" in r.files else np.array([])
    if meds.size == 0:
        return meds, np.array([], bool), vsprof, dep
    delta = float(r["outlier_delta"]) if "outlier_delta" in r.files else float(DELTA_LOGL)
    best = np.nanmax(meds)
    kept = (best - meds) <= delta if np.isfinite(best) else np.ones(len(meds), bool)
    return meds, kept, vsprof, dep


# ----------------------------------------------------------------------------- figure
def plot_net(net, wells_cells, outdir, args):
    nrows = len(wells_cells)
    fig, axes = plt.subplots(nrows, 3, figsize=(16, 4.6 * nrows), constrained_layout=True,
                             squeeze=False)
    for row, (well, ix, iy, dist) in enumerate(wells_cells):
        name = well[0]
        overlay = wq.overlay_curves(net, name)
        per_nc = {}
        for nc in NCS:
            npz = os.path.join(outdir, f"well_{name}_nc{nc:02d}_cell_{ix}_{iy}.npz")
            if not os.path.exists(npz):
                continue
            r = np.load(npz, allow_pickle=True)
            if "chain_loglike_med" not in r.files:
                continue
            meds, kept, vsprof, cdep = chain_stats(r)
            per_nc[nc] = dict(r=r, meds=meds, kept=kept, vsprof=vsprof, cdep=cdep)

        # ---- col 1: per-chain median log-like, kept vs dropped ----
        ax = axes[row, 0]
        for nc in NCS:
            d = per_nc.get(nc)
            if d is None or d["meds"].size == 0:
                continue
            meds, kept = d["meds"], d["kept"]
            x = np.full(len(meds), nc, float) + np.linspace(-1.4, 1.4, len(meds))
            ax.scatter(x[kept], meds[kept], c="tab:green", s=40, zorder=3, edgecolor="k",
                       lw=0.4, label="kept" if nc == NCS[0] else None)
            ax.scatter(x[~kept], meds[~kept], c="tab:red", s=40, zorder=3, lw=1.0, marker="x",
                       label="dropped" if nc == NCS[0] else None)
            best = np.nanmax(meds)
            thr = best - DELTA_LOGL          # absolute Delta-logL boundary, same rule as the mask
            ax.hlines(thr, nc - 1.7, nc + 1.7, color=CCOL[nc], lw=1.3, ls="--")
            ax.text(nc, np.nanmax(meds), f"{int(kept.sum())}/{len(kept)}", ha="center",
                    va="bottom", fontsize=9, color=CCOL[nc])
        ax.set_xticks(NCS)
        ax.set_xlabel("number of chains")
        ax.set_ylabel("per-chain post-burnin median log-like")
        ax.set_title(f"{name}: chain fits (dashed = ΔlogL ≤ {DELTA_LOGL:g} outlier filter)")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.2)

        # ---- col 2: filtered posterior Vs(z) vs chain count ----
        ax = axes[row, 1]
        d24 = per_nc.get(24)
        if d24 is not None:
            r = d24["r"]
            ax.fill_betweenx(r["depth"], r["vs_p16"], r["vs_p84"], color=CCOL[24], alpha=0.15,
                             label="24-chain 16-84%")
        ref = None
        for nc in NCS:
            d = per_nc.get(nc)
            if d is None:
                continue
            r = d["r"]
            ax.plot(r["vs_median"], r["depth"], "-", color=CCOL[nc], lw=2,
                    label=f"{nc} chains (median)")
            if nc == 24:
                ref = (r["depth"], r["vs_median"])
        for v, zc, lab, col, ls in overlay:
            m = zc <= args.depth_max
            ax.plot(v[m], zc[m], color=col, ls=ls, lw=1.5, alpha=0.9, label=lab)
        if ref is not None:
            for nc in (4, 8, 16):
                d = per_nc.get(nc)
                if d is None:
                    continue
                drift = np.median(np.abs(np.interp(ref[0], d["r"]["depth"], d["r"]["vs_median"])
                                         - ref[1]))
                print(f"{net}/{name}: |median Vs({nc}ch)-Vs(24ch)| = {1000*drift:.0f} m/s "
                      f"(depth-median)")
        ax.set_xlim(args.vs_min, args.vs_max)
        ax.set_ylim(0, args.depth_max)
        ax.invert_yaxis()
        ax.set_xlabel("Vs [km/s]")
        ax.set_ylabel("depth [km]")
        ax.set_title(f"{name}: filtered posterior vs chain count ({dist:.2f} km from well)")
        ax.legend(fontsize=6.5, loc="lower left")
        ax.grid(alpha=0.2)

        # ---- col 3: scaling ----
        ax = axes[row, 2]
        ncs, n_kept, frac, disagree = [], [], [], []
        for nc in NCS:
            d = per_nc.get(nc)
            if d is None or d["meds"].size == 0:
                continue
            kept = d["kept"]
            ncs.append(nc)
            n_kept.append(int(kept.sum()))
            frac.append(float(kept.mean()))
            vk = d["vsprof"][kept] if kept.any() else d["vsprof"]
            disagree.append(float(np.nanmax(np.nanstd(vk, axis=0))) if len(vk) > 1 else np.nan)
        ax.plot(ncs, n_kept, "o-", color="tab:green", lw=1.8, label="kept (good-mode) chains")
        ax.plot(NCS, NCS, ":", color="0.6", lw=1.2, label="total chains")
        ax.set_xlabel("number of chains")
        ax.set_ylabel("chains")
        ax.set_xticks(NCS)
        ax.set_title(f"{name}: good-mode chains & inter-chain spread")
        axb = ax.twinx()
        axb.plot(ncs, disagree, "s--", color="tab:purple", lw=1.4, ms=5,
                 label="max inter-chain Vs std")
        axb.set_ylabel("max inter-chain Vs(z) std [km/s]", color="tab:purple")
        axb.tick_params(axis="y", labelcolor="tab:purple")
        axb.set_ylim(bottom=0)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.2)
        print(f"{net}/{name}: kept {dict(zip(ncs, n_kept))}, frac {[round(x,2) for x in frac]}, "
              f"chain_disagree {[round(x,3) for x in disagree]} km/s")

    fig.suptitle(
        f"{net.capitalize()} -- effect of BayHunter chain count on well-adjacent Vs inversion\n"
        f"physical period-trim  |  +/-50% LVZ/HVZ (lvz=hvz=0.5)  |  fund+overtone group  |  "
        f"outlier-chain filter: ΔlogL ≤ {DELTA_LOGL:g} from best chain\n"
        "more chains = more independent hits on the good-mode basin -> more robust, stable posterior",
        fontsize=12)
    out_png = os.path.join(outdir, f"CHAINCOUNT_wells_{net}.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"-> {out_png}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", required=True, choices=("riehen", "aargau"))
    ap.add_argument("--wells", default=None, help="comma list of well names; default = all that qualify")
    ap.add_argument("--phase", default="both", choices=("run", "plot", "both"))
    ap.add_argument("--waves", default="fundot", choices=("fundot", "fund"),
                    help="fundot = fund+overtone (default); fund = fundamental only "
                         "(use a distinct --outdir to avoid npz collisions)")
    ap.add_argument("--ncs", default=None,
                    help="comma list of chain counts to run (default: all of 4,8,16,24)")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--max-cell-dist", type=float, default=1.5)
    ap.add_argument("--overtone-min-t", type=float, default=1.0)
    ap.add_argument("--depth-max", type=float, default=6.0)
    ap.add_argument("--vs-min", type=float, default=0.3)
    ap.add_argument("--vs-max", type=float, default=3.6)
    ap.add_argument("--alpha", type=float, default=0.5)
    # each chain must be individually well-sampled, else the chain-count effect is
    # confounded with per-chain under-sampling (trans-D studies run O(1e5-1e6)/chain)
    ap.add_argument("--iter-burnin", type=int, default=200_000)
    ap.add_argument("--iter-main", type=int, default=100_000)
    ap.add_argument("--maxmodels", type=int, default=20_000)
    ap.add_argument("--pred-nsub", type=int, default=300)
    ap.add_argument("--bayhunter-python", default="/opt/anaconda3/envs/bayhunter/bin/python")
    ap.add_argument("--bayhunter-runner", default="run_bayhunter_cell.py")
    args = ap.parse_args()
    net = args.net
    suffix = "" if args.waves == "fundot" else f"_{args.waves}"
    outdir = args.outdir or f"{PROJ}/{net}/tomo/vs_inversion/wells/chaincount{suffix}"
    os.makedirs(outdir, exist_ok=True)

    hull = wq.stations_hull(net)
    want = None if not args.wells else set(w.strip() for w in args.wells.split(","))
    wells_cells = []
    for well in wq.WELLS[net]:
        name, lat, lon, welldep = well
        if want is not None and name not in want:
            continue
        if welldep < 1000 or not hull.contains_point((lon, lat)):
            continue
        ix, iy, dist = find_cell(net, lon, lat)
        if dist > args.max_cell_dist:
            print(f"skip {name}: nearest cell {dist:.2f} km > {args.max_cell_dist} km")
            continue
        wells_cells.append((well, ix, iy, dist))

    waves = {"fundot": ("fund", "overtone"), "fund": ("fund",)}[args.waves]
    ncs = NCS if not args.ncs else [int(x) for x in args.ncs.split(",")]
    if args.phase in ("run", "both"):
        for well, ix, iy, dist in wells_cells:
            for nc in ncs:
                run_one(net, well, ix, iy, nc, outdir, args, waves)
    if args.phase in ("plot", "both"):
        plot_net(net, wells_cells, outdir, args)
    print("done.")


if __name__ == "__main__":
    main()
