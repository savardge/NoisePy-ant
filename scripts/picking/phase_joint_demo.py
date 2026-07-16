"""STEP-2 proof: does adding phase velocity to the group inversion improve DEPTH resolution?

For a few well-covered Aargau cells, invert twice with BayHunter:
  (A) GROUP-only  : far-field group curves (fund+overtone) -- the current step-1 model;
  (B) GROUP+PHASE : the same group curves PLUS the phase-velocity curves (fund+overtone) from
                    the step-2 phase tomography, as a joint surf96 group+phase target.
Everything else identical. Then compare posterior Vs(z): median + 68% band, and the 68%
half-width vs depth. The claim to test: phase (longer usable periods, valid to ~1 lambda)
shrinks the deep (3-6 km) posterior where group-only is unconstrained.

Cells are auto-selected as those well-covered in BOTH tomographies whose PHASE curve reaches a
longer period than the far-field GROUP curve (i.e. where phase can actually add depth).

Run (BayHunter env):
  PYTHONPATH=~/Codes/NoisePy-ant /opt/anaconda3/envs/bayesbay_dev/bin/python phase_joint_demo.py \
      --net aargau --ncells 4
Outputs: Projects/<net>/tomo/phase/joint_demo/{cell_<ix>_<iy>_{group,joint}.npz, compare.png}
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from noisepy import vs_inversion as vi
from noisepy import period_resolution as pr

PROJ = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
GROUP_PROD = {"aargau": "farfield2p5/swtomotv-output/production",
              "riehen": "farfield2p5/swtomotv-output/production"}
PHASE_PROD = {"aargau": "phase/swtomotv-output/production",
              "riehen": "phase/swtomotv-output/production"}
YAML = {"aargau": "farfield2p5/aargau_swtomotv_ff2p5.yaml",
        "riehen": "farfield2p5/riehen_swtomotv_ff2p5.yaml"}
BH_PY = "/opt/anaconda3/envs/bayhunter/bin/python"
RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_bayhunter_cell.py")
CFG_COMMON = dict(depth_max=6.0, vs_bounds=[0.3, 3.6], n_layers=[1, 20],
                  maxfrac=vi.MAX_ADJ_FRAC, nchains=4, iter_burnin=25000, iter_main=12000,
                  maxmodels=20000, pred_nsub=40, skip_pred=False, _timeout=1800.0)


def _write_curve(path, TUS):
    T, U, S = TUS
    np.savetxt(path, np.column_stack([T, U, S]), fmt="%.6f")


def run_cell(net, ix, iy, lon, lat, gcurves, pcurves, outdir):
    """Run group-only then group+phase for one cell; return (group_npz, joint_npz)."""
    wd = os.path.join(outdir, f"work_{ix}_{iy}")
    os.makedirs(wd, exist_ok=True)
    gfiles = {}
    for w, tus in gcurves.items():
        p = os.path.join(wd, f"g_{w}.txt"); _write_curve(p, tus); gfiles[w] = p
    pfiles = {}
    for w, tus in pcurves.items():
        p = os.path.join(wd, f"p_{w}.txt"); _write_curve(p, tus); pfiles[w] = p
    env = dict(os.environ, OBJC_DISABLE_INITIALIZE_FORK_SAFETY="YES",
               VECLIB_MAXIMUM_THREADS="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    outs = {}
    for tag, cfg_extra in (("group", {}),
                           ("joint", {"curves_phase": pfiles})):
        out_npz = os.path.join(outdir, f"cell_{ix}_{iy}_{tag}.npz")
        cfg = dict(curves=gfiles, out_npz=out_npz,
                   savepath=os.path.join(wd, f"bh_{tag}"),
                   cell=[ix, iy, lon, lat], measure="group", **cfg_extra, **CFG_COMMON)
        cfgp = os.path.join(wd, f"cfg_{tag}.json")
        json.dump(cfg, open(cfgp, "w"))
        print(f"  cell ({ix},{iy}) {tag} ...", flush=True)
        r = subprocess.run([BH_PY, RUNNER, cfgp], env=env, capture_output=True, text=True,
                           timeout=CFG_COMMON["_timeout"] + 120)
        if not os.path.exists(out_npz):
            print(r.stderr[-1500:]); return None
        outs[tag] = out_npz
    return outs["group"], outs["joint"]


def pick_cells(net, ncells):
    """Cells well-covered in both tomographies where phase reaches a longer period than group."""
    gprod = os.path.join(PROJ, net, "tomo", GROUP_PROD[net])
    pprod = os.path.join(PROJ, net, "tomo", PHASE_PROD[net])
    import yaml as _yaml
    from swtomotv.geometry import make_grid, xy2ll
    cfg = _yaml.safe_load(open(os.path.join(PROJ, net, "tomo", YAML[net])))
    grid = make_grid(tuple(cfg["bounds"]), float(cfg["dx_km"]))
    # scan a coarse set of interior cells; keep those with good joint coverage + phase reach
    nx, ny = len(grid.x), len(grid.y)
    cand = []
    for ix in range(2, nx - 2, 2):
        for iy in range(2, ny - 2, 2):
            g = vi.load_cell_curves(gprod, ix, iy, min_periods=6)
            p = vi.load_cell_curves(pprod, ix, iy, min_periods=5)
            if not (g.has("fund") and p.has("fund")):
                continue
            gTmax, pTmax = g.curves["fund"][0].max(), p.curves["fund"][0].max()
            if pTmax <= gTmax + 0.3:                      # phase must extend beyond group
                continue
            x, y = grid.x[ix] + grid.dx / 2, grid.y[iy] + grid.dx / 2
            lat, lon = xy2ll(x, y, *grid.origin)
            cand.append((pTmax - gTmax, ix, iy, float(lon), float(lat),
                         len(g.curves["fund"][0]), len(p.curves["fund"][0])))
    cand.sort(reverse=True)
    # spread the picks across the grid (avoid clustering): greedy min-distance
    picks = []
    for c in cand:
        if all((c[1] - q[1]) ** 2 + (c[2] - q[2]) ** 2 > 25 for q in picks):
            picks.append(c)
        if len(picks) >= ncells:
            break
    return gprod, pprod, picks


def trim(cell, net):
    return pr.trim_reliable(cell, net, "physical", {"alpha": 0.5, "R_frac": 0.5, "depth_max": 6.0})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", default="aargau", choices=("aargau", "riehen"))
    ap.add_argument("--ncells", type=int, default=4)
    args = ap.parse_args()
    pr.PROD_ROOT[args.net] = os.path.join(PROJ, args.net, "tomo", GROUP_PROD[args.net])
    gprod, pprod, picks = pick_cells(args.net, args.ncells)
    if not picks:
        raise SystemExit("no cells with phase reach beyond group; check phase production coverage")
    outdir = os.path.join(PROJ, args.net, "tomo", "phase", "joint_demo")
    os.makedirs(outdir, exist_ok=True)
    print(f"{len(picks)} demo cells:", [(p[1], p[2]) for p in picks], flush=True)
    results = []
    for dT, ix, iy, lon, lat, ng, npp in picks:
        g = trim(vi.load_cell_curves(gprod, ix, iy), args.net)
        p = vi.load_cell_curves(pprod, ix, iy)               # phase: keep full period reach
        gc = {w: g.curves[w] for w in ("fund", "overtone") if g.has(w)}
        pc = {w: p.curves[w] for w in ("fund", "overtone") if p.has(w)}
        if "fund" not in gc or "fund" not in pc:
            continue
        out = run_cell(args.net, ix, iy, lon, lat, gc, pc, outdir)
        if out:
            results.append((ix, iy, out[0], out[1], gc, pc))
    if results:
        make_figure(args.net, results, outdir)


def make_figure(net, results, outdir):
    n = len(results)
    fig, axs = plt.subplots(2, n, figsize=(4.2 * n, 9), squeeze=False)
    for k, (ix, iy, gnpz, jnpz, gc, pc) in enumerate(results):
        G, J = np.load(gnpz), np.load(jnpz)
        z = G["depth"]
        aV = axs[0, k]
        for R, col, lab in ((G, "tab:red", "group only"), (J, "tab:blue", "group+phase")):
            aV.plot(R["vs_median"], z, color=col, lw=2, label=lab)
            aV.fill_betweenx(z, R["vs_p16"], R["vs_p84"], color=col, alpha=0.18)
        aV.invert_yaxis(); aV.set_title(f"cell ({ix},{iy})", fontsize=10)
        aV.set_xlabel("Vs [km/s]");  aV.set_ylabel("z [km]") if k == 0 else None
        aV.legend(fontsize=8)
        # phase reach annotation
        gTmax = max(gc[w][0].max() for w in gc); pTmax = max(pc[w][0].max() for w in pc)
        aV.axhline(0.5 * 1.1 * float(pc["fund"][1].mean()) * pTmax, color="tab:blue",
                   ls=":", lw=0.8)
        aV.text(0.03, 0.02, f"group Tmax {gTmax:.1f}s\nphase Tmax {pTmax:.1f}s",
                transform=aV.transAxes, fontsize=7, va="bottom")
        aW = axs[1, k]
        aW.plot(0.5 * (G["vs_p84"] - G["vs_p16"]), z, color="tab:red", lw=2, label="group only")
        aW.plot(0.5 * (J["vs_p84"] - J["vs_p16"]), z, color="tab:blue", lw=2, label="group+phase")
        aW.invert_yaxis(); aW.set_xlabel("68% half-width [km/s]")
        aW.set_ylabel("z [km]") if k == 0 else None
        aW.axhspan(3.0, 6.0, color="gray", alpha=0.1); aW.legend(fontsize=8)
    fig.suptitle(f"{net.capitalize()} — joint group+phase vs group-only Vs (step-2 depth-"
                 f"resolution test; gray band = deep zone phase should tighten)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(outdir, "compare.png")
    fig.savefig(out, dpi=145); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    main()
