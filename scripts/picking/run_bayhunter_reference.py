"""1D Vs(z) BayHunter inversion of a network's PHASE-velocity reference dispersion curves
(the array-wide, mode-separated fundamental/overtone picks in Projects/{net}/vsg_modesep/
ref_fundamental_phase.txt / ref_overtone_phase.txt), not a per-cell tomography curve.

Unlike run_vs_inversion.py (which inverts per-cell GROUP-velocity curves from swtomotv
production maps), this inverts the whole-network PHASE-velocity reference with BayHunter only,
sweeping two axes as a sensitivity/robustness check:
  * waves:      fundamental only  vs  fundamental + 1st overtone jointly
  * constraint: unconstrained     vs  <=50% adjacent-layer contrast (LVZ+HVZ allowed either way)
-> 4 runs per network.

Usage (bayhunter env has BayHunter+disba; PYTHONPATH must reach noisepy):
  PYTHONPATH=~/Codes/Noisepy-ant /opt/anaconda3/envs/bayhunter/bin/python \
    run_bayhunter_reference.py --net riehen,aargau --outroot <out>
"""
import argparse
import os

import numpy as np
from noisepy import vs_inversion as vi

REF_DIR = {
    "riehen": os.path.expanduser("~/Codes/extract_higher_modes/Projects/riehen/vsg_modesep"),
    "aargau": os.path.expanduser("~/Codes/extract_higher_modes/Projects/aargau/vsg_modesep"),
}
WAVESETS = {"fund": ("fund",), "fundot": ("fund", "overtone")}
CONSTRAINTS = {"unconstrained": None, "maxfrac50": 0.5}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", default="riehen,aargau")
    ap.add_argument("--outroot", required=True)
    ap.add_argument("--depth-max", type=float, default=6.0)
    ap.add_argument("--vs-min", type=float, default=0.3)
    ap.add_argument("--vs-max", type=float, default=3.6)
    ap.add_argument("--n-chains", type=int, default=10)
    ap.add_argument("--iter-burnin", type=int, default=120_000)
    ap.add_argument("--iter-main", type=int, default=60_000)
    ap.add_argument("--runner", default=os.path.join(os.path.dirname(__file__),
                                                      "run_bayhunter_cell.py"))
    ap.add_argument("--bayhunter-python", default=None,
                    help="only needed if this driver itself is run from another env")
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    nets = [n.strip() for n in args.net.split(",") if n.strip()]

    for net in nets:
        refdir = REF_DIR[net]
        paths = {"fund": os.path.join(refdir, "ref_fundamental_phase.txt"),
                 "overtone": os.path.join(refdir, "ref_overtone_phase.txt")}
        cell = vi.load_reference_curves(paths, label=net)
        cell = vi.decimate_periods(cell, max_periods=55)   # surf96 caps observed data at 60 pts
        print(f"\n=== {net} reference phase-velocity curves (decimated to <=55 pts/wave) ===")
        for w in cell.curves:
            T, U, S = cell.curves[w]
            print(f"  {w}: {len(T)} periods  T {T.min():.2f}-{T.max():.2f}s  "
                  f"c {U.min():.2f}-{U.max():.2f} km/s")

        results = {}
        for wave_key, waves in WAVESETS.items():
            for con_key, maxfrac in CONSTRAINTS.items():
                tag = f"{wave_key}_{con_key}"
                outdir = os.path.join(args.outroot, net, tag)
                os.makedirs(outdir, exist_ok=True)
                npz = os.path.join(outdir, "bayhunter_result.npz")
                if args.reuse and os.path.exists(npz):
                    print(f"\n--- {net}/{tag} (reused) ---")
                    r = vi.load_result(npz); r["cell"] = cell
                else:
                    print(f"\n--- {net}/{tag}: waves={waves} maxfrac={maxfrac} ---")
                    r = vi.run_bayhunter(
                        cell, npz, args.runner, args.bayhunter_python or "python",
                        waves=waves, depth_max=args.depth_max,
                        vs_bounds=(args.vs_min, args.vs_max), maxfrac=maxfrac,
                        nchains=args.n_chains, iter_burnin=args.iter_burnin,
                        iter_main=args.iter_main, measure="phase",
                        workdir=os.path.join(outdir, "work"))
                    r["cell"] = cell
                vi.plot_inversion(r, os.path.join(outdir, "bayhunter_inversion.png"),
                                  title=f"{net} reference phase — {tag}", measure="phase")
                print(f"  {r.get('n_models')} models, {r.get('runtime_s', float('nan')):.0f}s, "
                      f"misfit {vi.data_misfit(r)}")
                results[tag] = r

        _compare_sweep(net, results, os.path.join(args.outroot, net, "sweep_comparison.png"))
    print("\ndone.")


def _compare_sweep(net, results, path):
    """4-panel comparison: Vs(z) for all 4 configs overlaid + a misfit table."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"fund_unconstrained": "C0", "fund_maxfrac50": "C1",
              "fundot_unconstrained": "C2", "fundot_maxfrac50": "C3"}
    fig, axs = plt.subplots(1, 2, figsize=(11, 6), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axs[0]
    for tag, r in results.items():
        c = colors.get(tag, "k")
        ax.fill_betweenx(r["depth"], r["vs_p16"], r["vs_p84"], color=c, alpha=0.15)
        ax.plot(r["vs_median"], r["depth"], color=c, lw=2, label=tag)
    ax.invert_yaxis()
    ax.set(xlabel="Vs [km/s]", ylabel="depth [km]", title=f"{net}: BayHunter Vs(z), phase-velocity ref")
    ax.legend(fontsize=8)
    ax = axs[1]; ax.axis("off")
    rows = [["config", "n_layers", "chi(fund)", "chi(ot)"]]
    for tag, r in results.items():
        mis = vi.data_misfit(r)
        nl = r.get("n_layers_post", np.array([np.nan]))
        rows.append([tag, f"{np.mean(nl):.1f}±{np.std(nl):.1f}",
                     f"{mis.get('fund', np.nan):.2f}", f"{mis.get('overtone', np.nan):.2f}"])
    tb = ax.table(cellText=rows, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.6)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


if __name__ == "__main__":
    main()
