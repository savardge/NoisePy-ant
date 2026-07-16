"""Repeatable production group-velocity maps from V6 mode-separated picks, via swtomotv.

Drives swtomotv's VALIDATED numerical core (kernel.build_G + inversion.tv_two_step, the
MATLAB-parity-gated functions -- see swtomotv/AGENTS.md) with a transparent, fully logged
parameter policy, instead of the package's legacy-coupled coherence synthesis. Every choice
is documented in PARAMETERS.md and echoed to the per-period table below.

Parameter policy (see PARAMETERS.md for rationale):
  * grid, bounds, dx      : from the swtomotv dataset YAML (--config)
  * data covariance Cd    : blanket (rel_err * tau)^2, rel_err = 0.10 (fixed; user choice)
  * prior correlation LC  : FIXED per run (--lc; default max(1.0, 3*dx) km)
  * prior slowness std se : FIXED (--se, default 0.025 s/km). For these structure-dominated
                            datasets neither the L-curve nor the discrepancy principle yields an
                            interior optimum (chi2_red never reaches 1 even at least damping --
                            residuals are dominated by real 3-D structure the 2-D straight-ray
                            theory cannot fit, not by pick noise), so both auto-selectors rail.
                            se is therefore a documented analyst choice; the per-period chi2(se)
                            scan is saved (discrepancy_*.png) so the trade-off is auditable and
                            --se is easy to revise.
  * inversion             : one-step TV (no Liu-Yao reweighting), homogeneous 1/v_moy prior
  * display mask          : legacy ray-count coverage AND resolution-diagonal in the top
                            (1 - --res-drop-q) of covered cells (drop the worst-resolved
                            quantile -- self-scaling with damping/period; edge artifacts go)

Outputs, under {output_root}/production/{wave}/ :
  map_T{T}.npz           per-period: period, vel (nx,ny masked), mask, res_diag, unc_s, se, LC,
                         chi2_red, var_red, N, coverage
  production_{wave}.csv  per-period parameter + QC table (the audit trail)
  Lcurve_{wave}.png      the L-curve per period with the selected corner marked
  maps_{wave}.png        multi-period masked velocity-map panel

Usage:
  python run_production.py --config <swtomotv_yaml> --wave fund [--lc 1.5] [--res-min 0.05]
  (repeat per wave/network; see PARAMETERS.md for the exact reproduce commands)
"""
from __future__ import annotations
import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from swtomotv.config import DatasetConfig, MethodConfig
from swtomotv.geometry import make_grid, ll2xy
from swtomotv.data.picks import build_cache, load_cache, available_periods
from swtomotv.kernel import build_G
from swtomotv.inversion import cell_distance_matrix, cm_exp_inverse, tv_two_step

SE_GRID = (0.005, 0.007, 0.010, 0.014, 0.020, 0.028, 0.040, 0.055)  # s/km (sweep grid + one step)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="swtomotv dataset YAML")
    ap.add_argument("--wave", required=True, help="wave key in the YAML (fund / overtone)")
    ap.add_argument("--lc", type=float, default=None, help="prior correlation length km (default max(1,3*dx))")
    ap.add_argument("--lc-mode", default="fixed", choices=("fixed", "fresnel"),
                    help="fixed = one LC for all periods (legacy, --lc). fresnel = PHYSICS-derived "
                         "per-period LC(T) = clip(sqrt(lambda(T)*L/2), --lc, --lc-max): the 1st "
                         "Fresnel-zone half-width is the width the wave actually senses, so a "
                         "fixed LC lets the prior admit structure finer than the data can see "
                         "(Riehen: LC 1.5 km vs Fresnel 2.6->5.0 km over T 1->4.3 s). Unlike the "
                         "railed sigma_eff/L-curve sweep this needs no interior optimum.")
    ap.add_argument("--lc-max", type=float, default=8.0, help="cap on LC(T) for --lc-mode fresnel")
    ap.add_argument("--fresnel-path-km", type=float, default=None,
                    help="representative interstation path L for the Fresnel width (default: the "
                         "median path length of this period's rays)")
    ap.add_argument("--mask-mode", default="coverage", choices=("coverage", "res"),
                    help="coverage (legacy) = ray-hit cells AND res>=thresh; at fine dx + sparse "
                         "long-T rays this paints 1-cell-wide ray corridors that imply resolution "
                         "the wavelength cannot carry, while hiding neighbours the LC prior DOES "
                         "constrain. res = threshold on the resolution diagonal alone (it already "
                         "encodes the smoothing), which is self-consistent with the prior.")
    ap.add_argument("--se", type=float, default=0.025, help="prior slowness std sigma_eff s/km (fixed)")
    ap.add_argument("--res-drop-q", type=float, default=0.25,
                    help="hide this quantile of worst-resolved COVERED cells (default 0.25)")
    ap.add_argument("--rel-err", type=float, default=0.10, help="blanket relative travel-time error (Cd)")
    ap.add_argument("--fast", action="store_true",
                    help="skip the per-period sigma_eff audit scan + per-period figures (fixed lc/se); "
                         "much faster for fine grids")
    args = ap.parse_args()

    ds = DatasetConfig.from_yaml(args.config)
    method = MethodConfig(rel_err=args.rel_err)
    grid = make_grid(ds.bounds, ds.dx_km)
    LC = args.lc if args.lc is not None else max(1.0, 3 * ds.dx_km)
    outdir = ds.output_root / "production" / args.wave
    outdir.mkdir(parents=True, exist_ok=True)
    ds.ensure_dirs()

    # clean picks may have been re-exported; rebuild caches for this wave from the current CSV
    build_cache(ds, grid, args.wave, force=True)
    periods = available_periods(ds, args.wave)

    sta = pd.read_csv(ds.cache_dir / "stations_in_grid.csv")
    sx, sy = ll2xy(sta.latitude.values, sta.longitude.values, *grid.origin)
    xc, yc = grid.x + grid.dx / 2, grid.y + grid.dx / 2      # cell centers (legacy plotting)
    figdir = outdir / "figures"                              # one PNG per period
    figdir.mkdir(exist_ok=True)
    DIST = cell_distance_matrix(grid.x, grid.y)
    # inv(exp(-r/LC)) is O(ncell^3) and depends ONLY on LC (se just rescales it), so cache it per
    # distinct LC: --lc-mode fresnel needs one per period, but LC(T) is rounded to 0.25 km bins so
    # only a handful of distinct inversions are ever done.
    _CMI_CACHE = {}

    def cmi_for(lc):
        key = round(float(lc), 2)
        if key not in _CMI_CACHE:
            _CMI_CACHE[key] = cm_exp_inverse(DIST, key)
        return _CMI_CACHE[key]

    def lc_for(T, v_moy, dists):
        """LC(T): fixed, or the 1st Fresnel-zone half-width sqrt(lambda*L/2) (clipped)."""
        if args.lc_mode == "fixed":
            return LC
        L = args.fresnel_path_km or (float(np.median(dists)) if len(dists) else 8.0)
        lam = float(v_moy) * float(T)
        return float(np.clip(round(np.sqrt(lam * L / 2.0) / 0.25) * 0.25, LC, args.lc_max))

    rows, maps = [], []
    Lfig, Lax = plt.subplots(figsize=(7, 6))
    for T in periods:
        z = load_cache(ds, args.wave, T)
        tau, v_moy = z["tau"], float(z["v_moy"])
        N = len(tau)
        if N < 40:                          # too few rays for a meaningful map
            continue
        # rebuild kernels (pick set changed on re-export -> cached G is stale)
        G, mask, G_sum = build_G(ds, method, grid, args.wave, T, use_cache=False)
        Cd = (args.rel_err * tau) ** 2
        LC_T = lc_for(T, v_moy, z["dist"])       # period-dependent (fresnel) or fixed
        CMi = cmi_for(LC_T)
        # diagnostic scan: reduced chi-square vs se (audit trail; se itself is fixed = args.se).
        # --fast skips it (8 extra full inversions/period) since se is fixed anyway.
        chis = None if args.fast else \
            [float(np.sum(tv_two_step(G, tau, v_moy, CMi, se, two_step=False)[1]["misfit_post"] ** 2
                          / Cd) / N) for se in SE_GRID]
        se_star = args.se
        # production solve at the fixed se, with posterior + resolution
        m2, stats, extras = tv_two_step(G, tau, v_moy, CMi, se_star, want_post=True, two_step=False)
        V = 1.0 / grid.vec_to_map(m2)
        R = grid.vec_to_map(extras["res_diag"])
        U = grid.vec_to_map(extras["unc_s"])
        # display mask. coverage: ray-hit cells AND res>=quantile (legacy). res: res>=quantile of
        # ALL cells whose res_diag is finite -- keeps LC-correlated neighbours of rays that the
        # coverage test drops, and avoids the 1-cell ray-corridor artifact at fine dx / long T.
        covered = mask > 0
        rbase = covered if args.mask_mode == "coverage" else np.isfinite(R) & (R > 0)
        rthr = np.quantile(R[rbase], args.res_drop_q) if rbase.any() else 0.0
        show = rbase & (R >= rthr)
        Vm = np.where(show, V, np.nan)
        chi2_red = float(np.sum(stats["misfit_post"] ** 2 / (args.rel_err * tau) ** 2) / N)
        cov = int(np.sum(show))
        np.savez_compressed(outdir / f"map_T{T:.1f}.npz",
                            period=T, vel=Vm, vel_full=np.where(covered, V, np.nan),
                            mask=show, res_diag=R, unc_s=U, res_thresh=rthr,
                            se=se_star, LC=LC_T, chi2_red=chi2_red,
                            var_red=stats["var_red"], N=N, coverage=cov)
        rows.append(dict(T=T, N=N, se_eff=se_star, LC=LC_T,
                         var_red=round(stats["var_red"], 3),
                         restit_post=round(stats["restit_post"], 2), chi2_red=round(chi2_red, 2),
                         cells_shown=cov, cells_covered=int((mask > 0).sum())))
        maps.append((T, Vm))
        if not args.fast:                      # per-period audit figure (skipped in --fast)
            f1, a1 = plt.subplots(figsize=(6.6, 6))
            vlo, vhi = (np.nanpercentile(Vm, [5, 95]) if np.isfinite(Vm).any() else (None, None))
            p1 = a1.pcolormesh(xc, yc, Vm.T, cmap="RdYlBu", vmin=vlo, vmax=vhi, shading="auto")
            plt.colorbar(p1, ax=a1, label="group velocity [km/s]", shrink=0.85)
            a1.plot(sx, sy, "^", ms=3.5, mfc="k", mec="w", mew=0.3, zorder=3)
            a1.set_aspect("equal")
            a1.set(xlabel="x [km]", ylabel="y [km]")
            a1.set_title(f"{ds.name}  {args.wave}  T={T:.1f} s\n"
                         f"N={N} rays  |  LC={LC_T:g} km  sigma_eff={se_star}\n"
                         f"var_red={stats['var_red']:.2f}  chi2={chi2_red:.1f}  "
                         f"shown {cov}/{int((mask>0).sum())}", fontsize=9)
            f1.tight_layout(); f1.savefig(figdir / f"map_T{T:.1f}.png", dpi=130); plt.close(f1)
            Lax.plot(SE_GRID, chis, "-", lw=0.5, color="0.85", zorder=1)
        print(f"{ds.name}/{args.wave} T={T:.1f} N={N:5d} se={se_star:.3f} "
              f"LC={LC_T:g} var_red={stats['var_red']:.2f} restit_post={stats['restit_post']:.1f}% "
              f"chi2_red={chi2_red:.1f} shown={cov}/{int((mask>0).sum())}")

    tab = pd.DataFrame(rows)
    tab.to_csv(outdir / f"production_{args.wave}.csv", index=False)
    Lax.axhline(1.0, color="r", ls="--", lw=1.2, zorder=2, label="discrepancy target chi2=1")
    Lax.axvline(args.se, color="b", lw=1.4, zorder=2, label=f"fixed sigma_eff={args.se}")
    Lax.set(xscale="log", yscale="log", xlabel="prior slowness std sigma_eff [s/km]",
            ylabel="reduced chi-square", title=f"{ds.name} {args.wave}: chi2(sigma_eff) per period "
            "(grey) -- rails below 1: se is a fixed analyst choice (blue)")
    Lax.legend(fontsize=8)
    Lfig.tight_layout(); Lfig.savefig(outdir / f"discrepancy_{args.wave}.png", dpi=130); plt.close(Lfig)

    # multi-period map panel
    if maps:
        n = len(maps); nc = min(5, n); nr = int(np.ceil(n / nc))
        fig, axs = plt.subplots(nr, nc, figsize=(3.4 * nc, 3.2 * nr), squeeze=False)
        for ax, (T, Vm) in zip(axs.ravel(), maps):
            vlo, vhi = np.nanpercentile(Vm, [5, 95])
            pc = ax.pcolormesh(xc, yc, Vm.T, cmap="RdYlBu", vmin=vlo, vmax=vhi, shading="auto")
            plt.colorbar(pc, ax=ax, shrink=0.8)
            ax.plot(sx, sy, "^", ms=1.5, mfc="k", mec="none")
            ax.set_aspect("equal"); ax.set_title(f"T={T:.1f}s", fontsize=9)
        for ax in axs.ravel()[len(maps):]:
            ax.axis("off")
        fig.suptitle(f"{ds.name} {args.wave}: production group-velocity maps "
                     f"({args.lc_mode}-LC, se={args.se}, coverage + drop-worst-"
                     f"{int(100*args.res_drop_q)}%-resolution mask)", y=1.0)
        fig.tight_layout()
        fig.savefig(outdir / f"maps_{args.wave}.png", dpi=120); plt.close(fig)
    print(f"\nwrote {len(rows)} period maps + production_{args.wave}.csv + figures under {outdir}")


if __name__ == "__main__":
    main()
