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


CMAP = "magma"        # perceptually uniform; for ABSOLUTE velocity (no meaningful midpoint)
ALPHA_HIDDEN = 0.28   # plausibility-flagged cells: shown, but visibly de-emphasised
CMAP_ANOM = "RdBu"    # diverging, centred on 0, for the per-map RELATIVE anomaly, where a
                      # midpoint DOES exist. Reversed so slow=blue, fast=red.


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="swtomotv dataset YAML")
    ap.add_argument("--wave", required=True, help="wave key in the YAML (fund / overtone)")
    ap.add_argument("--lc", type=float, default=None, help="prior correlation length km (default max(1,3*dx))")
    ap.add_argument("--lc-mode", default="fixed", choices=("fixed", "fresnel", "coverage"),
                    help="fixed = one LC for all periods (legacy, --lc). fresnel = PHYSICS-derived "
                         "per-period LC(T) = clip(sqrt(lambda(T)*L/2), --lc, --lc-max): the 1st "
                         "Fresnel-zone half-width is the width the wave actually senses, so a "
                         "fixed LC lets the prior admit structure finer than the data can see "
                         "(Riehen: LC 1.5 km vs Fresnel 2.6->5.0 km over T 1->4.3 s). Unlike the "
                         "railed sigma_eff/L-curve sweep this needs no interior optimum.")
    ap.add_argument("--lc-cov-minrays", type=float, default=0.0,
                    help="ABSOLUTE ray-density floor for --lc-mode coverage (0 = rely on "
                         "--lc-cov-frac alone)")
    ap.add_argument("--lc-cov-frac", type=float, default=0.5,
                    help="a cell counts as constrained when its ray density reaches this "
                         "FRACTION of the period's median non-zero density. Relative "
                         "because absolute counts differ ~20x between these networks.")
    ap.add_argument("--lc-cov-q", type=float, default=90.0,
                    help="percentile of the interior distance-to-nearest-ray used as LC in "
                         "--lc-mode coverage. 90 leaves ~10%% of interior cells further "
                         "than one correlation length from any ray.")
    ap.add_argument("--lc-cov-factor", type=float, default=1.0,
                    help="safety multiplier on the coverage-derived LC")
    ap.add_argument("--lc-scale", type=float, default=1.0,
                    help="multiplier on the Fresnel half-width in --lc-mode fresnel "
                         "(1.0 = the textbook sqrt(lambda*L/2)). Lower it to relax the "
                         "long-period smoothing; the short end is set by --lc anyway.")
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
    ap.add_argument("--vplaus", default="off",
                    help="plausibility mask from THIS period's pick distribution, hiding "
                         "cells the data cannot support. Modes:\n"
                         "  off        no mask (default)\n"
                         "  mad:K      keep median +- K*1.4826*MAD of the picks\n"
                         "  pct:P      keep the [P, 100-P] percentile band of the picks\n"
                         "  dens:F     keep velocities where the pick histogram count is\n"
                         "             >= F * its peak -- multimodality-safe, since every\n"
                         "             dense branch is kept and only sparse tails are cut\n"
                         "Motivation: the resolution mask misses these. Haute-Sorne's "
                         "eastern edge reaches 14.7 km/s at T=2.03 s with res_diag 0.021, "
                         "and clearing it via --res-drop-q costs 47%% of the map; a pick-"
                         "support cut removes exactly those cells. It also catches cells "
                         "that are WELL resolved but outside the data range, which "
                         "resolution cannot flag by construction.")
    ap.add_argument("--res-drop-q", type=float, default=0.25,
                    help="hide this quantile of worst-resolved COVERED cells (default 0.25)")
    ap.add_argument("--rel-err", type=float, default=0.10,
                    help="blanket relative travel-time error used when --cd-mode blanket")
    ap.add_argument("--cd-mode", choices=("blanket", "measured", "scaled"), default="blanket",
                    help="data covariance. blanket: Cd=(rel_err*tau)^2, every pick equal "
                         "(the legacy behaviour -- and until now the ONLY behaviour, since "
                         "inversion.tv_two_step read a module constant REL_ERR=0.10 and "
                         "--rel-err changed only the reported chi2). measured: per-datum "
                         "Cd=tau_std^2 from the substack-jackknife pick repeatability "
                         "(tau_std = dist*sigma_v/v^2, cached by data/picks.py), so good "
                         "picks outweigh noisy ones. scaled: the same per-datum shape, "
                         "multiplied by one constant per period so median chi2_red -> 1 -- "
                         "keeps the jackknife's quality RANKING but stops asserting an "
                         "absolute error budget the residuals contradict (substack "
                         "repeatability misses systematic error: off-great-circle paths, "
                         "3D effects, straight-ray approximation).")
    ap.add_argument("--cd-scale-iters", type=int, default=6,
                    help="fixed-point iterations for --cd-mode scaled. Scaling Cd by a "
                         "constant is NOT a no-op: the prior CM is fixed, so it shifts the "
                         "data-vs-prior balance and the model moves, hence iterate.")
    ap.add_argument("--no-figs", action="store_true",
                    help="skip the per-period map PNGs (the npz are always written)")
    ap.add_argument("--cd-floor", type=float, default=1e-3,
                    help="floor on tau_std [s] for --cd-mode measured; matches "
                         "products/uncertainty.py. Guards zero-repeatability picks from "
                         "acquiring infinite weight.")
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
    # Stale-cache guard. Cache and map file names embed the period via ds.tfmt, so changing
    # `period_decimals` renames everything. Leftovers from the previous convention are NOT
    # ignored: available_periods globs this directory, so it sees each period twice and then
    # asks for a cache that was never written under the new naming (T0.5 stale vs the real
    # rung T0.508 -> lookup of "T0.500" -> FileNotFoundError, mid-run). Fail fast instead.
    import re as _re
    _cd = ds.cache_dir
    if _cd.exists():
        _want = ds.period_decimals
        _bad = []
        for _f in _cd.glob("%s_T*_std*.npz" % args.wave):
            _m = _re.search(r"_T(\d+\.(\d+))_std", _f.name)
            if _m and len(_m.group(2)) != _want:
                _bad.append(_f.name)
        if _bad:
            raise SystemExit(
                "FATAL: %d cache files in %s use %d-decimal periods but this config sets "
                "period_decimals=%d (e.g. %s).\nDelete the cache directory and re-run; "
                "build_cache regenerates it.\nAlso remove any map_T*.npz/.png written under "
                "the old naming, or the output will mix both conventions."
                % (len(_bad), _cd, len(_re.search(r"_T\d+\.(\d+)_std", _bad[0]).group(1)),
                   _want, _bad[0]))
    build_cache(ds, grid, args.wave, force=True)
    periods = available_periods(ds, args.wave)

    sta = pd.read_csv(ds.cache_dir / "stations_in_grid.csv")
    sx, sy = ll2xy(sta.latitude.values, sta.longitude.values, *grid.origin)
    xc, yc = grid.x + grid.dx / 2, grid.y + grid.dx / 2      # cell centers (legacy plotting)
    # all figures go under the dataset-level figures/ root (ds.fig_dir, created by
    # ensure_dirs) in a per-wave subdir -- NOT under production/{wave}/ (which
    # previously left output_root/figures/ permanently empty).
    figdir = ds.fig_dir / args.wave
    figdir.mkdir(parents=True, exist_ok=True)
    # phase datasets carry the phase pick tables but reuse swtomotv's group_velocity
    # column, so the only honest source for the axis label is the dataset name.
    MEASURE_LABEL = "phase-velocity" if "phase" in ds.name.lower() else "group-velocity"
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

    def lc_coverage(gsum, T):
        """Smallest LC that still bridges the gaps in this period's ray coverage.

        With CM = sigma_eff^2 exp(-r/LC) the prior AMPLITUDE does not depend on LC (see
        swtomotv docs/01_math.md) -- LC only sets how far a ray's constraint spreads. Make
        it too small and interior cells that no ray touches stay at the homogeneous prior
        mean, producing a velocity pattern that is an artifact of the prior rather than of
        the data. Make it too large and real structure is smeared.

        So measure the gaps directly: distance transform from the ray-hit cells, evaluated
        over the FILLED coverage footprint (interior holes only -- the exterior is not
        something we claim to resolve), and take the --lc-cov-q percentile. LC at that value
        means all but (100-q)% of interior cells sit within one correlation length of a ray.
        """
        from scipy import ndimage
        # Use G_sum (the ray-density map), NOT `mask`: build_G's mask is already
        # thresholded to NaN/1.0 at method.min_density, so comparing it against a ray count
        # is always False and the function silently returns the floor.
        # A cell clipped by ONE ray is not constrained either -- at dx 0.5 km nearly every
        # interior cell is touched -- hence the --lc-cov-minrays threshold.
        g = np.asarray(gsum, float)
        g = np.where(np.isfinite(g), g, 0.0)
        # RELATIVE threshold. Absolute ray counts are not comparable: Haute-Sorne's median
        # non-zero G_sum is ~600-800 where Riehen's is 21-40 (170k paths vs 19.5k, and a
        # coarser grid), so any fixed number is unreachable for one network and trivial for
        # the other. Scale by this period's own median density instead.
        nz = g[g > 0]
        ref = float(np.median(nz)) if nz.size else 0.0
        thr = max(args.lc_cov_minrays, args.lc_cov_frac * ref)
        hit = g >= thr
        if not hit.any():
            return LC
        # Domain = convex hull of the ray-covered cells: the area the map CLAIMS. Using
        # fill_holes(hit) makes this self-referential -- the domain becomes the constrained
        # set, distances are ~0 by construction, and every threshold returns the floor.
        idx = np.argwhere(g > 0)
        domain = None
        if len(idx) >= 3:
            try:
                from scipy.spatial import ConvexHull, Delaunay
                hull = Delaunay(idx[ConvexHull(idx).vertices])
                ii, jj = np.meshgrid(np.arange(g.shape[0]), np.arange(g.shape[1]),
                                     indexing="ij")
                domain = (hull.find_simplex(np.column_stack([ii.ravel(), jj.ravel()])) >= 0
                          ).reshape(g.shape)
            except Exception:
                domain = None
        if domain is None or not domain.any():
            domain = ndimage.binary_fill_holes(hit)
        if not domain.any():
            return LC
        # distance (in cells) from every cell to the nearest ray-hit cell
        dist_cells = ndimage.distance_transform_edt(~hit)
        d_km = dist_cells[domain] * float(grid.dx)
        if not d_km.size:
            return LC
        need = float(np.percentile(d_km, args.lc_cov_q)) * args.lc_cov_factor
        lc = float(np.clip(round(need / 0.25) * 0.25, LC, args.lc_max))
        print("    [lc-cov] T=%-6g rays/cell>=%g on %5.1f%% of the interior | "
              "gap p50 %.2f p%g %.2f km -> LC %.2f"
              % (T, thr, 100.0 * hit[domain].mean(),
                 float(np.percentile(d_km, 50)), args.lc_cov_q, need, lc), flush=True)
        return lc

    def lc_for(T, v_moy, dists, G_sum=None):
        """LC(T): fixed, coverage-driven, or the 1st Fresnel half-width sqrt(lambda*L)/2.

        FRESNEL CONVENTION (corrected 2026-08-03 against the source papers). A point at
        perpendicular distance n from the midpoint of a path of length L has detour
        d = 2n^2/L, so n = sqrt(d*L/2). The first Fresnel zone is defined by the detour
        criterion, and the literature does NOT agree on it:

          Yoshizawa & Kennett 2002, GJI 149, 440, eq (24)   d = lam/2   n = sqrt(lam*L)/2
          Spetzler, Trampert & Snieder 2002, GJI 149, 755,
            eq (1), their n = 8/3                           d = 3lam/8  n = sqrt(3lam*L)/4

        This function used sqrt(lam*L/2), which is the d = lam (FULL wavelength) case --
        the second Fresnel zone, sqrt(2) larger than Yoshizawa & Kennett and 1.63x
        Spetzler -- while its docstring claimed it was the first-zone half-width. Every
        run before 2026-08-03 (_lccov, _lcinfl, _histfilt, _lc0.5, _lcfix1) carries that
        sqrt(2). Now uses the Yoshizawa & Kennett lam/2 convention.

        --lc-scale then multiplies that half-width. Yoshizawa & Kennett identify the
        INFLUENCE ZONE -- the region over which the wave stays coherent in phase, which is
        what a prior correlation length should track -- as one third of the WIDTH of the
        first Fresnel zone, i.e. a half-extent of sqrt(lam*L)/6. Against the corrected
        half-width that is --lc-scale 0.333.

        CAVEAT ON SCALE. Both papers are continental/global: 25-150 s over 2000-4000 km,
        L/lam of order 10-40, and the paraxial derivation assumes n << L. These arrays run
        L/lam = 1.7-8, and at Riehen T=4.8 s the half-width is 6.1 km against L=16 km
        (n/L = 0.38), so the paraxial assumption fails and the 1/3 rule is being
        extrapolated well outside where it was calibrated.
        """
        if args.lc_mode == "fixed":
            return LC
        if args.lc_mode == "coverage":
            return lc_coverage(G_sum, T)
        L = args.fresnel_path_km or (float(np.median(dists)) if len(dists) else 8.0)
        lam = float(v_moy) * float(T)
        # --lc-max still caps the long-period end: 39% of Haute-Sorne periods, 28% of
        # Aargau and 9% of Riehen sat exactly at 8 km under the old (sqrt(2) larger)
        # formula, which on an 18-41 km grid is a correlation length of order the array.
        raw = args.lc_scale * np.sqrt(lam * L) / 2.0
        return float(np.clip(round(raw / 0.25) * 0.25, LC, args.lc_max))

    rows, maps, pickv = [], [], []
    Lfig, Lax = plt.subplots(figsize=(7, 6))
    for T in periods:
        z = load_cache(ds, args.wave, T)
        tau, v_moy = z["tau"], float(z["v_moy"])
        N = len(tau)
        if N < 40:                          # too few rays for a meaningful map
            continue
        # rebuild kernels (pick set changed on re-export -> cached G is stale)
        G, mask, G_sum = build_G(ds, method, grid, args.wave, T, use_cache=False)
        if args.cd_mode in ("measured", "scaled"):
            ts = np.asarray(z["tau_std"], dtype=float)
            # a pick with no usable repeatability falls back to the blanket error rather
            # than to a fabricated tiny sigma
            bad = ~np.isfinite(ts) | (ts <= 0)
            ts = np.where(bad, args.rel_err * tau, np.maximum(ts, args.cd_floor))
            Cd = ts ** 2
            n_bad = int(bad.sum())
            cd_scale = 1.0
        else:
            Cd = (args.rel_err * tau) ** 2
            n_bad = 0
            cd_scale = 1.0
        LC_T = lc_for(T, v_moy, z["dist"], G_sum)  # fresnel / fixed / coverage-driven
        CMi = cmi_for(LC_T)
        if args.cd_mode == "scaled":
            # Fixed point, and it must run AFTER CMi exists: chi2 = sum(mis^2/Cd)/N, so
            # multiplying Cd by the current chi2 drives it toward 1. Scaling Cd by a constant
            # is not a no-op -- CM is fixed, so the data-vs-prior balance shifts and the model
            # moves -- hence iterate. want_post=False; the posterior is computed once below.
            for _ in range(max(1, args.cd_scale_iters)):
                _, _st, _ = tv_two_step(G, tau, v_moy, CMi, args.se, two_step=False, cd=Cd)
                _chi = float(np.sum(_st["misfit_post"] ** 2 / Cd) / N)
                if not np.isfinite(_chi) or _chi <= 0:
                    break
                Cd = Cd * _chi
                cd_scale *= _chi
                if abs(_chi - 1.0) < 0.02:
                    break
            cd_scale = float(np.sqrt(cd_scale))   # report as a sigma multiplier
        # diagnostic scan: reduced chi-square vs se (audit trail; se itself is fixed = args.se).
        # --fast skips it (8 extra full inversions/period) since se is fixed anyway.
        chis = None if args.fast else \
            [float(np.sum(tv_two_step(G, tau, v_moy, CMi, se, two_step=False)[1]["misfit_post"] ** 2
                          / Cd) / N) for se in SE_GRID]
        se_star = args.se
        # production solve at the fixed se, with posterior + resolution
        m2, stats, extras = tv_two_step(G, tau, v_moy, CMi, se_star, want_post=True,
                                        two_step=False,
                                        cd=Cd if args.cd_mode != "blanket" else None)
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
        n_vplaus = 0
        vhidden = np.full_like(V, np.nan)
        if args.vplaus and args.vplaus != "off":
            gv = np.asarray(z["gv"], float)
            gv = gv[np.isfinite(gv)]
            mode, _, par = args.vplaus.partition(":")
            lo_v, hi_v = -np.inf, np.inf
            if gv.size:
                if mode == "mad":
                    k = float(par or 3.0)
                    med = np.median(gv)
                    sig = 1.4826 * np.median(np.abs(gv - med))
                    lo_v, hi_v = med - k * sig, med + k * sig
                elif mode == "pct":
                    pq = float(par or 1.0)
                    lo_v, hi_v = np.percentile(gv, [pq, 100.0 - pq])
                elif mode == "dens":
                    fr = float(par or 0.02)
                    edges = np.arange(0.195, max(6.0, gv.max() + 0.1), 0.02)
                    h, _ = np.histogram(gv, bins=edges)
                    keepbin = h >= fr * h.max() if h.max() else np.zeros(len(h), bool)
                    if keepbin.any():
                        idx = np.where(keepbin)[0]
                        lo_v, hi_v = edges[idx[0]], edges[idx[-1] + 1]
                else:
                    raise SystemExit("--vplaus: unknown mode %r" % mode)
            drop = show & ~((V >= lo_v) & (V <= hi_v))
            n_vplaus = int(drop.sum())
            # Do NOT delete them: `show` narrows so the deliverable `vel` excludes them,
            # but the flagged cells are kept in `vel_hidden` and drawn semi-transparent, so
            # a reader can see what the mask removed instead of taking it on trust.
            vhidden = np.where(drop, V, np.nan)
            show = show & ~drop
            print("    [vplaus] T=%-6g %s -> keep %.2f-%.2f km/s | hides %d of %d shown"
                  % (T, args.vplaus, lo_v, hi_v, n_vplaus, int((rbase & (R >= rthr)).sum())),
                  flush=True)
        Vm = np.where(show, V, np.nan)
        chi2_red = float(np.sum(stats["misfit_post"] ** 2 / Cd) / N)
        cov = int(np.sum(show))
        np.savez_compressed(outdir / f"map_T{ds.tfmt(T)}.npz",
                            period=T, vel=Vm, vel_full=np.where(covered, V, np.nan),
                            mask=show, res_diag=R, unc_s=U, res_thresh=rthr,
                            se=se_star, LC=LC_T, chi2_red=chi2_red,
                            vplaus=args.vplaus, n_vplaus_hidden=n_vplaus,
                            vel_hidden=vhidden,
                            cd_mode=args.cd_mode, cd_n_fallback=n_bad,
                            cd_scale=cd_scale,
                            cd_median=float(np.median(np.sqrt(Cd))),
                            var_red=stats["var_red"], N=N, coverage=cov)
        rows.append(dict(T=T, N=N, se_eff=se_star, LC=LC_T,
                         var_red=round(stats["var_red"], 3),
                         restit_post=round(stats["restit_post"], 2), chi2_red=round(chi2_red, 2),
                         cells_shown=cov, cells_covered=int((mask > 0).sum())))
        maps.append((T, Vm, vhidden))
        # pick velocities for this period, from the same cache the inversion used -- so the
        # comparison figure below is guaranteed to show the picks that actually entered it
        pickv.append((T, np.asarray(z["gv"], float)))
        # Per-period map figure. This used to sit inside `if not args.fast:` together with
        # the sigma_eff scan line below -- but only the SCAN needs `chis`; the map does not.
        # Bundling them meant --fast (which every production run uses, since se is fixed)
        # silently produced no per-period maps at all. Now only the scan line is gated.
        if not args.no_figs:
            f1, a1 = plt.subplots(figsize=(6.6, 6))
            vlo, vhi = (np.nanpercentile(Vm, [5, 95]) if np.isfinite(Vm).any() else (None, None))
            p1 = a1.pcolormesh(xc, yc, Vm.T, cmap=CMAP, vmin=vlo, vmax=vhi, shading="auto")
            if np.isfinite(vhidden).any():
                a1.pcolormesh(xc, yc, vhidden.T, cmap=CMAP, vmin=vlo, vmax=vhi,
                              shading="auto", alpha=ALPHA_HIDDEN)
            plt.colorbar(p1, ax=a1, label=f"{MEASURE_LABEL} [km/s]", shrink=0.85)
            a1.plot(sx, sy, "^", ms=3.5, mfc="k", mec="w", mew=0.3, zorder=3)
            a1.set_aspect("equal")
            a1.set(xlabel="x [km]", ylabel="y [km]")
            a1.set_title(f"{ds.name}  {args.wave}  T={T:g} s\n"
                         f"N={N} rays  |  LC={LC_T:g} km  sigma_eff={se_star}\n"
                         f"var_red={stats['var_red']:.2f}  chi2={chi2_red:.1f}  "
                         f"shown {cov}/{int((mask>0).sum())}", fontsize=9)
            f1.tight_layout(); f1.savefig(figdir / f"map_T{ds.tfmt(T)}.png", dpi=130); plt.close(f1)
        if chis is not None:                   # only meaningful when the scan actually ran
            Lax.plot(SE_GRID, chis, "-", lw=0.5, color="0.85", zorder=1)
        print(f"{ds.name}/{args.wave} T={T:<6g} N={N:5d} se={se_star:.3f} "
              f"LC={LC_T:g} var_red={stats['var_red']:.2f} restit_post={stats['restit_post']:.1f}% "
              f"chi2_red={chi2_red:.1f} shown={cov}/{int((mask>0).sum())}")

    tab = pd.DataFrame(rows)
    tab.to_csv(outdir / f"production_{args.wave}.csv", index=False)
    if args.fast:
        # --fast skips the sigma_eff audit scan, so the discrepancy figure would
        # contain no data curves at all -- don't write an empty frame.
        plt.close(Lfig)
    else:
        Lax.axhline(1.0, color="r", ls="--", lw=1.2, zorder=2,
                    label="discrepancy-principle target: reduced χ² = 1")
        Lax.axvline(args.se, color="b", lw=1.4, zorder=2,
                    label=f"production prior slowness std σ_eff = {args.se} s/km (fixed)")
        Lax.plot([], [], "-", lw=0.5, color="0.6",
                 label="one grey curve per period: χ²(σ_eff) audit scan")
        Lax.set(xscale="log", yscale="log",
                xlabel="prior slowness std σ_eff (s/km)  —  larger = rougher maps allowed",
                ylabel="reduced χ²  =  Σ(residual/σ_data)² / N")
        Lax.set_title(f"{ds.name} {args.wave} — discrepancy-principle audit\n"
                      "each grey curve: data misfit vs damping for one period; the fixed "
                      "production σ_eff (blue)\nis an analyst choice — curves railing below "
                      "χ²=1 would indicate over-fitting headroom", fontsize=9, loc="left")
        Lax.legend(fontsize=8, loc="upper right")
        Lfig.tight_layout()
        Lfig.savefig(figdir / f"discrepancy_{args.wave}.png", dpi=130)
        plt.close(Lfig)

    # ---- velocity distribution vs period: PICKS (data space) beside MAP CELLS (model
    # space). The pick histogram says what was measured on station pairs; the map histogram
    # says what the inversion put on the grid. They should occupy the same band -- a model
    # distribution narrower than the picks means the prior absorbed the spread, one that
    # is wider or offset means the inversion is producing velocities the data never saw.
    if maps and pickv and not args.no_figs:
        rungs = np.array([t for t, _, _ in maps], float)
        if len(rungs) > 2:
            mid = np.sqrt(rungs[1:] * rungs[:-1])
            tedges = np.concatenate(([rungs[0] ** 2 / mid[0]], mid,
                                     [rungs[-1] ** 2 / mid[-1]]))
            allv = np.concatenate([v[np.isfinite(v)] for _, v in pickv]
                                  + [m[np.isfinite(m)].ravel() for _, m, _ in maps])
            vhi = np.nanpercentile(allv, 99.5) if allv.size else 4.0
            vedges = np.arange(0.195, vhi + 0.1, 0.02)   # half-node offset, 0.01 grid
            f2, a2 = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
            for ax, (src, data) in zip(a2, (("picks (station pairs)", pickv),
                                            ("map cells (inversion)", [(t, m) for t, m, _ in maps]))):
                Tl, Vl = [], []
                for t, v in data:
                    v = np.asarray(v, float).ravel()
                    v = v[np.isfinite(v)]
                    Tl.append(np.full(v.size, t)); Vl.append(v)
                Tl = np.concatenate(Tl); Vl = np.concatenate(Vl)
                H, _, _ = np.histogram2d(Tl, Vl, bins=[tedges, vedges])
                pos = H[H > 0]
                pc = ax.pcolormesh(tedges, vedges, np.where(H > 0, H, np.nan).T, cmap=CMAP,
                                   vmin=0, vmax=np.percentile(pos, 99) if pos.size else 1,
                                   shading="flat")
                plt.colorbar(pc, ax=ax, shrink=0.85,
                             label="%s per cell (p99 clip)"
                                   % ("picks" if "picks" in src else "grid cells"))
                med = [np.median(np.asarray(v, float).ravel()[
                           np.isfinite(np.asarray(v, float).ravel())]) for _, v in data]
                ax.plot(rungs, med, "-", color="deepskyblue", lw=1.8, label="median")
                ax.set_xlabel("period [s]  (CWT scale rungs)")
                ax.set_title("%s  --  %s" % (src, args.wave), fontsize=10)
                ax.legend(fontsize=8)
            a2[0].set_ylabel("%s [km/s]" % MEASURE_LABEL)
            f2.suptitle("%s %s: velocity distribution vs period, data space vs model space"
                        % (ds.name, args.wave), y=1.0)
            f2.tight_layout()
            f2.savefig(figdir / f"vdist_{args.wave}.png", dpi=130)
            plt.close(f2)

    # multi-period map panel, in TWO versions:
    #   per-period scaling  -- each panel on its own 5-95 percentile, best for seeing
    #                          structure at every period (a slow period is not washed out)
    #   common scaling      -- one shared colour range over all periods, the only way to
    #                          read the ABSOLUTE velocity change with period across panels
    # Both use a perceptually uniform sequential map: velocity here is an absolute
    # quantity, not an anomaly, so a diverging map like RdYlBu implies a midpoint that
    # does not exist and distorts apparent gradients.
    if maps:
        allv = np.concatenate([m[np.isfinite(m)].ravel() for _, m, _ in maps
                               if np.isfinite(m).any()]) if maps else np.array([])
        glo, ghi = (np.nanpercentile(allv, [2, 98]) if allv.size else (None, None))
        # Symmetric limit for the anomaly panels. NOT the pooled percentile: a single
        # pathological period (e.g. a short-T map whose mean is dragged by a few extreme
        # cells) then sets the scale for everything and flattens all real structure -- the
        # first version of this figure came out at +/-399%. Take each map's own p98 and use
        # the MEDIAN of those, so one bad period saturates its own panel instead of
        # destroying the others.
        _per_map = []
        for _, m, _ in maps:
            mu = np.nanmean(m)
            if np.isfinite(mu) and mu:
                a = np.abs(100.0 * (m[np.isfinite(m)] - mu) / mu)
                if a.size:
                    _per_map.append(np.nanpercentile(a, 98))
        alim = float(np.nanmedian(_per_map)) if _per_map else 1.0
        alim = max(alim, 1.0)
        for scaling in ("perperiod", "common", "anomaly"):
            n = len(maps); nc = min(5, n); nr = int(np.ceil(n / nc))
            fig, axs = plt.subplots(nr, nc, figsize=(3.4 * nc, 3.2 * nr), squeeze=False)
            for ax, (T, Vm, Vh) in zip(axs.ravel(), maps):
                if scaling == "anomaly":
                    # per-map relative anomaly: each period referenced to ITS OWN mean, so
                    # panels show lateral structure with the strong velocity-vs-period
                    # trend removed -- and a single symmetric scale makes the % amplitude
                    # comparable across periods.
                    mu = np.nanmean(Vm)
                    Z = 100.0 * (Vm - mu) / mu if np.isfinite(mu) and mu else Vm * np.nan
                    pc = ax.pcolormesh(xc, yc, Z.T, cmap=CMAP_ANOM, vmin=-alim, vmax=alim,
                                       shading="auto")
                    if np.isfinite(Vh).any():
                        Zh = 100.0 * (Vh - mu) / mu if np.isfinite(mu) and mu else Vh * np.nan
                        ax.pcolormesh(xc, yc, Zh.T, cmap=CMAP_ANOM, vmin=-alim, vmax=alim,
                                      shading="auto", alpha=ALPHA_HIDDEN)
                else:
                    if scaling == "common":
                        vlo, vhi = glo, ghi
                    else:
                        vlo, vhi = np.nanpercentile(Vm, [5, 95])
                    pc = ax.pcolormesh(xc, yc, Vm.T, cmap=CMAP, vmin=vlo, vmax=vhi,
                                       shading="auto")
                if scaling == "perperiod":
                    plt.colorbar(pc, ax=ax, shrink=0.8)
                ax.plot(sx, sy, "^", ms=1.5, mfc="k", mec="none")
                ax.set_aspect("equal"); ax.set_title(f"T={T:g}s", fontsize=9)
            for ax in axs.ravel()[len(maps):]:
                ax.axis("off")
            scale_txt = {
                "common": f"COMMON colour scale {glo:.2f}-{ghi:.2f} km/s (p2-p98 over all periods)",
                "anomaly": (f"RELATIVE anomaly vs each map's OWN mean, common scale "
                            f"+/-{alim:.1f}% (median over periods of each map's p98 "
                            f"|anomaly|; outlier periods saturate)"),
            }.get(scaling, "per-panel colour scale (p5-p95 of that period)")
            fig.suptitle(f"{ds.name} {args.wave}: production {MEASURE_LABEL} maps "
                         f"({args.lc_mode}-LC, se={args.se}, coverage + drop-worst-"
                         f"{int(100*args.res_drop_q)}%-resolution mask)\n{scale_txt}", y=1.0)
            if scaling in ("common", "anomaly"):
                fig.subplots_adjust(right=0.90)
                cax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
                fig.colorbar(pc, cax=cax,
                             label=(f"d{MEASURE_LABEL}/mean [%]" if scaling == "anomaly"
                                    else f"{MEASURE_LABEL} [km/s]"))
            else:
                fig.tight_layout()
            suffix = {"perperiod": "", "common": "_common", "anomaly": "_anomaly"}[scaling]
            fig.savefig(figdir / f"maps_{args.wave}{suffix}.png", dpi=120)
            plt.close(fig)
    print(f"\nwrote {len(rows)} period maps + production_{args.wave}.csv + figures under {outdir}")


if __name__ == "__main__":
    main()
