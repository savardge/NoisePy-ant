#!/usr/bin/env python
"""
Frequency-Bessel (F-J) dispersion imaging from the per-source VSG NPZ files saved by
phaseshift_dispersion.py --save-sources, using CC-FJpy (Wang et al. 2019; Li et al. 2021).

Unlike the per-source slant stack (phase shift) + image stacking route, the F-J transform
is an array method: ALL unique station-pair CCFs are pooled into a single
frequency-Bessel integral

    I(f, c) = sum_r  Re[ u(r, f) ] * J0(2*pi*f*r/c) * r dr

so each component yields one dispersion image directly. Per-pair symmetric (folded)
time-domain CCFs are cosine-transformed to Re[cross-spectrum]; each pair appears in two
virtual-source gathers and is averaged once.

Note: the J0 kernel is strictly correct for ZZ (Rayleigh). TT (Love) images are computed
with the same kernel as a qualitative test -- interpret with care.

Examples
--------
    # Aargau, vertical + transverse:
    /opt/anaconda3/envs/das-ambient-noise/bin/python fj_dispersion.py \\
        ~/Data/aargau/phasevelocity_VSG --component ZZ,TT

    # Riehen RI-only, all components, 2.5 Hz band, with picks:
    /opt/anaconda3/envs/das-ambient-noise/bin/python fj_dispersion.py \\
        ~/Data/riehen/phasevelocity_VSG --component ZZ,ZR,RZ,RR,TT \\
        --network RI --fmax 2.5 --picks

    # Riehen, custom band/velocities, Hankel variant:
    /opt/anaconda3/envs/das-ambient-noise/bin/python fj_dispersion.py \\
        ~/Data/riehen/phasevelocity_VSG --component ZZ --fmax 2.0 --func 1
"""

import argparse
import csv
import glob
import logging
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

logger = logging.getLogger("fjdisp")


def configure_logging(level):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
def pool_pairs(src_dir, min_offset=None, max_offset=None, networks=None):
    """Pool unique station-pair symmetric CCFs from all per-source NPZ gathers.

    networks : set of network code strings (e.g. {'RI'}), or None for all networks.
    Both source and receiver must belong to an accepted network when networks is set.

    Returns (sym [npairs, nt] float64, r_km [npairs], dt, npz_count).
    """
    npz_files = sorted(glob.glob(os.path.join(src_dir, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files in {src_dir}")

    # Filter source files by network if requested.
    if networks:
        npz_files = [p for p in npz_files
                     if os.path.splitext(os.path.basename(p))[0].split(".")[0] in networks]
    if not npz_files:
        raise ValueError(f"No NPZ files pass network filter {networks} in {src_dir}")

    acc = {}    # key -> [sum_trace, count, dist_km]
    dt_ref, nt_ref = None, None
    for path in npz_files:
        d = np.load(path, allow_pickle=True)
        src = str(d["src"])
        codes = [str(c) for c in d["rx_codes"]]
        sym = d["sym"].astype(np.float64)
        x = d["x"].astype(np.float64)
        dt = float(d["dt"])

        if dt_ref is None:
            dt_ref, nt_ref = dt, sym.shape[1]
        elif abs(dt - dt_ref) > 1e-9:
            raise ValueError(f"dt mismatch in {path}: {dt} vs {dt_ref}")
        nt = min(nt_ref, sym.shape[1])

        for i, code in enumerate(codes):
            # Filter receiver by network.
            if networks and code.split(".")[0] not in networks:
                continue
            if min_offset is not None and x[i] < min_offset:
                continue
            if max_offset is not None and x[i] > max_offset:
                continue
            key = tuple(sorted((src, code)))
            if key in acc:
                acc[key][0][:nt] += sym[i, :nt]
                acc[key][1] += 1
            else:
                acc[key] = [sym[i].copy(), 1, x[i]]

    if not acc:
        raise ValueError(f"No pairs passed all filters in {src_dir}")

    keys = sorted(acc)
    sym_all = np.stack([acc[k][0] / acc[k][1] for k in keys])
    r_km = np.array([acc[k][2] for k in keys])
    logger.info("Pooled %d unique pairs from %d sources (offsets %.2f-%.2f km)",
                len(keys), len(npz_files), r_km.min(), r_km.max())
    return sym_all, r_km, dt_ref, len(npz_files)


def pool_pairs_h5(comp, h5_method, networks, index, min_offset=None, max_offset=None,
                  polarity_fix=True, min_ngood=1):
    """Pool DEDUPLICATED symmetric CCFs for one component straight from the h5 stack files.

    Each unique station pair is read exactly once (sym = 0.5*(causal+acausal)), so the whole
    array is pooled at NPZ-mode pair counts (no double-counting) and the chosen h5 stack
    method (Allstack_pws / Allstack_linear / ...) is honoured. `networks` (set of codes)
    restricts both stations. `index` is a build_file_index() map. Returns
    (sym [npairs, nt] float64, r_km, dt, n_pairs).
    """
    from phaseshift_dispersion import read_pair_lags, fold_sym

    net = set(networks) if networks else None
    codes = [c for c in index if net is None or c.split(".")[0] in net]
    seen = set()
    rows, offs = [], []
    dt_ref = nt_ref = None
    for code in sorted(codes):
        for fpath, other in index.get(code, []):
            if net is not None and other.split(".")[0] not in net:
                continue
            key = tuple(sorted((code, other)))
            if key in seen:
                continue
            seen.add(key)
            cd = read_pair_lags(fpath, h5_method, [comp], polarity_fix)
            if comp not in cd:
                continue
            causal, acausal, dist, dt, ngood = cd[comp]
            if ngood < min_ngood:
                continue
            if min_offset is not None and dist < min_offset:
                continue
            if max_offset is not None and dist > max_offset:
                continue
            sym = fold_sym(comp, causal, acausal)
            if dt_ref is None:
                dt_ref, nt_ref = dt, sym.shape[0]
            elif abs(dt - dt_ref) > 1e-9:
                logger.warning("dt mismatch %s; skipping pair", key)
                continue
            nt = min(nt_ref, sym.shape[0])
            row = np.zeros(nt_ref)
            row[:nt] = sym[:nt]
            rows.append(row)
            offs.append(dist)
    if not rows:
        raise ValueError(f"No pairs for {comp} [{h5_method}] under network filter {networks}")
    r_km = np.asarray(offs)
    logger.info("Pooled %d unique pairs for %s [%s] (offsets %.2f-%.2f km)",
                len(rows), comp, h5_method, r_km.min(), r_km.max())
    return np.asarray(rows), r_km, dt_ref, len(rows)


def pool_pairs_h5_multi(comps, h5_method, networks, index, min_offset=None, max_offset=None,
                        polarity_fix=True, min_ngood=1):
    """Pool DEDUPLICATED symmetric CCFs for SEVERAL components in ONE sweep of the h5 files.

    Each unique pair file is opened exactly once and split into all requested components
    (vs. pool_pairs_h5, which re-reads every file once per component). Returns
    {comp: (sym [npairs, nt] float32, r_km, dt, n_pairs)} for each comp with >=1 pair.
    """
    from phaseshift_dispersion import read_pair_lags, fold_sym

    net = set(networks) if networks else None
    codes = [c for c in index if net is None or c.split(".")[0] in net]
    seen = set()
    acc = {c: {"rows": [], "offs": []} for c in comps}
    dt_ref = nt_ref = None
    for code in sorted(codes):
        for fpath, other in index.get(code, []):
            if net is not None and other.split(".")[0] not in net:
                continue
            key = tuple(sorted((code, other)))
            if key in seen:
                continue
            seen.add(key)
            cd = read_pair_lags(fpath, h5_method, comps, polarity_fix)
            for comp in comps:
                if comp not in cd:
                    continue
                causal, acausal, dist, dt, ngood = cd[comp]
                if ngood < min_ngood:
                    continue
                if min_offset is not None and dist < min_offset:
                    continue
                if max_offset is not None and dist > max_offset:
                    continue
                # Offset pool: no virtual source, so fold with the uniform first-station
                # convention (read_pair_lags keys causal to the filename ordering). This is
                # self-consistent for the odd cross terms -> no per-pair orientation needed.
                sym = fold_sym(comp, causal, acausal)
                if dt_ref is None:
                    dt_ref, nt_ref = dt, sym.shape[0]
                elif abs(dt - dt_ref) > 1e-9:
                    logger.warning("dt mismatch %s; skipping pair", key)
                    continue
                nt = min(nt_ref, sym.shape[0])
                row = np.zeros(nt_ref, dtype=np.float32)
                row[:nt] = sym[:nt]
                acc[comp]["rows"].append(row)
                acc[comp]["offs"].append(dist)
    out = {}
    for comp in comps:
        rows = acc[comp]["rows"]
        if not rows:
            logger.warning("No pairs for %s [%s] under network filter %s",
                           comp, h5_method, networks)
            continue
        r_km = np.asarray(acc[comp]["offs"])
        logger.info("Pooled %d unique pairs for %s [%s] (offsets %.2f-%.2f km)",
                    len(rows), comp, h5_method, r_km.min(), r_km.max())
        out[comp] = (np.asarray(rows), r_km, dt_ref, len(rows))
    return out


def cosine_transform(sym, dt, fmin, fmax, nfft=None, tmax=None, taper_frac=0.1):
    """Re[cross-spectrum] of the symmetric CCF via rfft, band-limited to [fmin, fmax]."""
    npairs, nt = sym.shape
    if tmax is not None:
        nkeep = min(nt, int(round(tmax / dt)) + 1)
        sym = sym[:, :nkeep]
        nt = nkeep
    ntap = max(2, int(round(taper_frac * nt)))
    w = np.ones(nt)
    w[-ntap:] = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, ntap)))
    sym = sym * w[None, :]
    nfft = nfft or nt
    f = np.fft.rfftfreq(nfft, dt)
    uf = np.real(np.fft.rfft(sym, n=nfft, axis=1))
    keep = (f >= fmin) & (f <= fmax)
    return uf[:, keep].astype(np.float32), f[keep]


def normalize_rows(uf, mode):
    if mode == "none":
        return uf
    if mode == "rowmax":
        peak = np.max(np.abs(uf), axis=1, keepdims=True)
        return uf / np.where(peak > 0, peak, 1.0)
    if mode == "whiten":
        from scipy.ndimage import uniform_filter1d
        env = uniform_filter1d(np.abs(uf), size=21, axis=1)
        return uf / np.where(env > 0, env, 1.0)
    raise ValueError(mode)


# ---------------------------------------------------------------------------
def plot_image(ax, img, f, vel, xscale, title, fmin, fmax):
    """Pcolormesh with per-frequency normalisation. Returns PolyCollection."""
    peak = np.max(img, axis=0, keepdims=True)
    norm = img / np.where(peak > 0, peak, 1.0)
    pm = ax.pcolormesh(f, vel, norm, cmap="jet", vmin=0, vmax=1.0,
                       shading="auto", rasterized=True)
    ax.set_xscale(xscale)
    ax.set_xlim(fmin, fmax)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Phase velocity [km/s]")
    ax.set_title(title)
    return pm


def overlay_picks(ax, f_axis, picks, min_score, label=True):
    """Scatter topology picks on an existing axes. picks = {'idx_f', 'vel', 'score'}."""
    if picks is None or len(picks["vel"]) == 0:
        return
    pf = f_axis[picks["idx_f"]]
    s = 4 + 26 * (picks["score"] - min_score) / max(1.0 - min_score, 1e-9)
    ax.scatter(pf, picks["vel"], c="white", s=s, marker="o",
               linewidths=0.4, edgecolors="k", zorder=5,
               label=f"topology picks (score≥{min_score:.2f})" if label else None)
    if label:
        ax.legend(loc="upper right", fontsize=7, framealpha=0.6)


def picks_in_band(picker, img, vel, f_axis, min_score, f_lo, f_hi):
    """Run topology picking only on frequency columns within [f_lo, f_hi].

    `img` is (nv, nf). Returns picks dict with `idx_f` mapped back to the FULL f-grid,
    so picking is confined to the array-resolvable band (cheaper, and no picks where the
    aperture / station spacing cannot constrain phase velocity)."""
    band = np.where((f_axis >= f_lo) & (f_axis <= f_hi))[0]
    empty = {"idx_f": np.array([], dtype=int), "vel": np.array([]), "score": np.array([])}
    if band.size == 0:
        return empty
    sub = picker(img[:, band], vel, min_score)
    if len(sub["idx_f"]) == 0:
        return empty
    return {"idx_f": band[np.asarray(sub["idx_f"], dtype=int)],
            "vel": sub["vel"], "score": sub["score"]}


def _resolution_band(ax, f_lo, f_hi):
    """Mark the array-resolution band: dashed lines at f_lo (aperture) and f_hi (spacing)."""
    for fr, lbl in ((f_lo, f"f$_{{min}}$={f_lo:.2f}"), (f_hi, f"f$_{{max}}$={f_hi:.2f}")):
        if fr is None or not np.isfinite(fr) or fr <= 0:
            continue
        ax.axvline(fr, color="k", ls="--", lw=1.2, zorder=6)
        ax.text(fr, ax.get_ylim()[1], f" {lbl} Hz", color="k", fontsize=7,
                va="top", ha="left", rotation=90, zorder=6)


def picks_to_csv_rows(picks, f_axis, comp, method):
    """Convert a picks dict to a list of row dicts for CSV writing."""
    rows = []
    for idx, vel, score in zip(picks["idx_f"], picks["vel"], picks["score"]):
        freq = float(f_axis[idx])
        rows.append({
            "component": comp,
            "method": method,
            "freq_hz": freq,
            "period_s": 1.0 / freq if freq > 0 else np.nan,
            "vel_kms": float(vel),
            "score": float(score),
        })
    return rows


# ---------------------------------------------------------------------------
def emit_network_fj(comp, sym, r_km, dt, n_units, tag, title_extra, args, vel, c_ms,
                    outdir, ccfj, picker, vs_root):
    """Compute + persist + plot the whole-array F-J image for one (component, source).

    `tag` is appended to every output filename (e.g. '_RI_pws'); `title_extra` to titles.
    Returns the list of picks-CSV rows produced (empty if --picks off).
    """
    uf, f = cosine_transform(sym, dt, args.fmin, args.fmax, nfft=args.nfft, tmax=args.tmax)
    if args.fstride > 1:
        uf, f = uf[:, ::args.fstride], f[::args.fstride]
    uf = normalize_rows(uf, args.norm)
    npairs = uf.shape[0]
    logger.info("[%s%s] F-J transform: %d pairs x %d freqs x %d velocities",
                comp, title_extra, npairs, len(f), len(c_ms))
    img = ccfj.fj_noise(np.ascontiguousarray(uf), (r_km * 1e3).astype(np.float32),
                        c_ms, f.astype(np.float32),
                        fstride=1, itype=args.itype, func=args.func, num=args.num)

    # Array-resolution band: f_min from the aperture (longest resolvable wavelength = max
    # offset) and f_max from the spatial Nyquist of the smallest interstation spacing
    # (lambda_min = 2*dmin). Picking is confined to this band, so no time is spent and no
    # picks are reported where the array geometry cannot constrain phase velocity.
    aperture = float(np.max(r_km))
    pos = r_km[r_km > 0]
    min_spacing = float(np.min(pos)) if pos.size else 0.0
    f_res_lo = args.vave / aperture if aperture > 0 else args.fmin
    f_res_hi = args.vave / (2.0 * min_spacing) if min_spacing > 0 else args.fmax
    f_pick_lo = max(f_res_lo, args.fmin)
    f_pick_hi = min(f_res_hi, args.fmax)

    from phaseshift_dispersion import ODD_LAG_COMPONENTS
    lag_parity = "odd" if comp in ODD_LAG_COMPONENTS else "even"
    np.savez_compressed(os.path.join(outdir, f"fj_{comp}{tag}.npz"),
                        fj=img.astype(np.float32), f=f, vel=vel, component=comp,
                        n_pairs=npairs, r_km=r_km, func=args.func, itype=args.itype,
                        norm=args.norm, tmax=args.tmax or -1.0,
                        network_filter=args.network or "all",
                        f_res_lo=f_pick_lo, f_res_hi=f_pick_hi,
                        aperture_km=aperture, min_spacing_km=min_spacing, vave=args.vave,
                        lag_parity=lag_parity)

    csv_rows = []
    fj_picks = None
    if args.picks:
        fj_picks = picks_in_band(picker, img, vel, f, args.pick_min_score, f_pick_lo, f_pick_hi)
        logger.info("[%s%s] F-J picks: %d in band %.3f-%.3f Hz "
                    "(aperture %.1f km, dmin %.2f km)", comp, title_extra,
                    len(fj_picks["vel"]), f_pick_lo, f_pick_hi, aperture, min_spacing)
        csv_rows.extend(picks_to_csv_rows(fj_picks, f, comp, f"fj{tag}"))

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    pm = plot_image(ax, img, f, vel, "log",
                    f"F-J dispersion  {comp}{title_extra}  ({npairs} pairs)",
                    args.fmin, args.fmax)
    ax.set_ylim(args.vmin, args.vmax)
    if fj_picks is not None:
        overlay_picks(ax, f, fj_picks, args.pick_min_score)
    _resolution_band(ax, f_pick_lo, f_pick_hi)
    fig.colorbar(pm, ax=ax, label="Normalized amplitude")
    fig.savefig(os.path.join(outdir, f"fj_{comp}{tag}_freqlog.png"), dpi=150)
    plt.close(fig)

    # Optional comparison vs the precomputed slant-stack image (NPZ-mode only).
    stacks_path = os.path.join(vs_root, comp, f"stacks_{comp}.npz") if vs_root else None
    if stacks_path and os.path.isfile(stacks_path):
        d = np.load(stacks_path, allow_pickle=True)
        if args.stack_method in d:
            ps_img = np.asarray(d[args.stack_method], dtype=float)
            ps_f, ps_vel = np.asarray(d["f"], dtype=float), np.asarray(d["vel"], dtype=float)
            ps_picks = (picks_in_band(picker, ps_img, ps_vel, ps_f, args.pick_min_score,
                                      f_pick_lo, f_pick_hi) if args.picks else None)
            if ps_picks is not None:
                csv_rows.extend(picks_to_csv_rows(ps_picks, ps_f, comp, f"slant{tag}"))
            fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True,
                                     constrained_layout=True)
            plot_image(axes[0], img, f, vel, "log", f"F-J transform  {comp}{title_extra}",
                       args.fmin, args.fmax)
            if fj_picks is not None:
                overlay_picks(axes[0], f, fj_picks, args.pick_min_score)
            pm = plot_image(axes[1], ps_img, ps_f, ps_vel, "log",
                            f"Slant stack ({args.stack_method})  {comp}", args.fmin, args.fmax)
            if ps_picks is not None:
                overlay_picks(axes[1], ps_f, ps_picks, args.pick_min_score)
            axes[0].set_ylim(args.vmin, args.vmax)
            axes[1].set_ylim(args.vmin, args.vmax)
            _resolution_band(axes[0], f_pick_lo, f_pick_hi)
            _resolution_band(axes[1], f_pick_lo, f_pick_hi)
            fig.colorbar(pm, ax=axes, label="Normalized amplitude", shrink=0.9)
            fig.savefig(os.path.join(outdir, f"fj_vs_phaseshift_{comp}{tag}.png"), dpi=150)
            plt.close(fig)
    logger.info("[%s%s] done", comp, title_extra)
    return csv_rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="VSG root dir (contains <comp>/sources/*.npz)")
    p.add_argument("--component", required=True,
                   help="Component(s), comma-separated, e.g. 'ZZ' or 'ZZ,ZR,RZ,RR,TT'")
    p.add_argument("--network", default=None,
                   help="Comma-separated network codes to keep (e.g. 'RI'). "
                        "Both source and receiver must match. Default: all networks.")
    p.add_argument("--stackdir", default=None,
                   help="Read DEDUPLICATED pairs straight from this h5 STACK_* dir instead of "
                        "the sym NPZs. Enables per-h5-method F-J (--h5-method).")
    p.add_argument("--h5-method", default="Allstack_pws,Allstack_linear",
                   help="With --stackdir: h5 stack method(s), comma-separated. Each yields a "
                        "method-tagged F-J image set (default: Allstack_pws,Allstack_linear)")
    p.add_argument("--min-ngood", type=int, default=1,
                   help="With --stackdir: skip pairs with ngood below this")
    p.add_argument("--no-polarity-fix", action="store_true",
                   help="With --stackdir: disable cross-network polarity sign flip")
    p.add_argument("--fmin", type=float, default=0.1, help="Min frequency [Hz]")
    p.add_argument("--fmax", type=float, default=5.0, help="Max frequency [Hz]")
    p.add_argument("--vave", type=float, default=3.0,
                   help="Representative phase velocity [km/s] for the array-resolution "
                        "picking band: f_min=vave/aperture, f_max=vave/(2*min_spacing) "
                        "(default 3.0; matches phaseshift_dispersion/fj_subarray)")
    p.add_argument("--vmin", type=float, default=0.2, help="Min phase velocity [km/s]")
    p.add_argument("--vmax", type=float, default=6.0, help="Max phase velocity [km/s]")
    p.add_argument("--dv", type=float, default=0.01, help="Velocity step [km/s]")
    p.add_argument("--min-offset", type=float, default=None, help="Min pair offset [km]")
    p.add_argument("--max-offset", type=float, default=None, help="Max pair offset [km]")
    p.add_argument("--tmax", type=float, default=None,
                   help="Truncate symmetric lag at this time [s] before FFT")
    p.add_argument("--nfft", type=int, default=None, help="Zero-pad FFT length")
    p.add_argument("--fstride", type=int, default=1,
                   help="Keep every Nth frequency column (speed)")
    p.add_argument("--norm", choices=["none", "rowmax", "whiten"], default="rowmax",
                   help="Per-pair spectrum normalization (default: rowmax)")
    p.add_argument("--itype", type=int, choices=[0, 1], default=1,
                   help="0 trapezoidal integral, 1 linear approximation (default)")
    p.add_argument("--func", type=int, choices=[0, 1], default=0,
                   help="0 Bessel J0 (default), 1 Hankel (modified F-J, Xi et al. 2021)")
    p.add_argument("--num", type=int, default=os.cpu_count() or 8,
                   help="CPU threads for ccfj (note: CPU build is effectively single-threaded)")
    p.add_argument("--stack-method", default="pws",
                   help="Key in stacks_<comp>.npz for the comparison panel (default: pws)")
    p.add_argument("--picks", action="store_true",
                   help="Run topology findpeaks picking on F-J and slant-stack images, "
                        "overlay on plots, and save picks CSV")
    p.add_argument("--pick-min-score", type=float, default=0.5,
                   help="Topology picking: min persistence score (default: 0.5)")
    p.add_argument("--output-dir", default=None,
                   help="Output directory (default: <root>/../CC-FJpy/network_<net|all>)")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    configure_logging(args.log_level)
    import ccfj

    picker = None
    if args.picks:
        from phaseshift_dispersion import extract_picks_topology
        picker = extract_picks_topology

    networks = None
    if args.network:
        networks = {n.strip().upper() for n in args.network.split(",") if n.strip()}
        logger.info("Network filter: %s", networks)

    root = os.path.abspath(os.path.expanduser(args.root))
    net_tag = args.network.upper() if args.network else "all"
    outdir = args.output_dir or os.path.join(os.path.dirname(root),
                                             "CC-FJpy", f"network_{net_tag}")
    os.makedirs(outdir, exist_ok=True)

    vel = np.arange(args.vmin, args.vmax + args.dv / 2, args.dv)
    c_ms = (vel * 1e3).astype(np.float32)
    comps = [c.strip().upper() for c in args.component.split(",") if c.strip()]
    all_csv_rows = []

    if args.stackdir:
        # ---- h5 mode: deduplicated whole-array F-J, one set per h5 stack method ----
        from phaseshift_dispersion import build_file_index
        stackdir = os.path.abspath(os.path.expanduser(args.stackdir))
        logger.info("Indexing %s ...", stackdir)
        index = build_file_index(stackdir)
        methods = [m.strip() for m in args.h5_method.split(",") if m.strip()]
        for method in methods:
            mshort = method[len("Allstack_"):] if method.startswith("Allstack_") else method
            # One sweep of the h5 files per method pools ALL components at once.
            pooled = pool_pairs_h5_multi(
                comps, method, networks, index,
                min_offset=args.min_offset, max_offset=args.max_offset,
                polarity_fix=not args.no_polarity_fix, min_ngood=args.min_ngood)
            for comp in comps:
                if comp not in pooled:
                    continue
                sym, r_km, dt, npairs = pooled[comp]
                tag = f"_{net_tag}_{mshort}"
                all_csv_rows.extend(emit_network_fj(
                    comp, sym, r_km, dt, npairs, tag, f" [{net_tag} {mshort}]",
                    args, vel, c_ms, outdir, ccfj, picker, vs_root=None))
            del pooled
    else:
        # ---- NPZ mode: pooled sym from the VSG --save-sources output ----
        for comp in comps:
            src_dir = os.path.join(root, comp, "sources")
            if not os.path.isdir(src_dir):
                logger.warning("Sources dir not found, skipping: %s", src_dir)
                continue
            sym, r_km, dt, nsrc = pool_pairs(src_dir, args.min_offset, args.max_offset,
                                             networks=networks)
            tag = f"_{net_tag}" if args.network else ""
            title_extra = f" [{net_tag}]" if args.network else ""
            all_csv_rows.extend(emit_network_fj(
                comp, sym, r_km, dt, nsrc, tag, title_extra,
                args, vel, c_ms, outdir, ccfj, picker, vs_root=root))

    if args.picks and all_csv_rows:
        csv_tag = f"_{net_tag}" if args.network else ""
        csv_path = os.path.join(outdir, f"fj_picks{csv_tag}.csv")
        fieldnames = ["component", "method", "freq_hz", "period_s", "vel_kms", "score"]
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_csv_rows)
        logger.info("Wrote %d picks -> %s", len(all_csv_rows), csv_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
