#!/usr/bin/env python
"""
Subarray F-J vs slant-stack dispersion comparison from raw ASDF (.h5) cross-correlations.

Given ONE center station and a radius, select every station within that radius and treat
the resulting subarray as a mini virtual-source array. For each requested lag branch
(symmetric / causal / acausal) and channel combination, this:

  1. builds the per-source virtual gathers directly from the h5 stack files (so the causal
     and acausal one-sided branches are available -- the cached per-source NPZs hold only
     the symmetric fold),
  2. pools all subarray pairs into a single frequency-Bessel (F-J) transform image
     (CC-FJpy, Wang et al. 2019), and in the same pass accumulates the per-source
     phase-shift dispersion images into a phase-weighted slant stack,
  3. picks both images with topology persistence (findpeaks), and
  4. writes a 3-panel figure -- F-J image | slant-stack image | station map -- where the
     map shows the whole network (grey) with the subarray highlighted in red over a
     swisstopo NationalMapColor basemap (hillshade relief + rivers + canton boundaries).

Causal vs acausal asymmetry diagnoses noise-source directionality; sym maximises SNR.

Example
-------
    /opt/anaconda3/envs/das-ambient-noise/bin/python fj_subarray.py \\
        /Volumes/Data/unige/riehen/crosscorrelations/STACK_CHRI_normZ \\
        --station-csv /Volumes/Data/unige/riehen/crosscorrelations/stations_nodes_noisepy.csv \\
        --center RI.BAS01 --radius 4 \\
        --component ZZ,TT --lags sym,causal,acausal --network RI \\
        --stack-method Allstack_pws --fmax 2.5 \\
        --output-dir ~/Data/riehen/CC-FJpy/subarrays
"""

import argparse
import csv
import logging
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import contextily as cx

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from phaseshift_dispersion import (  # noqa: E402
    EPS,
    load_station_csv,
    build_file_index,
    build_source_gathers,
    setup_freq_axis,
    phase_shift_image,
    init_stack_state,
    accumulate,
    finalize_stacks,
    extract_picks_topology,
    haversine_km,
)
from fj_dispersion import (  # noqa: E402
    cosine_transform,
    normalize_rows,
    plot_image,
    overlay_picks,
    picks_to_csv_rows,
)

logger = logging.getLogger("fjsub")

_WGS84_R = 6378137.0  # Web-Mercator sphere radius


def configure_logging(level):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")


def method_short(stack_method):
    """'Allstack_pws' -> 'pws', 'Allstack_linear' -> 'linear' (else the raw name)."""
    return stack_method[len("Allstack_"):] if stack_method.startswith("Allstack_") \
        else stack_method


def aperture_fmin(max_offset_km, vave):
    """Minimum resolvable frequency [Hz] from the array aperture.

    Longest resolvable wavelength = aperture (max interstation distance); at an average
    phase velocity `vave` [km/s] this maps to f_min = vave / aperture. Same convention as
    phaseshift_dispersion's auto frequency band (vave/omax). Picks below f_min sample
    wavelengths longer than the array can constrain and are discarded.
    """
    return vave / max_offset_km if max_offset_km > 0 else 0.0


def cut_picks_below(picks, f_axis, f_min):
    """Drop picks at frequencies below f_min (uses each picks dict's own freq axis)."""
    if picks is None or len(picks["idx_f"]) == 0:
        return picks
    keep = f_axis[picks["idx_f"]] >= f_min
    return {"idx_f": picks["idx_f"][keep], "vel": picks["vel"][keep],
            "score": picks["score"][keep]}


def web_mercator(lon, lat):
    """Closed-form lon/lat (deg) -> EPSG:3857 metres (avoids a pyproj dependency)."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    x = _WGS84_R * np.radians(lon)
    y = _WGS84_R * np.log(np.tan(np.pi / 4.0 + np.radians(lat) / 2.0))
    return x, y


# ---------------------------------------------------------------------------
def resolve_center(coords, center):
    """Resolve a center argument ('RI.BAS01' or bare 'BAS01') to a full NET.STA code."""
    if center in coords:
        return center
    matches = [c for c in coords if c.split(".")[-1] == center]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"Center station {center!r} not found in coords")
    raise ValueError(f"Center {center!r} is ambiguous: {matches}")


def select_subarray(coords, center_code, radius_km, network_codes):
    """Return codes within radius_km of the center (great-circle), restricted to network_codes."""
    clon, clat = coords[center_code]
    out = []
    for code in network_codes:
        lon, lat = coords[code]
        if haversine_km(clon, clat, lon, lat) <= radius_km:
            out.append(code)
    return sorted(out)


def pool_subarray(index, subarray, dtype, comp, lag, polarity_fix,
                  min_offset, min_ngood, vel, fmin, fmax, nfft_arg, amp_mode):
    """Build subarray gathers from h5 and produce both the F-J input and the slant stack.

    Each subarray station is a virtual source; receivers are restricted to the subarray.
    Every source gather contributes its traces to the F-J pool (offset r) AND its
    phase-shift image to the cross-source stack -- consistent treatment for one-sided lags
    (a causal A->B and the reciprocal B->A are distinct branches, both kept).

    Returns dict with F-J inputs (sym_all, r_km, dt), slant-stack outputs (ps_stacks, f_ps),
    and bookkeeping (n_src, n_pairs), or None if no data.
    """
    subarray_set = set(subarray)
    fj_rows, fj_off = [], []
    dt_ref = nt_ref = None
    nfft = f_ps = f_mask_idx = None
    state = None
    n_src = 0

    for src in subarray:
        gathers = build_source_gathers(
            index, src, subarray_set, dtype, [comp], polarity_fix,
            min_offset, min_ngood, lag=lag)
        g = gathers.get(comp)
        if g is None:
            continue
        sym, x, dt = g["sym"], g["x"], g["dt"]

        if dt_ref is None:
            dt_ref, nt_ref = dt, sym.shape[1]
            nfft, f_axis = setup_freq_axis(nt_ref, dt, nfft_arg)
            keep = (f_axis >= fmin) & (f_axis <= fmax)
            f_mask_idx = np.where(keep)[0]
            f_ps = f_axis[f_mask_idx]
            state = init_stack_state(vel.size, f_ps.size)
        elif abs(dt - dt_ref) > 1e-9:
            logger.warning("dt mismatch for source %s; skipping", src)
            continue

        # F-J pool: pad/truncate to the reference length.
        nt = min(nt_ref, sym.shape[1])
        for i in range(sym.shape[0]):
            row = np.zeros(nt_ref)
            row[:nt] = sym[i, :nt]
            fj_rows.append(row)
            fj_off.append(x[i])

        # Slant-stack: per-source phase-shift image accumulated into the stack state.
        E = phase_shift_image(sym, x, dt, f_mask_idx, nfft, vel, amp_mode=amp_mode)
        accumulate(state, E, root_n=2.0)
        n_src += 1

    if n_src == 0 or not fj_rows:
        return None

    ps_stacks = finalize_stacks(state, pws_power=2.0, root_n=2.0)
    return {
        "sym_all": np.asarray(fj_rows),
        "r_km": np.asarray(fj_off),
        "dt": dt_ref,
        "ps_stacks": ps_stacks,
        "f_ps": f_ps,
        "n_src": n_src,
        "n_pairs": len(fj_rows),
    }


# ---------------------------------------------------------------------------
def plot_subarray_map(ax, coords, network_codes, subarray, center_code,
                      pad_frac=0.08, whole_network=False):
    """Station map over swisstopo NationalMapColor.

    whole_network=False: network (grey) + subarray (red) + center (star).
    whole_network=True : every network station in red, no center star.
    """
    net_lon = np.array([coords[c][0] for c in network_codes])
    net_lat = np.array([coords[c][1] for c in network_codes])
    nx, ny = web_mercator(net_lon, net_lat)

    if whole_network:
        ax.scatter(nx, ny, s=18, c="red", edgecolors="black", linewidths=0.3,
                   zorder=4, label=f"network ({len(network_codes)})")
    else:
        sub_lon = np.array([coords[c][0] for c in subarray])
        sub_lat = np.array([coords[c][1] for c in subarray])
        sx, sy = web_mercator(sub_lon, sub_lat)
        cx_, cy_ = web_mercator(*coords[center_code])
        ax.scatter(nx, ny, s=14, c="0.35", edgecolors="white", linewidths=0.3,
                   zorder=3, label=f"network ({len(network_codes)})")
        ax.scatter(sx, sy, s=34, c="red", edgecolors="black", linewidths=0.4,
                   zorder=4, label=f"subarray ({len(subarray)})")
        ax.scatter([cx_], [cy_], s=160, marker="*", c="red", edgecolors="black",
                   linewidths=0.6, zorder=5, label="center")

    spanx = nx.max() - nx.min() or 1000.0
    spany = ny.max() - ny.min() or 1000.0
    ax.set_xlim(nx.min() - pad_frac * spanx, nx.max() + pad_frac * spanx)
    ax.set_ylim(ny.min() - pad_frac * spany, ny.max() + pad_frac * spany)
    try:
        cx.add_basemap(ax, source=cx.providers.SwissFederalGeoportal.NationalMapColor,
                       attribution_size=4)
    except Exception as exc:  # noqa: BLE001 - offline / tile fetch failure
        logger.warning("Basemap fetch failed (%s); plain background.", exc)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
    ax.set_title("Station map")


def _aperture_line(ax, f_res):
    """Dashed black vertical line marking the aperture-limited minimum resolvable freq."""
    ax.axvline(f_res, color="k", ls="--", lw=1.3, zorder=6)
    ax.text(f_res, ax.get_ylim()[1], f" f$_{{min}}$={f_res:.2f} Hz",
            color="k", fontsize=7, va="top", ha="left", rotation=0)


def make_figure(outpath, fj_img, f_fj, ps_img, f_ps, vel, fj_picks, ps_picks,
                coords, network_codes, subarray, center_code,
                comp, lag, min_score, fmin, fmax, stack_method, n_src, n_pairs, f_res,
                whole_network=False, title_label=None):
    """Three-panel comparison figure: F-J | slant stack | map.

    f_res : aperture-limited minimum resolvable frequency; drawn as a dashed black line.
    Picks passed in are already cut below f_res. whole_network switches the map + title.
    """
    fig = plt.figure(figsize=(19, 5.6))
    gs = GridSpec(1, 3, width_ratios=[1.15, 1.15, 1.0], wspace=0.22)

    ax0 = fig.add_subplot(gs[0])
    plot_image(ax0, fj_img, f_fj, vel, "log", f"F-J transform  {comp}  ({lag})", fmin, fmax)
    ax0.set_ylim(vel.min(), vel.max())
    if fj_picks is not None:
        overlay_picks(ax0, f_fj, fj_picks, min_score)
    _aperture_line(ax0, f_res)

    ax1 = fig.add_subplot(gs[1])
    pm = plot_image(ax1, ps_img, f_ps, vel, "log",
                    f"Slant stack ({stack_method})  {comp}  ({lag})", fmin, fmax)
    ax1.set_ylim(vel.min(), vel.max())
    if ps_picks is not None:
        overlay_picks(ax1, f_ps, ps_picks, min_score)
    _aperture_line(ax1, f_res)
    fig.colorbar(pm, ax=ax1, label="Normalized amplitude", pad=0.02)

    ax2 = fig.add_subplot(gs[2])
    plot_subarray_map(ax2, coords, network_codes, subarray, center_code,
                      whole_network=whole_network)

    if whole_network:
        head = f"Whole network {title_label}"
    else:
        head = f"Subarray {center_code}  r≤{_RADIUS_LABEL} km"
    fig.suptitle(f"{head}  |  {comp}  |  lag={lag}  |  {n_src} sources, "
                 f"{n_pairs} pair-traces", fontsize=12)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


_RADIUS_LABEL = "?"  # set per-center for the suptitle


# ---------------------------------------------------------------------------
def center_subdir(args, center_code):
    return os.path.join(os.path.expanduser(args.output_dir),
                        f"{center_code}_r{args.radius:g}km")


def picks_csv_path(args, center_code, mshort):
    return os.path.join(center_subdir(args, center_code),
                        f"picks_{center_code}_r{args.radius:g}km_{mshort}.csv")


def process_unit(label, csv_stem, subarray, sub_dir, args, coords, network_codes, index,
                 vel, c_ms, ccfj, method, whole_network=False):
    """Run all components x lags for ONE station set (subarray or whole network) and ONE
    h5 stack method. Outputs are method-tagged (pws/linear): fj_<comp>_<lag>_<mshort>.npz,
    fj_subarray_<comp>_<lag>_<mshort>.png, and picks_<csv_stem>_<mshort>.csv in sub_dir.
    `label` drives the figure title; `csv_stem` the picks-CSV filename.
    Returns True if any picks CSV was written.
    """
    mshort = method_short(method)
    logger.info("Unit %s [%s] -> %d stations", label, mshort, len(subarray))
    lags = [l.strip().lower() for l in args.lags.split(",") if l.strip()]
    comps = [c.strip().upper() for c in args.component.split(",") if c.strip()]
    os.makedirs(sub_dir, exist_ok=True)
    all_csv_rows = []

    for comp in comps:
        for lag in lags:
            pooled = pool_subarray(
                index, subarray, method, comp, lag,
                polarity_fix=not args.no_polarity_fix,
                min_offset=args.min_offset, min_ngood=args.min_ngood,
                vel=vel, fmin=args.fmin, fmax=args.fmax,
                nfft_arg=args.nfft, amp_mode=args.amp_mode)
            if pooled is None:
                logger.warning("No data for %s %s/%s [%s]; skipping",
                               label, comp, lag, mshort)
                continue

            uf, f_fj = cosine_transform(pooled["sym_all"], pooled["dt"],
                                        args.fmin, args.fmax, nfft=args.nfft, tmax=args.tmax)
            uf = normalize_rows(uf, args.norm)
            fj_img = ccfj.fj_noise(np.ascontiguousarray(uf),
                                   (pooled["r_km"] * 1e3).astype(np.float32),
                                   c_ms, f_fj.astype(np.float32),
                                   fstride=1, itype=args.itype, func=args.func, num=args.num)
            ps_img = pooled["ps_stacks"]["pws"]
            f_ps = pooled["f_ps"]

            aperture = float(pooled["r_km"].max())
            f_res = aperture_fmin(aperture, args.vave)
            logger.info("[%s %s/%s %s] %d pairs, aperture=%.2f km -> f_min=%.3f Hz",
                        label, comp, lag, mshort, uf.shape[0], aperture, f_res)

            fj_picks = cut_picks_below(
                extract_picks_topology(fj_img, vel, args.pick_min_score), f_fj, f_res)
            ps_picks = cut_picks_below(
                extract_picks_topology(ps_img, vel, args.pick_min_score), f_ps, f_res)
            all_csv_rows.extend(picks_to_csv_rows(fj_picks, f_fj, comp, f"fj_{lag}"))
            all_csv_rows.extend(picks_to_csv_rows(ps_picks, f_ps, comp, f"slant_{lag}"))

            npz = os.path.join(sub_dir, f"fj_{comp}_{lag}_{mshort}.npz")
            np.savez_compressed(npz, fj=fj_img.astype(np.float32), f=f_fj, vel=vel,
                                ps=ps_img.astype(np.float32), f_ps=f_ps,
                                component=comp, lag=lag, center=label,
                                radius_km=(-1.0 if whole_network else args.radius),
                                subarray=np.array(subarray, dtype=object),
                                n_src=pooled["n_src"], n_pairs=pooled["n_pairs"],
                                r_km=pooled["r_km"], stack_method=method,
                                aperture_km=aperture, f_res=f_res, vave=args.vave)

            png = os.path.join(sub_dir, f"fj_subarray_{comp}_{lag}_{mshort}.png")
            make_figure(png, fj_img, f_fj, ps_img, f_ps, vel, fj_picks, ps_picks,
                        coords, network_codes, subarray,
                        None if whole_network else label,
                        comp, lag, args.pick_min_score, args.fmin, args.fmax,
                        method, pooled["n_src"], pooled["n_pairs"], f_res,
                        whole_network=whole_network, title_label=label)

    if all_csv_rows:
        fields = ["component", "method", "freq_hz", "period_s", "vel_kms", "score"]
        with open(os.path.join(sub_dir, f"picks_{csv_stem}_{mshort}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(all_csv_rows)
    return bool(all_csv_rows)


# ---------------------------------------------------------------------------
def main(argv=None):
    global _RADIUS_LABEL
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stackdir", help="STACK_* dir with per-station subdirs of pair .h5 files")
    p.add_argument("--station-csv", required=True, help="noisepy stations CSV (lon/lat)")
    p.add_argument("--center", default=None, help="Center station code (e.g. RI.BAS01 or BAS01)")
    p.add_argument("--all-centers", action="store_true",
                   help="Use every network station as a center (ignores --center)")
    p.add_argument("--all-stations", "--whole-network", dest="whole_network",
                   action="store_true",
                   help="Pool the ENTIRE network (after --network filter) as one array; "
                        "no center/radius, map shows all stations in red")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip (center|network, method) units whose picks CSV already exists")
    p.add_argument("--radius", type=float, default=None, help="Subarray radius [km]")
    p.add_argument("--component", required=True, help="Component(s), comma-separated")
    p.add_argument("--lags", default="sym,causal,acausal",
                   help="Lag branches, comma-separated subset of sym,causal,acausal")
    p.add_argument("--network", default=None,
                   help="Restrict to network code(s), comma-separated (e.g. RI)")
    p.add_argument("--stack-method", default="Allstack_pws",
                   help="ASDF AuxiliaryData stack type(s), comma-separated. Each produces its "
                        "own method-tagged figures/NPZs/CSV (e.g. Allstack_pws,Allstack_linear)")
    p.add_argument("--fmin", type=float, default=0.1, help="Min frequency [Hz]")
    p.add_argument("--fmax", type=float, default=2.5, help="Max frequency [Hz]")
    p.add_argument("--vmin", type=float, default=0.2, help="Min phase velocity [km/s]")
    p.add_argument("--vmax", type=float, default=6.0, help="Max phase velocity [km/s]")
    p.add_argument("--dv", type=float, default=0.01, help="Velocity step [km/s]")
    p.add_argument("--vave", type=float, default=3.0,
                   help="Average phase velocity [km/s] for the aperture-limited f_min "
                        "= vave/aperture (default 3.0; matches phaseshift_dispersion)")
    p.add_argument("--min-offset", type=float, default=0.0, help="Min pair offset [km]")
    p.add_argument("--min-ngood", type=int, default=1, help="Skip pairs with ngood below this")
    p.add_argument("--amp-mode", choices=["phase_only", "sqrt", "raw"], default="phase_only",
                   help="Slant-stack amplitude weighting (default: phase_only)")
    p.add_argument("--no-polarity-fix", action="store_true",
                   help="Disable cross-network polarity sign flip")
    p.add_argument("--nfft", type=int, default=None, help="Zero-pad FFT length")
    p.add_argument("--tmax", type=float, default=None,
                   help="Truncate symmetric lag at this time [s] before the F-J FFT")
    p.add_argument("--norm", choices=["none", "rowmax", "whiten"], default="rowmax",
                   help="Per-pair F-J spectrum normalization (default: rowmax)")
    p.add_argument("--itype", type=int, choices=[0, 1], default=1, help="F-J integral type")
    p.add_argument("--func", type=int, choices=[0, 1], default=0, help="0 Bessel, 1 Hankel")
    p.add_argument("--num", type=int, default=os.cpu_count() or 8, help="ccfj threads")
    p.add_argument("--pick-min-score", type=float, default=0.5,
                   help="Topology picking min persistence score (default: 0.5)")
    p.add_argument("--output-dir", required=True,
                   help="Base output dir; a <center>_r<radius>km subfolder is created")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    configure_logging(args.log_level)
    import ccfj

    coords = load_station_csv(os.path.expanduser(args.station_csv))
    networks = None
    if args.network:
        networks = {n.strip().upper() for n in args.network.split(",") if n.strip()}
    network_codes = sorted(c for c in coords
                           if networks is None or c.split(".")[0] in networks)
    if not network_codes:
        p.error(f"No stations match network filter {networks}")

    if not args.whole_network and not args.all_centers and args.center is None:
        p.error("Provide --center <code>, --all-centers, or --whole-network")
    if not args.whole_network and args.radius is None:
        p.error("--radius is required for subarray mode (--center / --all-centers)")

    methods = [m.strip() for m in args.stack_method.split(",") if m.strip()]
    out_base = os.path.expanduser(args.output_dir)

    # Build the work list of (label, csv_stem, subarray, sub_dir, whole_network) units.
    units = []
    if args.whole_network:
        net_tag = "_".join(sorted(networks)) if networks else "all"
        label = f"network_{net_tag}"
        units.append((label, label, list(network_codes), out_base, True))
        logger.info("Whole-network mode: %d stations -> %s", len(network_codes), label)
    else:
        centers = list(network_codes) if args.all_centers \
            else [resolve_center(coords, args.center)]
        if args.all_centers:
            logger.info("All-centers mode: %d candidate centers", len(centers))
        for center_code in centers:
            subarray = select_subarray(coords, center_code, args.radius, network_codes)
            if len(subarray) < 3:
                logger.warning("Center %s: only %d stations within %.2f km; skipping.",
                               center_code, len(subarray), args.radius)
                continue
            stem = f"{center_code}_r{args.radius:g}km"
            units.append((center_code, stem, subarray, center_subdir(args, center_code), False))

    logger.info("Indexing %s ...", args.stackdir)
    index = build_file_index(args.stackdir)

    vel = np.arange(args.vmin, args.vmax + args.dv / 2, args.dv)
    c_ms = (vel * 1e3).astype(np.float32)
    if args.radius is not None:
        _RADIUS_LABEL = f"{args.radius:g}"

    n_ok = n_skip = 0
    total = len(units) * len(methods)
    k = 0
    for label, stem, subarray, sub_dir, whole in units:
        for method in methods:
            k += 1
            mshort = method_short(method)
            csv_path = os.path.join(sub_dir, f"picks_{stem}_{mshort}.csv")
            if args.skip_existing and os.path.isfile(csv_path):
                logger.info("(%d/%d) %s [%s] exists; skipping", k, total, label, mshort)
                n_skip += 1
                continue
            logger.info("(%d/%d) processing %s [%s]", k, total, label, mshort)
            try:
                if process_unit(label, stem, subarray, sub_dir, args, coords,
                                 network_codes, index, vel, c_ms, ccfj, method,
                                 whole_network=whole):
                    n_ok += 1
                else:
                    n_skip += 1
            except Exception as exc:  # noqa: BLE001 - keep the batch going
                logger.error("Unit %s [%s] failed: %s", label, mshort, exc)
                n_skip += 1

    logger.info("Done: %d (unit,method) produced output, %d skipped/failed.",
                n_ok, n_skip)
    return 0


if __name__ == "__main__":
    sys.exit(main())
