#!/usr/bin/env python
"""
Phase-shift / slant-stack dispersion imaging on cross-correlation virtual-source gathers.

Implements the array phase-shift transform (Park, Miller & Xia 1998) on ambient-noise
cross-correlation gathers and the phase-weighted slant stack of Cheng et al. (2021, GJI) to
produce a single network-average surface-wave phase-velocity dispersion image.

For each station S taken as a virtual source, the symmetric (folded) cross-correlations of all
selected pairs (S, R) form a record section u(x, t) with offset x = interstation distance. Each
gather is transformed to an f-v image

    E_s(f, v) = sum_j  exp( i 2 pi f x_j / v ) * P_j(f),     P_j = U_j / |U_j|   (phase_only)

with U_j(f) = rfft of the trace at offset x_j. |E_s(f, v)| has peaks along the dispersion
branch. The per-source images are then stacked across all sources with four strategies (linear
mean, phase-weighted stack, complex-PWS, and nth-root) so coherent features common to many
virtual sources are enhanced while incoherent sidelobes / aliasing ghosts are suppressed.

The ASDF (.h5) cross-correlation files are read directly with h5py (no pyasdf dependency):
data lives at /AuxiliaryData/<Allstack_method>/<comp> with metadata in the group attributes.

Example
-------
    python phaseshift_dispersion.py \
        /Volumes/Data/unige/riehen/crosscorrelations/STACK_CHRI_normZ \
        --component ZZ --stations RI.BAS26,RI.BAS27,RI.BAS28,RI.BET01 \
        --output-dir /tmp/ps_test --output-plot /tmp/ps_stack.png

References
----------
Park, Miller & Xia (1998), SEG Expanded Abstracts (phase-shift / MASW).
Cheng et al. (2021), GJI, "phase-weighted slant stacking".
Schimmel & Paulssen (1997), GJI (phase-weighted stack).
"""

import argparse
import glob
import logging
import os
import sys

import numpy as np
import h5py
from scipy.fft import rfft, rfftfreq, next_fast_len

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# reference_model lives in scripts/picking; add it to the path for the dispersion overlay.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "picking"))
try:
    from reference_model import reference_curves
except Exception:  # disba missing or import error -> overlay simply skipped
    reference_curves = None

logger = logging.getLogger("phaseshift")

EPS = 1e-12


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------
def configure_logging(level):
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def load_station_csv(path):
    """Return dict 'NET.STA' -> (lon, lat) from a noisepy stations csv."""
    import pandas as pd

    df = pd.read_csv(path)
    coords = {}
    for _, row in df.iterrows():
        code = f"{row['network']}.{row['station']}"
        coords[code] = (float(row["longitude"]), float(row["latitude"]))
    return coords


def resolve_station_list(arg, available):
    """
    Resolve --stations (comma list | file path | 'all') against the set of station codes
    that actually have data on disk. Bare station names (no network) are matched by suffix.
    """
    if arg is None or arg.lower() == "all":
        return set(available)

    if os.path.isfile(arg):
        with open(arg) as fh:
            raw = [ln for ln in fh if not ln.strip().startswith("#")]
        text = " ".join(raw)
    else:
        text = arg
    tokens = [t.strip() for t in text.replace(",", " ").split() if t.strip()]

    selected = set()
    by_name = {}
    for code in available:
        by_name.setdefault(code.split(".")[-1], []).append(code)
    for tok in tokens:
        if tok in available:
            selected.add(tok)
        elif tok in by_name:  # bare station name
            selected.update(by_name[tok])
        else:
            logger.warning("Requested station '%s' not found on disk; skipping.", tok)
    return selected


# ---------------------------------------------------------------------------
# Pair-file discovery (handles both filename orderings; pairs stored only once)
# ---------------------------------------------------------------------------
def parse_pair(filepath):
    """'NET.STA1_NET.STA2.h5' -> ('NET.STA1', 'NET.STA2')."""
    base = os.path.basename(filepath)
    if base.endswith(".h5"):
        base = base[:-3]
    a, b = base.split("_")
    return a, b


def build_file_index(stackdir):
    """
    Scan all pair files ONCE and map every station to the files it participates in.

    Pairs are stored once on disk (under one station's subdir), so each file is registered
    against BOTH of its stations -> index[code] = [(filepath, other_code), ...]. This replaces
    the per-source globbing (which was O(sources x dirs)) with a single directory walk.
    """
    index = {}
    files = glob.glob(os.path.join(stackdir, "*", "*.h5"))
    for f in files:
        try:
            a, b = parse_pair(f)
        except ValueError:
            continue
        index.setdefault(a, []).append((f, b))
        index.setdefault(b, []).append((f, a))
    logger.info("Indexed %d pair files across %d stations.", len(files), len(index))
    return index


def haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance [km] between scalar or array lon/lat (degrees)."""
    r = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def estimate_offsets(coords, selected):
    """All pairwise inter-station distances [km] among selected stations, from coordinates.

    Used only to fix the frequency band and the near-field cut up front, with no file I/O."""
    codes = [c for c in selected if c in coords]
    if len(codes) < 2:
        return np.array([])
    lon = np.array([coords[c][0] for c in codes])
    lat = np.array([coords[c][1] for c in codes])
    iu, ju = np.triu_indices(len(codes), k=1)
    return haversine_km(lon[iu], lat[iu], lon[ju], lat[ju])


# ---------------------------------------------------------------------------
# Reading cross-correlations (direct h5py; ASDF layout)
# ---------------------------------------------------------------------------
def read_pair_lags(sfile, dtype, comps, polarity_fix=True):
    """
    Open one pair file ONCE and split all requested components into causal / acausal lags.

    Both are returned on a positive lag-time axis (0..maxlag): `causal` = positive lags of the
    stored CC (= first-station -> second-station propagation), `acausal` = negated/folded
    negative lags (= second-station -> first-station, by reciprocity CC_AB(-t)=CC_BA(+t)).
    Returns dict comp -> (causal, acausal, dist, dt, ngood). {} on failure / missing stack type.
    """
    a, b = parse_pair(sfile)
    sign = -1.0 if (polarity_fix and a.split(".")[0] != b.split(".")[0]) else 1.0
    out = {}
    try:
        with h5py.File(sfile, "r") as ds:
            grp = ds["AuxiliaryData"][dtype]
            for comp in comps:
                if comp not in grp:
                    continue
                node = grp[comp]
                data = sign * np.asarray(node[()], dtype=np.float64)
                imid = data.shape[0] // 2
                causal = data[imid:]                  # +lags: A -> B
                acausal = np.flip(data[: imid + 1])   # -lags folded: B -> A
                out[comp] = (causal, acausal, float(node.attrs["dist"]),
                             float(node.attrs["dt"]), int(node.attrs["ngood"]))
    except Exception as exc:  # noqa: BLE001 - corrupt file / missing stack type
        logger.debug("Skip %s: %s", os.path.basename(sfile), exc)
        return {}
    return out


# Rayleigh vertical (Z) and radial (R) motion are in 90-degree phase quadrature, so in an
# ISOTROPIC, equipartitioned ambient field the vertical-radial CROSS components are ODD in lag
# (SPAC kernel J1(kr), correct fold (causal-acausal)/2), while the diagonal autocorrelations
# (ZZ, RR) and Love TT are EVEN (J0 kernel, (causal+acausal)/2).
#
# EMPIRICAL CAVEAT (Aargau, June 2026): the field there is strongly DIRECTIONAL — per-pair
# corr(causal, acausal) ~ 0 for every component, i.e. the two lag sides are largely independent
# one-sided EGFs, not +/- copies. In that regime each lag side carries a one-sided Rayleigh
# cross arrival that ADDS under (causal+acausal)/2 (additive folding gave a HIGHER, cleaner
# dispersion ridge than the odd fold, and the odd fold cancelled in the cross-SOURCE PWS step).
# So the EVEN ("additive") fold is the operational default here; the textbook ODD fold is opt-in
# via --cross-fold odd for genuinely isotropic fields. (Separately, the J0-vs-J1 kernel mismatch
# biases cross-component velocity at low kr; that needs a J1-kernel F-J transform, not re-folding.)
ODD_LAG_COMPONENTS = frozenset({"ZR", "RZ"})
CROSS_FOLD = "even"   # "even" = additive (default, directional fields); "odd" = (c-a)/2


def fold_sym(comp, causal, acausal):
    """Lag-symmetric stack. Cross comps use CROSS_FOLD ('even' additive, or 'odd' (c-a)/2).

    For the odd fold the `(causal-acausal)/2` result is referenced to the FIRST station of the
    pair; `select_lag` negates it when the chosen virtual source is the second station so a
    gather stacks coherently. The even (additive) fold is orientation-free.
    """
    if comp in ODD_LAG_COMPONENTS and CROSS_FOLD == "odd":
        return 0.5 * (causal - acausal)
    return 0.5 * (causal + acausal)


def select_lag(causal, acausal, src_is_first, lag, comp=None):
    """Pick the trace for the requested lag convention, oriented for virtual source `src`.

    'sym'     -> parity-correct fold (fold_sym): (c+a)/2 for even comps, (c-a)/2 for the
                 odd vertical-radial cross terms. Order-independent, best SNR.
    'causal'  -> source->receiver one-sided gather (reciprocity picks the correct side).
    'acausal' -> receiver->source one-sided gather (the opposite branch).
    """
    if lag == "sym":
        folded = fold_sym(comp, causal, acausal)
        # fold_sym references the odd (c-a)/2 to the FIRST station in the file. A virtual
        # source is the second station in ~half its pairs; because the cross term is
        # antisymmetric, swapping source<->receiver flips its sign. Orient every trace to
        # the source so the gather stacks coherently (even comps are orientation-free).
        if comp in ODD_LAG_COMPONENTS and CROSS_FOLD == "odd" and not src_is_first:
            folded = -folded
        return folded
    sr = causal if src_is_first else acausal   # source -> receiver
    return sr if lag == "causal" else (acausal if src_is_first else causal)


def build_source_gathers(index, src, selected, dtype, comps, polarity_fix,
                         min_offset, min_ngood, lag="sym"):
    """
    Assemble virtual-source gathers for `src` for ALL requested components in a single disk
    pass (each pair file opened once). Returns dict comp -> gather, where each gather is
    {src, codes, x [km], sym (n_rx, nt_sym), dt, t_sym}. Comps with no traces are omitted.
    The `lag` convention (sym|causal|acausal) is resolved per pair using the file ordering so
    that 'causal' is always the source->receiver wavefield (exploiting CC reciprocity).
    """
    acc = {c: {"rows": [], "x": [], "codes": [], "dt": None, "nt": None} for c in comps}
    for fpath, other in index.get(src, []):
        if other == src or other not in selected:
            continue
        a, _ = parse_pair(fpath)
        src_is_first = (src == a)
        comp_data = read_pair_lags(fpath, dtype, comps, polarity_fix)
        for comp, (causal, acausal, dist, dt, ngood) in comp_data.items():
            if dist < min_offset or ngood < min_ngood:
                continue
            trace = select_lag(causal, acausal, src_is_first, lag, comp)
            a_ = acc[comp]
            if a_["dt"] is None:
                a_["dt"], a_["nt"] = dt, trace.shape[0]
            elif abs(dt - a_["dt"]) > 1e-9 or trace.shape[0] != a_["nt"]:
                logger.warning("Inconsistent dt/npts for %s-%s (%s); skipping.", src, other, comp)
                continue
            a_["rows"].append(trace)
            a_["x"].append(dist)
            a_["codes"].append(other)

    gathers = {}
    for comp, a in acc.items():
        if a["dt"] is None or not a["rows"]:
            continue
        offsets = np.asarray(a["x"])
        order = np.argsort(offsets)
        gathers[comp] = {
            "src": src,
            "codes": [a["codes"][i] for i in order],
            "x": offsets[order],
            "sym": np.asarray(a["rows"])[order],
            "dt": a["dt"],
            "t_sym": np.arange(a["nt"]) * a["dt"],
        }
    return gathers


# ---------------------------------------------------------------------------
# Phase-shift transform
# ---------------------------------------------------------------------------
def setup_freq_axis(nt_sym, dt, nfft_arg):
    nfft = max(next_fast_len(nt_sym), nfft_arg or 0)
    f = rfftfreq(nfft, dt)
    return nfft, f


def phase_shift_image(sym, x, dt, f_mask_idx, nfft, vel, amp_mode="phase_only"):
    """
    Compute the complex phase-shift dispersion image E(f, v) for one gather.

    Parameters
    ----------
    sym : (n_rx, nt_sym) folded cross-correlations
    x   : (n_rx,) offsets [km]
    f_mask_idx : indices into the rfft frequency axis to keep
    vel : (nv,) trial phase velocities [km/s]

    Returns
    -------
    E : (nv, nf) complex array, nf = len(f_mask_idx)
    """
    spec = rfft(sym, n=nfft, axis=1)[:, f_mask_idx]  # (n_rx, nf)
    amp = np.abs(spec)
    if amp_mode == "phase_only":
        P = np.where(amp > EPS, spec / np.maximum(amp, EPS), 0.0)
    elif amp_mode == "sqrt":
        P = np.where(amp > EPS, spec / np.sqrt(np.maximum(amp, EPS)), 0.0)
    elif amp_mode == "raw":
        P = spec
    else:
        raise ValueError(f"Unknown amp_mode: {amp_mode}")

    f = rfftfreq(nfft, dt)[f_mask_idx]       # (nf,)
    twopi_fx = 2.0 * np.pi * np.outer(x, f)  # (n_rx, nf)
    E = np.empty((vel.size, f.size), dtype=np.complex128)
    for iv, v in enumerate(vel):
        E[iv] = np.sum(np.exp(1j * twopi_fx / v) * P, axis=0)
    return E


def normalize_per_freq(amp):
    """Normalise each frequency column to its max (so the ridge is visible across the band)."""
    peak = np.max(amp, axis=0, keepdims=True)
    return amp / np.where(peak > EPS, peak, 1.0)


# ---------------------------------------------------------------------------
# Cross-source stacking
# ---------------------------------------------------------------------------
def init_stack_state(nv, nf):
    return {
        "phasor": np.zeros((nv, nf), dtype=np.complex128),  # sum E/|E|
        "amp": np.zeros((nv, nf)),                          # sum |E| (per-freq normalised)
        "root": np.zeros((nv, nf)),                         # sum |E|^(1/n)
        "ns": np.zeros((nv, nf)),                           # coverage count
    }


def accumulate(state, E, root_n):
    amp = np.abs(E)
    state["phasor"] += np.where(amp > EPS, E / np.maximum(amp, EPS), 0.0)
    anorm = normalize_per_freq(amp)
    state["amp"] += anorm
    state["root"] += anorm ** (1.0 / root_n)
    state["ns"] += amp > EPS


def finalize_stacks(state, pws_power, root_n):
    ns = np.maximum(state["ns"], 1.0)
    linear = state["amp"] / ns
    coherence = np.abs(state["phasor"]) / ns          # in [0, 1]
    pws = (coherence ** pws_power) * linear
    complex_pws = coherence * linear
    root = (state["root"] / ns) ** root_n
    return {
        "linear": linear,
        "pws": pws,
        "complex_pws": complex_pws,
        "root": root,
        "coverage": state["ns"],
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _disp_axes(f, vel, xaxis):
    """Return (xvals, xlabel) and a sensible x-limit for the dispersion image."""
    if xaxis == "period":
        with np.errstate(divide="ignore"):
            xv = np.where(f > 0, 1.0 / f, np.nan)
        return xv, "Period [s]"
    return f, "Frequency [Hz]"


_NO_REFERENCE = False  # set to True by --no-reference flag


def _ref_phase_curve(component):
    """Reference phase-velocity curve (fundamental mode) for the overlay, or None."""
    if _NO_REFERENCE or reference_curves is None:
        return None
    wave = "love" if component.upper() in ("TT", "RT", "TR", "TZ", "ZT") else "rayleigh"
    try:
        ref = reference_curves()
        per, vel = ref[(wave, "phase", 0)]
        if len(per):
            return wave, np.asarray(per), np.asarray(vel)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Reference curve unavailable: %s", exc)
    return None


def _overlay_reference(ax, ref, xaxis, vlim):
    if ref is None:
        return
    wave, per, vel = ref
    xv = per if xaxis == "period" else 1.0 / per
    m = (vel >= vlim[0]) & (vel <= vlim[1])
    ax.plot(xv[m], vel[m], "w--", lw=1.4, label=f"{wave} ref (mode 0)")


def plot_station_map(ax, gather, coords, basemap_name, map_pad, use_map):
    """Map of the source (star) and its receivers (triangles) on a topo/OSM basemap."""
    src = gather["src"]
    rx_codes = gather["codes"]
    pts = [coords[c] for c in rx_codes if c in coords]
    if src in coords:
        slon, slat = coords[src]
    elif pts:
        slon, slat = np.mean(pts, axis=0)
    else:
        ax.text(0.5, 0.5, "no coords", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    lons = [p[0] for p in pts] + [slon]
    lats = [p[1] for p in pts] + [slat]
    lon0, lon1 = min(lons), max(lons)
    lat0, lat1 = min(lats), max(lats)
    dlon = (lon1 - lon0) or 0.02
    dlat = (lat1 - lat0) or 0.02
    padx, pady = map_pad * dlon, map_pad * dlat
    ax.set_xlim(lon0 - padx, lon1 + padx)
    ax.set_ylim(lat0 - pady, lat1 + pady)

    if pts:
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], marker="v", s=24,
                   c="k", edgecolors="white", linewidths=0.4, zorder=5, label="receivers")
    ax.scatter([slon], [slat], marker="*", s=240, c="red", edgecolors="white",
               linewidths=0.6, zorder=6, label="source")

    if use_map:
        try:
            import contextily as cx

            provider = cx.providers
            for part in basemap_name.split("."):
                provider = getattr(provider, part)
            cx.add_basemap(ax, crs="EPSG:4326", source=provider, attribution_size=4)
        except Exception as exc:  # noqa: BLE001 - offline / tile failure -> plain scatter
            logger.warning("Basemap fetch failed (%s); using plain scatter.", exc)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("auto")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.8)


def plot_wiggle_gather(ax, t, x, sym, tmax, dt, fmin, fmax):
    """Variable-area wiggle plot of an offset-sorted gather (offset on y, lag time on x).

    Traces are band-passed to the analysis band for display so the surface-wave moveout is
    legible (the dispersion transform itself uses the unfiltered spectra)."""
    disp = sym
    try:
        from obspy.signal.filter import bandpass

        disp = np.array([bandpass(tr, fmin, fmax, df=1.0 / dt, corners=4, zerophase=True)
                         for tr in sym])
    except Exception as exc:  # noqa: BLE001
        logger.debug("Display bandpass skipped: %s", exc)
    symn = disp / np.maximum(np.max(np.abs(disp), axis=1, keepdims=True), EPS)
    ntr = x.size
    span = (x.max() - x.min()) or 1.0
    gain = 2.2 * span / max(ntr, 1)  # trace excursion ~ a couple of trace spacings
    win = t <= tmax
    tw = t[win]
    for i in range(ntr):
        trace = x[i] + gain * symn[i, win]
        ax.plot(tw, trace, "k-", lw=0.4)
        ax.fill_between(tw, x[i], trace, where=trace >= x[i], color="k", lw=0, alpha=0.7)
    for vline in (0.5, 1.0, 2.0):
        ax.plot(x / vline, x, "c-", lw=0.8, alpha=0.8)
        ax.text(min(x[-1] / vline, tmax), x[-1], f"{vline:g} km/s",
                color="c", fontsize=7, va="bottom", ha="right")
    pad = gain
    ax.set(xlim=(0, tmax), ylim=(x.min() - pad, x.max() + pad),
           xlabel="Lag time [s]", ylabel="Offset [km]", title="Symmetric gather (wiggle)")


# Per-source figure display settings (diagnostic panels only; the transform itself is unfiltered).
WIGGLE_BAND = (0.2, 5.0)       # Hz, band-pass applied to the wiggle gather for legibility
SOURCE_DISP_FMAX = 2.0         # Hz, upper limit of the per-source dispersion panel (log f axis)


def plot_source_png(outpath, gather, E, f, vel, xaxis, component, coords,
                    basemap_name, map_pad, use_map, tmax_plot):
    """Three-panel per-source figure: map | gather wiggle | dispersion image.

    The wiggle gather is band-passed to WIGGLE_BAND; the dispersion panel shows a
    logarithmic frequency axis capped at SOURCE_DISP_FMAX (the `xaxis` argument is
    retained for signature compatibility but the per-source panel is always log-f)."""
    amp = normalize_per_freq(np.abs(E))
    ref = _ref_phase_curve(component)
    x = gather["x"]
    t = gather["t_sym"]
    # Adaptive window: capture the slowest plausible moveout, but never exceed tmax_plot.
    tmax = min(tmax_plot, 1.3 * x.max() / 0.4)

    fig = plt.figure(figsize=(17, 5.2))
    gs = GridSpec(1, 3, width_ratios=[1.0, 1.1, 1.2], wspace=0.28)

    ax_map = fig.add_subplot(gs[0])
    plot_station_map(ax_map, gather, coords, basemap_name, map_pad, use_map)
    ax_map.set_title(f"{gather['src']}  ({len(x)} receivers)")

    ax_g = fig.add_subplot(gs[1])
    plot_wiggle_gather(ax_g, t, x, gather["sym"], tmax, gather["dt"], *WIGGLE_BAND)

    # Dispersion panel: logarithmic frequency axis, capped at SOURCE_DISP_FMAX.
    ax_d = fig.add_subplot(gs[2])
    keep = (f > 0) & (f <= SOURCE_DISP_FMAX)
    fk = f[keep]
    pm = ax_d.pcolormesh(fk, vel, amp[:, keep], cmap="jet", shading="auto")
    _overlay_reference(ax_d, ref, "freq", (vel.min(), vel.max()))
    fig.colorbar(pm, ax=ax_d, label="normalised |E|", pad=0.02)
    ax_d.set_xscale("log")
    ax_d.set(xlabel="Frequency [Hz]", ylabel="Phase velocity [km/s]",
             xlim=(fk.min(), SOURCE_DISP_FMAX), ylim=(vel.min(), vel.max()),
             title="Phase-shift dispersion")
    if ref is not None:
        ax_d.legend(loc="upper right", fontsize=7)

    fig.suptitle(f"Virtual source {gather['src']} | comp {component} | "
                 f"offsets {x.min():.2f}-{x.max():.2f} km", fontsize=11)
    fig.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_source_npz(outdir, comp, gather, E, f, vel, coords):
    """Save per-source gather + dispersion image as a compressed NPZ for later replotting.

    Saved to <outdir>/<comp>/sources/<NET>.<STA>.npz.  Float64 waveforms are cast to float32
    and the complex E image is split into E_real / E_imag (float32) to keep files small and
    avoid portability issues with complex128 in older numpy versions.

    Coordinates are extracted from the `coords` dict (NET.STA -> (lon, lat)) and stored inline
    so that plot_source_gather.py needs no external CSV.
    """
    sdir = os.path.join(outdir, comp, "sources")
    os.makedirs(sdir, exist_ok=True)
    src = gather["src"]
    rx_codes = gather["codes"]

    # Extract source and receiver coordinates (fall back to NaN if missing from CSV).
    src_lon, src_lat = coords.get(src, (np.nan, np.nan))
    rx_lons = np.array([coords.get(c, (np.nan, np.nan))[0] for c in rx_codes])
    rx_lats = np.array([coords.get(c, (np.nan, np.nan))[1] for c in rx_codes])

    np.savez_compressed(
        os.path.join(sdir, f"{src}.npz"),
        sym=gather["sym"].astype(np.float32),
        x=gather["x"],
        t_sym=gather["t_sym"],
        dt=gather["dt"],
        f=f,
        vel=vel,
        E_real=E.real.astype(np.float32),
        E_imag=E.imag.astype(np.float32),
        src=np.str_(src),
        rx_codes=np.array(rx_codes, dtype=object),
        src_lon=src_lon,
        src_lat=src_lat,
        rx_lons=rx_lons,
        rx_lats=rx_lats,
    )


def _continuity_filter(idx_f, vels, scores, neighbor_cols=5, neighbor_dv=0.12, min_neighbors=4):
    """Keep only picks supported by picks at similar velocity in nearby frequency columns.

    A real dispersion branch is laterally continuous: each of its picks has neighbours
    within ±neighbor_dv km/s over the ±neighbor_cols adjacent columns. Salt-and-pepper
    picks from incoherent columns are isolated and get dropped.
    """
    keep = np.zeros(idx_f.size, dtype=bool)
    for i in range(idx_f.size):
        near = (np.abs(idx_f - idx_f[i]) <= neighbor_cols) & (np.abs(idx_f - idx_f[i]) > 0)
        n = np.count_nonzero(near & (np.abs(vels - vels[i]) <= neighbor_dv))
        keep[i] = n >= min_neighbors
    return idx_f[keep], vels[keep], scores[keep]


def extract_picks_topology(raw_img, vel, min_score=0.5):
    """Pick dispersion branches per frequency column by topological persistence (findpeaks).

    Unlike a single-ridge tracker, persistence picking returns EVERY peak whose
    score exceeds `min_score`, so multiple coexisting branches (e.g. fundamental +
    first overtone) are picked simultaneously rather than forcing one curve that
    jumps between modes at an osculation point.

    Each column is normalised to its own max before scoring, so the persistence
    score (0-1) measures peak prominence relative to that column's dynamic range
    and `min_score` means the same thing at every frequency. Because that makes
    noise peaks in incoherent columns score high too, a continuity filter then
    drops picks with no support at similar velocity in adjacent columns.

    raw_img : (nv, nf) RAW (not per-frequency-normalised) stacked image.
    Returns dict of scattered picks {'idx_f', 'vel', 'score'} (1-D, same length).
    """
    from findpeaks import findpeaks
    logging.getLogger("findpeaks").setLevel(logging.WARNING)
    logging.getLogger("findpeaks.stats").setLevel(logging.WARNING)

    A = np.ma.filled(raw_img, 0.0) if np.ma.isMaskedArray(raw_img) else np.asarray(raw_img, float)
    nv, nf = A.shape
    fp = findpeaks(method="topology", verbose="silent")
    max_col_picks = 4   # a column with more "significant" peaks than this is incoherent speckle
    out_j, out_v, out_s = [], [], []
    for j in range(nf):
        X = A[:, j]
        peak = X.max()
        if peak <= EPS:
            continue
        try:
            df = fp.fit(X / peak)["persistence"]
        except Exception as exc:  # noqa: BLE001 - degenerate column
            logger.debug("findpeaks failed on column %d: %s", j, exc)
            continue
        col = []
        for y, score in zip(df["y"], df["score"]):
            iy = int(y)
            if iy <= 0 or iy >= nv - 1 or score < min_score:  # skip image-edge picks
                continue
            col.append((j, float(vel[iy]), float(score)))
        # Real branch structure means 1-3 prominent peaks per column. Many peaks all
        # clearing min_score = incoherent speckle column -> contributes no picks.
        if len(col) > max_col_picks:
            continue
        for j_, v_, s_ in col:
            out_j.append(j_)
            out_v.append(v_)
            out_s.append(s_)
    idx_f, vels, scores = _continuity_filter(
        np.asarray(out_j, dtype=int), np.asarray(out_v), np.asarray(out_s))
    return {"idx_f": idx_f, "vel": vels, "score": scores}


def save_picks(outpath, picks, f, vel, component):
    """Save the topology picks (per stacking method) for later re-plotting / curve fitting.

    Stores the f/period/vel axes plus per-method scattered picks as <key>_f [Hz],
    <key>_period [s], <key>_vel [km/s], <key>_score (persistence, 0-1)."""
    f = np.asarray(f, dtype=float)
    with np.errstate(divide="ignore"):
        period = np.where(f > 0, 1.0 / f, np.nan)
    arrays = {"f": f, "period": period, "vel": np.asarray(vel), "component": component}
    for key, pk in picks.items():
        pf = f[pk["idx_f"]]
        arrays[f"{key}_f"] = pf
        with np.errstate(divide="ignore"):
            arrays[f"{key}_period"] = np.where(pf > 0, 1.0 / pf, np.nan)
        arrays[f"{key}_vel"] = pk["vel"]
        arrays[f"{key}_score"] = pk["score"]
    os.makedirs(os.path.dirname(os.path.abspath(outpath)), exist_ok=True)
    np.savez(outpath, **arrays)
    logger.info("Saved topology picks -> %s", outpath)


def plot_stack_grid(outpath, stacks, f, vel, xaxis, component, n_sources, min_sources,
                    xscale="linear", xlim=None, title_suffix="", overlay_ridge=True,
                    pick_min_score=0.5, picks_cache=None):
    """2x2 grid comparing the four stacking strategies, with topology picks overlaid.

    Multi-branch picking: every persistence peak with score >= pick_min_score is shown
    (marker size scales with score), so coexisting modes appear as separate point clouds.

    xscale : 'linear' or 'log' for the x-axis. xlim : optional (lo, hi) in the x-units
    (Hz for xaxis='freq', s for xaxis='period'). picks_cache : optional dict of picks
    from a previous call (identical across axis variants) to skip recomputation.
    Returns dict[key] = {'idx_f','vel','score'} scattered picks for saving."""
    xv, xlabel = _disp_axes(f, vel, xaxis)
    ref = _ref_phase_curve(component)
    mask = stacks["coverage"] < min_sources

    panels = [("linear", "Linear mean"),
              ("pws", "Phase-weighted (PWS)"),
              ("complex_pws", "Complex-PWS"),
              ("root", "Nth-root")]

    picks = picks_cache if picks_cache is not None else {}
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True, sharey=True)
    order = np.argsort(xv) if xaxis == "period" else None
    for ax, (key, title) in zip(axes.ravel(), panels):
        raw_img = np.ma.masked_array(stacks[key], mask=mask)
        img = normalize_per_freq(raw_img)       # display only
        if order is not None:
            pm = ax.pcolormesh(xv[order], vel, img[:, order], cmap="jet", shading="auto")
        else:
            pm = ax.pcolormesh(xv, vel, img, cmap="jet", shading="auto")
        _overlay_reference(ax, ref, xaxis, (vel.min(), vel.max()))

        # Topology picking on the RAW image — not the display-normalised one.
        if key not in picks:
            picks[key] = extract_picks_topology(raw_img, vel, pick_min_score)
        pk = picks[key]
        if overlay_ridge and pk["idx_f"].size:
            # Marker size scales with persistence score within [min_score, 1].
            s = 4 + 26 * (pk["score"] - pick_min_score) / max(1 - pick_min_score, EPS)
            ax.scatter(xv[pk["idx_f"]], pk["vel"], s=s, c="white", edgecolors="k",
                       linewidths=0.3, alpha=0.8, zorder=5,
                       label=f"topology picks (score>={pick_min_score:g})")

        fig.colorbar(pm, ax=ax, label="normalised", pad=0.02)
        ax.set(title=title, ylim=(vel.min(), vel.max()))
        ax.set_xscale(xscale)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ax in axes[-1]:
            ax.set_xlabel(xlabel)
        if ax in axes[:, 0]:
            ax.set_ylabel("Phase velocity [km/s]")
    axes[0, 0].legend(loc="upper right", fontsize=8)

    fig.suptitle(f"Stacked phase-shift dispersion | comp {component} | "
                 f"{n_sources} virtual sources{title_suffix}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote stacked dispersion image -> %s", outpath)
    return picks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stackdir", help="STACK_* directory with per-station subdirs of pair .h5 files")
    p.add_argument("--component", required=True,
                   help="Component(s) to image, comma-separated, e.g. 'ZZ' or 'ZZ,RZ,ZR,RR,TT'. "
                        "All requested comps share one disk pass (each pair file read once).")
    p.add_argument("--stations", default="all",
                   help="Comma list | file (one NET.STA per line) | 'all'")
    p.add_argument("--station-csv", default=None,
                   help="stations csv for map coords (default: <stackdir>/../stations_nodes_noisepy.csv)")
    p.add_argument("--stack-method", default="Allstack_linear",
                   choices=["Allstack_linear", "Allstack_pws", "Allstack_robust",
                            "Allstack_nroot", "Allstack_auto_covariance"],
                   help="Which on-disk stack to read (default: Allstack_linear = raw Green's fn)")
    p.add_argument("--freqmin", type=float, default=None, help="Min frequency [Hz]")
    p.add_argument("--freqmax", type=float, default=None, help="Max frequency [Hz]")
    p.add_argument("--vmin", type=float, default=0.2, help="Min phase velocity [km/s]")
    p.add_argument("--vmax", type=float, default=4.5, help="Max phase velocity [km/s]")
    p.add_argument("--dv", type=float, default=0.01, help="Velocity step [km/s]")
    p.add_argument("--nfft", type=int, default=None, help="Zero-pad FFT length (densify freq axis)")
    p.add_argument("--vave", type=float, default=3.0, help="Average velocity for auto freq band [km/s]")
    p.add_argument("--xaxis", choices=["freq", "period"], default="freq", help="Display x-axis")
    p.add_argument("--amp-mode", choices=["phase_only", "sqrt", "raw"], default="phase_only",
                   help="Amplitude handling per trace (phase_only = Cheng)")
    p.add_argument("--lag", choices=["sym", "causal", "acausal"], default="sym",
                   help="Lag convention per gather: sym=folded (default, best SNR); "
                        "causal=source->receiver one-sided gather (uses CC reciprocity); "
                        "acausal=receiver->source one-sided gather")
    p.add_argument("--image-stack", default="pws",
                   choices=["linear", "pws", "complex_pws", "root"],
                   help="(informational) headline stacking method; all four are always plotted")
    p.add_argument("--pws-power", type=float, default=2.0, help="PWS sharpness exponent nu")
    p.add_argument("--root-n", type=float, default=2.0, help="Nth-root stack order")
    p.add_argument("--min-offset", type=float, default=None,
                   help="Drop traces with offset below this [km] (near-field cut; default auto)")
    p.add_argument("--min-receivers", type=int, default=5,
                   help="Skip a source with fewer valid receivers")
    p.add_argument("--min-ngood", type=int, default=1, help="Skip pairs with ngood below this")
    p.add_argument("--min-sources", type=int, default=3,
                   help="Mask stacked cells covered by fewer sources")
    p.add_argument("--pick-min-score", type=float, default=0.5,
                   help="Topology picking: min persistence score (0-1) for a peak to be "
                        "kept; picks all branches above this (default 0.5)")
    p.add_argument("--tmax-plot", type=float, default=40.0, help="Max lag shown in gather panel [s]")
    p.add_argument("--no-polarity-fix", action="store_true",
                   help="Disable the cross-network (CH/RI) sign flip")
    p.add_argument("--cross-fold", choices=["even", "odd"], default="even",
                   help="Lag fold for the vertical-radial cross comps (ZR/RZ): 'even' = "
                        "additive (c+a)/2 (default, best for directional ambient fields); "
                        "'odd' = (c-a)/2 (textbook J1 antisymmetric fold, isotropic fields)")
    p.add_argument("--no-reference", action="store_true",
                   help="Do not overlay the reference dispersion curve")
    p.add_argument("--no-per-source", action="store_true", help="Skip per-source PNGs")
    p.add_argument("--save-sources", action="store_true",
                   help="Save per-source gather + dispersion image as NPZ "
                        "(written to <output-dir>/<comp>/sources/<STA>.npz)")
    p.add_argument("--no-map", action="store_true", help="Skip basemap tiles (plain scatter)")
    p.add_argument("--basemap", default="OpenTopoMap", help="contextily provider name")
    p.add_argument("--map-pad", type=float, default=0.15, help="Map bbox padding fraction")
    p.add_argument("--output-dir", default="phaseshift_out",
                   help="Root output dir; results go to <output-dir>/<comp>/")
    p.add_argument("--output-plot", default=None,
                   help="Stacked PNG path (single-component runs only; overrides default)")
    p.add_argument("--output-data", default=None,
                   help="Stacked NPZ path (single-component runs only; overrides default)")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    configure_logging(args.log_level)

    global _NO_REFERENCE, CROSS_FOLD
    _NO_REFERENCE = args.no_reference
    CROSS_FOLD = args.cross_fold

    stackdir = os.path.abspath(args.stackdir)
    comps = [c.strip().upper() for c in args.component.split(",") if c.strip()]
    dtype = args.stack_method
    polarity_fix = not args.no_polarity_fix
    single = len(comps) == 1

    # Station coordinates (for the map and for the coordinate-derived frequency band).
    csv = args.station_csv or os.path.join(os.path.dirname(stackdir), "stations_nodes_noisepy.csv")
    coords = {}
    if os.path.isfile(csv):
        coords = load_station_csv(csv)
        logger.info("Loaded %d station coords from %s", len(coords), csv)
    else:
        logger.warning("Station csv not found (%s); maps -> scatter only.", csv)

    # One-time file index (replaces per-source globbing).
    index = build_file_index(stackdir)
    selected = resolve_station_list(args.stations, set(index.keys()))
    sources = sorted(selected)
    logger.info("%d components %s | %d stations selected as virtual sources.",
                len(comps), comps, len(sources))
    if not sources:
        logger.error("No stations selected; nothing to do.")
        return 1

    # Frequency band & near-field cut from station coordinates (no file I/O).
    off = estimate_offsets(coords, selected)
    if args.min_offset is not None:
        min_offset = args.min_offset
    elif off.size:
        min_offset = float(np.percentile(off, 2))
        logger.info("Auto near-field cut --min-offset = %.3f km.", min_offset)
    else:
        min_offset = 0.0
    if off.size:
        omin, omax = max(off.min(), 1e-6), off.max()
    else:
        omin, omax = None, None
    fmin = args.freqmin if args.freqmin is not None else (args.vave / omax if omax else None)
    fmax = args.freqmax if args.freqmax is not None else (args.vave / omin if omin else None)
    if fmin is None or fmax is None:
        logger.error("Cannot derive frequency band: provide --freqmin/--freqmax or a station csv.")
        return 1
    vel = np.arange(args.vmin, args.vmax + args.dv, args.dv)

    # Per-component accumulators, initialised lazily once dt/npts are known from the first gather.
    grid = {}  # filled with nfft, f_full, f_mask_idx, f, dt on first gather
    states = {c: None for c in comps}
    n_used = {c: 0 for c in comps}

    def comp_dir(c):
        d = os.path.join(args.output_dir, c)
        os.makedirs(d, exist_ok=True)
        return d

    for src in sources:
        gathers = build_source_gathers(index, src, selected, dtype, comps,
                                       polarity_fix, min_offset, args.min_ngood, args.lag)
        if not gathers:
            continue
        # Lazily fix the frequency grid the first time we see real data.
        if not grid:
            any_g = next(iter(gathers.values()))
            dt = any_g["dt"]
            nfft, f_full = setup_freq_axis(any_g["sym"].shape[1], dt, args.nfft)
            fmax_eff = min(fmax, 0.5 / dt)  # Nyquist guard
            f_mask_idx = np.where((f_full >= fmin) & (f_full <= fmax_eff))[0]
            if f_mask_idx.size == 0:
                logger.error("Empty frequency band [%g, %g] Hz.", fmin, fmax_eff)
                return 1
            grid = {"dt": dt, "nfft": nfft, "f_mask_idx": f_mask_idx, "f": f_full[f_mask_idx]}
            logger.info("Grid: %d freqs [%.3g-%.3g Hz], %d vels [%.2f-%.2f km/s], nfft=%d",
                        grid["f"].size, fmin, fmax_eff, vel.size, args.vmin, args.vmax, nfft)

        for comp, g in gathers.items():
            if g["x"].size < args.min_receivers:
                continue
            if states[comp] is None:
                states[comp] = init_stack_state(vel.size, grid["f"].size)
            E = phase_shift_image(g["sym"], g["x"], grid["dt"], grid["f_mask_idx"],
                                  grid["nfft"], vel, args.amp_mode)
            accumulate(states[comp], E, args.root_n)
            n_used[comp] += 1
            if args.save_sources:
                save_source_npz(args.output_dir, comp, g, E, grid["f"], vel, coords)
            if not args.no_per_source:
                sdir = os.path.join(comp_dir(comp), "sources")
                os.makedirs(sdir, exist_ok=True)
                png = os.path.join(sdir, f"{g['src']}.png")
                plot_source_png(png, g, E, grid["f"], vel, args.xaxis, comp, coords,
                                args.basemap, args.map_pad, not args.no_map, args.tmax_plot)

    if not grid:
        logger.error("No source gather met the selection / --min-receivers=%d.",
                     args.min_receivers)
        return 1

    rc = 0
    for comp in comps:
        if states[comp] is None:
            logger.warning("Component %s: no usable sources; skipped.", comp)
            rc = 1
            continue
        stacks = finalize_stacks(states[comp], args.pws_power, args.root_n)
        cdir = comp_dir(comp)
        stack_png = (args.output_plot if (single and args.output_plot)
                     else os.path.join(cdir, f"stacked_{comp}.png"))
        os.makedirs(os.path.dirname(os.path.abspath(stack_png)), exist_ok=True)
        picks = plot_stack_grid(stack_png, stacks, grid["f"], vel, args.xaxis, comp,
                                n_used[comp], args.min_sources,
                                pick_min_score=args.pick_min_score)
        npz = (args.output_data if (single and args.output_data)
               else os.path.join(cdir, f"stacks_{comp}.npz"))
        os.makedirs(os.path.dirname(os.path.abspath(npz)), exist_ok=True)
        np.savez(npz, f=grid["f"], vel=vel, component=comp, n_sources=n_used[comp],
                 **{k: np.asarray(v) for k, v in stacks.items()})
        save_picks(os.path.join(cdir, f"picks_{comp}.npz"), picks, grid["f"], vel, comp)
        logger.info("Component %s: %d sources -> %s", comp, n_used[comp], npz)

    return rc


if __name__ == "__main__":
    sys.exit(main())
