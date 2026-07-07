#!/usr/bin/env python
"""
Re-plot per-source 3-panel figures (map | wiggle gather | dispersion image) from the NPZ
files saved by phaseshift_dispersion.py --save-sources.

This lets you generate or regenerate individual source figures instantly — no re-reading of
the original h5 cross-correlation files needed.

Examples
--------
# Plot one station, one component:
    python plot_source_gather.py ~/Data/aargau/phasevelocity_VSG \\
        --station AA.3006384 --component ZZ

# Plot all sources for two components:
    python plot_source_gather.py ~/Data/aargau/phasevelocity_VSG \\
        --component ZZ,TT --all

# Period axis, no map tiles (offline):
    python plot_source_gather.py ~/Data/aargau/phasevelocity_VSG \\
        --station AA.3006384 --component ZZ --xaxis period --no-map

# Write PNGs to a different directory:
    python plot_source_gather.py ~/Data/aargau/phasevelocity_VSG \\
        --station AA.3006384 --component ZZ --outdir /tmp/figs
"""

import argparse
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from phaseshift_dispersion import (  # noqa: E402
    configure_logging,
    load_station_csv,
    plot_source_png,
)

import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
def load_source_npz(npz_path):
    """Load a per-source NPZ and reconstruct the gather dict and complex E image."""
    d = np.load(npz_path, allow_pickle=True)
    E = d["E_real"].astype(np.float64) + 1j * d["E_imag"].astype(np.float64)
    gather = {
        "src":     str(d["src"]),
        "codes":   list(d["rx_codes"]) if "rx_codes" in d else [],
        "sym":     d["sym"].astype(np.float64),
        "x":       d["x"].astype(np.float64),
        "t_sym":   d["t_sym"].astype(np.float64),
        "dt":      float(d["dt"]),
        "src_lon": float(d["src_lon"]) if "src_lon" in d else np.nan,
        "src_lat": float(d["src_lat"]) if "src_lat" in d else np.nan,
        "rx_lons": d["rx_lons"].astype(np.float64) if "rx_lons" in d else np.array([]),
        "rx_lats": d["rx_lats"].astype(np.float64) if "rx_lats" in d else np.array([]),
    }
    f   = d["f"].astype(np.float64)
    vel = d["vel"].astype(np.float64)
    return gather, E, f, vel


def _build_coords_from_npz(gather):
    """Reconstruct a minimal coords dict from the arrays stored inside the NPZ."""
    coords = {}
    src = gather["src"]
    if not np.isnan(gather["src_lon"]):
        coords[src] = (gather["src_lon"], gather["src_lat"])
    for code, lon, lat in zip(gather["codes"], gather["rx_lons"], gather["rx_lats"]):
        if not np.isnan(lon):
            coords[code] = (float(lon), float(lat))
    return coords


def plot_one(npz_path, outpath, comp, xaxis, freqmax, basemap_name, map_pad,
             use_map, tmax_plot, extra_coords):
    """Load one NPZ, cap the frequency axis, and write a 3-panel PNG."""
    gather, E, f, vel = load_source_npz(npz_path)

    # Optional high-frequency cap (re-slice both f and E columns).
    keep = f <= freqmax
    f  = f[keep]
    E  = E[:, keep]

    # Build coords: NPZ-embedded coordinates take priority; extra_coords fills any gaps.
    coords = _build_coords_from_npz(gather)
    coords.update({k: v for k, v in extra_coords.items() if k not in coords})

    plot_source_png(
        outpath, gather, E, f, vel,
        xaxis=xaxis,
        component=comp,
        coords=coords,
        basemap_name=basemap_name,
        map_pad=map_pad,
        use_map=use_map,
        tmax_plot=tmax_plot,
    )
    logger.info("Wrote %s", outpath)


# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("root",
                   help="Root output directory (contains <comp>/sources/*.npz sub-dirs)")
    p.add_argument("--component", required=True,
                   help="Component(s), comma-separated, e.g. 'ZZ' or 'ZZ,TT'")
    p.add_argument("--station", default=None,
                   help="Single station to plot, e.g. 'AA.3006384'. "
                        "Use --all to process every NPZ found.")
    p.add_argument("--all", action="store_true",
                   help="Plot all sources found in <comp>/sources/")
    p.add_argument("--xaxis", choices=["freq", "period"], default="freq",
                   help="Horizontal axis type (default: freq)")
    p.add_argument("--freqmax", type=float, default=5.0,
                   help="Cap the frequency axis at this value [Hz] (default: 5.0)")
    p.add_argument("--tmax-plot", type=float, default=40.0,
                   help="Max lag shown in wiggle gather [s] (default: 40)")
    p.add_argument("--no-map", action="store_true",
                   help="Skip basemap tile fetch (plain scatter fallback)")
    p.add_argument("--basemap", default="OpenTopoMap",
                   help="contextily provider name (default: OpenTopoMap)")
    p.add_argument("--map-pad", type=float, default=0.15,
                   help="Bbox padding fraction for the map panel (default: 0.15)")
    p.add_argument("--station-csv", default=None,
                   help="Optional CSV with extra station coordinates "
                        "(supplements coords stored inside the NPZ)")
    p.add_argument("--outdir", default=None,
                   help="Directory for output PNGs (default: same as source NPZ dir)")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    configure_logging(args.log_level)

    if not args.station and not args.all:
        p.error("Specify --station <NET.STA> or --all")

    # Optional extra coords from an external CSV.
    extra_coords = {}
    if args.station_csv and os.path.isfile(args.station_csv):
        extra_coords = load_station_csv(args.station_csv)
        logger.info("Loaded %d extra coords from %s", len(extra_coords), args.station_csv)

    comps = [c.strip().upper() for c in args.component.split(",") if c.strip()]
    root  = os.path.abspath(args.root)

    n_ok = n_skip = 0
    for comp in comps:
        src_dir = os.path.join(root, comp, "sources")
        if not os.path.isdir(src_dir):
            logger.warning("Sources dir not found: %s", src_dir)
            continue

        if args.all:
            npz_files = sorted(glob.glob(os.path.join(src_dir, "*.npz")))
        else:
            sta = args.station
            npz_files = [os.path.join(src_dir, f"{sta}.npz")]

        for npz_path in npz_files:
            if not os.path.isfile(npz_path):
                logger.warning("NPZ not found: %s", npz_path)
                n_skip += 1
                continue

            sta_code = os.path.splitext(os.path.basename(npz_path))[0]
            out_dir  = args.outdir or src_dir
            os.makedirs(out_dir, exist_ok=True)
            png_path = os.path.join(out_dir, f"{sta_code}_{comp}.png")

            try:
                plot_one(npz_path, png_path, comp, args.xaxis, args.freqmax,
                         args.basemap, args.map_pad, not args.no_map,
                         args.tmax_plot, extra_coords)
                n_ok += 1
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed %s/%s: %s", comp, sta_code, exc)
                n_skip += 1

    logger.info("Done: %d plotted, %d skipped/failed.", n_ok, n_skip)
    return 0 if n_skip == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
