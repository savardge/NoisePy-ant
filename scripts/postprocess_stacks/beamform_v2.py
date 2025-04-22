#!/usr/bin/env python3
"""
beamforming.py

Compute noise source directivity via beamforming on ambient noise cross-correlation stacks.

Inspired by:
Bowden, D. C., Sager, K., Fichtner, A., & Chmiel, M. (2021). Connecting beamforming and
kernel-based noise source inversion. Geophysical Journal International, 224(3), 1607–1620.

Usage:
    python beamforming.py input_ncf.npz \
        --sl 0.75 --nux 121 --nuy 121 \
        --freqmin 0.02 --freqmax 0.5 \
        --output-beam beam.npz \
        --output-plot beam.png \
        [--no-plot] \
        [--use-argmin] \
        [--corners 4] \
        [--log-level DEBUG]
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from obspy.signal.filter import bandpass


def configure_logging(level: str) -> None:
    """Configure root logger."""
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(format=fmt, level=getattr(logging, level.upper()))


def load_ncf(ncf_file: Path) -> dict:
    """Load ambient noise cross-correlation stack data from NPZ file."""
    data = np.load(ncf_file)
    return {
        'az': data['azimuth'],
        'baz': data['backazimuth'],
        'r': data['r'],
        'ccf': data['ncts'],
        'dt': data['dt'].item(),
        't': data['t']
    }


def apply_bandpass(ccf: np.ndarray, dt: float, freqmin: float, freqmax: float,
                   corners: int = 4, zerophase: bool = True) -> np.ndarray:
    """Bandpass-filter each cross-correlation trace."""
    ccf_f = np.empty_like(ccf, dtype=np.float32)
    df = 1.0 / dt
    for i in range(ccf.shape[0]):
        ccf_f[i, :] = bandpass(
            ccf[i, :], freqmin, freqmax, df=df,
            corners=corners, zerophase=zerophase
        )
    return ccf_f


def setup_slowness_grid(sl: float, nux: int, nuy: int) -> tuple:
    """Create slowness grid vectors and spacings."""
    ux = np.linspace(-sl, sl, nux)
    uy = np.linspace(-sl, sl, nuy)
    dux = ux[1] - ux[0]
    duy = uy[1] - uy[0]
    return ux, uy, dux, duy


def compute_beam(ccf_filt: np.ndarray, az: np.ndarray, r: np.ndarray,
                 ux: np.ndarray, uy: np.ndarray, t: np.ndarray,
                 use_argmin: bool = False) -> np.ndarray:
    """Compute beam power for positive lags (azimuth-based)."""
    # Precompute geometry factors
    cosaz_r = np.cos(np.deg2rad(az)) * r
    sinaz_r = np.sin(np.deg2rad(az)) * r

    # Time grid parameters
    dt = t[1] - t[0]
    t0 = t[0]
    npairs = az.size

    # Build 2D grid of slowness combos
    nux, nuy = ux.size, uy.size
    uxg, uyg = np.meshgrid(ux, uy, indexing='xy')
    ux_flat = uxg.ravel()
    uy_flat = uyg.ravel()

    # Compute effective times: shape (ngrids, npairs)
    time_eff = -(np.outer(ux_flat, np.cos(np.deg2rad(az)) * r) +
                 np.outer(uy_flat, np.sin(np.deg2rad(az)) * r))

    # Convert to sample indices
    if not use_argmin:
        idx = np.rint((time_eff - t0) / dt).astype(int)
        idx = np.clip(idx, 0, t.size - 1)
    else:
        idx = np.abs(time_eff[:, :, None] - t[None, None, :]).argmin(axis=2)

    # Gather and sum beam contributions
    rows = np.arange(npairs)[None, :]
    vals = ccf_filt[rows, idx]
    beam_flat = vals.sum(axis=1)

    # Reshape back to grid
    beam = beam_flat.reshape((nux, nuy))
    return beam


def plot_beam(beam: np.ndarray, ux: np.ndarray, uy: np.ndarray,
              dux: float, duy: float, title: str, outfile: Path) -> None:
    """Plot beam power map and save figure."""
    fig, ax = plt.subplots(figsize=(10, 8))
    mesh_x, mesh_y = np.meshgrid(ux, uy, indexing='xy')
    im = ax.pcolormesh(mesh_x - dux / 2,
                       mesh_y - duy / 2,
                       beam.T,
                       cmap='inferno', shading='auto')
    ax.set_aspect('equal')
    ax.set_xlabel('Slowness E–W [s/km]')
    ax.set_ylabel('Slowness N–S [s/km]')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='Beam Power')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Beamforming on ambient noise cross-correlation stacks"
    )
    parser.add_argument('ncf_file', type=Path,
                        help='Input NPZ with cross-correlation stacks')
    parser.add_argument('--sl', type=float, default=0.75,
                        help='Max absolute slowness [s/km]')
    parser.add_argument('--nux', type=int, default=121,
                        help='Grid points in X slowness')
    parser.add_argument('--nuy', type=int, default=121,
                        help='Grid points in Y slowness')
    parser.add_argument('--corners', type=int, default=4,
                        help='Filter corners for bandpass')
    parser.add_argument('--use-argmin', action='store_true',
                        help='Use argmin instead of uniform dt index')
    parser.add_argument('--freqmin', type=float, default=None,
                        help='Minimum frequency for bandpass [Hz]')
    parser.add_argument('--freqmax', type=float, default=None,
                        help='Maximum frequency for bandpass [Hz]')
    parser.add_argument('--no-plot', action='store_true',
                        help='Suppress figure output')
    parser.add_argument('--output-beam', type=Path,
                        default=None, help='NPZ to save beam power')
    parser.add_argument('--output-plot', type=Path,
                        default=None, help='PNG figure filename')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    args = parser.parse_args()

    configure_logging(args.log_level)
    data = load_ncf(args.ncf_file)

    # Determine bandpass frequencies
    if args.freqmin is not None and args.freqmax is not None:
        freqmin = args.freqmin
        freqmax = args.freqmax
    else:
        vs_ave = 3.0
        freqmin = 1.0 / (np.max(data['r']) / vs_ave)
        freqmax = 1.0 / (np.min(data['r']) / vs_ave)
    logging.info(f"Filtering between {freqmin:.3f}–{freqmax:.3f} Hz")

    ccf_filt = apply_bandpass(
        data['ccf'], data['dt'], freqmin, freqmax,
        corners=args.corners, zerophase=True
    )

    ux, uy, dux, duy = setup_slowness_grid(args.sl, args.nux, args.nuy)
    beam = compute_beam(
        ccf_filt, data['az'], data['r'], ux, uy,
        data['t'], use_argmin=args.use_argmin
    )

    # Save beam
    out_beam = args.output-beam or args.ncf_file.with_suffix('_beam.npz')
    np.savez(out_beam, beam=beam, ux=ux, uy=uy)
    logging.info(f"Beam power saved to {out_beam}")

    # Optional plot
    if not args.no-plot:
        out_fig = args.output-plot or out_beam.with_suffix('.png')
        plot_beam(beam, ux, uy, dux, duy,
                  title=args.ncf_file.name,
                  outfile=out_fig)
        logging.info(f"Figure saved to {out_fig}")


if __name__ == '__main__':
    main()
