#!/usr/bin/env python
"""
Azimuth-binned 2-psi (Smith-Dahlen) anisotropy test on slant-stack phase velocities.

Reads the per-source gather NPZs written by phaseshift_dispersion.py --save-sources,
pools all unique station pairs, bins them by inter-station azimuth psi (mod 180 deg),
and slant-stacks WITHIN each azimuth bin (phase-shift transform at a fixed frequency)
to obtain an azimuthally-resolved phase velocity c(T, psi). At each target period it
fits the Smith-Dahlen relation

    c(T, psi) = c0(T) * [ 1 + a(T) * cos( 2 (psi - phi(T)) ) ]

(Rayleigh azimuthal anisotropy is 2-psi dominated; Love would be 4-psi). Per-bin
uncertainty is estimated by bootstrap resampling of the pairs in the bin, giving the
inter-measurement scatter against which the 2-psi amplitude must be resolved. A 4-psi
fit and an F-test (2-psi vs isotropic constant) are reported as cross-checks.

This is the two-station / array realisation of the framework used by Lin & Schmandt
(2014) and the classic Smith & Dahlen (1973) azimuthal anisotropy parameterisation.

CAVEAT: pooled azimuth bins sample different ray paths, so apparent c(psi) variation
can be lateral heterogeneity aliased into azimuth rather than true anisotropy. Stable
phi across periods, a 2-psi amplitude well above bootstrap scatter, and a small 4-psi
term together argue for anisotropy; this script reports all three so the call is explicit.

Example
-------
    python azimuthal_anisotropy_2psi.py ~/Data/aargau/phasevelocity_VSG \
        --component ZZ --periods 0.5,0.7,1.0,1.4,2.0 --nbins 12 --nboot 200 \
        --ref-csv ~/Data/aargau/phasevelocity_VSG/analysis/reference_curves.csv
"""

import argparse
import glob
import logging
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq, next_fast_len

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
EPS = 1e-12


# ---------------------------------------------------------------------------
def great_circle_azimuth(lon1, lat1, lon2, lat2):
    """Initial bearing (deg, 0=N, 90=E) from point 1 to point 2."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return np.degrees(np.arctan2(x, y)) % 360.0


def load_pairs(root, comp, min_offset, max_offset):
    """Pool unique station pairs from per-source NPZs.

    Returns dict with arrays: x [km], psi [deg, mod 180], and the stacked folded
    traces (n_pairs, nt) plus dt. Each physical pair is kept once (dedup by code set);
    by CC reciprocity the folded trace is identical for either ordering."""
    files = sorted(glob.glob(os.path.join(root, comp, "sources", "*.npz")))
    if not files:
        raise SystemExit(f"No source NPZs under {root}/{comp}/sources/")
    seen = {}
    dt = None
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        src = str(d["src"])
        slon, slat = float(d["src_lon"]), float(d["src_lat"])
        codes = list(d["rx_codes"])
        x = d["x"].astype(np.float64)
        sym = d["sym"].astype(np.float64)
        rlon = d["rx_lons"].astype(np.float64)
        rlat = d["rx_lats"].astype(np.float64)
        if dt is None:
            dt = float(d["dt"])
        for i, rc in enumerate(codes):
            key = frozenset((src, rc))
            if key in seen:
                continue
            if not (min_offset <= x[i] <= max_offset):
                continue
            if np.isnan(slon) or np.isnan(rlon[i]):
                continue
            az = great_circle_azimuth(slon, slat, rlon[i], rlat[i]) % 180.0
            seen[key] = (x[i], az, sym[i])
    xs = np.array([v[0] for v in seen.values()])
    az = np.array([v[1] for v in seen.values()])
    traces = np.array([v[2] for v in seen.values()])
    logger.info("%s: %d unique pairs (offsets %.1f-%.1f km)", comp, xs.size, xs.min(), xs.max())
    return {"x": xs, "psi": az, "traces": traces, "dt": dt}


def phase_only_band(traces, dt, target_freqs, bandwidth=0.15, nband=7):
    """Phase-only spectra U/|U| over a narrow band around each target frequency.

    Band-averaging the slant-stack power across +/- bandwidth*f stabilises the velocity
    estimate (a single FFT bin is noisy). Returns P (n_pairs, n_targets, nband) and the
    band frequencies fbands (n_targets, nband)."""
    nt = traces.shape[1]
    nfft = next_fast_len(nt)
    f = rfftfreq(nfft, dt)
    U = rfft(traces, n=nfft, axis=1)
    P = np.empty((traces.shape[0], len(target_freqs), nband), dtype=complex)
    fbands = np.empty((len(target_freqs), nband))
    for k, ft in enumerate(target_freqs):
        want = np.linspace(ft * (1 - bandwidth), ft * (1 + bandwidth), nband)
        idx = np.array([np.argmin(np.abs(f - w)) for w in want])
        Uk = U[:, idx]
        P[:, k, :] = Uk / np.maximum(np.abs(Uk), EPS)
        fbands[k] = f[idx]
    return P, fbands


def slant_velocity(P_band, x, fbands, vgrid):
    """Band-averaged phase velocity: argmax_v of sum over band of |sum_j e^{i2pi f x/v} P_j|.

    P_band : (n_pairs, nband), fbands : (nband,). Returns (c, power(v), railed_flag)."""
    power = np.zeros(vgrid.size)
    inv_v = 1.0 / vgrid
    for b, fb in enumerate(fbands):
        phase = 2.0 * np.pi * fb * np.outer(x, inv_v)        # (n_pairs, n_v)
        power += np.abs(np.sum(np.exp(1j * phase) * P_band[:, b, None], axis=0))
    j = int(np.argmax(power))
    railed = (j <= 1) or (j >= vgrid.size - 2)               # peak at window edge -> unreliable
    return vgrid[j], power, railed


def fit_n_psi(psi_deg, c, w, n):
    """Weighted LSQ fit c = c0 + A cos(n psi) + B sin(n psi). psi in degrees.

    Returns c0, amp (=sqrt(A^2+B^2)), phi (deg, fast axis for n=2), and the
    weighted residual sum of squares (chi2)."""
    ps = np.radians(psi_deg)
    G = np.column_stack([np.ones_like(ps), np.cos(n * ps), np.sin(n * ps)])
    W = np.diag(w)
    GtW = G.T @ W
    cov = np.linalg.inv(GtW @ G)
    m = cov @ (GtW @ c)
    resid = c - G @ m
    chi2 = float(resid @ W @ resid)
    c0, A, B = m
    amp = np.hypot(A, B)
    phi = 0.5 * np.degrees(np.arctan2(B, A)) % 180.0   # fast axis (n=2)
    # amplitude uncertainty via error propagation from cov(A,B)
    dA, dB = np.sqrt(cov[1, 1]), np.sqrt(cov[2, 2])
    amp_err = np.hypot(A * dA, B * dB) / max(amp, EPS)
    return {"c0": float(c0), "amp": float(amp), "amp_err": float(amp_err),
            "phi": float(phi), "chi2": chi2, "A": float(A), "B": float(B)}


def fit_const(c, w):
    c0 = np.sum(w * c) / np.sum(w)
    chi2 = float(np.sum(w * (c - c0) ** 2))
    return {"c0": float(c0), "chi2": chi2}


# ---------------------------------------------------------------------------
def analyse_period(pairs, P_band, fbands, T, vgrid, nbins, nboot, min_bin, rng):
    """Per-azimuth-bin band-averaged slant velocities + bootstrap scatter at one period.

    vgrid is already centred on the network-mean c for this period, so each bin locks to
    the same branch the full array sees. Edge-railed bins are dropped as unreliable."""
    psi = pairs["psi"]
    x = pairs["x"]
    edges = np.linspace(0, 180, nbins + 1)
    centers, cvals, csig, counts = [], [], [], []
    for b in range(nbins):
        sel = (psi >= edges[b]) & (psi < edges[b + 1])
        nsel = int(np.count_nonzero(sel))
        if nsel < min_bin:
            continue
        xb, Pb = x[sel], P_band[sel]
        c_meas, _, railed = slant_velocity(Pb, xb, fbands, vgrid)
        if railed:
            continue
        boots = np.empty(nboot)
        for k in range(nboot):
            ridx = rng.integers(0, nsel, nsel)
            boots[k], _, _ = slant_velocity(Pb[ridx], xb[ridx], fbands, vgrid)
        centers.append(0.5 * (edges[b] + edges[b + 1]))
        cvals.append(c_meas)
        csig.append(np.std(boots))
        counts.append(nsel)
    return (np.array(centers), np.array(cvals),
            np.maximum(np.array(csig), EPS), np.array(counts))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="phasevelocity_VSG root (contains <comp>/sources/*.npz)")
    p.add_argument("--component", default="ZZ")
    p.add_argument("--periods", default="0.5,0.7,1.0,1.4,2.0",
                   help="Target periods [s], comma-separated")
    p.add_argument("--nbins", type=int, default=12, help="Azimuth bins over 0-180 deg")
    p.add_argument("--nboot", type=int, default=200, help="Bootstrap resamples per bin")
    p.add_argument("--min-bin", type=int, default=20, help="Min pairs to keep an azimuth bin")
    p.add_argument("--min-offset", type=float, default=2.0, help="Near-field cut [km]")
    p.add_argument("--max-offset", type=float, default=25.0, help="Far-offset cut [km]")
    p.add_argument("--vwin", type=float, default=0.35,
                   help="Half-width of per-bin velocity window around network-mean c [km/s]")
    p.add_argument("--vbroad", type=float, default=0.5,
                   help="Half-width of the network-mean search around c_ref [km/s]; keep "
                        "small enough to exclude the overtone and short-period aliasing lobes")
    p.add_argument("--dv", type=float, default=0.005, help="Velocity search step [km/s]")
    p.add_argument("--ref-csv", default=None,
                   help="reference_curves.csv to window the velocity search (fundamental)")
    p.add_argument("--vmid", type=float, default=None,
                   help="Fixed search-window centre [km/s] if no --ref-csv")
    p.add_argument("--outdir", default=None, help="Output dir (default <root>/analysis)")
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args(argv)

    periods = [float(t) for t in args.periods.split(",") if t.strip()]
    target_freqs = np.array([1.0 / T for T in periods])
    outdir = args.outdir or os.path.join(args.root, "analysis")
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Velocity search-window centres per period (fundamental branch).
    vmid = {}
    if args.ref_csv and os.path.isfile(args.ref_csv):
        import pandas as pd
        df = pd.read_csv(args.ref_csv)
        sub = df[(df.branch == "fundamental") & (df.component == args.component)]
        for T in periods:
            i = (sub.period - T).abs().idxmin()
            vmid[T] = float(sub.loc[i, "c_ref"])
        logger.info("Velocity windows from %s", os.path.basename(args.ref_csv))
    else:
        for T in periods:
            vmid[T] = args.vmid if args.vmid else 2.0

    pairs = load_pairs(args.root, args.component, args.min_offset, args.max_offset)
    P, fbands = phase_only_band(pairs["traces"], pairs["dt"], target_freqs)

    # Network-mean c per period from ALL pairs, searched only within +/- vbroad of the
    # fundamental reference. The tight bracket keeps the estimate on the fundamental branch
    # (excludes the overtone at intermediate T) and off the short-period aliasing lobes,
    # so each period's narrow per-bin window self-centres on the correct mode.
    net_c0 = {}
    for k, T in enumerate(periods):
        vbroad = np.arange(vmid[T] - args.vbroad, vmid[T] + args.vbroad + args.dv, args.dv)
        c_net, _, railed = slant_velocity(P[:, k, :], pairs["x"], fbands[k], vbroad)
        net_c0[T] = c_net
        flag = " [railed-edge]" if railed else ""
        logger.info("T=%.2fs: network-mean c=%.3f km/s (ref %.3f)%s", T, c_net, vmid[T], flag)

    results = []
    panels = []   # (T, f_T, cen, cval, csig, f2) for the per-period figure
    for k, T in enumerate(periods):
        f_T = float(fbands[k].mean())
        vgrid = np.arange(net_c0[T] - args.vwin, net_c0[T] + args.vwin + args.dv, args.dv)
        cen, cval, csig, cnt = analyse_period(
            pairs, P[:, k, :], fbands[k], T, vgrid, args.nbins, args.nboot, args.min_bin, rng)
        if cen.size < 5:
            logger.warning("T=%.2fs: only %d usable bins; skipping fit", T, cen.size)
            continue
        w = 1.0 / csig ** 2
        f2 = fit_n_psi(cen, cval, w, 2)
        f4 = fit_n_psi(cen, cval, w, 4)
        fc = fit_const(cval, w)
        # F-test: 2-psi (3 params) vs constant (1 param)
        dof2 = cen.size - 3
        Fstat = ((fc["chi2"] - f2["chi2"]) / 2.0) / (f2["chi2"] / max(dof2, 1))
        peak2peak = 2.0 * f2["amp"]
        anis_pct = 100.0 * f2["amp"] / f2["c0"]
        scatter = float(np.median(csig))
        results.append({"T": T, "f": f_T, "nbins": cen.size, **f2,
                        "anis_pct": anis_pct, "p2p": peak2peak,
                        "scatter": scatter, "F": Fstat,
                        "amp4": f4["amp"], "phi4": f4["phi"], "chi2_4": f4["chi2"]})
        panels.append((T, f_T, cen, cval, csig, f2))
        logger.info("T=%.2fs: c0=%.3f  2psi amp=%.1f%%  phi=%.0f deg  "
                    "(scatter~%.1f%%)  4psi amp=%.1f%%  F=%.1f",
                    T, f2["c0"], anis_pct, f2["phi"],
                    100 * scatter / f2["c0"], 100 * f4["amp"] / f2["c0"], Fstat)

    # Per-period c(psi) panels — only the periods that produced a fit, wrapped to a grid.
    if panels:
        npan = len(panels)
        ncol = min(5, npan)
        nrow = int(np.ceil(npan / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.4 * nrow),
                                 squeeze=False)
        pp = np.linspace(0, 180, 181)
        for ax in axes.ravel():
            ax.axis("off")
        for ip, (T, f_T, cen, cval, csig, f2) in enumerate(panels):
            ax = axes[ip // ncol, ip % ncol]
            ax.axis("on")
            ax.errorbar(cen, cval, yerr=csig, fmt="o", ms=5, color="tab:blue",
                        capsize=2, label="binned c(psi)")
            cfit = f2["c0"] + f2["A"] * np.cos(np.radians(2 * pp)) + \
                f2["B"] * np.sin(np.radians(2 * pp))
            ax.plot(pp, cfit, "r-", lw=2, label="2psi fit")
            ax.axvline(f2["phi"], color="k", ls="--", lw=1, alpha=0.6)
            ax.set(title=f"T={T:g}s ({f_T:.2f} Hz)\n"
                   f"2psi={100 * f2['amp'] / f2['c0']:.1f}%  phi={f2['phi']:.0f}°",
                   xlabel="azimuth psi [deg]", xlim=(0, 180))
            if ip % ncol == 0:
                ax.set_ylabel("phase velocity c [km/s]")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
        fig.suptitle(f"Azimuthal 2-psi anisotropy | {args.component} | "
                     f"{pairs['x'].size} pairs, {args.nbins} azimuth bins, "
                     f"{args.nboot} bootstraps", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        figpath = os.path.join(outdir, f"anisotropy_2psi_{args.component}.png")
        fig.savefig(figpath, dpi=130, bbox_inches="tight")
        logger.info("Wrote %s", figpath)

    if results:
        import csv
        csvpath = os.path.join(outdir, f"anisotropy_2psi_{args.component}.csv")
        cols = ["T", "f", "nbins", "c0", "amp", "amp_err", "anis_pct", "p2p",
                "phi", "scatter", "F", "amp4", "phi4"]
        with open(csvpath, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            wr.writeheader()
            for r in results:
                wr.writerow({c: r[c] for c in cols})
        logger.info("Wrote %s", csvpath)

        # Summary: fast axis + 2psi amplitude vs period
        Ts = [r["T"] for r in results]
        fig2, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
        a1.errorbar(Ts, [r["anis_pct"] for r in results],
                    yerr=[100 * r["amp_err"] / r["c0"] for r in results],
                    fmt="o-", color="tab:red", capsize=3, label="2psi amplitude")
        a1.plot(Ts, [100 * r["scatter"] / r["c0"] for r in results], "s--",
                color="gray", label="median bootstrap scatter")
        a1.plot(Ts, [100 * r["amp4"] / r["c0"] for r in results], "^:",
                color="tab:green", label="4psi amplitude")
        a1.set(xlabel="Period [s]", ylabel="amplitude [% of c0]",
               title="Anisotropy amplitude vs scatter")
        a1.legend(fontsize=8); a1.grid(alpha=0.3)
        a2.errorbar(Ts, [r["phi"] for r in results], fmt="o-", color="tab:blue", capsize=3)
        a2.set(xlabel="Period [s]", ylabel="fast axis phi [deg]",
               title="Fast-axis azimuth vs period", ylim=(0, 180))
        a2.grid(alpha=0.3)
        fig2.tight_layout()
        sp = os.path.join(outdir, f"anisotropy_2psi_{args.component}_summary.png")
        fig2.savefig(sp, dpi=130, bbox_inches="tight")
        logger.info("Wrote %s", sp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
