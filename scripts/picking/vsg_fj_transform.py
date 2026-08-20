#!/usr/bin/env python3
"""Frequency-Bessel (F-J) transform of the VSG source gathers, network-stacked, for direct
comparison with the phase-shift (slant-stack) images the VSG reference curves were picked from.

WHY. The VSG "overtone" reference ridge sits at 3.5-4.5 km/s -- above the Vs(halfspace) ceiling
that every trapped Rayleigh mode obeys -- and is weakly dispersive. That is the signature of a
LEAKING (guided-P) mode (Li et al. 2022 GRL; Wu et al. 2025 & Shi et al. 2025 GJI). The forward
argument cannot confirm it alone; the beam can. The F-J transform (Wang et al. 2019; CC-FJpy is
the same group's implementation) uses the Bessel kernel that is the correct 2-D-array transform
for CCFs, so it separates crowded/weak branches better than a plane-wave slant stack -- and the
leaking-mode literature extracts these branches with exactly this transform. The question this
answers: does F-J show the ridge as a distinct branch above the ceiling, cleanly separated from
a genuine R1 below it?

INPUT. The same per-virtual-source npz the slant stack consumed:
    <Data>/<net>/phasevelocity_VSG/ZZ/sources/<src>.npz
        sym (n_rx, nt)  folded causal CCF traces;  x (n_rx) offsets [km];  dt
F-J wants the FREQUENCY-DOMAIN CCFs, real part, in SI: uf (n_rx, n_f) from rfft(sym),
r [m], c [m/s], f [Hz]. Verified against the package example (summed.npz: complex spectra,
r*1e3, c in m/s).

STACKING. Per source: |FJ| normalised per frequency column (max=1), then linear stack across
sources -- identical to vsg_modesep.py, so contrast is comparable. Also writes the UNnormalised
stack, since per-column normalisation can inflate weak columns.

Usage:
  python vsg_fj_transform.py --net aargau
  python vsg_fj_transform.py --net riehen --max-sources 60      # quick look
"""
import argparse, glob, os, sys, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = {"aargau": "/Users/genevievesavard/Data/aargau/phasevelocity_VSG",
        "riehen": "/Users/genevievesavard/Data/riehen/phasevelocity_VSG",
        "hautesorne": "/Users/genevievesavard/Data/hautesorne/phasevelocity_VSG"}
EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", required=True)
    ap.add_argument("--comp", default="ZZ")
    ap.add_argument("--max-sources", type=int, default=None)
    ap.add_argument("--fmin", type=float, default=0.15)
    ap.add_argument("--fmax", type=float, default=3.0)
    ap.add_argument("--nf", type=int, default=300)
    ap.add_argument("--cmin", type=float, default=0.5)
    ap.add_argument("--cmax", type=float, default=6.0)
    ap.add_argument("--nc", type=int, default=551)
    ap.add_argument("--min-offset", type=float, default=0.0,
                    help="km; the slant-stack analysis excluded < ~2 km (near field)")
    ap.add_argument("--itype", type=int, default=1, help="0 trapezoid, 1 linear-approx (README default)")
    ap.add_argument("--func", type=int, default=0, help="0 Bessel, 1 Hankel")
    a = ap.parse_args()

    import ccfj
    src_dir = f"{DATA[a.net]}/{a.comp}/sources"
    files = sorted(glob.glob(f"{src_dir}/*.npz"))
    if not files:
        raise SystemExit(f"no source gathers under {src_dir}")
    if a.max_sources:
        files = files[: a.max_sources]
    out_dir = f"{EHM}/{a.net}/vsg_modesep"
    os.makedirs(out_dir, exist_ok=True)

    # frequency & velocity grids. Frequencies must be a subset of the rfft grid, so build the
    # rfft grid from the first file and pick the nearest bins to a log-spaced target.
    z0 = np.load(files[0], allow_pickle=True)
    dt = float(z0["dt"]); nt = np.asarray(z0["sym"]).shape[1]
    nfft = 2 ** int(np.ceil(np.log2(nt)) + 1)
    fr = np.fft.rfftfreq(nfft, dt)
    ftarget = np.exp(np.linspace(np.log(a.fmin), np.log(a.fmax), a.nf))
    fidx = np.unique([int(np.argmin(np.abs(fr - fv))) for fv in ftarget])
    f = fr[fidx]
    c = np.linspace(a.cmin, a.cmax, a.nc)          # km/s

    acc = np.zeros((len(c), len(f)))              # per-column-normalised stack
    acc_raw = np.zeros((len(c), len(f)))          # unnormalised stack
    nused = 0; t0 = time.time()
    for k, fp in enumerate(files):
        z = np.load(fp, allow_pickle=True)
        sym = np.asarray(z["sym"], float); x = np.asarray(z["x"], float)
        keep = x >= a.min_offset
        if keep.sum() < 8:
            continue
        sym, x = sym[keep], x[keep]
        # F-J wants frequency-domain CCFs (real part) in SI units
        U = np.fft.rfft(sym, n=nfft, axis=1)[:, fidx]
        uf = np.ascontiguousarray(np.real(U), dtype=np.float32)
        try:
            img = ccfj.fj_noise(uf, np.ascontiguousarray(x * 1e3, dtype=np.float32),
                                np.ascontiguousarray(c * 1e3, dtype=np.float32),
                                np.ascontiguousarray(f, dtype=np.float32),
                                fstride=1, itype=a.itype, func=a.func)
        except Exception as e:
            print(f"  {os.path.basename(fp)}: fj_noise failed: {e}"); continue
        img = np.asarray(img, float)
        if img.shape != (len(c), len(f)):
            img = img.T if img.T.shape == (len(c), len(f)) else img
        img = np.abs(img)
        acc_raw += img
        mx = img.max(axis=0, keepdims=True); mx[mx == 0] = 1.0
        acc += img / mx
        nused += 1
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{len(files)} sources  ({time.time()-t0:.0f}s)", flush=True)
    if not nused:
        raise SystemExit("no source contributed")
    acc /= nused; acc_raw /= nused
    tag = f"fj_{a.comp}_sign+1"
    np.savez(os.path.join(out_dir, f"vsg_{tag}.npz"), f=f, vel=c, n_sources=nused,
             FJ=acc, FJ_raw=acc_raw, itype=a.itype, func=a.func, min_offset=a.min_offset)

    # ---- comparison figure with the slant-stack image and the picked references ----
    ss = os.path.join(out_dir, "vsg_modesep_stacks_sign+1.npz")
    refs = {}
    for w, fn in (("R0", "ref_fundamental_phase.txt"), ("‘R1’", "ref_overtone_phase.txt")):
        p = os.path.join(out_dir, fn)
        if os.path.exists(p):
            refs[w] = np.loadtxt(p)
    fig, axs = plt.subplots(1, 2 if os.path.exists(ss) else 1, figsize=(15, 6.2), squeeze=False)
    ax = axs[0][0]
    A = acc / np.percentile(acc, 99.5)
    ax.pcolormesh(f, c, np.clip(A, 0, 1), cmap="inferno", shading="auto")
    for w, r in refs.items():
        ax.plot(1.0 / r[:, 0], r[:, 1], "c--" if w == "R0" else "w-", lw=1.5,
                label=f"picked {w} (from slant stack)")
    ax.set_xscale("log"); ax.set_xlim(a.fmin, a.fmax); ax.set_ylim(a.cmin, a.cmax)
    ax.set_xlabel("frequency [Hz]"); ax.set_ylabel("phase velocity [km/s]")
    ax.set_title(f"{a.net} — F-J transform, {a.comp}, {nused} sources stacked", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper right")
    if os.path.exists(ss):
        s = np.load(ss, allow_pickle=True)
        ax = axs[0][1]
        S = s["ZZ"] / np.percentile(s["ZZ"], 99.5)
        ax.pcolormesh(s["f"], s["vel"], np.clip(S, 0, 1), cmap="inferno", shading="auto")
        for w, r in refs.items():
            ax.plot(1.0 / r[:, 0], r[:, 1], "c--" if w == "R0" else "w-", lw=1.5)
        ax.set_xscale("log"); ax.set_xlim(a.fmin, a.fmax); ax.set_ylim(a.cmin, a.cmax)
        ax.set_xlabel("frequency [Hz]")
        ax.set_title(f"phase-shift (slant stack), ZZ, {int(s['n_sources'])} sources", fontsize=10.5)
    fig.suptitle(f"{a.net}: F-J vs slant-stack network images (per-column normalised, "
                 f"clipped at p99.5)", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    png = os.path.join(out_dir, f"vsg_{tag}_vs_slantstack.png")
    fig.savefig(png, dpi=130, bbox_inches="tight")
    print(f"\nsources used {nused}/{len(files)}   wrote {png}")


if __name__ == "__main__":
    main()
