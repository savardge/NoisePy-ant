"""Network Love (TT) phase-velocity reference curve from the VSG phase-shift spectra.

This is the Love analog of vsg_modesep.py + pick_ridges.py, but WITHOUT mode separation: the
transverse-transverse (TT) cross-correlation IS the Love wavefield, so there is nothing to
synthesize. For each virtual source we take the per-source VSG phase-shift (slant-stack) spectrum
E(c, f) already stored in {paths.vsg_dir}/TT/sources/<src>.npz (E_real/E_imag on the common
(velocity, frequency) grid), per-frequency normalize |E|, and stack linearly across the whole
network. The Love fundamental is the dominant ridge on TT; we pick it per frequency by argmax in a
velocity window with the same dominance / roughness / min-run QC as pick_ridges.py, and write a
2-column reference `ref_love_phase.txt` (period[s], phase_velocity[km/s]) for
period_resolution.py / load_reference_curve.

Usage:  python vsg_love_reference.py --config ../../param_files/modesep_params.yaml
                                     [--vmin 1.2 --vmax 3.4] [--fmin 0.2 --fmax 2.2]
                                     [--dom-min 0.6 --rough-max 0.04] [--max-sources N] [--out DIR]

Note: the stored E is the same phase-shift product the Rayleigh VSG references were picked from
(before mode synthesis), so the Love reference is built on an identical slant-stack footing.
"""
import argparse
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import modesep_config

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--config", required=True, help="network YAML (param_files/modesep_*.yaml)")
ap.add_argument("--vmin", type=float, default=1.2, help="Love fundamental search vmin [km/s]")
ap.add_argument("--vmax", type=float, default=3.4, help="Love fundamental search vmax [km/s]")
ap.add_argument("--fmin", type=float, default=0.2, help="min frequency [Hz]")
ap.add_argument("--fmax", type=float, default=2.2, help="max frequency [Hz]")
ap.add_argument("--dom-min", type=float, default=0.6, help="min in-window/full-column dominance")
ap.add_argument("--rough-max", type=float, default=0.04, help="max rolling |dc| roughness [km/s]")
ap.add_argument("--min-run", type=int, default=10, help="min contiguous kept frequencies")
ap.add_argument("--rough-win", type=int, default=9, help="roughness rolling-window samples")
ap.add_argument("--max-sources", type=int, default=None, help="limit sources for quick tests")
ap.add_argument("--out", default=None, help="output dir (default: paths.ref_dir)")
# --- Love OVERTONE (1st higher mode) reference: a second, faster ridge on the same TT stack. ---
# There is no G_LR-style synthesis for Love (single-component SH), so the overtone is picked as a
# distinct high-velocity ridge in the SAME network TT f-c stack. It is genuinely weaker than the
# fundamental, so its dominance gate is relaxed (dominance here = overtone_peak / full-column max,
# i.e. the overtone amplitude relative to the fundamental). If QC rejects it, nothing is written and
# the unified picker leaves Love overtone phase unresolved (NaN) -- group still emits.
ap.add_argument("--no-overtone", action="store_true", help="skip the Love overtone ridge")
ap.add_argument("--ot-vmin", type=float, default=2.6, help="Love overtone search vmin [km/s]")
ap.add_argument("--ot-vmax", type=float, default=4.0, help="Love overtone search vmax [km/s]")
ap.add_argument("--ot-dom-min", type=float, default=0.35,
                help="min overtone/full-column dominance (relaxed vs fundamental)")
ap.add_argument("--ot-min-run", type=int, default=6, help="overtone min contiguous kept frequencies")
ap.add_argument("--ot-sep-min", type=float, default=0.3,
                help="min c_overtone - c_fundamental [km/s]: reject overtone picks that merely ride "
                     "the top of the fundamental ridge (window-floor artifact)")
# --from-stack skips the per-source restack and reuses a previously written vsg_love_stack.npz
# (f, vel, TT). Use it to (re)pick the overtone from an already-built fundamental stack cheaply.
ap.add_argument("--from-stack", default=None,
                help="reuse an existing vsg_love_stack.npz instead of restacking TT sources")
args = ap.parse_args()

cfg = modesep_config.load_config(args.config)
D = cfg["paths"]["vsg_dir"]
OUT = args.out or cfg["paths"]["ref_dir"]
NET = cfg["network"].get("name", cfg["network"].get("code", ""))
os.makedirs(OUT, exist_ok=True)

if args.from_stack:
    # ---- fast path: reuse a previously written network stack (skip the per-source restack) ----
    z = np.load(args.from_stack, allow_pickle=True)
    fgrid, vel, acc = np.asarray(z["f"], float), np.asarray(z["vel"], float), np.asarray(z["TT"], float)
    nused = int(z["n_sources"]) if "n_sources" in z else 0
    print(f"{NET}: loaded TT stack from {args.from_stack} ({nused} sources)")
else:
    srcdir = os.path.join(D, "TT", "sources")
    srcs = sorted(glob.glob(os.path.join(srcdir, "*.npz")))
    if args.max_sources:
        srcs = srcs[: args.max_sources]
    if not srcs:
        raise SystemExit(f"no TT VSG source npz under {srcdir}")

    z0 = np.load(srcs[0], allow_pickle=True)
    fgrid, vel = np.asarray(z0["f"], float), np.asarray(z0["vel"], float)

    # ---- network stack of the per-source TT phase-shift spectra (per-frequency normalized |E|) --
    acc = np.zeros((len(vel), len(fgrid)))
    nused = 0
    for fp in srcs:
        try:
            z = np.load(fp, allow_pickle=True)
            if np.asarray(z["f"]).shape != fgrid.shape or np.asarray(z["vel"]).shape != vel.shape:
                continue
            E = np.abs(np.asarray(z["E_real"], float) + 1j * np.asarray(z["E_imag"], float))
            mx = E.max(axis=0, keepdims=True)                 # per-frequency normalize
            acc += E / np.where(mx > 0, mx, 1.0)
            nused += 1
        except Exception as e:
            print(f"{os.path.basename(fp)}: skip ({e})", flush=True)
    acc /= max(nused, 1)
    print(f"{NET}: stacked {nused}/{len(srcs)} TT sources")


def pick_ridge(img, fmin, fmax, vmin, vmax, dom_min, rough_max, rough_win, min_run, max_gap=2):
    """Per-frequency argmax within (fmin..fmax, vmin..vmax) + dominance/roughness/min-run QC.
    Identical logic to pick_ridges.py::pick. Returns (freq, c_phase, dominance)."""
    fi = np.where((fgrid >= fmin) & (fgrid <= fmax))[0]
    vi = np.where((vel >= vmin) & (vel <= vmax))[0]
    col_max = img[:, fi].max(axis=0)
    win = img[np.ix_(vi, fi)]
    kmax = np.argmax(win, axis=0)
    peak = win[kmax, np.arange(len(fi))]
    dom = peak / np.where(col_max > 0, col_max, np.nan)
    cpick = vel[vi][kmax]
    good = dom >= dom_min
    dc = np.abs(np.diff(cpick)); dc = np.concatenate([dc[:1], dc])
    rough = np.convolve(dc, np.ones(rough_win) / rough_win, mode="same")
    good &= rough <= rough_max
    if good.any():
        idx = np.where(good)[0]
        runs = np.split(idx, np.where(np.diff(idx) > max_gap)[0] + 1)
        good = np.zeros_like(good)
        for r in runs:
            if len(r) >= min_run:
                good[r] = True
    return fgrid[fi][good], cpick[good], dom[good]


# ---- pick the Love fundamental (mandatory) and, if requested, the overtone (best-effort) ----
# Each branch is (label, outfile, vmin, vmax, dom_min, min_run, plot_color). The overtone shares
# the fundamental's roughness/rough-win QC but uses its own velocity window + relaxed dominance.
branches = [("fundamental", "ref_love_phase.txt", args.vmin, args.vmax,
             args.dom_min, args.min_run, "w")]
if not args.no_overtone:
    branches.append(("overtone", "ref_love_overtone_phase.txt", args.ot_vmin, args.ot_vmax,
                     args.ot_dom_min, args.ot_min_run, "k"))

picked = {}                                              # label -> (freq, c, dominance, window)
for label, fname, vlo, vhi, dommin, minrun, _col in branches:
    fp, cp, dp = pick_ridge(acc, args.fmin, args.fmax, vlo, vhi,
                            dommin, args.rough_max, args.rough_win, minrun)
    if label == "overtone" and len(fp) and "fundamental" in picked:
        # Separation gate: keep the overtone only where it is genuinely faster than the fundamental
        # ridge at the same frequency. Without this the window-floor picks (overtone ~ fundamental +
        # dv) survive the dominance/roughness QC and contaminate the reference with a flat artifact.
        ff, fc = picked["fundamental"][0], picked["fundamental"][1]
        c_fund_at = np.interp(fp, ff, fc, left=np.nan, right=np.nan)
        keep = np.isfinite(c_fund_at) & (cp - c_fund_at >= args.ot_sep_min)
        n_drop = int((~keep).sum())
        fp, cp, dp = fp[keep], cp[keep], dp[keep]
        if n_drop:
            print(f"{NET}: overtone separation gate dropped {n_drop} floor-artifact picks "
                  f"(c_ot - c_fund < {args.ot_sep_min} km/s)")
    if len(fp) == 0:
        if label == "fundamental":
            raise SystemExit("Love fundamental QC rejected all frequencies -- widen window / relax QC")
        print(f"{NET}: Love overtone ridge rejected by QC ({vlo}-{vhi} km/s) -- "
              f"no {fname} written; overtone phase will be unresolved downstream.")
        continue
    order = np.argsort(1.0 / fp)
    outfile = os.path.join(OUT, fname)
    np.savetxt(outfile, np.column_stack([(1.0 / fp)[order], cp[order]]), fmt="%.4f",
               header=f"period[s]  phase_velocity[km/s]  (Love {label}, network TT VSG stack)")
    picked[label] = (fp, cp, dp, (vlo, vhi))
    print(f"{NET}: {len(fp)} Love {label} picks | T {1/fp.max():.2f}-{1/fp.min():.2f} s "
          f"| c {cp.min():.2f}-{cp.max():.2f} km/s -> {outfile}")

# ---- QC figure: stacked TT image (freq + period) with the picked ridge(s) ----
colors = {"fundamental": "w", "overtone": "k"}
fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, xaxis in zip(axs, ("frequency", "period")):
    xf = (lambda q: q) if xaxis == "frequency" else (lambda q: 1.0 / q)
    ax.pcolormesh(xf(fgrid), vel, acc, cmap="jet", shading="auto")
    for label, (fp, cp, _dp, (vlo, vhi)) in picked.items():
        ax.plot(xf(fp), cp, colors[label] + ".", ms=4, label=f"picked Love {label}")
        x0, x1 = sorted([xf(args.fmin), xf(args.fmax)])
        ax.add_patch(plt.Rectangle((x0, vlo), x1 - x0, vhi - vlo,
                                   fill=False, ec=colors[label], ls=":", lw=1))
    ax.set(title=f"{NET} TT VSG stack ({nused} src) -- {xaxis}", xscale="log",
           xlabel=("Frequency [Hz]" if xaxis == "frequency" else "Period [s]"), ylim=(0.5, 5))
    ax.set_xlim(sorted([xf(0.16), xf(2.2)]))
    ax.legend(loc="upper right", fontsize=9)
axs[0].set_ylabel("Phase velocity [km/s]")
fig.tight_layout()
figpath = os.path.join(OUT, "ref_love_phase.png")
fig.savefig(figpath, dpi=130)
np.savez(os.path.join(OUT, "vsg_love_stack.npz"), f=fgrid, vel=vel, TT=acc, n_sources=nused)
print(f"wrote {figpath} and vsg_love_stack.npz")
