"""Validate the unified picker on a network: 2D pick-distribution plots for all eight pick types,
plus a Love-fundamental phase sign check.

Reads the <pair>_unified.csv files written by dispersion_unified.py and produces:
  * unified_pick_distributions.png -- an 8-panel period x velocity 2D histogram (2 waves x 2 modes x
    {group, phase}); the data-derived phase reference is overlaid on each phase panel.
  * unified_love_sign_check.png -- Love-fundamental measured phase velocity vs the data-derived
    Love reference. The median residual confirms the +pi/4 Love phase shift (analog of the Rayleigh
    PHASE_OFFSET~0 check). A residual near zero => +pi/4 is correct; ~half a fringe => flip the sign.

Usage:
    python validate_unified_picks.py --dir <dispersion_unified_dir> --ref-dir <vsg_modesep_dir>
                                     [--out-dir <where to write PNGs>] [--pick-method argmax]
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--dir", required=True, help="dispersion_unified output dir (has <src>/<pair>_unified.csv)")
ap.add_argument("--ref-dir", required=True, help="vsg_modesep dir with the four ref_*_phase.txt")
ap.add_argument("--out-dir", default=None, help="where to write PNGs (default: --dir)")
ap.add_argument("--pick-method", default="argmax", choices=["argmax", "topology", "both"],
                help="which pick method to histogram (argmax is cleanest)")
args = ap.parse_args()
OUT = args.out_dir or args.dir

REF_FILES = {("rayleigh", "fundamental"): "ref_fundamental_phase.txt",
             ("rayleigh", "overtone"): "ref_overtone_phase.txt",
             ("love", "fundamental"): "ref_love_phase.txt",
             ("love", "overtone"): "ref_love_overtone_phase.txt"}
refs = {}
for key, fn in REF_FILES.items():
    try:
        refs[key] = np.loadtxt(os.path.join(args.ref_dir, fn))
    except Exception:
        refs[key] = None


# --------------------------------------------------------------------------- load picks
files = sorted(glob.glob(os.path.join(args.dir, "*", "*_unified.csv")))
if not files:
    raise SystemExit(f"no *_unified.csv under {args.dir}")
print(f"loading {len(files)} pair CSVs ...")

# Collect (period, group, phase, overlap) per (wave, mode).
COLS = None
data = {k: {"T": [], "U": [], "c": [], "ov": []} for k in REF_FILES}
for fn in files:
    arr = np.genfromtxt(fn, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if arr.size == 0:
        continue
    arr = np.atleast_1d(arr)
    if COLS is None:
        COLS = arr.dtype.names
    pm = arr["pick_method"].astype(str)
    sel_pm = np.ones(len(arr), bool) if args.pick_method == "both" else (pm == args.pick_method)
    wt = arr["wave_type"].astype(str)
    md = arr["mode"].astype(str)
    for (wave, mode) in REF_FILES:
        m = sel_pm & (wt == wave) & (md == mode)
        if not m.any():
            continue
        data[(wave, mode)]["T"].append(arr["nominal_period"][m])
        data[(wave, mode)]["U"].append(arr["group_velocity"][m])
        data[(wave, mode)]["c"].append(arr["phase_velocity"][m])
        data[(wave, mode)]["ov"].append(arr["mode_overlap"][m])
for k in data:
    for f in data[k]:
        data[k][f] = np.concatenate(data[k][f]) if data[k][f] else np.array([])


# --------------------------------------------------------------------------- figure 1: 8 panels
Tb = np.arange(0.2, 6.05, 0.1)
Vb = np.arange(0.5, 5.05, 0.05)
ROWS = [("rayleigh", "fundamental"), ("rayleigh", "overtone"),
        ("love", "fundamental"), ("love", "overtone")]
fig, axs = plt.subplots(4, 2, figsize=(15, 18))
for ir, key in enumerate(ROWS):
    wave, mode = key
    d = data[key]
    for ic, (meas, arrname) in enumerate((("group", "U"), ("phase", "c"))):
        ax = axs[ir, ic]
        T, V = d["T"], d[arrname]
        good = np.isfinite(T) & np.isfinite(V) & (V > 0)
        T, V = T[good], V[good]
        n = len(T)
        if n:
            H, xe, ye = np.histogram2d(T, V, bins=[Tb, Vb])
            pm = ax.pcolormesh(xe, ye, np.where(H.T > 0, H.T, np.nan), cmap="viridis", norm=LogNorm())
            plt.colorbar(pm, ax=ax, label="picks / cell")
        # reference overlay (phase reference on the phase panel; its group form on the group panel)
        r = refs[key]
        if r is not None and r.ndim == 2 and len(r):
            if meas == "phase":
                ax.plot(r[:, 0], r[:, 1], "r--", lw=2, label="data-derived c_ref")
                ax.legend(fontsize=9, loc="upper left")
        ax.set(title=f"{wave} {mode} -- {meas}  (n={n:,})", xlabel="Period [s]",
               ylabel=f"{meas.capitalize()} velocity [km/s]", xlim=(0.2, 6), ylim=(0.5, 5.0))
fig.suptitle(f"Aargau unified picker: 2D pick distributions ({len(files)} pairs, "
             f"pick_method={args.pick_method})", y=0.997, fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.99))
p1 = os.path.join(OUT, "unified_pick_distributions.png")
fig.savefig(p1, dpi=120)
plt.close(fig)
print(f"wrote {p1}")


# --------------------------------------------------------------------------- figure 2: Love sign check
def sign_check(ax, key, title):
    d = data[key]
    r = refs[key]
    T, c = d["T"], d["c"]
    good = np.isfinite(T) & np.isfinite(c) & (c > 0)
    T, c = T[good], c[good]
    if r is None or not len(T):
        ax.set_title(title + " (no data)")
        return None
    cref = np.interp(T, r[:, 0], r[:, 1], left=np.nan, right=np.nan)
    ok = np.isfinite(cref)
    resid = c[ok] - cref[ok]
    med = float(np.median(resid)) if len(resid) else np.nan
    ax.hist(resid, bins=np.arange(-1.0, 1.02, 0.04), color="steelblue", edgecolor="none")
    ax.axvline(0, color="k", lw=1)
    ax.axvline(med, color="r", lw=2, label=f"median {med:+.3f} km/s")
    ax.set(title=f"{title}  (n={len(resid):,})", xlabel="measured c - reference c [km/s]",
           ylabel="picks")
    ax.legend(fontsize=9)
    return med


fig, axs = plt.subplots(1, 2, figsize=(13, 5))
m_love = sign_check(axs[0], ("love", "fundamental"), "Love fundamental phase - reference")
m_ray = sign_check(axs[1], ("rayleigh", "fundamental"), "Rayleigh fundamental phase - reference (control)")
fig.suptitle("Phase sign / bias check: measured phase velocity vs data-derived reference", y=1.0)
fig.tight_layout()
p2 = os.path.join(OUT, "unified_love_sign_check.png")
fig.savefig(p2, dpi=120)
plt.close(fig)
print(f"wrote {p2}")

# --------------------------------------------------------------------------- text summary
print("\n=== pick-count summary (pick_method=%s) ===" % args.pick_method)
for key in ROWS:
    d = data[key]
    ng = int(np.sum(np.isfinite(d["U"]) & (d["U"] > 0)))
    npz = int(np.sum(np.isfinite(d["c"]) & (d["c"] > 0)))
    print(f"  {key[0]:8s} {key[1]:11s}: group {ng:6d} | phase {npz:6d}")
print(f"\nLove fundamental phase median residual vs ref: {m_love}")
print(f"Rayleigh fundamental phase median residual vs ref: {m_ray}")
