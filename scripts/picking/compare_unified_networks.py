"""Cross-network diagnostics for the unified picking + QC outputs (Aargau vs Riehen).

Produces, in extract_higher_modes/Projects/unified_diagnostics/:
  * unified_compare_distributions.png -- side-by-side post-QC 2D pick histograms, one column per
    network, rows = {Rayleigh fund, Rayleigh overtone, Love fund} x {group, phase}, with the
    network's own data-derived phase reference overlaid on phase panels.
  * unified_compare_ray_maps.png -- per network x pick type: station map with every surviving
    pair drawn as a great-circle chord colored by its number of QC'd group picks (path-density /
    coverage-pattern view). Station coordinates harvested from the VSG per-source npz
    (src_lon/lat + rx_codes/rx_lons/rx_lats).
  * unified_compare_stats.txt -- survivor counts, pair coverage, path-length stats per network.

Usage: python compare_unified_networks.py [--nets aargau,riehen]
Reads <project>/dispersion_unified/picks_unified_QCd.csv (run qc_unified_picks.py first).
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.collections import LineCollection

ROOT = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
NETWORKS = {
    "aargau": {"title": "Aargau (AA)", "project": f"{ROOT}/aargau",
               "vsg": "/Users/genevievesavard/Data/aargau/phasevelocity_VSG"},
    "riehen": {"title": "Riehen (RI)", "project": f"{ROOT}/riehen",
               "vsg": "/Users/genevievesavard/Data/riehen/phasevelocity_VSG"},
    "hautesorne": {"title": "Haute-Sorne (SS)", "project": f"{ROOT}/hautesorne",
                   "vsg": "/Users/genevievesavard/Data/hautesorne/phasevelocity_VSG"},
}
REF_FILES = {("rayleigh", "fundamental"): "ref_fundamental_phase.txt",
             ("rayleigh", "overtone"): "ref_overtone_phase.txt",
             ("love", "fundamental"): "ref_love_phase.txt"}
PICK_TYPES = [("rayleigh", "fundamental"), ("rayleigh", "overtone"), ("love", "fundamental")]

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--nets", default="aargau,riehen")
args = ap.parse_args()
nets = [n.strip() for n in args.nets.split(",") if n.strip() in NETWORKS]
OUT = f"{ROOT}/unified_diagnostics"
os.makedirs(OUT, exist_ok=True)


def station_coords(vsg_dir):
    """{code: (lon, lat)} harvested from every component's VSG source npz."""
    coords = {}
    for fp in glob.glob(os.path.join(vsg_dir, "*", "sources", "*.npz")):
        try:
            z = np.load(fp, allow_pickle=True)
            coords[str(z["src"])] = (float(z["src_lon"]), float(z["src_lat"]))
            for c, lo, la in zip(z["rx_codes"], z["rx_lons"], z["rx_lats"]):
                coords[str(c)] = (float(lo), float(la))
        except Exception:
            continue
    return coords


data = {}
for net in nets:
    qcd = os.path.join(NETWORKS[net]["project"], "dispersion_unified", "picks_unified_QCd.csv")
    if not os.path.exists(qcd):
        print(f"WARN: {qcd} missing -- run qc_unified_picks.py for {net}; skipping")
        continue
    print(f"loading {net} ...")
    d = pd.read_csv(qcd)
    refs = {}
    for key, fn in REF_FILES.items():
        try:
            refs[key] = np.loadtxt(os.path.join(NETWORKS[net]["project"], "vsg_modesep", fn))
        except Exception:
            refs[key] = None
    data[net] = {"df": d, "refs": refs, "coords": station_coords(NETWORKS[net]["vsg"])}
if not data:
    raise SystemExit("no networks loaded")

# ------------------------------------------------------------------ fig 1: distributions
Tb = np.arange(0.2, 6.05, 0.1)
Vb = np.arange(0.5, 5.05, 0.05)
rows = [(w, m, meas) for (w, m) in PICK_TYPES for meas in ("group", "phase")]
fig, axs = plt.subplots(len(rows), len(data), figsize=(8.5 * len(data), 3.6 * len(rows)),
                        squeeze=False)
for ic, net in enumerate(data):
    d = data[net]["df"]
    for ir, (w, m, meas) in enumerate(rows):
        ax = axs[ir, ic]
        vcol = "group_velocity" if meas == "group" else "phase_velocity"
        ok = d[f"{meas}_ok"] == 1
        sub = d[(d.wave_type == w) & (d["mode"] == m) & ok]
        T, V = sub["nominal_period"].to_numpy(), sub[vcol].to_numpy()
        good = np.isfinite(T) & np.isfinite(V) & (V > 0)
        T, V = T[good], V[good]
        if len(T):
            H, xe, ye = np.histogram2d(T, V, bins=[Tb, Vb])
            pm = ax.pcolormesh(xe, ye, np.where(H.T > 0, H.T, np.nan), cmap="viridis",
                               norm=LogNorm())
            plt.colorbar(pm, ax=ax, label="picks / cell")
        r = data[net]["refs"].get((w, m))
        if meas == "phase" and r is not None and np.ndim(r) == 2 and len(r):
            ax.plot(r[:, 0], r[:, 1], "r--", lw=1.5, label="c_ref")
            ax.legend(fontsize=8, loc="upper left")
        ax.set(title=f"{NETWORKS[net]['title']}: {w} {m} -- {meas} (n={len(T):,})",
               xlim=(0.2, 6), ylim=(0.5, 5.0))
        if ic == 0:
            ax.set_ylabel(f"{meas} velocity [km/s]")
        if ir == len(rows) - 1:
            ax.set_xlabel("Period [s]")
fig.suptitle("Unified picks after QC -- network comparison", y=0.999, fontsize=15)
fig.tight_layout(rect=(0, 0, 1, 0.995))
p1 = os.path.join(OUT, "unified_compare_distributions.png")
fig.savefig(p1, dpi=110)
plt.close(fig)
print(f"wrote {p1}")

# ------------------------------------------------------------------ fig 2: ray-path maps
fig, axs = plt.subplots(len(data), len(PICK_TYPES),
                        figsize=(6.5 * len(PICK_TYPES), 6.2 * len(data)), squeeze=False)
stats_lines = []
for ir, net in enumerate(data):
    d = data[net]["df"]
    coords = data[net]["coords"]
    d = d[d.group_ok == 1]
    per_pair = (d.groupby(["pair", "wave_type", "mode"]).size()
                  .rename("npicks").reset_index())
    for ic, (w, m) in enumerate(PICK_TYPES):
        ax = axs[ir, ic]
        sub = per_pair[(per_pair.wave_type == w) & (per_pair["mode"] == m)]
        segs, counts = [], []
        missing = 0
        for _, r in sub.iterrows():
            try:
                s1, s2 = r["pair"].split("_")
            except ValueError:
                continue
            if s1 in coords and s2 in coords:
                segs.append([coords[s1], coords[s2]])
                counts.append(r["npicks"])
            else:
                missing += 1
        if segs:
            counts = np.asarray(counts, float)
            lc = LineCollection(segs, cmap="magma", norm=LogNorm(1, max(counts.max(), 2)),
                                linewidths=0.5, alpha=0.35)
            lc.set_array(counts)
            ax.add_collection(lc)
            plt.colorbar(lc, ax=ax, label="QC'd group picks / pair")
        lons = np.array([c[0] for c in coords.values()])
        lats = np.array([c[1] for c in coords.values()])
        fin = np.isfinite(lons) & np.isfinite(lats)
        lons, lats = lons[fin], lats[fin]
        ax.plot(lons, lats, "^", ms=3, color="0.25", mec="w", mew=0.2, zorder=3)
        asp = 1.0 / np.cos(np.deg2rad(np.mean(lats))) if len(lats) else 1.0
        ax.set(title=f"{NETWORKS[net]['title']}: {w} {m} -- {len(segs):,} pairs"
                     + (f" ({missing} no-coord)" if missing else ""),
               xlabel="Longitude", ylabel="Latitude",
               aspect=asp if np.isfinite(asp) and asp > 0 else 1.0)
        ax.margins(0.05)
        stats_lines.append(f"{net:8s} {w:8s} {m:11s}: {len(segs):,} pairs with picks, "
                           f"median picks/pair {np.median(counts) if len(segs) else 0:.0f}")
fig.suptitle("QC'd pick coverage -- ray paths colored by surviving group-pick count",
             y=0.998, fontsize=15)
fig.tight_layout(rect=(0, 0, 1, 0.99))
p2 = os.path.join(OUT, "unified_compare_ray_maps.png")
fig.savefig(p2, dpi=110)
plt.close(fig)
print(f"wrote {p2}")

# ------------------------------------------------------------------ stats
with open(os.path.join(OUT, "unified_compare_stats.txt"), "w") as f:
    for net in data:
        d = data[net]["df"]
        f.write(f"== {NETWORKS[net]['title']} ==\n")
        f.write(f"pairs in QC'd table: {d['pair'].nunique():,}\n")
        f.write(f"distance: median {d.groupby('pair')['distance'].first().median():.1f} km\n")
        for (w, m), sub in d.groupby(["wave_type", "mode"]):
            f.write(f"  {w:8s} {m:11s}: group {int((sub.group_ok == 1).sum()):,} "
                    f"| phase {int((sub.phase_ok == 1).sum()):,}\n")
        f.write("\n")
    f.write("\n".join(stats_lines) + "\n")
print(f"wrote {os.path.join(OUT, 'unified_compare_stats.txt')}")
