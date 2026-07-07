"""Lateral-structure maps from the mode-separated picks: does the fundamental split into a
slow / fast domain across a mapped fault, and does the higher mode concentrate on
fault-crossing paths?

Fundamental group velocity per pair (mean over confirmed periods in analysis.lateral_band) is
placed at the path MIDPOINT and as a per-station average; overtone occurrence per pair is
tested against crossing of the mapped faults (paths.faults_shapefile; the main fault =
analysis.fault_main_name defines the West/East split).

Graceful degradation: the fault analysis needs geopandas+shapely (base anaconda env only) AND a
faults_shapefile. Without either, panels A/B/C (Vg midpoints, station averages, overtone
midpoints) and the corr(Vg, longitude) statistic are still produced; the fault overlays,
West/East split, and fault-crossing statistics are skipped.

Inputs : paths.dispersion_dir/*/*_modes_validated.csv, coords from paths.vsg_dir,
         optional paths.faults_shapefile.
Outputs: (in project_dir) lateral_structure.png, lateral_structure_stats.txt

Usage (full):   /opt/anaconda3/bin/python lateral_structure_map.py --config <yaml>
Usage (no flt): python lateral_structure_map.py --config <yaml>   # any env; maps only
"""
import argparse
import glob
import os
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import modesep_config

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--config", required=True, help="network YAML (param_files/modesep_params.yaml)")
args = ap.parse_args()
cfg = modesep_config.load_config(args.config)
ROOT = cfg["paths"]["project_dir"]
V6 = cfg["paths"]["dispersion_dir"]
NETNAME = cfg["network"].get("name", "")
BAND = tuple(cfg["analysis"].get("lateral_band", (1.5, 2.5)))
BANDLAB = f"{BAND[0]:.1f}-{BAND[1]:.1f} s"
FAULTS = cfg["paths"].get("faults_shapefile")
FAULT_MAIN = cfg["analysis"].get("fault_main_name")

# ---- optional geopandas/shapely for the fault analysis ----
HAVE_GPD = False
if FAULTS and os.path.exists(FAULTS):
    try:
        import geopandas as gpd
        from shapely.geometry import LineString
        HAVE_GPD = True
    except Exception:
        print("geopandas/shapely unavailable -> maps only (no fault analysis). "
              "Run with /opt/anaconda3/bin/python for the full fault analysis.")

# ---------------- station coordinates + per-pair Vg(band) + overtone count ----------------
coords = modesep_config.vsg_station_coords(cfg["paths"]["vsg_dir"])
rows = []
for vf in glob.glob(os.path.join(V6, "*", "*_modes_validated.csv")):
    pr = os.path.basename(vf).replace("_modes_validated.csv", "")
    s1, s2 = pr.split("_")
    if s1 not in coords or s2 not in coords:
        continue
    try:
        v = pd.read_csv(vf)
    except Exception:
        continue
    if len(v) == 0:
        continue
    a = v[(v.fund_flag == "ok") & (v.period >= BAND[0]) & (v.period <= BAND[1])]
    lo1, la1 = coords[s1]; lo2, la2 = coords[s2]
    rows.append(dict(pair=pr, lo1=lo1, la1=la1, lo2=lo2, la2=la2,
                     mlon=(lo1 + lo2) / 2, mlat=(la1 + la2) / 2,
                     Uf=a.U_fund.mean() if len(a) else np.nan,
                     n_ot=int((v.ot_flag == "ok").sum())))
df = pd.DataFrame(rows)

# ---------------- faults + crossing test (optional) ----------------
gdf = flex = None
if HAVE_GPD:
    gdf = gpd.read_file(FAULTS)
    allf = gdf.geometry.unary_union
    paths = gpd.GeoSeries([LineString([(r.lo1, r.la1), (r.lo2, r.la2)]) for _, r in df.iterrows()])
    df["cross_any"] = paths.intersects(allf).values
    if FAULT_MAIN and (gdf.Name == FAULT_MAIN).any():
        flex = gdf[gdf.Name == FAULT_MAIN].geometry.unary_union
        df["cross_flex"] = paths.intersects(flex).values
        fx, fy = flex.xy
        df["side"] = np.where(
            df.mlon < [float(np.interp(la, np.array(fy), np.array(fx))) for la in df.mlat],
            "West", "East")

# per-station average Vg
from collections import defaultdict
acc = defaultdict(list)
for _, r in df.dropna(subset=["Uf"]).iterrows():
    acc[r.pair.split("_")[0]].append(r.Uf); acc[r.pair.split("_")[1]].append(r.Uf)
sta = {k: np.mean(w) for k, w in acc.items() if len(w) >= 5 and k in coords}

# ---------------- geographic frame ----------------
alon = np.array([c[0] for c in coords.values()]); alat = np.array([c[1] for c in coords.values()])
xlim = (alon.min() - 0.01, alon.max() + 0.01); ylim = (alat.min() - 0.01, alat.max() + 0.01)
ASP = 1.0 / np.cos(np.radians(alat.mean()))

def draw_faults(ax, label=False):
    if gdf is None:
        return
    for _, row in gdf.iterrows():
        g = row.geometry
        gs = [g] if g.geom_type == "LineString" else list(g.geoms)
        is_main = row.Name == FAULT_MAIN
        for gg in gs:
            x, y = gg.xy
            ax.plot(x, y, color=("k" if is_main else "0.45"),
                    lw=(2.6 if is_main else 1.0), ls=("-" if is_main else "--"),
                    zorder=5, alpha=0.9)
        if label and row.Name and xlim[0] < gg.centroid.x < xlim[1] \
                and ylim[0] < gg.centroid.y < ylim[1]:
            ax.annotate(row.Name, (gg.centroid.x, gg.centroid.y), fontsize=6.5,
                        color=("k" if is_main else "0.35"), zorder=6)

def frame(ax):
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect(ASP)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

fig, axs = plt.subplots(2, 2, figsize=(16, 13))
med = df.Uf.median(); vlo, vhi = df.Uf.quantile(0.08), df.Uf.quantile(0.92)

# --- A: midpoint Vg ---
ax = axs[0, 0]; d = df.dropna(subset=["Uf"])
sc = ax.scatter(d.mlon, d.mlat, c=d.Uf, cmap="RdYlBu", vmin=vlo, vmax=vhi, s=9, alpha=0.75, zorder=3)
draw_faults(ax, label=True)
plt.colorbar(sc, ax=ax, label=f"fundamental Vg [km/s], {BANDLAB}", shrink=0.8)
ax.set_title(f"A. Fundamental group velocity by path midpoint ({BANDLAB})\n"
             "blue=fast  red=slow" + ("; black=" + str(FAULT_MAIN) if flex is not None else ""))
frame(ax)

# --- B: station-average Vg ---
ax = axs[0, 1]
sx = [coords[k][0] for k in sta]; sy = [coords[k][1] for k in sta]; sv = [sta[k] for k in sta]
sc = ax.scatter(sx, sy, c=sv, cmap="RdYlBu", vmin=vlo, vmax=vhi, s=95, ec="k", lw=0.5, zorder=4)
draw_faults(ax, label=False)
plt.colorbar(sc, ax=ax, label=f"station-mean fundamental Vg [km/s], {BANDLAB}", shrink=0.8)
ax.set_title("B. Per-station average fundamental Vg\n(mean over all paths at each node)")
frame(ax)

# --- C: higher-mode occurrence ---
ax = axs[1, 0]
base = df[df.n_ot == 0]; ot = df[df.n_ot >= 1].sort_values("n_ot")
ax.scatter(base.mlon, base.mlat, c="0.8", s=5, zorder=2, label="no overtone")
sc = ax.scatter(ot.mlon, ot.mlat, c=ot.n_ot, cmap="magma_r", s=14,
                vmin=1, vmax=np.percentile(ot.n_ot, 95) if len(ot) else 1, zorder=3)
draw_faults(ax, label=False)
plt.colorbar(sc, ax=ax, label="# confirmed overtone periods (per pair)", shrink=0.8)
ax.set_title("C. Higher-mode occurrence by path midpoint")
frame(ax); ax.legend(loc="lower right", fontsize=8)

# --- D: two branches (W/E) if fault split available, else Vg distribution + corr ---
ax = axs[1, 1]
corr = np.corrcoef(df.dropna(subset=["Uf"]).mlon, df.dropna(subset=["Uf"]).Uf)[0, 1]
bins = np.arange(0.6, 2.6, 0.06)
if flex is not None and "side" in df:
    for side, col in (("West", "#c0392b"), ("East", "#2c6fbb")):
        s = df[(df.side == side)].Uf.dropna()
        ax.hist(s, bins=bins, alpha=0.6, color=col, density=True,
                label=f"{side} of {FAULT_MAIN} (n={len(s)}, med {s.median():.2f})")
        ax.axvline(s.median(), color=col, ls="--")
    ax.set_title(f"D. Two branches = West (slow) vs East (fast)   corr(Vg,lon)={corr:+.2f}")

    def rate(m):
        dd = df[m]; return len(dd), 100 * (dd.n_ot >= 1).mean(), 100 * (dd.n_ot >= 3).mean()
    lines = ["Higher-mode vs fault crossing:"]
    if "cross_flex" in df:
        lines.append("  cross %-14s: n=%d  %.0f%% ot>=1  %.0f%% ot>=3" %
                     (FAULT_MAIN[:14], *rate(df.cross_flex)))
        lines.append("  no cross          : n=%d  %.0f%% ot>=1  %.0f%% ot>=3" % rate(~df.cross_flex))
    lines.append("  cross any fault   : n=%d  %.0f%% ot>=1  %.0f%% ot>=3" % rate(df.cross_any))
    lines.append("  cross no fault    : n=%d  %.0f%% ot>=1  %.0f%% ot>=3" % rate(~df.cross_any))
    ax.text(0.02, 0.97, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=8,
            family="monospace", bbox=dict(fc="w", ec="0.6", alpha=0.9))
else:
    ax.hist(df.Uf.dropna(), bins=bins, color="0.5", density=True)
    ax.set_title(f"D. Fundamental Vg distribution   corr(Vg, longitude)={corr:+.2f}\n"
                 "(no fault split: geopandas/shapefile unavailable)")
ax.set(xlabel=f"fundamental Vg [km/s], {BANDLAB}", ylabel="density")
ax.legend(fontsize=8)

fig.suptitle(f"{NETNAME}: lateral velocity contrast and higher-mode generation", fontsize=14, y=0.995)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "lateral_structure.png"), dpi=140)
print("wrote lateral_structure.png")

with open(os.path.join(ROOT, "lateral_structure_stats.txt"), "w") as fh:
    fh.write(f"band {BANDLAB}; pairs with Vg {df.Uf.notna().sum()}\n")
    fh.write(f"corr(Vg, midpoint lon) = {corr:+.3f}  (positive => faster to the east)\n")
    if flex is not None and "side" in df:
        fh.write(f"West-of-{FAULT_MAIN} median Vg = {df[df.side=='West'].Uf.median():.3f}, "
                 f"East = {df[df.side=='East'].Uf.median():.3f} km/s\n\n")

        def rate(m):
            dd = df[m]; return len(dd), 100 * (dd.n_ot >= 1).mean(), 100 * (dd.n_ot >= 3).mean()
        tests = [(f"cross {FAULT_MAIN}", df.cross_flex), (f"no-cross {FAULT_MAIN}", ~df.cross_flex)] \
            if "cross_flex" in df else []
        tests += [("cross any fault", df.cross_any), ("cross no fault", ~df.cross_any)]
        for label, m in tests:
            n, a, b = rate(m); fh.write(f"{label:30s}: n={n:5d}  ot>=1 {a:4.1f}%  ot>=3 {b:4.1f}%\n")
    else:
        fh.write("(fault analysis skipped: no geopandas or no faults_shapefile)\n")
print("wrote lateral_structure_stats.txt")
