"""2D histograms of the final validated picks (fundamental / overtone, group & phase) and a
map of ray paths that present confirmed higher-mode picks.

Inputs : paths.dispersion_dir/*/*_{dispersion_all,modes_validated}.csv, station coords from
         paths.vsg_dir, reference curves from paths.ref_dir.
Outputs: (in project_dir) network_pick_histograms.png, map_overtone_rays.png,
         overtone_ray_counts.csv

Usage:  python network_pick_maps.py --config ../../param_files/modesep_params.yaml
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import modesep_config

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--config", required=True, help="network YAML (param_files/modesep_params.yaml)")
args = ap.parse_args()
cfg = modesep_config.load_config(args.config)
ROOT = cfg["paths"]["project_dir"]
V6 = cfg["paths"]["dispersion_dir"]
NETNAME = cfg["network"].get("name", "")
MINOT = int(cfg["analysis"].get("min_overtone_periods", 3))   # "robust overtone path" threshold
_refp = modesep_config.ref_curve_paths(cfg)
ref_c = {b: np.loadtxt(_refp[b]) for b in ("fundamental", "overtone")}

# ---------------- gather validated group picks + per-pair overtone counts ----------------
gf, go = [], []           # (T, U) confirmed fundamental / overtone
ot_count = {}             # pair -> n confirmed overtone periods
vfiles = sorted(glob.glob(os.path.join(V6, "*", "*_modes_validated.csv")))
for vf in vfiles:
    pr = os.path.basename(vf).replace("_modes_validated.csv", "")
    try:
        v = pd.read_csv(vf)
    except Exception:
        continue
    a = v[v.fund_flag == "ok"]
    gf.append(np.column_stack([a.period, a.U_fund]))
    b = v[v.ot_flag == "ok"]
    go.append(np.column_stack([b.period, b.U_overtone]))
    ot_count[pr] = len(b)
GF, GO = np.vstack(gf), np.vstack(go)

# ---------------- gather confirmed phase picks ----------------
pf, po = [], []
for f in sorted(glob.glob(os.path.join(V6, "*", "*_dispersion_all.csv"))):
    if "LINEAR" in f:
        continue
    vf = f.replace("_dispersion_all.csv", "_modes_validated.csv")
    if not os.path.exists(vf):
        continue
    try:
        d = pd.read_csv(f, usecols=["nominal_period", "phase_velocity", "lag",
                                    "component", "pick_method"])
        v = pd.read_csv(vf)
    except Exception:
        continue
    d["phase_velocity"] = pd.to_numeric(d["phase_velocity"], errors="coerce")
    s = d[(d.lag == "sym") & (d.pick_method == "argmax") & np.isfinite(d.phase_velocity)]
    okf = set(v[v.fund_flag == "ok"].period.round(2))
    oko = set(v[v.ot_flag == "ok"].period.round(2))
    q = s[(s.component == "G_LR0") & s.nominal_period.round(2).isin(okf)]
    pf.append(np.column_stack([q.nominal_period, q.phase_velocity]))
    q = s[(s.component == "G_LR1") & s.nominal_period.round(2).isin(oko)]
    po.append(np.column_stack([q.nominal_period, q.phase_velocity]))
PF, PO = np.vstack(pf), np.vstack(po)

# ---------------- figure 1: 2D histograms ----------------
Tb = np.arange(0.2, 6.05, 0.1)
Vb = np.arange(0.5, 4.55, 0.05)
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
panels = [(GF, "group, fundamental (validator-confirmed)", "fundamental", True),
          (GO, "group, overtone (validator-confirmed)", "overtone", True),
          (PF, "phase, fundamental (confirmed + gated)", "fundamental", False),
          (PO, "phase, overtone (confirmed + gated)", "overtone", False)]
for ax, (D, title, branch, is_group) in zip(axs.ravel(), panels):
    H, xe, ye = np.histogram2d(D[:, 0], D[:, 1], bins=[Tb, Vb])
    pm = ax.pcolormesh(xe, ye, np.where(H.T > 0, H.T, np.nan), cmap="viridis",
                       norm=matplotlib.colors.LogNorm())
    plt.colorbar(pm, ax=ax, label="picks per cell")
    r = ref_c[branch]
    if is_group:
        pass                                    # group refs derived; keep phase curve off group
    else:
        ax.plot(r[:, 0], r[:, 1], "r--", lw=2, label="VSG picked c_ref")
        ax.legend(fontsize=8, loc="upper right")
    ax.set(title=f"{title}  (n={len(D):,})", xlabel="Period [s]",
           ylabel=("Group" if is_group else "Phase") + " velocity [km/s]",
           xlim=(0.2, 6), ylim=(0.5, 4.5))
fig.suptitle(f"{NETNAME} network, all pairs: 2D histograms of final validated picks", y=0.995)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "network_pick_histograms.png"), dpi=130)
plt.close(fig)

# ---------------- station coordinates from the VSG per-source files ----------------
coords = modesep_config.vsg_station_coords(cfg["paths"]["vsg_dir"])
print(f"stations with coordinates: {len(coords)}")

# ---------------- figure 2: overtone ray map ----------------
segs_all, segs_ot, wts = [], [], []
for pr, n in ot_count.items():
    s1, s2 = pr.split("_")
    if s1 not in coords or s2 not in coords:
        continue
    seg = [coords[s1], coords[s2]]
    segs_all.append(seg)
    if n >= MINOT:                              # robust overtone path
        segs_ot.append(seg)
        wts.append(n)
wts = np.array(wts)
pd.DataFrame([(p, n) for p, n in ot_count.items()], columns=["pair", "n_ot_confirmed"]) \
    .to_csv(os.path.join(ROOT, "overtone_ray_counts.csv"), index=False)

fig, ax = plt.subplots(figsize=(11, 10))
ax.add_collection(LineCollection(segs_all, colors="0.85", linewidths=0.3, alpha=0.35,
                                 zorder=1))
lc = LineCollection(segs_ot, array=np.clip(wts, MINOT, 25), cmap="plasma",
                    linewidths=1.0, alpha=0.65, zorder=2)
ax.add_collection(lc)
plt.colorbar(lc, ax=ax, label="confirmed overtone periods per pair (clipped at 25)")
lons = np.array([c[0] for c in coords.values()])
lats = np.array([c[1] for c in coords.values()])
ax.plot(lons, lats, "^", ms=4, mfc="k", mec="w", mew=0.4, zorder=3, label="stations")
ax.set_xlim(lons.min() - 0.03, lons.max() + 0.03)
ax.set_ylim(lats.min() - 0.02, lats.max() + 0.02)
ax.set_aspect(1.0 / np.cos(np.radians(lats.mean())))
ax.set(xlabel="Longitude", ylabel="Latitude",
       title=f"{NETNAME}: ray paths with confirmed 1st-higher-mode picks (>={MINOT} periods): "
             f"{len(segs_ot)} of {len(segs_all)} pairs")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "map_overtone_rays.png"), dpi=130)

n_any = sum(1 for n in ot_count.values() if n >= 1)
n3 = sum(1 for n in ot_count.values() if n >= MINOT)
n5 = sum(1 for n in ot_count.values() if n >= 5)
print(f"pairs: {len(ot_count)} | overtone >=1 period: {n_any} ({100*n_any/len(ot_count):.0f}%) "
      f"| >={MINOT}: {n3} ({100*n3/len(ot_count):.0f}%) | >=5: {n5} ({100*n5/len(ot_count):.0f}%)")
print("wrote network_pick_histograms.png, map_overtone_rays.png, overtone_ray_counts.csv")
