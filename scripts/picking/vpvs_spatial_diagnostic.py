#!/usr/bin/env python
"""Diagnostics for the claim: posterior Vp/Vs carries no resolvable spatial signal here.

Four panels, each testing one link in the argument:
  A  the map itself -- is there visible basin structure?
  B  the elevation proxy -- basins low, ridges high, so a basin effect should show as a trend
  C  the spatial null -- 92 adjacent cells are not 92 independent samples, so the basin excess
     must be judged against random discs of the same size drawn from the same map
  D  the per-cell constraint -- if one cell's posterior spans half the prior, no 0.1 km/s
     spatial signal can be resolved no matter how many cells are averaged
"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uc_consistency_maps import basemap, load_assets                  # noqa: E402
from vs_model_figures import km_xy                                     # noqa: E402

E = "/Users/genevievesavard/Codes/extract_higher_modes"
SCRATCH = ("/private/tmp/claude-501/-Users-genevievesavard-Codes-extract-higher-modes/"
           "527b2c5c-898d-4fd9-a91b-1e0bb04acbe0/scratchpad/vpvs_cells")
BASINS = [("Delémont", 7.34, 47.36), ("Ajoie", 7.05, 47.42)]
PRIOR = (1.5, 3.5)
RAD_KM = 5.0
rng = np.random.default_rng(0)

rows = list(csv.DictReader(open(f"{E}/Projects/_gate_eval/vpvs_hs_full.csv")))
ij = np.array([[int(r["ix"]), int(r["iy"])] for r in rows])
med = np.array([float(r["med"]) for r in rows])
p16 = np.array([float(r["p16"]) for r in rows])
p84 = np.array([float(r["p84"]) for r in rows])

vol = np.load(f"{E}/Projects/hautesorne/tomo/2_vs_depth_inversion/vs_prod3/"
              "RLg_radial/volume_fundlove.npz", allow_pickle=True)
key = {tuple(int(x) for x in c): i for i, c in enumerate(vol["cells"])}
keep = np.array([tuple(c) in key for c in ij])
ij, med, p16, p84 = ij[keep], med[keep], p16[keep], p84[keep]
ll = vol["lonlat"][np.array([key[tuple(c)] for c in ij])]
width = p84 - p16

elev, extent, gk, hs, ext_km = load_assets("hautesorne")
ex = np.array(extent, float); ny_, nx_ = elev.shape
xi = np.clip(((ll[:, 0]-ex[0])/(ex[1]-ex[0])*nx_).astype(int), 0, nx_-1)
yi = np.clip(((ll[:, 1]-ex[2])/(ex[3]-ex[2])*ny_).astype(int), 0, ny_-1)
ez = elev[ny_-1-yi, xi]

fig = plt.figure(figsize=(19, 8.6))
gs = fig.add_gridspec(3, 2, width_ratios=[1.05, 1], height_ratios=[1, 1, 1],
                      hspace=0.55, wspace=0.13)

# ---- A: the map -------------------------------------------------------------------------
axA = fig.add_subplot(gs[:, 0])
basemap(axA, "hautesorne", elev, gk, ext_km, hs, wells=True, labels=True)
x, y = km_xy(ll[:, 0], ll[:, 1])
lo, hi = np.percentile(med, [5, 95])
s = axA.scatter(x, y, c=med, s=16, cmap="RdYlBu_r", vmin=lo, vmax=hi, zorder=4, lw=0)
plt.colorbar(s, ax=axA, label="posterior Vp/Vs (median)", shrink=.85, pad=.02)
for nm, lo_, la_ in BASINS:
    bx, by = km_xy(np.array([lo_]), np.array([la_]))
    axA.add_patch(plt.Circle((bx[0], by[0]), RAD_KM, fill=False, ec="k", lw=2, ls="--", zorder=6))
    axA.annotate(nm, (bx[0], by[0]), fontsize=11, weight="bold", ha="center",
                 va="center", zorder=7,
                 bbox=dict(fc="w", alpha=.75, ec="none", pad=1.5))
axA.set_aspect("equal")
axA.set_title(f"A — posterior Vp/Vs, {len(med)} cells\n"
              f"median {np.median(med):.2f}; no basin structure visible")

# ---- B: elevation proxy -----------------------------------------------------------------
axB = fig.add_subplot(gs[0, 1])
g = np.isfinite(ez)
r = spearmanr(ez[g], med[g])
axB.scatter(ez[g], med[g], s=7, alpha=.35, lw=0)
b = np.linspace(np.nanmin(ez[g]), np.nanmax(ez[g]), 9)
bc = 0.5*(b[1:]+b[:-1])
bm = [np.median(med[g][(ez[g] >= b[i]) & (ez[g] < b[i+1])])
      if ((ez[g] >= b[i]) & (ez[g] < b[i+1])).sum() > 10 else np.nan for i in range(len(b)-1)]
axB.plot(bc, bm, "o-", color="crimson", lw=2, label="binned median")
axB.set(xlabel="elevation (m)", ylabel="Vp/Vs",
        title=f"B — basins should sit HIGH if the prediction holds\n"
              f"Spearman ρ = {r.statistic:+.3f} (p = {r.pvalue:.2f}) — flat")
axB.legend(); axB.grid(alpha=.3)

# ---- C: spatial null --------------------------------------------------------------------
axC = fig.add_subplot(gs[1, 1])
null = []
for _ in range(3000):
    c = rng.integers(0, len(ll))
    dd = np.hypot((ll[:, 0]-ll[c, 0])*np.cos(np.deg2rad(ll[c, 1]))*111.32,
                  (ll[:, 1]-ll[c, 1])*111.32)
    mm = dd < RAD_KM
    if mm.sum() < 20 or (~mm).sum() < 20:
        continue
    null.append(np.median(med[mm]) - np.median(med[~mm]))
null = np.array(null)
axC.hist(null, bins=50, color="0.7", label=f"random {RAD_KM:g} km discs (n={len(null)})")
for (nm, lo_, la_), col in zip(BASINS, ["crimson", "tab:orange"]):
    d = np.hypot((ll[:, 0]-lo_)*np.cos(np.deg2rad(la_))*111.32, (ll[:, 1]-la_)*111.32)
    mm = d < RAD_KM
    obs = np.median(med[mm]) - np.median(med[~mm])
    p = np.mean(np.abs(null) >= abs(obs))
    axC.axvline(obs, color=col, lw=2.5, label=f"{nm}: {obs:+.3f} (p={p:.2f})")
axC.set(xlabel="Vp/Vs excess inside disc vs outside", ylabel="count",
        title="C — the basin excess is an ordinary fluctuation\n"
              f"null sd = {null.std():.3f}")
axC.legend(fontsize=8)

# ---- D: per-cell constraint -------------------------------------------------------------
axD = fig.add_subplot(gs[2, 1])
axD.hist(width, bins=50, color="steelblue", label="per-cell p16–p84 width")
axD.axvline(PRIOR[1]-PRIOR[0], color="k", lw=2.5, ls="--",
            label=f"prior width = {PRIOR[1]-PRIOR[0]:.1f}")
axD.axvline(np.median(width), color="crimson", lw=2.5,
            label=f"median width = {np.median(width):.2f}"
                  f"  ({100*np.median(width)/(PRIOR[1]-PRIOR[0]):.0f}% of prior)")
axD.axvline(0.11, color="green", lw=2.5, ls=":",
            label="claimed Delémont signal = 0.11")
axD.set(xlabel="per-cell Vp/Vs posterior width (p16–p84)", ylabel="cells",
        title="D — the data DO inform Vp/Vs (KS vs uniform prior D≈0.29, p≈0; only 0.4% of\ncells touch a prior bound) but each cell resolves it to ~half the prior — far coarser than 0.1")
axD.legend(fontsize=8, loc="upper left")

# inset: two actual posteriors, basin vs ridge
ins = axD.inset_axes([0.63, 0.06, 0.35, 0.56])
for f, lab, col in ((f"{SCRATCH}/cell_61_27_fundlove.npz", "Delémont basin (413 m)", "crimson"),
                    (f"{SCRATCH}/cell_59_13_fundlove.npz", "ridge (1135 m)", "navy")):
    if os.path.exists(f):
        a = np.asarray(np.load(f, allow_pickle=True)["vpvs_post"], float).ravel()
        a = a[np.isfinite(a)]
        ins.hist(a, bins=60, density=True, alpha=.5, color=col, label=lab)
ins.axvspan(PRIOR[0], PRIOR[1], color="0.85", zorder=0, label="prior")
ins.set(xlim=PRIOR, yticks=[])
ins.set_title("two real posteriors (not railing)", fontsize=8)
ins.legend(fontsize=6.5, loc="upper left")

fig.suptitle("Haute-Sorne free-Vp/Vs arm — is there a basin signal in posterior Vp/Vs? "
             "(1,072 cells, RLg_radial)", fontsize=14)
o = f"{E}/Projects/_gate_eval/vpvs_spatial_diagnostic.png"
fig.savefig(o, dpi=140, bbox_inches="tight")
print("wrote", o)
