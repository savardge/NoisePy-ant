#!/usr/bin/env python
"""Where does the mode-identification gate remove Love group samples?

The gate's justification is physical: the U>=c picks it removes were argued to be
branch-misidentified fundamentals, excited in basins and low-relief ground. That predicts the
drops are SPATIALLY COHERENT and anti-correlated with elevation -- not scattered. This maps
the per-cell drop count over DEM hillshade and swisstopo tectonic lines so the prediction can
be checked directly.
"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uc_consistency_maps import basemap, load_assets            # noqa: E402
from vs_model_figures import km_xy, E                            # noqa: E402

NETS = ["riehen", "aargau", "hautesorne"]
GATE = "_gate_eval/gatedrops_%s_love.csv"

fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
for ax, net in zip(axes, NETS):
    rows = list(csv.DictReader(open(os.path.join(E, GATE % net))))
    ij = np.array([[int(r["ix"]), int(r["iy"])] for r in rows])
    nd = np.array([int(r["dropped"]) for r in rows], float)

    vol = np.load(f"{E}/{net}/tomo/2_vs_depth_inversion/vs_prod3/L0g/volume_love.npz",
                  allow_pickle=True)
    key = {tuple(c): i for i, c in enumerate(vol["cells"])}
    sel = np.array([key[tuple(c)] for c in ij])
    ll = vol["lonlat"][sel]

    elev, extent, gk, hs, ext_km = load_assets(net)
    basemap(ax, net, elev, gk, ext_km, hs, wells=True, labels=True)
    x, y = km_xy(ll[:, 0], ll[:, 1])

    hit = nd > 0
    lim = max(np.percentile(nd[hit], 95), 1) if hit.any() else 1
    ax.scatter(x[~hit], y[~hit], s=6, c="0.75", alpha=.45, lw=0, zorder=3)
    s = ax.scatter(x[hit], y[hit], c=nd[hit], s=17, cmap="inferno_r", vmin=1, vmax=lim,
                   zorder=4, lw=0)
    plt.colorbar(s, ax=ax, label="periods dropped", shrink=.8)
    ax.set_aspect("equal")
    ax.set_title(f"{net} — {100*hit.mean():.0f}% of cells gated\n"
                 f"{int(nd.sum())} of samples dropped, max {int(nd.max())} in one cell")

    # the physical prediction: more drops where the ground is low
    ex = np.array(extent, float)
    ny_, nx_ = elev.shape
    xi = np.clip(((ll[:,0]-ex[0])/(ex[1]-ex[0])*nx_).astype(int), 0, nx_-1)
    yi = np.clip(((ll[:,1]-ex[2])/(ex[3]-ex[2])*ny_).astype(int), 0, ny_-1)
    ez = elev[ny_-1-yi, xi]
    g = np.isfinite(ez) & np.isfinite(nd)
    if np.unique(nd[g]).size > 1 and np.unique(ez[g]).size > 1:
        r = spearmanr(ez[g], nd[g])
        print(f"{net:11s} drops vs elevation: rho={r.statistic:+.3f} p={r.pvalue:.2g}  "
              f"(cells gated {100*hit.mean():.0f}%)")
        ax.text(.02, .02, f"ρ(elev, drops) = {r.statistic:+.2f}", transform=ax.transAxes,
                fontsize=9, va="bottom", bbox=dict(fc="w", alpha=.8, ec="0.6"))

fig.suptitle("Mode-identification gate: Love group samples removed per cell "
             "(grey = untouched). Prediction: coherent patches in low ground, not scatter.",
             fontsize=13)
fig.tight_layout()
o = f"{E}/_gate_eval/gate_drop_maps.png"
fig.savefig(o, dpi=140)
print("wrote", o)
