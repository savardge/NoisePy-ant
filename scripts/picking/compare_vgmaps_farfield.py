"""Before/after comparison of swtomotv production group-velocity maps: original picks vs the
far-field (>= 2.5 lambda) re-filtered picks, per network and wave, at selected periods.

Rows = periods, columns = (before, after, after - before). Same grid, same LC/sigma_eff, same
station set -- the ONLY difference is the pick filter, so the difference column isolates what
the near-field picks were doing to the maps. Expect: at short periods, small differences; at
long periods, the filtered maps become FASTER overall (the removed short paths were biased
slow) and lose coverage at the edges (fewer qualifying paths).

Run:  /opt/anaconda3/bin/python compare_vgmaps_farfield.py
Outputs: Projects/<net>/tomo/farfield2p5/compare_vgmaps_<wave>.png
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml as _yaml
from pyproj import Transformer

PROJROOT = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
OLD_ROOT = {"aargau": "swtomotv-output-500m", "riehen": "swtomotv-output-200m"}
R_EARTH_KM = 6371.0
_TR = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)


def xy2ll(x, y, lat0, lon0):
    lat = lat0 + np.degrees(np.asarray(y) / R_EARTH_KM)
    lon = lon0 + np.degrees(np.asarray(x) / (R_EARTH_KM *
                            np.cos(np.radians((lat + lat0) / 2))))
    return lat, lon


def grid_lv95_km(cfg):
    """Cell-center LV95-km meshes (nx, ny) for the dataset YAML's grid."""
    b, dx = cfg["bounds"], float(cfg["dx_km"])
    lat0, lon0 = b[0], b[2]
    xmax, ymax = (R_EARTH_KM * np.cos(np.radians((b[1] + lat0) / 2))
                  * np.radians(b[3] - lon0),
                  R_EARTH_KM * np.radians(b[1] - lat0))
    x = np.arange(0.0, np.ceil(xmax) + dx / 2, dx) + dx / 2
    y = np.arange(0.0, np.ceil(ymax) + dx / 2, dx) + dx / 2
    X, Y = np.meshgrid(x, y, indexing="ij")
    lat, lon = xy2ll(X, Y, lat0, lon0)
    e, n = _TR.transform(lon, lat)
    return np.asarray(e) / 1e3, np.asarray(n) / 1e3


def compare(net, wave, n_periods=4):
    tomo = os.path.join(PROJROOT, net, "tomo")
    ffdir = os.path.join(tomo, "farfield2p5")
    cfg = _yaml.safe_load(open(os.path.join(ffdir, f"{net}_swtomotv_ff2p5.yaml")))
    E, N = grid_lv95_km(cfg)
    old_dir = os.path.join(tomo, OLD_ROOT[net], "production", wave)
    new_dir = os.path.join(ffdir, "swtomotv-output", "production", wave)
    newcsv = pd.read_csv(os.path.join(new_dir, f"production_{wave}.csv"))
    Ts = newcsv[newcsv["N"] >= 100]["T"].values
    if not len(Ts):
        print(f"  {net} {wave}: no well-populated periods"); return
    sel = sorted({float(Ts[int(round(i))]) for i in np.linspace(0, len(Ts) - 1, n_periods)})
    st = pd.read_csv(os.path.join(tomo, "stations.csv"))
    se, sn = _TR.transform(st["longitude"].values, st["latitude"].values)
    fig, axs = plt.subplots(len(sel), 3, figsize=(15, 4.6 * len(sel)),
                            sharex=True, sharey=True, layout="constrained")
    axs = np.atleast_2d(axs)
    for i, T in enumerate(sel):
        fo = os.path.join(old_dir, f"map_T{T:g}.npz")
        fn = os.path.join(new_dir, f"map_T{T:g}.npz")
        if not (os.path.exists(fo) and os.path.exists(fn)):
            for a in axs[i]:
                a.set_axis_off()
            continue
        o, n_ = np.load(fo), np.load(fn)
        Vo, Vn = o["vel"], n_["vel"]
        both = np.isfinite(Vo) & np.isfinite(Vn)
        fin = np.concatenate([Vo[np.isfinite(Vo)], Vn[np.isfinite(Vn)]])
        vlo, vhi = np.nanpercentile(fin, 2), np.nanpercentile(fin, 98)
        dv = np.where(both, Vn - Vo, np.nan)
        dmax = max(0.02, float(np.nanpercentile(np.abs(dv), 98)))
        for a, D, cmap, vmin, vmax, lab in (
                (axs[i, 0], Vo, "RdYlBu", vlo, vhi,
                 f"before (N={int(o['N'])}, vr={float(o['var_red']):.2f})"),
                (axs[i, 1], Vn, "RdYlBu", vlo, vhi,
                 f"after >=2.5$\\lambda$ (N={int(n_['N'])}, vr={float(n_['var_red']):.2f})"),
                (axs[i, 2], dv, "coolwarm", -dmax, dmax,
                 f"after - before (median {1e3*np.nanmedian(dv):+.0f} m/s)")):
            pc = a.pcolormesh(E, N, D, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
            a.plot(se / 1e3, sn / 1e3, "v", ms=2, mfc="k", mec="none", ls="none")
            a.set_aspect("equal")
            a.set_title(f"T = {T:g} s — {lab}", fontsize=9)
            fig.colorbar(pc, ax=a, fraction=0.045, pad=0.02).set_label(
                "vg [km/s]" if cmap == "RdYlBu" else "dvg [km/s]")
        axs[i, 0].set_ylabel("N LV95 [km]")
    for a in axs[-1]:
        a.set_xlabel("E LV95 [km]")
    fig.suptitle(f"{net.capitalize()} {wave} — production vg maps, original vs far-field-"
                 f"filtered picks (same grid/LC/$\\sigma$; only the pick filter differs)",
                 fontsize=12)
    out = os.path.join(ffdir, f"compare_vgmaps_{wave}.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("  wrote", out)


if __name__ == "__main__":
    for net in ("aargau", "riehen"):
        for wave in ("fund", "overtone"):
            compare(net, wave)
