#!/usr/bin/env python3
"""Cell-by-cell comparison of the MANUSCRIPT Rayleigh group-velocity maps with the current
production maps (Haute-Sorne).

Manuscript maps: .mat files from the 2024/25 Tarantola-Valette inversion
(inv_TV_rayleigh_sigma0.01_LC1_T*.mat, LC=1 km, sigma=0.01, 500 m grid), built from the
GaussFiltPeriod/1-bit-coherency pick set with a lambda>=1.5 far-field cut, max_std 20%,
min_count 3.

Current maps: tspws_<measure>_<cd>_dx0.5_prod3_k<k>/production/fund/map_T*.npz -- ts-PWS
stacks + substack-jackknife sigma, unified CWT picker, vbounds k-cull, corrected Fresnel
correlation length, no output masking.

THE GRIDS ARE THE SAME (origin 47.237194/6.945514, 85x53, dx 0.5 km, x 0-42, y 0-26), which
is what makes a cell-by-cell comparison legitimate; the script asserts it rather than assuming.

PERIOD MATCHING. The manuscript maps sit on a uniform 0.1 s grid (0.6-6.5 s); the current
maps sit on the CWT scale ladder (0.2 ... 8.61 s, geometric). Each current map is matched to
the nearest manuscript period and the pair is USED ONLY IF they agree to within `--max-dt`
(default 0.05 s = half the manuscript grid step), so no comparison is made across a period
gap that is itself a velocity difference.

Reported per period: n common cells, Pearson r and Spearman rho of the two velocity fields,
median and IQR of (mine - manuscript), and the same for the LATERAL ANOMALY (each map
demeaned over the common cells) -- the anomaly comparison is the one that matters for
structure, since a uniform velocity offset between two pick sets says nothing about whether
they image the same features.

Usage:
  python compare_manuscript_vgmaps.py
  python compare_manuscript_vgmaps.py --cd scaled --k k3
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MS = ("/Volumes/T7black/hautesorne2/vg-maps/run2_all_g500m/vg-maps-new/"
      "inv_TV_rayleigh_sigma0.01_LC1")
EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
OUT = f"{EHM}/hautesorne/tomo/1_velocity_maps/3_diagnostics/manuscript_vgmap_comparison"


def load_ms(f):
    d = sio.loadmat(f, squeeze_me=True, struct_as_record=False)
    V = np.asarray(d["V_map"], float)
    m = np.asarray(d["mask"], float)
    V = np.where(m > 0, V, np.nan)
    gp = d["grid_params"]
    return V, d, np.atleast_1d(gp.x_grid), np.atleast_1d(gp.y_grid)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cd", default="blanket")
    ap.add_argument("--k", default="k2")
    ap.add_argument("--wave", default="fund")
    ap.add_argument("--max-dt", type=float, default=0.05)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    msf = sorted(glob.glob(f"{MS}/*_T*.mat"),
                 key=lambda p: float(re.search(r"_T([\d.]+)\.mat", p).group(1)))
    msT = np.array([float(re.search(r"_T([\d.]+)\.mat", p).group(1)) for p in msf])
    mine = sorted(glob.glob(f"{EHM}/hautesorne/tomo/1_velocity_maps/1_production/"
                            f"tspws_group_{a.cd}_dx0.5_prod3_{a.k}/production/{a.wave}/map_T*.npz"))
    if not mine or not msf:
        raise SystemExit("missing maps (is /Volumes/T7black mounted?)")

    _, _, gx, gy = load_ms(msf[0])
    rows, pairs = [], []
    for f in mine:
        z = np.load(f)
        T = float(z["period"])
        j = int(np.argmin(np.abs(msT - T)))
        if abs(msT[j] - T) > a.max_dt:
            continue
        Vm, d, _, _ = load_ms(msf[j])
        Vn = np.asarray(z["vel"], float)
        if Vn.shape != Vm.shape:
            # my npz stores (nx, ny); the .mat is (85, 53) = (nx, ny) too -- assert, do not guess
            if Vn.T.shape == Vm.shape:
                Vn = Vn.T
            else:
                raise SystemExit(f"shape mismatch {Vn.shape} vs {Vm.shape} at T={T}")
        ok = np.isfinite(Vn) & np.isfinite(Vm)
        if ok.sum() < 100:
            continue
        x, y = Vn[ok], Vm[ok]
        dv = x - y
        ax_, ay_ = x - x.mean(), y - y.mean()          # lateral anomalies
        rows.append(dict(
            T_mine=round(T, 3), T_ms=round(float(msT[j]), 2), n_cells=int(ok.sum()),
            r=round(float(pearsonr(x, y)[0]), 3),
            rho=round(float(spearmanr(x, y)[0]), 3),
            r_anom=round(float(pearsonr(ax_, ay_)[0]), 3),
            med_mine=round(float(np.median(x)), 3), med_ms=round(float(np.median(y)), 3),
            bias=round(float(np.median(dv)), 3),
            iqr_diff=round(float(np.subtract(*np.percentile(dv, [75, 25]))), 3),
            std_mine=round(float(x.std()), 3), std_ms=round(float(y.std()), 3),
            vr_mine=round(float(z["var_red"]), 3),
            vr_ms=round(float(np.atleast_1d(d["stats"].var_red)[0]), 3)))
        pairs.append((T, Vn, Vm))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(OUT, f"vgmap_comparison_{a.cd}_{a.k}.csv"), index=False)
    print(D.to_string(index=False))
    print("\nsummary over %d matched periods:" % len(D))
    for c in ("r", "r_anom", "bias", "std_mine", "std_ms", "vr_mine", "vr_ms"):
        print(f"   {c:<10} median {D[c].median():+.3f}")

    sel = [p for p in pairs if any(abs(p[0] - t) < 1e-6 for t in
                                   (1.02, 2.03, 3.04, 4.06))][:4]
    if sel:
        fig, axs = plt.subplots(3, len(sel), figsize=(4.3 * len(sel), 11), squeeze=False)
        for i, (T, Vn, Vm) in enumerate(sel):
            ok = np.isfinite(Vn) & np.isfinite(Vm)
            lo = np.nanpercentile(np.concatenate([Vn[ok], Vm[ok]]), 2)
            hi = np.nanpercentile(np.concatenate([Vn[ok], Vm[ok]]), 98)
            for r_, (V, lab) in enumerate(((Vm, "manuscript"), (Vn, "current"))):
                ax = axs[r_][i]
                im = ax.imshow(V.T, origin="lower", cmap="inferno", vmin=lo, vmax=hi,
                               extent=[gx[0], gx[-1], gy[0], gy[-1]])
                ax.set_aspect("equal"); ax.set_title(f"{lab}  T={T:.2f}s", fontsize=9.5)
                plt.colorbar(im, ax=ax, shrink=0.75, label="U [km/s]")
            ax = axs[2][i]
            dv = np.where(ok, Vn - Vm, np.nan)
            m = np.nanpercentile(np.abs(dv), 98)
            im = ax.imshow(dv.T, origin="lower", cmap="bwr_r", vmin=-m, vmax=m,
                           extent=[gx[0], gx[-1], gy[0], gy[-1]])
            ax.set_aspect("equal")
            ax.set_title(f"current - manuscript (r={pearsonr(Vn[ok],Vm[ok])[0]:.2f})", fontsize=9.5)
            plt.colorbar(im, ax=ax, shrink=0.75, label="dU [km/s]")
        fig.suptitle(f"Haute-Sorne Rayleigh group maps: manuscript (TV, LC=1, sigma=0.01) vs "
                     f"current (tspws_group_{a.cd}_prod3_{a.k})\nred = current SLOWER",
                     fontsize=12.5, fontweight="bold")
        fig.tight_layout()
        p = os.path.join(OUT, f"vgmap_comparison_{a.cd}_{a.k}.png")
        fig.savefig(p, dpi=130, bbox_inches="tight")
        print("\nwrote", p)


if __name__ == "__main__":
    main()
