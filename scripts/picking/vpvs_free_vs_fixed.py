"""Paired comparison of a radial arm with FIXED Vp/Vs (1.73) against its FREE-Vp/Vs twin.

Runs on whatever cells the free arm has finished (they finish independently, and shards are
strided, so a partial set is a spatial sample). Every number is per-cell paired and restricted
to the depth window that is reliable in BOTH arms.

Why this comparison exists: BayHunter uses ONE Vp/Vs for the whole profile, and Love waves are
blind to Vp while Rayleigh is not. So a wrong ratio biases only the Rayleigh side, and in a
joint R+L inversion the only free parameter that can absorb a Rayleigh-only offset is gamma.
Freeing the ratio therefore tests whether the deep negative gamma is structure or the ratio.

Writes: <out>/vpvs_compare.png (Vs + gamma distributions by depth, before/after) and a stats
table on stdout. Run on the cluster where the cells are:

  python vpvs_free_vs_fixed.py --fixed <cells dir> --free <cells dir> --out <dir>
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEPTHS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
FLOOR = 0.15


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fixed", required=True)
    ap.add_argument("--free", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="hautesorne RLg_radial")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    z = None
    G = {"fix": [], "free": []}; V = {"fix": [], "free": []}
    zmin, zmax, vp_med, vp_lo, vp_hi, cd_fix, cd_free = [], [], [], [], [], [], []
    files = sorted(glob.glob(os.path.join(a.free, "cell_*.npz")))
    for f in files:
        g = os.path.join(a.fixed, os.path.basename(f))
        if not os.path.exists(g):
            continue
        try:
            A = np.load(g, allow_pickle=True); B = np.load(f, allow_pickle=True)
        except Exception:
            continue
        if z is None:
            z = np.asarray(A["depth"], float)
        G["fix"].append(np.asarray(A["gamma_median"], float))
        G["free"].append(np.asarray(B["gamma_median"], float))
        V["fix"].append(np.asarray(A["vs_median"], float))
        V["free"].append(np.asarray(B["vs_median"], float))
        zmax.append(min(float(A["z_reliable_max"]), float(B["z_reliable_max"])))
        zmin.append(max(float(A["z_reliable_min"]) if "z_reliable_min" in A.files else 0.0,
                        float(B["z_reliable_min"]) if "z_reliable_min" in B.files else 0.0))
        vp = np.asarray(B["vpvs_post"], float).ravel()
        vp_med.append(np.median(vp)); vp_lo.append(np.percentile(vp, 16)); vp_hi.append(np.percentile(vp, 84))
        cd_fix.append(float(A["chain_disagree"])); cd_free.append(float(B["chain_disagree"]))
    n = len(zmax)
    print(f"{a.label}: {n} paired cells (free arm has {len(files)} finished)")
    if n < 5:
        return
    for k in G:
        G[k] = np.array(G[k]); V[k] = np.array(V[k])
    zmin, zmax = np.array(zmin), np.array(zmax)
    vp_med = np.array(vp_med); cd_fix = np.array(cd_fix); cd_free = np.array(cd_free)
    # mask outside the joint reliable window
    ok = (z[None, :] >= zmin[:, None]) & (z[None, :] <= zmax[:, None])
    for k in G:
        G[k] = np.where(ok, G[k], np.nan); V[k] = np.where(ok, V[k], np.nan)

    # ------------------------------------------------------------------ stats
    print(f"\nposterior Vp/Vs (free): median of cell medians {np.median(vp_med):.3f}, "
          f"p5 {np.percentile(vp_med,5):.3f}, p95 {np.percentile(vp_med,95):.3f}; "
          f"rail hi(>=3.4) {100*np.mean(vp_med>=3.4):.0f}%  lo(<=1.6) {100*np.mean(vp_med<=1.6):.0f}%")
    print(f"chain_disagree: fixed {np.median(cd_fix):.3f} -> free {np.median(cd_free):.3f}; "
          f"free better in {100*np.mean(cd_free<cd_fix):.0f}% of cells")
    print(f"\n{'z':>5}{'n':>6}{'gamma fix':>11}{'gamma free':>12}{'|g|>.15 fix':>13}{'|g|>.15 free':>14}"
          f"{'Vs fix':>9}{'Vs free':>9}{'dVs':>8}")
    for d in DEPTHS:
        k = int(np.argmin(np.abs(z - d)))
        gf, gr = G["fix"][:, k], G["free"][:, k]
        vf, vr = V["fix"][:, k], V["free"][:, k]
        m = np.isfinite(gf) & np.isfinite(gr)
        if m.sum() < 5:
            continue
        print(f"{d:>5.1f}{m.sum():>6}{np.median(gf[m]):>+11.3f}{np.median(gr[m]):>+12.3f}"
              f"{100*np.mean(abs(gf[m])>=FLOOR):>12.0f}%{100*np.mean(abs(gr[m])>=FLOOR):>13.0f}%"
              f"{np.median(vf[m]):>9.2f}{np.median(vr[m]):>9.2f}{np.median(vr[m]-vf[m]):>+8.3f}")

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    cf, cr = "tab:gray", "tab:red"

    # (a) gamma(z): median + 16-84 band, both arms
    ax = axes[0, 0]
    for k, c, lab in (("fix", cf, "Vp/Vs fixed 1.73"), ("free", cr, "Vp/Vs free 1.5-3.5")):
        med = np.nanmedian(G[k], axis=0); lo = np.nanpercentile(G[k], 16, axis=0)
        hi = np.nanpercentile(G[k], 84, axis=0)
        ax.plot(med, z, color=c, lw=2, label=lab); ax.fill_betweenx(z, lo, hi, color=c, alpha=0.18)
    ax.axvline(0, color="k", lw=0.8); ax.axvspan(-FLOOR, FLOOR, color="0.85", zorder=0)
    ax.set_ylim(z.max(), 0); ax.set_xlabel("gamma = (Vsh-Vsv)/Vsv"); ax.set_ylabel("depth [km]")
    ax.set_title("gamma(z): median and 16-84% across cells"); ax.legend(fontsize=8); ax.grid(alpha=.3)

    # (b) Vs(z): same
    ax = axes[0, 1]
    for k, c, lab in (("fix", cf, "fixed"), ("free", cr, "free")):
        med = np.nanmedian(V[k], axis=0); lo = np.nanpercentile(V[k], 16, axis=0)
        hi = np.nanpercentile(V[k], 84, axis=0)
        ax.plot(med, z, color=c, lw=2, label=lab); ax.fill_betweenx(z, lo, hi, color=c, alpha=0.18)
    ax.set_ylim(z.max(), 0); ax.set_xlabel("Vs (Vsv) [km/s]"); ax.set_title("Vs(z): median and 16-84%")
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    # (c) posterior Vp/Vs
    ax = axes[0, 2]
    ax.hist(vp_med, bins=30, color=cr, alpha=0.8)
    ax.axvline(1.73, color=cf, lw=2, ls="--", label="fixed value 1.73")
    ax.axvline(np.median(vp_med), color="k", lw=1.5, label=f"free median {np.median(vp_med):.2f}")
    ax.set_xlim(1.5, 3.5); ax.set_xlabel("posterior median Vp/Vs per cell"); ax.set_ylabel("cells")
    ax.set_title("what the data want the ratio to be"); ax.legend(fontsize=8)

    # (d) gamma distributions at three depths, paired
    ax = axes[1, 0]
    show = [1.0, 2.5, 3.0]
    bins = np.linspace(-0.4, 0.4, 41)
    for i, d in enumerate(show):
        k = int(np.argmin(np.abs(z - d)))
        gf, gr = G["fix"][:, k], G["free"][:, k]
        m = np.isfinite(gf) & np.isfinite(gr)
        ax.hist(gf[m], bins=bins, histtype="step", lw=1.6, color=cf, alpha=0.5 + 0.25*i,
                label=f"fixed z={d}")
        ax.hist(gr[m], bins=bins, histtype="step", lw=1.6, color=cr, alpha=0.5 + 0.25*i,
                label=f"free  z={d}")
    ax.axvspan(-FLOOR, FLOOR, color="0.9", zorder=0); ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("gamma"); ax.set_ylabel("cells"); ax.set_title("gamma distributions by depth")
    ax.legend(fontsize=7, ncol=2)

    # (e) Vs distributions at the same depths
    ax = axes[1, 1]
    bins = np.linspace(1.0, 4.0, 46)
    for i, d in enumerate(show):
        k = int(np.argmin(np.abs(z - d)))
        vf, vr = V["fix"][:, k], V["free"][:, k]
        m = np.isfinite(vf) & np.isfinite(vr)
        ax.hist(vf[m], bins=bins, histtype="step", lw=1.6, color=cf, alpha=0.5 + 0.25*i,
                label=f"fixed z={d}")
        ax.hist(vr[m], bins=bins, histtype="step", lw=1.6, color=cr, alpha=0.5 + 0.25*i,
                label=f"free  z={d}")
    ax.set_xlabel("Vs [km/s]"); ax.set_ylabel("cells"); ax.set_title("Vs distributions by depth")
    ax.legend(fontsize=7, ncol=2)

    # (f) per-cell change in gamma at 2.5 km vs the fitted ratio
    ax = axes[1, 2]
    k = int(np.argmin(np.abs(z - 2.5)))
    gf, gr = G["fix"][:, k], G["free"][:, k]
    m = np.isfinite(gf) & np.isfinite(gr)
    ax.scatter(vp_med[m], gr[m] - gf[m], s=10, color=cr, alpha=0.6)
    ax.axhline(0, color="k", lw=0.8); ax.axvline(1.73, color=cf, ls="--", lw=1)
    r = np.corrcoef(vp_med[m], gr[m] - gf[m])[0, 1] if m.sum() > 5 else np.nan
    ax.set_xlabel("posterior Vp/Vs (free)"); ax.set_ylabel("gamma_free - gamma_fixed at 2.5 km")
    ax.set_title(f"per-cell shift vs fitted ratio  (r={r:+.2f})"); ax.grid(alpha=.3)

    fig.suptitle(f"{a.label}: Vp/Vs fixed 1.73 vs free [1.5, 3.5] -- {n} paired cells, "
                 f"joint reliable window only", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(a.out, "vpvs_compare.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
