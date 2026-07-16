"""Before/after Vs comparison: original picks vs far-field (>= 2.5 lambda) filtered picks.

Everything else in the two pipelines is identical (grid, LC/sigma_eff, physical period trim,
BayHunter settings), so the differences isolate the near-field group-velocity bias fix.
Per network, one figure:
  row 1: stacked median Vs(z) profiles (old vs new; the old deep LVZ + 6 km rebound should
         flatten), plus posterior-width medians;
  row 2: kriged N-S section, OLD;
  row 3: kriged N-S section, NEW (same colour scale, equal km:km aspect, z_res veils).

Run:  /opt/anaconda3/bin/python compare_vs_farfield.py
Outputs: Projects/<net>/tomo/farfield2p5/compare_vs_<net>.png
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import smooth_compare as S

PROJROOT = S.PROJROOT
PAIRS = {"aargau": ("grid_physical_500m", "grid_physical_500m_ff2p5"),
         "riehen": ("grid_physical_200m", "grid_physical_200m_ff2p5")}


def kriged_section(R, coord):
    """Heteroscedastic-kriging NS section (nz, ny) at E=coord for a loaded run."""
    j = int(np.argmin(np.abs(R["gx"] - coord)))
    nz = len(R["depth"])
    sec = np.full((nz, len(R["gy"])), np.nan)
    cover = (R["dmin"] <= S.MASK_L * R["L"])[j, :]
    for k in range(nz):
        m, _ = S.level_krige_unc(R, R["vs"][:, k], R["sd"][:, k])
        sec[k, :] = np.where(cover, m[j, :], np.nan)
    return sec, R["gy"]


def compare(net):
    old_dir, new_dir = (os.path.join(PROJROOT, net, "tomo", "vs_inversion", d)
                        for d in PAIRS[net])
    Ro, Rn = S.load_run(net, old_dir), S.load_run(net, new_dir)
    zro, _ = S.load_zres(Ro)
    zrn, _ = S.load_zres(Rn)
    z = Ro["depth"]
    coord = float(np.percentile(Rn["xd"], 50))
    So, yo = kriged_section(Ro, coord)
    Sn, yn = kriged_section(Rn, coord)

    fig = plt.figure(figsize=(15, 12), layout="constrained")
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1, 1])
    # --- stacked median profiles
    a = fig.add_subplot(gs[0, 0])
    for R, col, lab in ((Ro, "tab:red", "before (all picks)"),
                        (Rn, "tab:blue", "after (>= 2.5$\\lambda$)")):
        med = np.nanmedian(R["vs"], axis=0)
        p16 = np.nanpercentile(R["vs"], 25, axis=0)
        p84 = np.nanpercentile(R["vs"], 75, axis=0)
        a.plot(med, R["depth"], color=col, lw=2, label=f"{lab} (n={len(R['xd'])} cells)")
        a.fill_betweenx(R["depth"], p16, p84, color=col, alpha=0.15)
    a.axhspan(4.0, 5.5, color="gray", alpha=0.15)
    a.invert_yaxis(); a.legend(fontsize=9)
    a.set_xlabel("median Vs [km/s]"); a.set_ylabel("z [km]")
    a.set_title("stacked cell profiles (median + IQR); gray band = old artifact LVZ")
    a = fig.add_subplot(gs[0, 1])
    for R, col, lab in ((Ro, "tab:red", "before"), (Rn, "tab:blue", "after")):
        a.plot(np.nanmedian(R["sd"], axis=0), R["depth"], color=col, lw=2, label=lab)
    a.invert_yaxis(); a.legend(fontsize=9)
    a.set_xlabel("median posterior 68% half-width [km/s]"); a.set_ylabel("z [km]")
    a.set_title("posterior uncertainty vs depth")
    # --- sections, shared colour scale
    fin = np.concatenate([So[np.isfinite(So)], Sn[np.isfinite(Sn)]])
    vlo, vhi = np.nanpercentile(fin, 2), np.nanpercentile(fin, 98)
    for row, (sec, yy, R, zr, tag) in enumerate(
            ((So, yo, Ro, zro, "BEFORE — all picks"),
             (Sn, yn, Rn, zrn, "AFTER — far-field (>= 2.5$\\lambda$) picks")), start=1):
        a = fig.add_subplot(gs[row, :])
        pc = a.pcolormesh(yy, z, sec, cmap=S.CMAP, vmin=vlo, vmax=vhi, shading="gouraud")
        if zr is not None:
            j = int(np.argmin(np.abs(R["gx"] - coord)))
            S._veil_below_zres(a, R["gy"], z, zr[j, :])
        a.invert_yaxis(); a.set_aspect("equal")
        a.set_ylabel("z [km]")
        a.set_title(f"{tag} — kriged NS section at E = {coord*1e3:.0f} m "
                    f"(dashed = resolution depth)", fontsize=10)
        fig.colorbar(pc, ax=a, fraction=0.03, pad=0.01).set_label("Vs [km/s]")
    a.set_xlabel("N LV95 [km] (S $\\rightarrow$ N)")
    fig.suptitle(f"{net.capitalize()} — effect of the far-field pick filter on the Vs model "
                 f"(same tomography/inversion settings; only the picks differ)", fontsize=13)
    out = os.path.join(PROJROOT, net, "tomo", "farfield2p5", f"compare_vs_{net}.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    for net in ("aargau", "riehen"):
        compare(net)
