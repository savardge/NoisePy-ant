"""Network consistency of V6 picks with the VSG-picked reference curves, and azimuthal
anisotropy of the residuals.

Inputs : paths.dispersion_dir/*/*_dispersion_all.csv + *_modes_validated.csv (full-network
         batch), paths.ref_dir/ref_{fundamental,overtone}_phase.txt (pick_reference_ridges.py).
Outputs: (in project_dir) network_consistency.png -- group & phase pick distributions vs refs
         network_anisotropy.png   -- 2-psi azimuthal analysis; isotropic vs anisotropic
         final_network_stats.txt  -- all printed statistics

Usage:  python network_consistency_analysis.py --config ../../param_files/modesep_params.yaml
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

import modesep_config

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--config", required=True, help="network YAML (param_files/modesep_params.yaml)")
args = ap.parse_args()
cfg = modesep_config.load_config(args.config)
ROOT = cfg["paths"]["project_dir"]
V6 = cfg["paths"]["dispersion_dir"]
NETNAME = cfg["network"].get("name", "")
_refp = modesep_config.ref_curve_paths(cfg)
_band = cfg["analysis"]
BAND_FUND = tuple(_band.get("group_ref_band_fundamental", (0.7, 3.6)))
BAND_OT = tuple(_band.get("group_ref_band_overtone", (1.1, 1.9)))

# ---------------- references: c_ref picked from VSG; U_ref by differentiation ----------------
ref_c = {b: np.loadtxt(_refp[b]) for b in ("fundamental", "overtone")}

def group_from_c(tab, Tband):
    """U(T) from a picked c(T) table: k = w/c, U = dw/dk.

    The picked c is steppy (velocity-grid quantised argmax), so differentiation needs heavy
    smoothing: dense omega grid + wide Savitzky-Golay, generous edge trim, and a validity
    band restricted to where the picked curve has real slope support.
    """
    T, c = tab[:, 0], tab[:, 1]
    order = np.argsort(1.0 / T)
    w = 2 * np.pi / T[order]; c = c[order]
    wi = np.linspace(w.min(), w.max(), 800)
    ci = np.interp(wi, w, c)
    ci = savgol_filter(ci, 121, 3)
    ci = savgol_filter(ci, 81, 2)
    k = wi / ci
    U = np.gradient(wi, k)
    Ti = 2 * np.pi / wi
    good = (Ti >= Tband[0]) & (Ti <= Tband[1])
    o = np.argsort(Ti[good])
    return Ti[good][o], U[good][o]

ref_U = {"fundamental": group_from_c(ref_c["fundamental"], BAND_FUND),
         "overtone": group_from_c(ref_c["overtone"], BAND_OT)}

# ---------------- load all pairs ----------------
gr_rows, ph_rows = [], []
files = sorted(glob.glob(os.path.join(V6, "*", "*_dispersion_all.csv")))
for f in files:
    if "LINEAR" in f:
        continue
    vfile = f.replace("_dispersion_all.csv", "_modes_validated.csv")
    if not os.path.exists(vfile):
        continue
    try:
        d = pd.read_csv(f, usecols=["nominal_period", "group_velocity", "phase_velocity",
                                    "azimuth", "distance", "lag", "component", "pick_method"])
        v = pd.read_csv(vfile)
    except Exception:
        continue
    d["phase_velocity"] = pd.to_numeric(d["phase_velocity"], errors="coerce")
    okf = set(v[v.fund_flag == "ok"].period.round(2))
    oko = set(v[v.ot_flag == "ok"].period.round(2))
    s = d[(d.lag == "sym") & (d.pick_method == "argmax")]
    for comp, confset, branch in (("G_LR0", okf, "fundamental"), ("G_LR1", oko, "overtone")):
        q = s[s.component == comp]
        conf = q.nominal_period.round(2).isin(confset)
        gr_rows.append(pd.DataFrame(dict(
            T=q.nominal_period, U=q.group_velocity, azi=q.azimuth, dist=q.distance,
            branch=branch, conf=conf)))
        qq = q[np.isfinite(q.phase_velocity)]
        if len(qq):
            ph_rows.append(pd.DataFrame(dict(
                T=qq.nominal_period, c=qq.phase_velocity, azi=qq.azimuth, dist=qq.distance,
                branch=branch, conf=qq.nominal_period.round(2).isin(confset))))
G = pd.concat(gr_rows, ignore_index=True)
P = pd.concat(ph_rows, ignore_index=True)
npairs = len({os.path.basename(f) for f in files if "LINEAR" not in f})

log = open(os.path.join(ROOT, "final_network_stats.txt"), "w")
def out(*a):
    print(*a)
    log.write(" ".join(str(x) for x in a) + "\n")

out(f"pairs loaded: {npairs} | group pick rows: {len(G)} | phase pick rows: {len(P)}")

# residuals vs reference
for df, val, reftab in ((G, "U", ref_U), (P, "c", ref_c)):
    df["ref"] = np.nan
    for b in ("fundamental", "overtone"):
        m = df.branch == b
        if reftab is ref_U:
            Tr, Vr = reftab[b]
        else:
            Tr, Vr = reftab[b][:, 0], reftab[b][:, 1]
        o = np.argsort(Tr)
        df.loc[m, "ref"] = np.interp(df.loc[m, "T"], Tr[o], Vr[o], left=np.nan, right=np.nan)
    df["res"] = df[val] - df["ref"]
    df["rres"] = df["res"] / df["ref"]

# ---------------- figure 1: consistency ----------------
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
for ax, (df, val, reftab, lab) in zip(
        axs[:, 0], [(G[G.conf], "U", ref_U, "group (validator-confirmed)"),
                    (P, "c", ref_c, "phase (gated picks)")]):
    for b, cm in (("fundamental", "Reds"), ("overtone", "Blues")):
        q = df[(df.branch == b) & np.isfinite(df[val])]
        if len(q):
            ax.hexbin(q["T"], q[val], gridsize=60, cmap=cm, mincnt=1,
                      extent=(0.2, 6, 0.5, 4.5), alpha=0.75)
    for b, c in (("fundamental", "k"), ("overtone", "0.35")):
        if reftab is ref_U:
            Tr, Vr = reftab[b]
        else:
            Tr, Vr = reftab[b][:, 0], reftab[b][:, 1]
        o = np.argsort(Tr)
        ax.plot(Tr[o], Vr[o], c, lw=2.2, ls="--")
    ax.set(xlim=(0.2, 6), ylim=(0.5, 4.5), xlabel="Period [s]",
           ylabel=("Group velocity [km/s]" if val == "U" else "Phase velocity [km/s]"),
           title=f"{lab}: picks (red=fund, blue=overtone) vs VSG-derived reference (dashed)")
for ax, (df, lab) in zip(axs[:, 1], [(G[G.conf], "group"), (P, "phase")]):
    for b, c in (("fundamental", "crimson"), ("overtone", "royalblue")):
        q = df[(df.branch == b) & np.isfinite(df.res)]
        q = q[np.abs(q.res) < 1.0]          # exclude gross outliers instead of clipping
        if len(q):
            ax.hist(q.res, bins=np.arange(-1, 1.02, 0.04), alpha=0.6,
                    color=c, label=f"{b} (n={len(q)}, med {q.res.median():+.3f}, "
                                   f"MAD {1.4826*np.median(np.abs(q.res-q.res.median())):.3f})")
        out(f"{lab} {b}: n={len(q)} median res={q.res.median():+.3f} km/s "
            f"MAD={1.4826*np.median(np.abs(q.res-q.res.median())):.3f}")
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel=f"{lab} residual vs reference [km/s]", ylabel="count",
           title=f"{lab} residual distributions")
    ax.legend(fontsize=8)
fig.suptitle(f"{NETNAME}: V6 picks vs VSG-derived reference curves", y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "network_consistency.png"), dpi=130)
plt.close(fig)

# ---------------- figure 2: azimuthal anisotropy (fundamental) ----------------
def fit_2psi(psi_deg, y, azi_deg=None):
    """y = A0 + 1psi + 2psi (+ optional 1psi needs the full 0-360 azimuth).

    Fits A0 + C1 cos(a) + S1 sin(a) + C2 cos(2a) + S2 sin(2a) with a = full azimuth when
    provided (1psi = directional noise-source bias diagnostic; 2psi = anisotropy). Returns
    (A0, amp2, fast2_deg, amp1, model2(psi)) where model2 is the A0+2psi part only.
    """
    a = np.radians(azi_deg if azi_deg is not None else psi_deg)
    X = np.column_stack([np.ones_like(a), np.cos(a), np.sin(a), np.cos(2*a), np.sin(2*a)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    A0, C1, S1, C2, S2 = coef
    amp2 = np.hypot(C2, S2)
    amp1 = np.hypot(C1, S1)
    fast = np.degrees(0.5 * np.arctan2(S2, C2)) % 180
    return A0, amp2, fast, amp1, \
        lambda q: A0 + C2*np.cos(2*np.radians(q)) + S2*np.sin(2*np.radians(q))

fig, axs = plt.subplots(2, 2, figsize=(14, 9))
panels = [(G[(G.conf) & (G.branch == "fundamental")].dropna(subset=["rres"]),
           "group dU/U (confirmed fundamental)"),
          (P[(P.branch == "fundamental")].dropna(subset=["rres"]),
           "phase dc/c (fundamental, gated)")]
for col, (q, lab) in enumerate(panels):
    q = q[np.abs(q.rres) < 0.35].copy()
    q["psi"] = q.azi % 180
    A0, amp, fast, amp1, model = fit_2psi(q.psi.values, q.rres.values, azi_deg=q.azi.values)
    out(f"2psi fit {lab}: n={len(q)} A0={A0:+.4f} amp2={amp:.4f} ({100*amp:.1f}%) "
        f"fast={fast:.0f} deg | amp1(1psi source-bias diag)={100*amp1:.1f}%")
    # per-period-band breakdown (stability test: real anisotropy -> stable fast axis)
    for tlo, thi in ((0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0)):
        qq = q[(q["T"] >= tlo) & (q["T"] < thi)]
        if len(qq) < 200:
            continue
        b0, b2, bf, b1, _ = fit_2psi(qq.psi.values, qq.rres.values, azi_deg=qq.azi.values)
        out(f"   band {tlo:.1f}-{thi:.1f}s: n={len(qq):6d} amp2={100*b2:4.1f}% "
            f"fast={bf:5.0f} deg  amp1={100*b1:4.1f}%")
    ax = axs[0, col]
    ax.hexbin(q.psi, 100 * q.rres, gridsize=45, cmap="Greys", mincnt=1)
    bins = np.arange(0, 181, 15)
    bc = 0.5 * (bins[:-1] + bins[1:])
    bm = [100 * q.rres[(q.psi >= a) & (q.psi < b)].mean() for a, b in zip(bins[:-1], bins[1:])]
    be = [100 * q.rres[(q.psi >= a) & (q.psi < b)].sem() for a, b in zip(bins[:-1], bins[1:])]
    ax.errorbar(bc, bm, yerr=be, fmt="o", color="crimson", label="15-deg bin mean")
    qq = np.linspace(0, 180, 200)
    ax.plot(qq, 100 * model(qq), "b-", lw=2,
            label=f"2-psi fit: {100*amp:.1f}% , fast {fast:.0f} deg")
    ax.axhline(100 * A0, color="b", ls=":", lw=1)
    ax.set(xlabel="pair azimuth mod 180 [deg]", ylabel="relative residual [%]",
           title=f"{lab}\nvs azimuth", ylim=(-20, 20))
    ax.legend(fontsize=8)
    # isotropic vs anisotropic distributions
    ax = axs[1, col]
    raw = 100 * q.rres
    iso = 100 * (q.rres - model(q.psi) + A0)       # 2psi term removed, mean kept
    for y, c, lab2 in ((raw, "0.4", f"raw (std {raw.std():.2f}%)"),
                       (iso, "seagreen", f"anisotropy-corrected (std {iso.std():.2f}%)")):
        ax.hist(np.clip(y, -25, 25), bins=np.arange(-25, 25.5, 1), histtype="step",
                lw=2, color=c, label=lab2, density=True)
    out(f"  variance reduction: {100*(1 - iso.var()/raw.var()):.1f}%")
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel="relative residual [%]", ylabel="density",
           title="isotropic (corrected) vs raw (anisotropic) distribution")
    ax.legend(fontsize=8)
fig.suptitle(f"{NETNAME}: azimuthal anisotropy of V6 residuals vs VSG network reference "
             "(2-psi; residuals contain path heterogeneity + anisotropy)", y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "network_anisotropy.png"), dpi=130)
log.close()
print("wrote network_consistency.png, network_anisotropy.png, final_network_stats.txt")
