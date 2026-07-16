"""PER-PAIR noise-regime clustering of CCF substacks, and what each regime's stack reveals.

The question this answers: within a single station pair, cluster that pair's own ~200 time-window
substacks by wavefield character, stack each cluster separately, and ask whether a given cluster's
stack HIGHLIGHTS a particular wave type -- i.e. does some regime give a cleaner Rayleigh (ZZ /
G_LR0) or a cleaner Love (TT) dispersion image and better SNR than simply stacking everything?

Each pair is clustered INDEPENDENTLY (its own k chosen by silhouette). No cross-pair consensus is
required or attempted: an earlier pooled-consensus test showed window quality fluctuates
incoherently across pairs (epoch-median ellipticity varies only 0.05-0.07 network-wide while
WITHIN a pair it spans 0.01-0.15), so regimes are pair-local by nature -- which is exactly what
per-pair clustering assumes.

Per pair: cluster -> stack each cluster + an all-window stack (same linear code path, so the only
variable is which windows go in) -> rotate ENZ->RTZ -> for each of ZZ (Rayleigh), TT (Love) and
G_LR0 (mode-separated Rayleigh fundamental) measure:
    snr_med    median narrowband SNR over the resolvable band          (HIGHER = better)
    ridge_dev  median |U_argmax(T) - U_ref(T)| vs the network VSG group reference, over periods
               where the far-field rule holds                          (LOWER = cleaner dispersion)
    n_pick     number of ridge points surviving the wavelength rule    (HIGHER = more usable band)

Aggregated tests:
  1. does the BEST cluster beat the all-window stack, per wave type?
  2. SPECIALIZATION: is the best cluster for Rayleigh a DIFFERENT cluster than for Love? (that is
     the signature of regimes illuminating different wave types)

CONTROL (essential -- without it every number above is uninterpretable): each pair also gets a
SIZE-MATCHED RANDOM partition (its window labels shuffled into clusters of identical sizes, groups
"r0","r1",...). "Best cluster beats all" is a post-hoc max over k clusters, so the winner's curse
inflates it even for meaningless clusters; and a cluster holds fewer windows than the all-stack, so
its SNR is handicapped ~sqrt(n_cluster/n_all). The random partition carries BOTH effects and none of
the physics, so the only meaningful statistic is REAL vs RANDOM, not real vs all. Likewise the
specialization chance level is NOT 50%: for k clusters it is (k-1)/k, so it is computed per pair
from that pair's own k and reported alongside.

Outputs (in Projects/<net>/regime_pilot/): per_pair_cluster_metrics.csv, per_pair_regime_report.txt,
per_pair_regime_summary.png, per_pair_regime_examples.png (FTAN images per cluster for example pairs).

Usage: /opt/anaconda3/bin/python per_pair_regime_stack.py --net {riehen,aargau} [--n-pairs 60]
       (base env: needs sklearn + pycwt + findpeaks; PYTHONPATH=NoisePy-ant)
"""
import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from noisepy import dispersion  # noqa: E402
from noisepy import unified_picking as up  # noqa: E402
from noisepy.stacking import rotation  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

ENZ_ORDER = ["EE", "EN", "EZ", "NE", "NN", "NZ", "ZE", "ZN", "ZZ"]
RTZ_ORDER = ["ZR", "ZT", "ZZ", "RR", "RT", "RZ", "TR", "TT", "TZ"]
NETS = {
    "aargau": {"stack": "/Volumes/T7blue/aargau-data/STACK_CHAA_normZ",
               "proj": "/Users/genevievesavard/Codes/extract_higher_modes/Projects/aargau"},
    "riehen": {"stack": "/Volumes/T7blue/riehen-data/STACK_CHRI_normZ",
               "proj": "/Users/genevievesavard/Codes/extract_higher_modes/Projects/riehen"},
}
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--net", required=True, choices=list(NETS))
ap.add_argument("--n-pairs", type=int, default=60)
ap.add_argument("--kmin", type=int, default=2)
ap.add_argument("--kmax", type=int, default=4)
ap.add_argument("--min-win", type=int, default=15, help="min windows for a cluster to be stacked")
args = ap.parse_args()
PROJ = NETS[args.net]["proj"]
ROOT = os.path.join(PROJ, "regime_pilot")
CFG = up.Config

# ---------------------------------------------------------------- references (group form)
per_grid_ref = np.arange(0.2, 8.0, 0.1)
REFG = {}
for comp, fn in (("ZZ", "ref_fundamental_phase.txt"), ("G_LR0", "ref_fundamental_phase.txt"),
                 ("TT", "ref_love_phase.txt")):
    try:
        REFG[comp] = up.phase_ref_to_group_ref(
            dispersion.load_reference_curve(os.path.join(PROJ, "vsg_modesep", fn)), per_grid_ref)
    except Exception as e:
        print(f"WARN: no group reference for {comp} ({e})")
        REFG[comp] = None

z = np.load(os.path.join(ROOT, "window_features.npz"), allow_pickle=True)
F, meta = z["F"], z["meta"]
pairs_all = [str(p) for p in z["pairs"]]
print(f"[{args.net}] {len(F):,} windows over {len(pairs_all)} pairs; "
      f"clustering EACH pair independently (k={args.kmin}..{args.kmax} by silhouette)", flush=True)

sel_pairs = [ip for ip in np.unique(meta[:, 0]) if (meta[:, 0] == ip).sum() >= 4 * args.min_win]
sel_pairs = sel_pairs[:: max(1, len(sel_pairs) // args.n_pairs)][: args.n_pairs]


def measure(sig, dist, dt, comp):
    """SNR + dispersion-ridge agreement vs the network reference for one stacked trace."""
    out = {"snr_med": np.nan, "ridge_dev": np.nan, "n_pick": 0}
    Tmax = dist / CFG.vave
    per_grid = np.arange(CFG.Tmin, Tmax, CFG.dT)
    if len(per_grid) < 3:
        return out
    try:
        snr, _, _, _ = dispersion.nb_filt_gauss(sig, dt, 1.0 / per_grid, dist,
                                                alpha=CFG.gauss_alpha, vmin=CFG.vmin, vmax=CFG.vmax)
        out["snr_med"] = float(np.median(snr))
        cwt = dispersion.compute_cwt(sig, dist, dt, Tmin=CFG.Tmin, vmin=CFG.vmin, vmax=CFG.vmax,
                                     vave=CFG.vave)
        amp, per, vel, coi = dispersion.disp_image_from_cwt(
            cwt, dist, Tmin=CFG.Tmin, dT=CFG.dT, vmin=CFG.vmin, vmax=CFG.vmax, dvel=CFG.dvel,
            vave=CFG.vave)
        nper, gv, sc = dispersion.extract_dispersion(amp, per, vel, dist, vmax=CFG.vmax,
                                                     maxgap=CFG.maxgap, minlambda=2.0,
                                                     segments=True, min_seg=CFG.MIN_SEG)
        nper, gv, sc = dispersion.remove_picks_coi(np.asarray(nper), np.asarray(gv),
                                                   np.asarray(sc), vel, coi)
        out["n_pick"] = int(len(nper))
        R = REFG.get(comp)
        if R is not None and len(nper):
            uref = np.asarray(R(np.asarray(nper, float)), float)
            d = np.abs(np.asarray(gv, float) - uref)
            d = d[np.isfinite(d)]
            if len(d):
                out["ridge_dev"] = float(np.median(d))
        return out, (amp, per, vel)
    except Exception:
        return out, None


rows = []
examples = {}
for n, ip in enumerate(sel_pairs):
    pair = pairs_all[ip]
    src = pair.split("_")[0]
    f = os.path.join(NETS[args.net]["stack"], src, pair + ".h5")
    if not os.path.exists(f):
        continue
    m = meta[:, 0] == ip
    Fp, eps = F[m], meta[m, 1]
    good = np.isfinite(Fp).all(axis=1)
    Fp, eps = Fp[good], eps[good]
    if len(Fp) < 4 * args.min_win:
        continue
    # ---- cluster THIS pair only ----
    mu, sd = Fp.mean(0), Fp.std(0)
    Fz = np.nan_to_num((Fp - mu) / np.where(sd > 0, sd, 1.0))
    best = None
    for k in range(args.kmin, args.kmax + 1):
        if len(Fz) < k * args.min_win:
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Fz)
        try:
            s = silhouette_score(Fz, km.labels_)
        except Exception:
            continue
        if best is None or s > best[0]:
            best = (s, k, km.labels_)
    if best is None:
        continue
    sil, K, labels = best
    groups = {"all": set(eps.tolist())}
    for c in range(K):
        ce = set(eps[labels == c].tolist())
        if len(ce) >= args.min_win:
            groups[f"c{c}"] = ce
    # size-matched random control partition (same cluster sizes, physics-free)
    rng = np.random.RandomState(1234 + int(ip))
    shuf = rng.permutation(len(eps))
    pos = 0
    for c in range(K):
        nc = int((labels == c).sum())
        idx = shuf[pos:pos + nc]
        pos += nc
        if nc >= args.min_win:
            groups[f"r{c}"] = set(eps[idx].tolist())
    # ---- stack each group, rotate, measure ----
    try:
        with h5py.File(f, "r") as h:
            g = h["AuxiliaryData"]
            tkeys = sorted(k_ for k_ in g if k_.startswith("T"))
            a = g[tkeys[0]]["ZZ"].attrs
            dist, dt = float(a["dist"]), float(a["dt"])
            azi, baz = float(a["azi"]), float(a["baz"])
            acc = {gn: None for gn in groups}
            cnt = {gn: 0 for gn in groups}
            for tk in tkeys:
                grp = g[tk]
                if any(c_ not in grp for c_ in ENZ_ORDER):
                    continue
                ep = int(tk[1:])
                big = None
                for gn, eset in groups.items():
                    if ep in eset:
                        if big is None:
                            big = np.stack([np.asarray(grp[c_][:], np.float64) for c_ in ENZ_ORDER])
                        acc[gn] = big.copy() if acc[gn] is None else acc[gn] + big
                        cnt[gn] += 1
    except Exception as e:
        print(f"  {pair}: read failed ({e})", flush=True)
        continue
    for gn in groups:
        if acc[gn] is None or cnt[gn] < args.min_win:
            continue
        rt = rotation(acc[gn].astype(np.float32), {"azi": azi, "baz": baz}, {})
        d = {c_: rt[i] for i, c_ in enumerate(RTZ_ORDER)}
        sym = {c_: 0.5 * (v[len(v) // 2:] + v[: len(v) // 2 + 1][::-1]) for c_, v in d.items()}
        # mode-separated Rayleigh fundamental from this group's own stack
        try:
            c0, _c1 = dispersion.phase_corrected_components(sym["ZZ"], sym["RR"], sym["RZ"],
                                                            sym["ZR"])
            g_lr0 = dispersion.tf_pws(c0, dt)
        except Exception:
            g_lr0 = None
        for comp, sig in (("ZZ", sym["ZZ"]), ("TT", sym["TT"]), ("G_LR0", g_lr0)):
            if sig is None:
                continue
            res = measure(sig, dist, dt, comp)
            met, img = res if isinstance(res, tuple) else (res, None)
            rows.append({"pair": pair, "group": gn, "k": K, "silhouette": sil,
                         "n_win": cnt[gn], "comp": comp, **met})
            if len(examples) < 3 and gn == "all" and comp == "ZZ":
                examples[pair] = {"dist": dist, "dt": dt, "groups": {}}
            if pair in examples and img is not None and comp in ("ZZ", "TT"):
                examples[pair]["groups"].setdefault(gn, {})[comp] = img
    if (n + 1) % 10 == 0:
        print(f"  {n + 1}/{len(sel_pairs)} pairs clustered+stacked+measured", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "per_pair_cluster_metrics.csv"), index=False)
print(f"[{args.net}] {len(df)} (pair, group, comp) measurements", flush=True)

# ---------------------------------------------------------------- aggregate tests
lines = [f"=== per-pair regime stacking: {args.net} ===",
         f"pairs analyzed: {df.pair.nunique()} | k per pair: "
         f"{df.groupby('pair').k.first().value_counts().sort_index().to_dict()} | "
         f"median silhouette {df.groupby('pair').silhouette.first().median():.3f}", ""]
def _best(g, metric, prefix, better):
    v = g[g.group.str.startswith(prefix)][metric].dropna()
    if not len(v):
        return np.nan
    return float(v.max() if better == "higher" else v.min())


for comp in ["ZZ", "G_LR0", "TT"]:
    sub = df[df.comp == comp]
    if sub.empty:
        continue
    lines.append(f"--- {comp} ---")
    for metric, better in (("snr_med", "higher"), ("ridge_dev", "lower"), ("n_pick", "higher")):
        w_all = t_all = w_rnd = t_rnd = 0
        d_all, d_rnd = [], []
        for pair, g in sub.groupby("pair"):
            if "all" not in set(g.group):
                continue
            base = g[g.group == "all"][metric].values[0]
            breal = _best(g, metric, "c", better)
            brand = _best(g, metric, "r", better)
            if np.isfinite(base) and np.isfinite(breal):
                t_all += 1
                w_all += int((breal > base) if better == "higher" else (breal < base))
                d_all.append((breal - base) / abs(base) * 100 if base else np.nan)
            if np.isfinite(brand) and np.isfinite(breal):
                t_rnd += 1
                w_rnd += int((breal > brand) if better == "higher" else (breal < brand))
                d_rnd.append((breal - brand) / abs(brand) * 100 if brand else np.nan)
        if t_all:
            lines.append(f"  {metric:9s} best-real vs ALL   : {w_all}/{t_all} "
                         f"({100*w_all/t_all:3.0f}%) median {np.nanmedian(d_all):+6.1f}%   "
                         f"[post-hoc max -> inflated, not a test]")
        if t_rnd:
            lines.append(f"  {metric:9s} best-real vs RANDOM: {w_rnd}/{t_rnd} "
                         f"({100*w_rnd/t_rnd:3.0f}%) median {np.nanmedian(d_rnd):+6.1f}%   "
                         f"<-- THE TEST (50% = clusters carry no information)")
    lines.append("")
# SPECIALIZATION: is the best cluster for Rayleigh a different one than for Love?
# Chance is (k-1)/k per pair, NOT 50% -- accumulate the per-pair expectation.
same = diff = 0
exp_chance = []
for pair, g in df.groupby("pair"):
    r = g[(g.comp == "G_LR0") & g.group.str.startswith("c")].dropna(subset=["snr_med"])
    l = g[(g.comp == "TT") & g.group.str.startswith("c")].dropna(subset=["snr_med"])
    if len(r) < 2 or len(l) < 2:
        continue
    kk = min(len(r), len(l))
    exp_chance.append((kk - 1) / kk)
    if r.loc[r.snr_med.idxmax(), "group"] == l.loc[l.snr_med.idxmax(), "group"]:
        same += 1
    else:
        diff += 1
if same + diff:
    ch = 100 * float(np.mean(exp_chance))
    obs = 100 * diff / (same + diff)
    lines.append("SPECIALIZATION (best-SNR cluster, Rayleigh G_LR0 vs Love TT):")
    lines.append(f"  different cluster wins for R vs L: {diff}/{same+diff} pairs ({obs:.0f}%)")
    lines.append(f"  chance for the observed k-mix     : {ch:.0f}%  -> "
                 f"{'ABOVE chance' if obs > ch + 10 else 'consistent with CHANCE (no specialization)'}")
report = "\n".join(lines)
print("\n" + report)
with open(os.path.join(ROOT, "per_pair_regime_report.txt"), "w") as fo:
    fo.write(report + "\n")

# ---------------------------------------------------------------- figures
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, comp in zip(axs, ["ZZ", "G_LR0", "TT"]):
    sub = df[df.comp == comp]
    for gi, (lab, col) in enumerate([("all", "0.4"), ("clusters", "tab:blue")]):
        v = (sub[sub.group == "all"].snr_med if lab == "all"
             else sub[sub.group != "all"].snr_med)
        ax.hist(np.log10(v.dropna() + 1e-9), bins=30, alpha=0.55, color=col, label=lab)
    ax.set(title=f"{comp}: log10 median SNR", xlabel="log10 SNR")
    ax.legend()
fig.suptitle(f"{args.net}: per-pair cluster stacks vs all-window stack")
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "per_pair_regime_summary.png"), dpi=120)
plt.close(fig)

if examples:
    npair = len(examples)
    ncol = max(len(e["groups"]) for e in examples.values())
    fig, axs = plt.subplots(2 * npair, ncol, figsize=(3.4 * ncol, 3.0 * 2 * npair), squeeze=False)
    for ip_, (pair, e) in enumerate(examples.items()):
        for ic, (gn, imgs) in enumerate(sorted(e["groups"].items())):
            for j, comp in enumerate(["ZZ", "TT"]):
                ax = axs[2 * ip_ + j, ic]
                if comp not in imgs:
                    ax.axis("off")
                    continue
                amp, per, vel = imgs[comp]
                ax.pcolormesh(per, vel, amp.T, cmap="jet", shading="auto")
                R = REFG.get(comp)
                if R is not None:
                    uu = np.asarray(R(per), float)
                    ax.plot(per, uu, "w--", lw=1)
                ax.set(title=f"{pair[:18]} {gn} {comp}", xlim=(0.2, 5), ylim=(0.5, 4))
    fig.suptitle(f"{args.net}: FTAN per cluster stack (white = network group reference)", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "per_pair_regime_examples.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
print(f"wrote figures + report to {ROOT}")
