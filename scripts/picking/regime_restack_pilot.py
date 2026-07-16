"""Noise-regime clustered CCF restacking pilot for the 3C geophone networks (Aargau/Riehen).

Ports the masw-das / das-ambient-noise time-regime idea (cluster time windows by wavefield
character, restack per regime) to the 9-component geophone CCFs, using the per-window substacks
that ALREADY live in the stack H5 files (`AuxiliaryData/T<epoch>`, full unrotated ENZ tensor,
~200 windows/pair) -- no re-correlation needed.

Novelty vs the DAS version: the features are PHYSICAL 9C quantities per (pair, window) --
RTZ energy partition, TT-vs-cross-term envelope ratio (per-window env_ratio), Z-R 90-degree
elliptical coherence (the exact quantity the G_LR mode synthesis relies on), causal/acausal
asymmetry (directivity) -- so the regimes are interpretable (elliptical-clean / SH-dominant /
directional) rather than generic clusters.

Pipeline (one invocation = one network):
  1. pair selection (dist >= --min-dist; Riehen stratified graben-W / basement-E by midpoint lon)
  2. pass A: per (pair, window) features (rotate ENZ->RTZ per window via stacking.rotation)
  3. per-pair z-score -> pooled KMeans, k chosen by silhouette over --kmin..--kmax
  4. cross-pair epoch consensus (modal label + agreement; DAS ODH-4 lesson: if mean agreement
     ~ 1/k there are no global regimes -- reported, and the run warns)
  5. pass B: per-regime linear restack of the ENZ substacks over consensus epochs + an ALL-window
     baseline through the SAME code path (isolates regime selection from stacking method), rotate,
     write picker-compatible H5 trees:  <out>/<stack_all|stack_regime<r>>/<NET>.<src>/<pair>.h5
     with AuxiliaryData/Allstack_linear/<comp> (run dispersion_unified with DISP_STACK=linear).

Outputs under <out>: pairs_selected.csv, window_features.npz, regime_assignment.csv,
regime_clustering.png, plus the stack trees.

Usage:
  python regime_restack_pilot.py --net riehen --n-pairs 150 [--kmin 2 --kmax 5]
  python regime_restack_pilot.py --net aargau --n-pairs 100
Run with an env that has sklearn+h5py+scipy (e.g. /opt/anaconda3/bin/python), PYTHONPATH=NoisePy-ant.
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np
from scipy.signal import hilbert

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from noisepy.stacking import rotation  # noqa: E402  (ENZ->RTZ, validated production convention)

ENZ_ORDER = ["EE", "EN", "EZ", "NE", "NN", "NZ", "ZE", "ZN", "ZZ"]
RTZ_ORDER = ["ZR", "ZT", "ZZ", "RR", "RT", "RZ", "TR", "TT", "TZ"]
CROSS_T = ["RT", "TR", "TZ", "ZT"]                    # Love cross-term context

NETS = {
    "aargau": {"stack": "/Volumes/T7blue/aargau-data/STACK_CHAA_normZ", "code": "AA",
               "vsg": "/Users/genevievesavard/Data/aargau/phasevelocity_VSG",
               "out": "/Users/genevievesavard/Codes/extract_higher_modes/Projects/aargau/regime_pilot",
               "split_lon": None},
    "riehen": {"stack": "/Volumes/T7blue/riehen-data/STACK_CHRI_normZ", "code": "RI",
               "vsg": "/Users/genevievesavard/Data/riehen/phasevelocity_VSG",
               "out": "/Users/genevievesavard/Codes/extract_higher_modes/Projects/riehen/regime_pilot",
               "split_lon": 7.645},                   # W = URG graben, E = basement
}

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--net", required=True, choices=list(NETS))
ap.add_argument("--n-pairs", type=int, default=150)
ap.add_argument("--min-dist", type=float, default=2.0, help="km; avoids too-short per_grid pairs")
ap.add_argument("--kmin", type=int, default=2)
ap.add_argument("--kmax", type=int, default=5)
ap.add_argument("--agree-min", type=float, default=0.6, help="epoch consensus agreement threshold")
ap.add_argument("--out", default=None)
args = ap.parse_args()
NET = NETS[args.net]
OUT = args.out or NET["out"]
os.makedirs(OUT, exist_ok=True)


# --------------------------------------------------------------------------- station coords
def station_coords(vsg_dir):
    coords = {}
    for fp in glob.glob(os.path.join(vsg_dir, "ZZ", "sources", "*.npz")):
        try:
            z = np.load(fp, allow_pickle=True)
            coords[str(z["src"])] = (float(z["src_lon"]), float(z["src_lat"]))
            for c, lo, la in zip(z["rx_codes"], z["rx_lons"], z["rx_lats"]):
                coords[str(c)] = (float(lo), float(la))
        except Exception:
            continue
    return coords


# --------------------------------------------------------------------------- 1. pair selection
print(f"[{args.net}] selecting pairs ...", flush=True)
coords = station_coords(NET["vsg"])
cands = sorted(glob.glob(os.path.join(NET["stack"], f"{NET['code']}.*",
                                      f"{NET['code']}.*_{NET['code']}.*.h5")))
rows = []
for f in cands:
    pair = os.path.basename(f)[:-3]
    try:
        s1, s2 = pair.split("_")
    except ValueError:
        continue
    if s1 not in coords or s2 not in coords:
        continue
    mlon = 0.5 * (coords[s1][0] + coords[s2][0])
    rows.append((f, pair, mlon))
# stratify (or plain decimation) and check dist on the decimated shortlist only
if NET["split_lon"]:
    west = [r for r in rows if r[2] < NET["split_lon"]]
    east = [r for r in rows if r[2] >= NET["split_lon"]]
    half = args.n_pairs // 2
    short = (west[:: max(1, len(west) // (2 * half))] [:2 * half]
             + east[:: max(1, len(east) // (2 * half))] [:2 * half])
else:
    short = rows[:: max(1, len(rows) // (2 * args.n_pairs))][:2 * args.n_pairs]
sel = []
for f, pair, mlon in short:
    try:
        with h5py.File(f, "r") as h:
            g = h["AuxiliaryData"]
            tk = [k for k in g if k.startswith("T")]
            if not tk:
                continue
            dist = float(g[tk[0]][ENZ_ORDER[-1]].attrs["dist"])
        if dist >= args.min_dist:
            stratum = ("W" if NET["split_lon"] and mlon < NET["split_lon"] else "E")
            sel.append((f, pair, mlon, dist, stratum))
    except Exception:
        continue
    if len(sel) >= args.n_pairs:
        # keep strata balanced: stop once both strata are half-full (or unstratified target hit)
        if not NET["split_lon"]:
            break
        nw = sum(1 for s in sel if s[4] == "W")
        if nw >= args.n_pairs // 2 and len(sel) - nw >= args.n_pairs // 2:
            break
sel = sel[: args.n_pairs + 20]
with open(os.path.join(OUT, "pairs_selected.csv"), "w") as fo:
    fo.write("pair,mlon,dist,stratum\n")
    for _, pair, mlon, dist, st in sel:
        fo.write(f"{pair},{mlon:.4f},{dist:.2f},{st}\n")
print(f"  {len(sel)} pairs (strata: "
      f"{sum(1 for s in sel if s[4]=='W')} W / {sum(1 for s in sel if s[4]=='E')} E)", flush=True)


# --------------------------------------------------------------------------- 2. pass A: features
def window_features(rtz, dist, dt):
    """Physical 9C features for one rotated window. rtz: dict comp -> two-sided trace."""
    npts = len(rtz["ZZ"])
    mid = npts // 2
    i0, i1 = int(dist / 5.0 / dt), max(int(dist / 0.5 / dt), int(dist / 5.0 / dt) + 20)
    i1 = min(i1, mid - 1)

    def sym(x):
        return 0.5 * (x[mid:] + x[:mid + 1][::-1])

    def sig_energy(x):
        s = sym(x)[i0:i1]
        return float(np.sum(s * s))

    E = {c: sig_energy(rtz[c]) for c in RTZ_ORDER}
    tot = sum(E.values()) or 1.0
    # envelope ratio TT vs cross terms (per-window env_ratio)
    env = {c: np.abs(hilbert(sym(rtz[c])[i0:i1])) for c in ["TT"] + CROSS_T}
    mx_cross = max(float(e.max()) for c, e in env.items() if c != "TT") or 1e-20
    env_ratio = float(env["TT"].max()) / mx_cross
    # Z-R elliptical coherence: |<ZZ, H{RZ}>| normalized (90-deg-shifted RZ should match ZZ)
    zz = sym(rtz["ZZ"])[i0:i1]
    rz90 = np.imag(hilbert(sym(rtz["RZ"])))[i0:i1]
    denom = (np.linalg.norm(zz) * np.linalg.norm(rz90)) or 1e-20
    ell = float(abs(np.dot(zz, rz90)) / denom)
    rr = sym(rtz["RR"])[i0:i1]
    denom2 = (np.linalg.norm(zz) * np.linalg.norm(rr)) or 1e-20
    czzrr = float(abs(np.dot(zz, rr)) / denom2)

    def asym(x):
        p = float(np.sum(x[mid + i0:mid + i1] ** 2))
        n = float(np.sum(x[mid - i1:mid - i0] ** 2))
        return (p - n) / ((p + n) or 1e-20)

    feats = [np.log10(tot + 1e-20),
             E["ZZ"] / tot, E["RR"] / tot, E["TT"] / tot,
             (E["RZ"] + E["ZR"]) / tot, sum(E[c] for c in CROSS_T) / tot,
             np.log10(env_ratio + 1e-20), ell, czzrr,
             asym(rtz["ZZ"]), asym(rtz["TT"])]
    return feats


FEAT_NAMES = ["logE", "fZZ", "fRR", "fTT", "fRZZR", "fXT", "log_env_ratio",
              "ellipticity", "cZZRR", "asym_ZZ", "asym_TT"]

print(f"[{args.net}] pass A: extracting window features ...", flush=True)
F, meta = [], []                                       # meta: (pair_idx, epoch)
for ip, (f, pair, mlon, dist, st) in enumerate(sel):
    try:
        with h5py.File(f, "r") as h:
            g = h["AuxiliaryData"]
            tkeys = sorted(k for k in g if k.startswith("T"))
            a = g[tkeys[0]]["ZZ"].attrs
            dt, azi, baz = float(a["dt"]), float(a["azi"]), float(a["baz"])
            for tk in tkeys:
                grp = g[tk]
                if any(c not in grp for c in ENZ_ORDER):
                    continue
                big = np.stack([np.asarray(grp[c][:], np.float32) for c in ENZ_ORDER])
                rt = rotation(big, {"azi": azi, "baz": baz}, {})
                rtz = {c: rt[i] for i, c in enumerate(RTZ_ORDER)}
                F.append(window_features(rtz, dist, dt))
                meta.append((ip, int(tk[1:])))
    except Exception as e:
        print(f"  {pair}: features failed ({e})", flush=True)
    if (ip + 1) % 25 == 0:
        print(f"  {ip + 1}/{len(sel)} pairs featured", flush=True)
F = np.asarray(F, float)
meta = np.asarray(meta, int)
np.savez(os.path.join(OUT, "window_features.npz"), F=F, meta=meta,
         feat_names=FEAT_NAMES, pairs=[s[1] for s in sel])
print(f"  {len(F):,} (pair, window) feature rows", flush=True)

# --------------------------------------------------------------------------- 3. per-pair z-score + KMeans
from sklearn.cluster import KMeans                     # noqa: E402
from sklearn.metrics import silhouette_score           # noqa: E402

Fz = F.copy()
for ip in np.unique(meta[:, 0]):
    m = meta[:, 0] == ip
    mu, sd = F[m].mean(0), F[m].std(0)
    Fz[m] = (F[m] - mu) / np.where(sd > 0, sd, 1.0)
Fz = np.nan_to_num(Fz)

best = None
sil_by_k = {}
sub = np.random.RandomState(0).choice(len(Fz), min(len(Fz), 20000), replace=False)
for k in range(args.kmin, args.kmax + 1):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Fz)
    s = silhouette_score(Fz[sub], km.labels_[sub])
    sil_by_k[k] = s
    print(f"  k={k}: silhouette {s:.3f}", flush=True)
    if best is None or s > best[1]:
        best = (k, s, km.labels_)
K, SIL, labels = best
print(f"[{args.net}] chose k={K} (silhouette {SIL:.3f})", flush=True)

# --------------------------------------------------------------------------- 4. epoch consensus
epochs = np.unique(meta[:, 1])
assign = {}                                            # epoch -> (modal label, agreement, n_pairs)
for ep in epochs:
    m = meta[:, 1] == ep
    ls = labels[m]
    if len(ls) < max(5, len(sel) // 10):               # epoch seen by too few pairs
        continue
    vals, cnts = np.unique(ls, return_counts=True)
    imax = int(np.argmax(cnts))
    assign[ep] = (int(vals[imax]), float(cnts[imax] / len(ls)), int(len(ls)))
agree = np.array([v[1] for v in assign.values()])
mean_agree = float(agree.mean()) if len(agree) else 0.0
print(f"[{args.net}] consensus: {len(assign)} epochs, mean agreement {mean_agree:.2f} "
      f"(random = {1.0 / K:.2f}) -- {'GLOBAL REGIMES PRESENT' if mean_agree > 1.5 / K else 'WEAK/NO global regimes (ODH-4 scenario)'}",
      flush=True)
regime_epochs = {r: sorted(ep for ep, (l, a, n) in assign.items()
                           if l == r and a >= args.agree_min) for r in range(K)}
with open(os.path.join(OUT, "regime_assignment.csv"), "w") as fo:
    fo.write("epoch,label,agreement,n_pairs\n")
    for ep in sorted(assign):
        l, a, n = assign[ep]
        fo.write(f"{ep},{l},{a:.3f},{n}\n")
for r in range(K):
    print(f"  regime {r}: {len(regime_epochs[r])} consensus epochs", flush=True)

# --------------------------------------------------------------------------- diagnostics figure
import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import datetime as dtmod                               # noqa: E402

fig, axs = plt.subplots(2, 2, figsize=(15, 9))
ax = axs[0, 0]
ax.plot(list(sil_by_k), list(sil_by_k.values()), "o-")
ax.axvline(K, color="r", ls="--")
ax.set(title="silhouette vs k", xlabel="k", ylabel="silhouette")
ax = axs[0, 1]
eps = sorted(assign)
tt = [dtmod.datetime.utcfromtimestamp(e) for e in eps]
ax.scatter(tt, [assign[e][0] for e in eps], c=[assign[e][1] for e in eps],
           cmap="viridis", s=12, vmin=1.0 / K, vmax=1.0)
ax.set(title=f"epoch regime timeline (color=agreement; mean {mean_agree:.2f})", ylabel="regime")
ax = axs[1, 0]
hod = [(dtmod.datetime.utcfromtimestamp(e).hour) for e in eps]
for r in range(K):
    hh = [h for h, e in zip(hod, eps) if assign[e][0] == r]
    ax.hist(hh, bins=np.arange(25) - 0.5, alpha=0.6, label=f"regime {r}")
ax.legend()
ax.set(title="regime vs hour of day (UTC)", xlabel="hour")
ax = axs[1, 1]
# feature-space view: ellipticity vs log_env_ratio, colored by label (window-level)
i1, i2 = FEAT_NAMES.index("ellipticity"), FEAT_NAMES.index("log_env_ratio")
ax.scatter(Fz[sub, i1], Fz[sub, i2], c=labels[sub], cmap="tab10", s=2, alpha=0.4)
ax.set(title="window features (z-scored)", xlabel="Z-R ellipticity", ylabel="log env_ratio")
fig.suptitle(f"{args.net}: noise-regime clustering ({len(sel)} pairs, k={K})", y=0.99)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "regime_clustering.png"), dpi=120)
plt.close(fig)

# --------------------------------------------------------------------------- 5. pass B: restack
print(f"[{args.net}] pass B: restacking ...", flush=True)
trees = {"all": None} | {f"regime{r}": set(regime_epochs[r]) for r in range(K)
                         if len(regime_epochs[r]) >= 20}
for name in trees:
    os.makedirs(os.path.join(OUT, f"stack_{name}"), exist_ok=True)
n_done = 0
for f, pair, mlon, dist, st in sel:
    try:
        with h5py.File(f, "r") as h:
            g = h["AuxiliaryData"]
            tkeys = sorted(k for k in g if k.startswith("T"))
            a = g[tkeys[0]]["ZZ"].attrs
            dt, azi, baz = float(a["dt"]), float(a["azi"]), float(a["baz"])
            acc = {name: None for name in trees}
            cnt = {name: 0 for name in trees}
            for tk in tkeys:
                grp = g[tk]
                if any(c not in grp for c in ENZ_ORDER):
                    continue
                ep = int(tk[1:])
                big = np.stack([np.asarray(grp[c][:], np.float64) for c in ENZ_ORDER])
                for name, eset in trees.items():
                    if eset is None or ep in eset:
                        acc[name] = big if acc[name] is None else acc[name] + big
                        cnt[name] += 1
        src = pair.split("_")[0]
        for name in trees:
            if acc[name] is None or cnt[name] < 10:
                continue
            rt = rotation(acc[name].astype(np.float32), {"azi": azi, "baz": baz}, {})
            od = os.path.join(OUT, f"stack_{name}", src)
            os.makedirs(od, exist_ok=True)
            with h5py.File(os.path.join(od, pair + ".h5"), "w") as ho:
                gg = ho.create_group("AuxiliaryData/Allstack_linear")
                for i, c in enumerate(RTZ_ORDER):
                    ds = gg.create_dataset(c, data=rt[i].astype(np.float32))
                    for k, v in (("dist", dist), ("dt", dt), ("azi", azi), ("baz", baz),
                                 ("nwin", cnt[name])):
                        ds.attrs[k] = v
        n_done += 1
        if n_done % 25 == 0:
            print(f"  {n_done}/{len(sel)} pairs restacked", flush=True)
    except Exception as e:
        print(f"  {pair}: restack failed ({e})", flush=True)
print(f"[{args.net}] done: trees = {list(trees)} -> {OUT}", flush=True)
