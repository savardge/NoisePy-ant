"""PER-PAIR quality-selective CCF restacking (the pilot's second, physics-targeted variant).

Why this exists: the clustered-regime variant (regime_restack_pilot.py, ported from the DAS work)
returned a clean NEGATIVE on both geophone networks -- silhouette ~0.15, cross-pair consensus
agreement 0.41 (random 0.25), and the consensus collapsing to a single dominant label with no
diurnal structure. The feature diagnostics said why: window wavefield quality varies strongly
WITHIN a pair (ellipticity p10 0.01 -> p90 0.15, a 15x spread) but INCOHERENTLY across the array
(epoch-median ellipticity spans only 0.05-0.07 network-wide). There are no global regimes to find,
so cross-pair consensus was doomed -- but per-pair selection has plenty of signal to exploit.

Design (the control is the point):
    all    = every window                       (baseline: maximum stacking depth)
    top50  = best 50% of windows by --metric    (selection ON, half depth)
    bot50  = worst 50% of windows by --metric   (selection INVERTED, half depth)
top50 vs bot50 is the clean test: SAME window count, so any difference is purely selection quality,
not stacking depth. top50 vs all tests whether selection beats more data.

Circularity guard: selecting on ellipticity and then scoring with xmode_amp is partly tautological
(both probe Z-R coherence). The non-circular scores are pick yield, SNR, the graben rf_leak
coincidence, and above all agreement with the EXTERNAL VSG phase reference -- see regime_pilot_score.py.

Reuses the features already computed by regime_restack_pilot.py pass A (window_features.npz), so it
is cheap: no re-featuring, no re-correlation.

Usage: python regime_select_restack.py --net {riehen,aargau} [--metric ellipticity|log_env_ratio]
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from noisepy.stacking import rotation  # noqa: E402

ENZ_ORDER = ["EE", "EN", "EZ", "NE", "NN", "NZ", "ZE", "ZN", "ZZ"]
RTZ_ORDER = ["ZR", "ZT", "ZZ", "RR", "RT", "RZ", "TR", "TT", "TZ"]
NETS = {
    "aargau": {"stack": "/Volumes/T7blue/aargau-data/STACK_CHAA_normZ",
               "out": "/Users/genevievesavard/Codes/extract_higher_modes/Projects/aargau/regime_pilot"},
    "riehen": {"stack": "/Volumes/T7blue/riehen-data/STACK_CHRI_normZ",
               "out": "/Users/genevievesavard/Codes/extract_higher_modes/Projects/riehen/regime_pilot"},
}
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--net", required=True, choices=list(NETS))
ap.add_argument("--metric", default="ellipticity")
ap.add_argument("--min-win", type=int, default=10)
args = ap.parse_args()
OUT = NETS[args.net]["out"]

z = np.load(os.path.join(OUT, "window_features.npz"), allow_pickle=True)
F, meta = z["F"], z["meta"]
names = [str(x) for x in z["feat_names"]]
pairs = [str(p) for p in z["pairs"]]
mi = names.index(args.metric)
print(f"[{args.net}] {len(F):,} windows, {len(pairs)} pairs, selecting on '{args.metric}'", flush=True)

# ---- per-pair split: which epochs are top/bottom half for THIS pair ----
sel_epochs = {}                                        # pair -> {"top50": set, "bot50": set}
for ip in np.unique(meta[:, 0]):
    m = meta[:, 0] == ip
    v, eps = F[m, mi], meta[m, 1]
    good = np.isfinite(v)
    v, eps = v[good], eps[good]
    if len(v) < 2 * args.min_win:
        continue
    med = np.median(v)
    sel_epochs[pairs[ip]] = {"top50": set(eps[v >= med].tolist()),
                             "bot50": set(eps[v < med].tolist())}
print(f"  {len(sel_epochs)} pairs with enough windows to split", flush=True)

TREES = ["all", "top50", "bot50"]
for t in TREES:
    os.makedirs(os.path.join(OUT, f"stack_{t}"), exist_ok=True)

n_done = 0
for pair, split in sel_epochs.items():
    src = pair.split("_")[0]
    f = os.path.join(NETS[args.net]["stack"], src, pair + ".h5")
    if not os.path.exists(f):
        continue
    try:
        with h5py.File(f, "r") as h:
            g = h["AuxiliaryData"]
            tkeys = sorted(k for k in g if k.startswith("T"))
            a = g[tkeys[0]]["ZZ"].attrs
            dist, dt = float(a["dist"]), float(a["dt"])
            azi, baz = float(a["azi"]), float(a["baz"])
            acc = {t: None for t in TREES}
            cnt = {t: 0 for t in TREES}
            for tk in tkeys:
                grp = g[tk]
                if any(c not in grp for c in ENZ_ORDER):
                    continue
                ep = int(tk[1:])
                big = None
                for t in TREES:
                    if t == "all" or ep in split[t]:
                        if big is None:
                            big = np.stack([np.asarray(grp[c][:], np.float64) for c in ENZ_ORDER])
                        acc[t] = big.copy() if acc[t] is None else acc[t] + big
                        cnt[t] += 1
        for t in TREES:
            if acc[t] is None or cnt[t] < args.min_win:
                continue
            rt = rotation(acc[t].astype(np.float32), {"azi": azi, "baz": baz}, {})
            od = os.path.join(OUT, f"stack_{t}", src)
            os.makedirs(od, exist_ok=True)
            with h5py.File(os.path.join(od, pair + ".h5"), "w") as ho:
                gg = ho.create_group("AuxiliaryData/Allstack_linear")
                for i, c in enumerate(RTZ_ORDER):
                    ds = gg.create_dataset(c, data=rt[i].astype(np.float32))
                    for k, v in (("dist", dist), ("dt", dt), ("azi", azi), ("baz", baz),
                                 ("nwin", cnt[t])):
                        ds.attrs[k] = v
        n_done += 1
        if n_done % 25 == 0:
            print(f"  {n_done}/{len(sel_epochs)} pairs restacked", flush=True)
    except Exception as e:
        print(f"  {pair}: failed ({e})", flush=True)
print(f"[{args.net}] done -> {OUT}/stack_{{{','.join(TREES)}}}", flush=True)
