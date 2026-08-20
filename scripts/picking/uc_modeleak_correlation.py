"""Does the U>c inconsistency track MODE CONTAMINATION of the fundamental pick?

The tomography maps only carry velocity/coverage/resolution, so the earlier correlation test
had to use indirect proxies -- and its strongest hit (overtone/fundamental velocity ratio) is
partly circular, because a fundamental biased fast by overtone leakage lowers that ratio by
construction. The QC'd unified pick tables carry the direct measurements instead:

  mode_overlap  overlap between the fundamental and overtone FTAN branches at that pick
  xmode_amp     cross-mode amplitude leaking into the pick window
  snr_nbG_other the competing branch's narrowband SNR
  ot_flag       the overtone-leak flag the QC layer raised

Picks are located by station-pair MIDPOINT, which is a coarse stand-in for the true path
sensitivity -- fine for asking whether contamination is elevated in a REGION, not for
attributing it to an individual cell.

  python uc_modeleak_correlation.py --net riehen --wave love --uc <uc_maps dir>
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vs_model_figures import E                                       # noqa: E402

COLS = ["nominal_period", "wave_type", "mode", "pair", "snr_nbG", "snr_nbG_other",
        "mode_overlap", "xmode_amp", "ot_flag", "env_ratio", "group_ok"]


def station_xy(net):
    for p in (f"{E}/{net}/tomo/1_velocity_maps/0_inputs/configs/stations.csv",
              f"{E}/{net}/tomo/1_velocity_maps/inputs/stations.csv"):
        if os.path.exists(p):
            s = pd.read_csv(p)
            c = {x.lower(): x for x in s.columns}
            return {str(r[c.get("id", "id")]): (float(r[c["longitude"]]), float(r[c["latitude"]]))
                    for _, r in s.iterrows()}
    raise SystemExit("no stations.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", required=True)
    ap.add_argument("--wave", required=True, choices=("fund", "love"))
    ap.add_argument("--uc", required=True)
    ap.add_argument("--tmin", type=float, required=True)
    ap.add_argument("--tmax", type=float, required=True)
    ap.add_argument("--chunk", type=int, default=2_000_000)
    a = ap.parse_args()

    meta = f"{E}/{a.net}/tomo/1_velocity_maps/0_inputs/configs/picks_{a.wave}_uni.csv.meta.json"
    import json
    src = json.load(open(meta))["source"]
    print(f"picks: {src}")

    wt = "love" if a.wave == "love" else "rayleigh"
    st = station_xy(a.net)
    z = np.load(os.path.join(a.uc, f"uc_{a.net}_{a.wave}.npz"), allow_pickle=True)
    cells, lonlat = z["cells"], z["lonlat"]
    T, ratio = np.asarray(z["T"], float), np.asarray(z["ratio"], float)
    band = (T >= a.tmin) & (T <= a.tmax)
    viol = np.nanmean(ratio[:, band] > 1, axis=1)

    acc = {k: np.zeros(len(cells)) for k in ("n", "mode_overlap", "xmode_amp",
                                             "snr_nbG", "snr_ratio", "ot_flag")}
    # nearest-cell lookup on a coarse lon/lat grid
    lo0, la0 = lonlat[:, 0], lonlat[:, 1]
    for ch in pd.read_csv(src, usecols=lambda c: c in COLS, chunksize=a.chunk):
        ch = ch[(ch.wave_type == wt) & (ch["mode"].astype(str).str.contains("fund", case=False))]
        ch = ch[(ch.nominal_period >= a.tmin) & (ch.nominal_period <= a.tmax)]
        if "group_ok" in ch:
            ch = ch[ch.group_ok.astype(str).str.lower().isin(("true", "1", "1.0"))]
        if not len(ch):
            continue
        ss = ch.pair.astype(str).str.split("_", n=1, expand=True)
        mx, my = [], []
        for s1, s2 in zip(ss[0], ss[1]):
            p1, p2 = st.get(s1), st.get(s2)
            if p1 is None or p2 is None:
                mx.append(np.nan); my.append(np.nan)
            else:
                mx.append(0.5 * (p1[0] + p2[0])); my.append(0.5 * (p1[1] + p2[1]))
        mx = np.array(mx); my = np.array(my)
        ok = np.isfinite(mx)
        if not ok.any():
            continue
        idx = np.array([int(np.argmin((lo0 - x) ** 2 + ((la0 - y) * 1.4) ** 2))
                        for x, y in zip(mx[ok], my[ok])])
        sub = ch[ok]
        np.add.at(acc["n"], idx, 1.0)
        for col, key in (("mode_overlap", "mode_overlap"), ("xmode_amp", "xmode_amp"),
                         ("snr_nbG", "snr_nbG")):
            if col in sub:
                np.add.at(acc[key], idx, np.nan_to_num(sub[col].to_numpy(float)))
        if "snr_nbG_other" in sub and "snr_nbG" in sub:
            r = np.nan_to_num(sub["snr_nbG_other"].to_numpy(float)) / np.maximum(
                np.nan_to_num(sub["snr_nbG"].to_numpy(float)), 1e-6)
            np.add.at(acc["snr_ratio"], idx, r)
        if "ot_flag" in sub:
            np.add.at(acc["ot_flag"], idx,
                      (sub["ot_flag"].astype(str).str.lower().isin(("true", "1", "1.0"))
                       ).to_numpy(float))
        print(f"   ...{int(acc['n'].sum())} picks binned", flush=True)

    n = acc["n"]
    good = n >= 20
    print(f"\n{a.net}/{a.wave}  band {a.tmin}-{a.tmax} s   "
          f"{int(n.sum())} picks over {good.sum()} cells with >=20 picks")

    def sp(x, y):
        m = np.isfinite(x) & np.isfinite(y) & good
        if m.sum() < 30:
            return np.nan, int(m.sum())
        # A CONSTANT input must return nan, not a correlation. argsort of equal values yields
        # the index order, so a constant column silently correlates with anything that varies
        # spatially -- it produced an identical fake r=0.919 for three empty columns before
        # this guard existed.
        if np.unique(x[m]).size < 2 or np.unique(y[m]).size < 2:
            return np.nan, int(m.sum())
        rx = np.argsort(np.argsort(x[m])).astype(float)
        ry = np.argsort(np.argsort(y[m])).astype(float)
        rx -= rx.mean(); ry -= ry.mean()
        d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
        return (float((rx * ry).sum() / d) if d else np.nan), int(m.sum())

    print(f"    {'per-cell mean of':<30}{'Spearman r vs U>c':>20}{'n':>7}")
    for key, lab in (("mode_overlap", "mode_overlap"), ("xmode_amp", "xmode_amp"),
                     ("snr_ratio", "snr_other/snr (leak)"), ("ot_flag", "ot_flag rate"),
                     ("snr_nbG", "snr_nbG  [quality ctrl]")):
        v = np.where(n > 0, acc[key] / np.maximum(n, 1), np.nan)
        r, m = sp(viol, v)
        mark = "  <<<" if np.isfinite(r) and abs(r) >= 0.3 and "ctrl" not in lab else ""
        print(f"    {lab:<30}{r:>20.3f}{m:>7}{mark}")
    r, m = sp(viol, n)
    print(f"    {'pick count  [density ctrl]':<30}{r:>20.3f}{m:>7}")


if __name__ == "__main__":
    main()
