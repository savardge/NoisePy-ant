"""Aggregate the per-pair phase-velocity measurements (phase_batch.py) into swtomotv phase-
tomography inputs -- the phase analogue of export_tomo_picks.py.

Phase velocity is valid to ~1 wavelength (Ekstrom 2009 / Luo 2015), so unlike the group export
there is NO far-field r/lambda filter: the whole point is to keep the long-period/short-path
picks the group filter discards. The picks are already (a) de-duplicated by CWT scale and
labelled at their true scale period, and (b) physical (c > refined U_ref), from phase_batch.py.

QC applied here:
  * drop pairs touching a station flagged in station_qc.csv (same as the group export);
  * optional per-(pair,wave) smoothness gate (drop picks whose |c - median-neighbour| exceeds
    SMOOTH_KMS -- kills the residual 1.5-2.5 s wrong-2*pi*N-branch outliers).

Writes into Projects/<net>/tomo/phase/:
  picks_fund_phase.csv / picks_overtone_phase.csv  (swtomotv schema; PHASE c in group_velocity)
  <net>_swtomotv_phase.yaml                         (cloned grid; phase picks + phase output_root)

Usage:  python export_phase_picks.py --net aargau [--smooth 0.25]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

PROJ = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
FINE_YAML = {"aargau": "farfield2p5/aargau_swtomotv_ff2p5.yaml",
             "riehen": "farfield2p5/riehen_swtomotv_ff2p5.yaml"}
COLS = ["station_pair", "stasrc", "starcv", "inst_period", "group_velocity",
        "std", "count", "std_percent", "distance", "azimuth"]
# physical phase-velocity bounds per wave (drop cycle-skip / wrong-branch outliers) [km/s]
CBOUND = {"fund": (1.0, 4.2), "overtone": (1.8, 5.2)}


def load_all(net):
    rows = []
    for f in glob.glob(os.path.join(PROJ, net, "tomo", "phase_picks", "*.csv")):
        spair = os.path.basename(f)[:-4]
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if not len(d):
            continue
        # pair stem is <s1>_<s2>; station ids contain one dot (AA.xxxx / RI.xxxx)
        parts = spair.split("_")
        s1, s2 = parts[0], parts[1]
        d["station_pair"] = spair; d["stasrc"] = s1; d["starcv"] = s2
        rows.append(d)
    if not rows:
        raise SystemExit(f"no phase_picks under {net}; run phase_batch.py first")
    return pd.concat(rows, ignore_index=True)


def smooth_gate(df, wave, kms):
    """Drop picks that jump from their in-pair neighbours by > kms km/s (branch outliers)."""
    df = df.reset_index(drop=True)               # positional index for `keep` alignment
    keep = np.ones(len(df), dtype=bool)
    for _, g in df.groupby("station_pair"):
        if len(g) < 3:
            continue
        gs = g.sort_values("inst_period")
        c = gs.group_velocity.values
        med = np.median(np.abs(np.diff(c)))
        # local deviation from a 3-point running median
        run = pd.Series(c).rolling(3, center=True, min_periods=2).median().values
        bad = np.abs(c - run) > max(kms, 3 * med)
        keep[gs.index[bad]] = False
    return df[keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", required=True, choices=("aargau", "riehen"))
    ap.add_argument("--smooth", type=float, default=0.30,
                    help="per-pair smoothness gate [km/s]; 0 disables")
    args = ap.parse_args()
    tomo = os.path.join(PROJ, args.net, "tomo")
    outdir = os.path.join(tomo, "phase")
    os.makedirs(outdir, exist_ok=True)

    allp = load_all(args.net)
    # exclude pairs touching a flagged station (station_qc.csv lives at the project root, same
    # file the group far-field export used via --exclude-flagged; fall back to tomo/)
    qcf = os.path.join(PROJ, args.net, "station_qc.csv")
    if not os.path.exists(qcf):
        qcf = os.path.join(tomo, "station_qc.csv")
    bad = set()
    if os.path.exists(qcf):
        q = pd.read_csv(qcf, index_col=0)
        bad = set(q.index[q.flag.notna() & (q.flag.astype(str).str.strip() != "")])
        allp = allp[~allp.stasrc.isin(bad) & ~allp.starcv.isin(bad)]
        print(f"excluded {len(bad)} flagged stations")

    for wave, fn in (("fund", "picks_fund_phase.csv"),
                     ("overtone", "picks_overtone_phase.csv")):
        d = allp[allp.wave == wave].copy()
        d["inst_period"] = d.period.round(3)
        d["group_velocity"] = d.c_phase          # PHASE velocity in the canonical velocity column
        d["distance"] = d.dist
        d["std"] = 0.0; d["count"] = 1; d["std_percent"] = 0.0
        clo, chi = CBOUND[wave]
        n_pre = len(d)
        d = d[(d.group_velocity >= clo) & (d.group_velocity <= chi)]
        print(f"  {wave}: velocity bound [{clo},{chi}] kept {len(d)}/{n_pre}")
        d = d[COLS]
        if args.smooth > 0 and len(d):
            n0 = len(d); d = smooth_gate(d, wave, args.smooth)
            print(f"  {wave}: smoothness gate kept {len(d)}/{n0}")
        d.to_csv(os.path.join(outdir, fn), index=False)
        if len(d):
            print(f"{fn}: {len(d)} phase picks | {d.station_pair.nunique()} pairs | "
                  f"T {d.inst_period.min():.2f}-{d.inst_period.max():.2f} s | "
                  f"c {d.group_velocity.min():.2f}-{d.group_velocity.max():.2f} km/s")
        else:
            print(f"{fn}: 0 picks")

    # clone the fine-grid YAML: phase pick files + phase output_root
    src = os.path.join(tomo, FINE_YAML[args.net])
    txt = open(src).read()
    lines = []
    for ln in txt.splitlines():
        if ln.strip().startswith("fund:") and "picks" in ln:
            lines.append(f"  fund:     {os.path.join(outdir, 'picks_fund_phase.csv')}")
        elif ln.strip().startswith("overtone:") and "picks" in ln:
            lines.append(f"  overtone: {os.path.join(outdir, 'picks_overtone_phase.csv')}")
        elif ln.startswith("output_root:"):
            lines.append("output_root: " + os.path.join(outdir, "swtomotv-output"))
        elif ln.startswith("name:"):
            lines.append(f"name: {args.net}_phase")
        else:
            lines.append(ln)
    dst = os.path.join(outdir, f"{args.net}_swtomotv_phase.yaml")
    with open(dst, "w") as f:
        f.write("# Phase-velocity tomography dataset (step 2). Picks = phase c (no far-field\n"
                "# filter; valid to ~1 lambda). Cloned grid from " + FINE_YAML[args.net] + ".\n"
                + "\n".join(lines) + "\n")
    print("wrote", dst)


if __name__ == "__main__":
    main()
