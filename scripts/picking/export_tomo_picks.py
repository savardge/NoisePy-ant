"""Export validated mode-separated picks to swtomotv tomography inputs.

Reads every *_modes_validated.csv under paths.dispersion_dir and writes, into
{project_dir}/tomo/:
  picks_fund.csv      confirmed fundamental picks (fund_use == 1)
  picks_overtone.csv  confirmed 1st-higher-mode picks (ot_use == 1)
  stations.csv        id, latitude, longitude, elevation (from the VSG npz coords)
  bounds.txt          suggested grid bounds (station extent + margin) for the swtomotv YAML

Pick CSV schema = swtomotv canonical PickColumns:
  station_pair, stasrc, starcv, inst_period, group_velocity, std, count,
  std_percent, distance, azimuth
The V6 batch measures one group velocity per (pair, period) (pws stack only), so there is no
repeatability ensemble: std = 0, count = 1, std_percent = 0. Use swtomotv filters (100, 1) and
the legacy blanket Cd = (rel_err * tau)^2. azimuth = source->receiver great-circle azimuth
recomputed from the station coordinates (deg E of N).

Usage:  python export_tomo_picks.py --config ../../param_files/modesep_params.yaml
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd

import modesep_config

# Empirical near-field rejection rule (Projects/azimuthal_source_bias/, 2026-07):
# short paths measure the group velocity systematically SLOW (folded-correlation
# truncation near zero lag); the bias converges period-dependently, confirmed by
# the three-station closure test (closure_test.py). Keep picks with
# r/lambda >= cutoff(T), lambda = T * median-U(T) of the exported wave itself.
# Overtone: closure shows convergence like the fundamental; uniform cutoff.
FARFIELD_RULE = {
    "riehen": {"fund": [(0.0, 2.5, 3.0), (2.5, 99.0, 2.5)],
               "overtone": [(0.0, 99.0, 2.5)]},
    # Aargau fund T>=3: 2.5 lambda, which the 23 km aperture cannot satisfy beyond
    # T~4 s -> the biased declining long-period branch drops out entirely. A 2-lambda
    # concession at T>=4 s was tried (2026-07-12) and REINTRODUCED the 4-5 km LVZ
    # artifact (blanket Cd cannot downweight); deep Aargau needs phase/overtone data.
    "aargau": {"fund": [(0.0, 2.0, 4.5), (2.0, 3.0, 3.0), (3.0, 99.0, 2.5)],
               "overtone": [(0.0, 99.0, 2.5)]},
}


def farfield_filter(df, net, wave):
    """Drop near-field picks per FARFIELD_RULE; returns the filtered frame."""
    rule = FARFIELD_RULE[net][wave]
    med = df.groupby(df.inst_period.round(2))["group_velocity"].transform("median")
    rml = df.distance / (med * df.inst_period)
    cut = pd.Series(np.nan, index=df.index)
    for tlo, thi, c in rule:
        m = (df.inst_period >= tlo) & (df.inst_period < thi)
        cut[m] = c
    keep = rml >= cut
    print(f"  farfield {wave}: {keep.sum()}/{len(df)} picks kept "
          f"({100 * keep.mean():.0f}%); rule {rule}")
    return df[keep]


ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--config", required=True, help="network YAML (param_files/modesep_*.yaml)")
ap.add_argument("--exclude-flagged", action="store_true",
                help="drop pairs touching any station flagged in {project_dir}/station_qc.csv "
                     "(coupling / orientation-polarity / mode-sep anomalies)")
ap.add_argument("--farfield", action="store_true",
                help="apply the empirical period-dependent r/lambda near-field rejection rule "
                     "(FARFIELD_RULE; see Projects/azimuthal_source_bias/pick_rejection_rule.png)")
args = ap.parse_args()
cfg = modesep_config.load_config(args.config)
V6 = cfg["paths"]["dispersion_dir"]
PROJ = cfg["paths"]["project_dir"]
OUT = os.path.join(PROJ, "tomo")
os.makedirs(OUT, exist_ok=True)

coords = modesep_config.vsg_station_coords(cfg["paths"]["vsg_dir"])

# Optional: exclude pairs that touch a QC-flagged station (network_station_qc.py output).
# The flag column is empty for healthy stations; any non-empty flag => drop the station.
bad = set()
if args.exclude_flagged:
    qc = os.path.join(PROJ, "station_qc.csv")
    q = pd.read_csv(qc, index_col=0)
    bad = set(q.index[q.flag.notna() & (q.flag.astype(str).str.strip() != "")])
    print(f"excluding {len(bad)} flagged stations: {', '.join(sorted(bad))}")


def azimuth_deg(lo1, la1, lo2, la2):
    """Source->receiver azimuth, deg E of N (spherical)."""
    f1, f2 = np.radians(la1), np.radians(la2)
    dl = np.radians(lo2 - lo1)
    y = np.sin(dl) * np.cos(f2)
    x = np.cos(f1) * np.sin(f2) - np.sin(f1) * np.cos(f2) * np.cos(dl)
    return float(np.degrees(np.arctan2(y, x)) % 360.0)


fund_rows, ot_rows = [], []
nfiles = 0
for vf in sorted(glob.glob(os.path.join(V6, "*", "*_modes_validated.csv"))):
    pr = os.path.basename(vf).replace("_modes_validated.csv", "")
    s1, s2 = pr.split("_")
    if s1 not in coords or s2 not in coords:
        continue
    if s1 in bad or s2 in bad:              # drop pairs touching a flagged station
        continue
    try:
        v = pd.read_csv(vf)
    except Exception:
        continue
    if len(v) == 0:
        continue
    nfiles += 1
    azi = azimuth_deg(*coords[s1], *coords[s2])
    common = dict(station_pair=pr, stasrc=s1, starcv=s2,
                  std=0.0, count=1, std_percent=0.0, azimuth=round(azi, 2))
    for _, r in v[v.fund_use == 1].iterrows():
        fund_rows.append(dict(common, inst_period=r.period,
                              group_velocity=r.U_fund, distance=r.distance))
    for _, r in v[v.ot_use == 1].iterrows():
        ot_rows.append(dict(common, inst_period=r.period,
                            group_velocity=r.U_overtone, distance=r.distance))

cols = ["station_pair", "stasrc", "starcv", "inst_period", "group_velocity",
        "std", "count", "std_percent", "distance", "azimuth"]
net_key = cfg["network"]["name"].lower()
exported = {}
for rows, fn, wave in ((fund_rows, "picks_fund.csv", "fund"),
                       (ot_rows, "picks_overtone.csv", "overtone")):
    df = pd.DataFrame(rows, columns=cols)
    if args.farfield and len(df):
        df = farfield_filter(df, net_key, wave)
    df.to_csv(os.path.join(OUT, fn), index=False)
    exported[wave] = df
    if len(df):
        print(f"{fn}: {len(df)} picks | {df.station_pair.nunique()} pairs | "
              f"T {df.inst_period.min():.1f}-{df.inst_period.max():.1f} s | "
              f"gv {df.group_velocity.min():.2f}-{df.group_velocity.max():.2f} km/s")
    else:
        print(f"{fn}: 0 picks")

# stations restricted to those actually used in any exported (post-filter) pick
used = set()
for df in exported.values():
    used |= set(df.stasrc) | set(df.starcv)
sta = pd.DataFrame([dict(id=s, latitude=coords[s][1], longitude=coords[s][0],
                         elevation=0.0) for s in sorted(used)])
sta.to_csv(os.path.join(OUT, "stations.csv"), index=False)
print(f"stations.csv: {len(sta)} stations (from {nfiles} pair files)")

m = 0.01  # ~1 km margin so stations sit strictly inside the grid box
b = (sta.latitude.min() - m, sta.latitude.max() + m,
     sta.longitude.min() - m, sta.longitude.max() + m)
with open(os.path.join(OUT, "bounds.txt"), "w") as fh:
    fh.write("suggested swtomotv bounds [min_lat, max_lat, min_lon, max_lon]:\n")
    fh.write(f"bounds: [{b[0]:.6f}, {b[1]:.6f}, {b[2]:.6f}, {b[3]:.6f}]\n")
print(f"bounds: [{b[0]:.6f}, {b[1]:.6f}, {b[2]:.6f}, {b[3]:.6f}]")
