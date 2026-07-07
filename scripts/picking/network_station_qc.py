"""Station-level QC: find stations with systematically anomalous outputs.

Signatures:
  * poor coupling / site noise  -> low SNR term + low fundamental-confirmation term
  * misorientation / polarity   -> corrupted R/Z phase relations: high mode_mixing
    (suppression-test failure) and low confirmation at NORMAL SNR (Nayak & Thurber 2020
    advertise exactly this metadata-QC use of the method)

Method: per-pair metrics from *_modes_validated.csv (dispersion_batch_modesep.py output), then
per-station effect terms a_i from least squares on  metric_pair = mu + a_i + a_j  (each pair
constrains two stations). Robust z-scores flag outliers.

Inputs : paths.dispersion_dir/*/*_modes_validated.csv, station coords from paths.vsg_dir.
Outputs: {project_dir}/station_qc.csv, station_qc.png

Usage:  python network_station_qc.py --config ../../param_files/modesep_params.yaml
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import modesep_config

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--config", required=True, help="network YAML (param_files/modesep_params.yaml)")
args = ap.parse_args()
cfg = modesep_config.load_config(args.config)
ROOT = cfg["paths"]["project_dir"]
V6 = cfg["paths"]["dispersion_dir"]
CODE = cfg["network"]["code"]
_strip = f"{CODE}."

rows = []
for vf in sorted(glob.glob(os.path.join(V6, "*", "*_modes_validated.csv"))):
    pr = os.path.basename(vf).replace("_modes_validated.csv", "")
    try:
        v = pd.read_csv(vf)
    except Exception:
        continue
    s1, s2 = pr.split("_")
    n = len(v)
    if n == 0:
        continue
    snr = pd.to_numeric(v.snr_fund, errors="coerce")
    rows.append(dict(
        pair=pr, s1=s1, s2=s2, n=n,
        frac_fund=(v.fund_flag == "ok").mean(),
        frac_ot=(v.ot_flag == "ok").mean(),
        frac_mix=(v.ot_flag == "mode_mixing").mean(),
        snr=np.nanmedian(snr) if np.isfinite(snr).any() else np.nan,
        frac_glr0=np.isfinite(pd.to_numeric(v.U_fund, errors="coerce")).mean()))
P = pd.DataFrame(rows)
stations = sorted(set(P.s1) | set(P.s2))
sidx = {s: i for i, s in enumerate(stations)}
print(f"pairs: {len(P)}  stations: {len(stations)}")

def station_terms(metric):
    """Least squares metric = mu + a_i + a_j with sum(a)=0; returns per-station terms."""
    ok = np.isfinite(P[metric].values)
    q = P[ok]
    y = q[metric].values.astype(float)
    A = np.zeros((len(q) + 1, len(stations) + 1))
    A[:len(q), 0] = 1.0
    for r, (i1, i2) in enumerate(zip(q.s1.map(sidx), q.s2.map(sidx))):
        A[r, 1 + i1] = 1.0
        A[r, 1 + i2] = 1.0
    A[len(q), 1:] = 1.0                       # constraint sum(a)=0
    yy = np.concatenate([y, [0.0]])
    coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
    return coef[1:], coef[0]

S = pd.DataFrame(index=stations)
S["n_pairs"] = [((P.s1 == s) | (P.s2 == s)).sum() for s in stations]
for m in ("frac_fund", "frac_ot", "frac_mix", "snr"):
    terms, mu = station_terms(m)
    S[m + "_term"] = terms
    print(f"{m}: network mean {mu:.3f}")

def rz(x):
    med = np.nanmedian(x)
    mad = 1.4826 * np.nanmedian(np.abs(x - med))
    return (x - med) / (mad if mad > 0 else np.nanstd(x))

for m in ("frac_fund", "frac_ot", "frac_mix", "snr"):
    S[m + "_z"] = rz(S[m + "_term"].values)

# classification
S["flag"] = ""
coup = (S.snr_z < -2.5) & (S.frac_fund_z < -1.5)
orient = (S.frac_mix_z > 2.5) & (S.snr_z > -1.5)
badfund = (S.frac_fund_z < -2.5) & (S.snr_z > -1.5)
S.loc[coup, "flag"] = "coupling/noise"
S.loc[orient, "flag"] = "orientation/polarity?"
S.loc[badfund & ~orient, "flag"] = "mode-sep anomaly"
S.sort_values("frac_fund_z").to_csv(os.path.join(ROOT, "station_qc.csv"),
                                    float_format="%.3f")

# coordinates (from the per-virtual-source VSG npz files)
coords = modesep_config.vsg_station_coords(cfg["paths"]["vsg_dir"])

fig, axs = plt.subplots(1, 3, figsize=(19, 6))
# (a) map colored by fundamental-confirmation station term
ax = axs[0]
xy = np.array([coords.get(s, (np.nan, np.nan)) for s in stations])
sc = ax.scatter(xy[:, 0], xy[:, 1], c=S.frac_fund_z, cmap="RdYlGn", vmin=-4, vmax=4, s=45,
                edgecolors="k", linewidths=0.4)
plt.colorbar(sc, ax=ax, label="fundamental-confirmation station term [robust z]")
for s in S.index[(S.flag != "")]:
    if s in coords:
        ax.annotate(s.replace(_strip, ""), coords[s], fontsize=6, xytext=(3, 3),
                    textcoords="offset points")
ax.set_aspect(1 / np.cos(np.radians(np.nanmean(xy[:, 1]))))
ax.set(xlabel="Longitude", ylabel="Latitude", title="station term: fundamental confirmation")
# (b) coupling vs orientation separation
ax = axs[1]
ax.scatter(S.snr_z, S.frac_mix_z, s=30, c="0.6")
for s, r in S[S.flag != ""].iterrows():
    ax.scatter(r.snr_z, r.frac_mix_z, s=60,
               c={"coupling/noise": "royalblue", "orientation/polarity?": "crimson",
                  "mode-sep anomaly": "darkorange"}[r.flag])
    ax.annotate(s.replace(_strip, ""), (r.snr_z, r.frac_mix_z), fontsize=7,
                xytext=(4, 3), textcoords="offset points")
ax.axvline(-2.5, color="royalblue", ls=":", lw=1)
ax.axhline(2.5, color="crimson", ls=":", lw=1)
ax.set(xlabel="SNR station term [z]  (low = coupling/noise)",
       ylabel="mode_mixing station term [z]  (high = orientation/polarity)",
       title="pathology separation (colored = flagged)")
# (c) ranked fundamental-confirmation terms
ax = axs[2]
o = np.argsort(S.frac_fund_z.values)
ax.bar(range(len(stations)), S.frac_fund_z.values[o],
       color=["crimson" if S.flag.values[i] != "" else "0.6" for i in o])
ax.set(xlabel="stations (ranked)", ylabel="frac_fund term [z]",
       title="ranked station terms (red = flagged)")
fig.suptitle(f"{cfg['network'].get('name', '')}: station-level QC", y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "station_qc.png"), dpi=130)

print("\nflagged stations:")
cols = ["n_pairs", "frac_fund_term", "frac_fund_z", "snr_z", "frac_mix_z", "frac_ot_z", "flag"]
print(S[S.flag != ""].sort_values("frac_fund_z")[cols].to_string())
print("\nwrote station_qc.csv, station_qc.png")
