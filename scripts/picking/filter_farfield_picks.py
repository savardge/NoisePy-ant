"""Far-field re-filter of the V6 group-velocity picks: keep only picks measured at
interstation distance >= MIN_LAMBDA surface-wave wavelengths, per period and wave.

Why (July 2026 diagnosis, see the closure-test figures in Projects/<net>/tomo/):
group velocities measured on short paths are biased LOW -- the three-station closure test
(Luo et al. 2015, GJI 201, Sec. 4) on these very picks shows median traveltime residuals of
+15% (Aargau fund, shortest leg 1-1.5 lambda), +9% (1.5-2), +4.7% (2-2.5), converging only
around 2.5-3 lambda. That slow bias grows with period (dominating T > 3 s where most paths
are < 2 lambda) and is what carved the spurious model-wide LVZ at 4-5 km depth in the Vs
volumes. Luo et al.'s relaxation of the distance cut-off to ~1 lambda applies to PHASE
velocity only; for FTAN group velocities their own premise (Bensen et al. 2007, >= 3 lambda)
is what our closure test supports.

Criterion: distance >= MIN_LAMBDA * lambda_ref(T), lambda_ref(T) = median vg(T) * T computed
from the ORIGINAL pick pool per wave (a per-pick wavelength would preferentially keep slow
picks at fixed distance -- circular). MIN_LAMBDA = 2.5: the closure residual reaches the
far-field floor there for Riehen (+0.6%) and is within ~2% of it for Aargau.

Outputs (self-contained, originals untouched), under Projects/<net>/tomo/farfield2p5/:
  picks_fund.csv / picks_overtone.csv    the filtered picks
  <net>_swtomotv_ff2p5.yaml              swtomotv config (fine grid, new output_root)
  filter_summary_<net>.png               picks/period + mean vg(T), before vs after
  README.md                              this rationale + reproduce commands

Run:  /opt/anaconda3/bin/python filter_farfield_picks.py [--min-lambda 2.5]
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJROOT = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
FINE = {"aargau": ("aargau_swtomotv_500m.yaml", "500m"),
        "riehen": ("riehen_swtomotv_200m.yaml", "200m")}
WAVES = ("fund", "overtone")


def lam_ref(picks):
    """Per-period reference wavelength [km] from the ORIGINAL pick pool."""
    return {T: float(s["group_velocity"].median() * T)
            for T, s in picks.groupby("inst_period")}


def filter_net(net, min_lambda, outdir):
    os.makedirs(outdir, exist_ok=True)
    summary = {}
    for wave in WAVES:
        src = os.path.join(PROJROOT, net, "tomo", f"picks_{wave}.csv")
        p = pd.read_csv(src)
        lam = lam_ref(p)
        keep = p.apply(lambda r: r["distance"] >= min_lambda * lam[r["inst_period"]], axis=1)
        q = p[keep]
        q.to_csv(os.path.join(outdir, f"picks_{wave}.csv"), index=False)
        summary[wave] = (p, q, lam)
        print(f"  {net} {wave}: {len(p)} -> {len(q)} picks "
              f"({100 * len(q) / len(p):.0f}%), T range kept: "
              f"{q['inst_period'].min():g}-{q['inst_period'].max():g} s")
    return summary


def write_yaml(net, outdir):
    """Clone the fine-grid dataset YAML: filtered pick files + separate output_root."""
    src = os.path.join(PROJROOT, net, "tomo", FINE[net][0])
    dst = os.path.join(outdir, f"{net}_swtomotv_ff2p5.yaml")
    txt = open(src).read()
    for wave in WAVES:
        txt = txt.replace(f"{PROJROOT}/{net}/tomo/1_velocity_maps/inputs/picks_{wave}.csv",
                          os.path.join(outdir, f"picks_{wave}.csv"))
    old_root = [l for l in txt.splitlines() if l.startswith("output_root:")][0]
    txt = txt.replace(old_root, "output_root: " + os.path.join(outdir, "swtomotv-output"))
    txt = txt.replace(f"name: {net}", f"name: {net}_ff2p5")
    txt = ("# Far-field (>= 2.5 lambda) re-filtered dataset -- see README.md in this folder.\n"
           "# Cloned from " + FINE[net][0] + "; identical grid/bounds; only picks + output differ.\n"
           + txt)
    open(dst, "w").write(txt)
    print("  wrote", dst)
    return dst


def summary_figure(net, summary, min_lambda, outdir):
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.5))
    for j, wave in enumerate(WAVES):
        p, q, lam = summary[wave]
        Ts = sorted(p["inst_period"].unique())
        nb = [len(p[p.inst_period == T]) for T in Ts]
        na = [len(q[q.inst_period == T]) for T in Ts]
        a = axs[0, j]
        a.bar(Ts, nb, width=0.08, color="lightgray", label="before")
        a.bar(Ts, na, width=0.08, color="tab:blue", label=f">= {min_lambda:g}$\\lambda$")
        a.set_yscale("log"); a.set_xlabel("period [s]"); a.set_ylabel("picks")
        a.set_title(f"{wave}: pick counts"); a.legend()
        a = axs[1, j]
        mb = [p[p.inst_period == T]["group_velocity"].mean() for T in Ts]
        ma = [q[q.inst_period == T]["group_velocity"].mean() if na[i] >= 10 else np.nan
              for i, T in enumerate(Ts)]
        a.plot(Ts, mb, "o-", color="gray", ms=3, label="before (all distances)")
        a.plot(Ts, ma, "o-", color="tab:red", ms=3,
               label=f"after (>= {min_lambda:g}$\\lambda$, n>=10)")
        a.set_xlabel("period [s]"); a.set_ylabel("mean vg [km/s]")
        a.set_title(f"{wave}: mean dispersion"); a.legend()
    fig.suptitle(f"{net.capitalize()} — far-field pick filter (distance >= "
                 f"{min_lambda:g} wavelengths); the 'before' long-period decline is the "
                 f"near-field bias", fontsize=12)
    fig.tight_layout()
    out = os.path.join(outdir, f"filter_summary_{net}.png")
    fig.savefig(out, dpi=140); plt.close(fig); print("  wrote", out)


README = """# Far-field re-filtered picks (>= {ml:g} wavelengths)

Created {date} by filter_farfield_picks.py (NoisePy-ant/scripts/picking).

## Why this run exists

The production Vs volumes showed a low-velocity zone at 4-5 km depth spanning the whole
model. Diagnosis (see closure_test_{net}_fund.png in ../): FTAN group velocities measured on
interstation paths shorter than ~2.5 wavelengths are biased LOW (near-field/interference bias;
three-station closure residuals up to +15% in traveltime for legs of 1-1.5 lambda, converging
at 2.5-3 lambda). Short paths dominate the long-period pick pool, so the mean dispersion curve
declines beyond T ~ 3 s, which every 1-D inversion maps into a fake deep LVZ (nearly obscuring
the real graben signature in Aargau). Luo et al. (2015), which motivated the earlier low
wavelength-multiple criterion, demonstrates the ~1-lambda relaxation for PHASE velocities only
and explicitly keeps the >= 3 lambda recommendation (Bensen et al. 2007) for group velocities.

## What differs from the parent dataset (../)

ONLY the pick lists: distance >= {ml:g} * lambda_ref(T), with lambda_ref(T) = median vg(T)*T of
the original pool, per wave. Grid, bounds, dx, station file, QC exclusions, tomography
parameters (LC, sigma_eff) and the Vs-inversion setup are IDENTICAL to the parent production
run, so any difference in the resulting maps/volumes is attributable to the filter alone.

Consequence to expect: usable fundamental periods now end near {tmaxf:g} s ({net}), so the
models lose (biased) depth coverage — the resolution veil will sit shallower, honestly.

## Reproduce

    PY=/opt/anaconda3/envs/das-ambient-noise/bin/python   # swtomotv env
    export PYTHONPATH=/Users/genevievesavard/Codes/Noisepy-ant
    $PY run_production.py --config {yaml} --wave fund     --lc {lc} --se 0.025
    $PY run_production.py --config {yaml} --wave overtone --lc {lc} --se 0.025
    /opt/anaconda3/envs/bayesbay_dev/bin/python grid_vs_inversion.py \\
        --production {outdir}/swtomotv-output/production \\
        --config {yaml} --net {net} --criterion physical \\
        --outdir {vsdir} \\
        --bayhunter-python /opt/anaconda3/envs/bayhunter/bin/python \\
        --bayhunter-runner run_bayhunter_cell.py
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--min-lambda", type=float, default=2.5)
    args = ap.parse_args()
    import datetime
    lcs = {"aargau": 3.0, "riehen": 1.5}
    for net in ("aargau", "riehen"):
        print(f"== {net} ==")
        outdir = os.path.join(PROJROOT, net, "tomo", "farfield2p5")
        summary = filter_net(net, args.min_lambda, outdir)
        yamlpath = write_yaml(net, outdir)
        summary_figure(net, summary, args.min_lambda, outdir)
        q = summary["fund"][1]
        cnt = q.groupby("inst_period").size()
        tmaxf = float(cnt[cnt >= 30].index.max())
        vsdir = os.path.join(PROJROOT, net, "tomo", "vs_inversion",
                             f"grid_physical_{FINE[net][1]}_ff2p5")
        with open(os.path.join(outdir, "README.md"), "w") as f:
            f.write(README.format(ml=args.min_lambda, net=net, yaml=yamlpath, lc=lcs[net],
                                  outdir=outdir, vsdir=vsdir, tmaxf=tmaxf,
                                  date=datetime.date.today().isoformat()))
        print("  wrote README.md; fund far-field T_max (n>=30):", tmaxf, "s")


if __name__ == "__main__":
    main()
