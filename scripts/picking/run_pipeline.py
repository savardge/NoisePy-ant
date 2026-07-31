#!/usr/bin/env python3
"""Per-network pipeline driver: picking -> QC -> tomography export from ONE yaml.

Config: param_files/pipeline_<network>.yaml (all values live there; a missing key falls
back to the underlying tool's CLI default -- nothing network-specific is hard-coded here).

Stages:
    pick    dispersion_unified.py with the picker env (DISP_VMIN/DISP_STACK/DISP_LOVE_OT),
            --config paths.picker_config, --out paths.picks_tree
    qc      qc_unified_picks.py -> <picks_tree>/qc_<label>/ (+ qc_params_used.yaml) and
            repoints <picks_tree>/qc_current
    export  export_unified_tomo_picks.py reading qc_current, writing paths.export_dir

Usage:
    python run_pipeline.py --config ../../param_files/pipeline_riehen.yaml \
        --stage pick|qc|export|all [--dry-run]
"""
import argparse
import os
import shutil
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = {"pick": os.path.join(HERE, "dispersion_unified.py"),
         "qc": os.path.join(HERE, "qc_unified_picks.py"),
         "export": os.path.join(HERE, "export_unified_tomo_picks.py")}

QC_FLAGS = {"snr_min": "--snr-min", "vbounds_fund": "--vbounds-fund",
            "vbounds_ot": "--vbounds-ot", "farfield": "--farfield",
            "farfield_max": "--farfield-max", "band_edge_rungs": "--band-edge-rungs",
            "vave": "--vave", "u_bin": "--u-bin", "leak_tol": "--leak-tol",
            "leak_sep_factor": "--leak-sep-factor", "leak_tmax": "--leak-tmax",
            "env_min": "--env-min", "xmode_max": "--xmode-max", "disable": "--disable"}
QC_BOOLS = {"group_scale_dedupe": "--group-scale-dedupe",
            "fold_love_overtone": "--fold-love-overtone"}
EXPORT_FLAGS = {"measure": "--measure", "period_axis": "--period-axis",
                "max_std": "--max-std", "out_suffix": "--out-suffix",
                "vbounds": "--vbounds", "bounds_file": "--bounds-file"}


def sh(cmd, env=None, dry=False):
    print("  " + " ".join(cmd))
    if not dry:
        e = dict(os.environ)
        e.update(env or {})
        subprocess.run(cmd, check=True, env=e)


def stage_pick(cfg, dry):
    p = cfg.get("picker", {})
    env = {}
    if "vmin" in p:
        env["DISP_VMIN"] = str(p["vmin"])
    if "stack" in p:
        env["DISP_STACK"] = str(p["stack"])
    if p.get("love_overtone"):
        env["DISP_LOVE_OT"] = "1"
    env["PYTHONPATH"] = os.path.abspath(os.path.join(HERE, "..", ".."))
    print("[pick] env: %s" % {k: v for k, v in env.items() if k != "PYTHONPATH"})
    cmd = [sys.executable, TOOLS["pick"], "--config", cfg["paths"]["picker_config"],
           "--out", cfg["paths"]["picks_tree"]]
    if "nproc" in p:
        cmd += ["--nproc", str(p["nproc"])]
    sh(cmd, env=env, dry=dry)


def stage_qc(cfg, dry):
    q = cfg.get("qc", {})
    tree = cfg["paths"]["picks_tree"]
    out = os.path.join(tree, "qc_" + cfg["label"])
    cmd = [sys.executable, TOOLS["qc"], "--dir", tree, "--out-dir", out,
           "--ref-dir", cfg["paths"]["ref_dir"]]
    for k, f in QC_FLAGS.items():
        if k in q and str(q[k]) != "":
            cmd += [f, str(q[k])]
    for k, f in QC_BOOLS.items():
        if q.get(k):
            cmd += [f]
    print("[qc] -> %s" % out)
    sh(cmd, dry=dry)
    if dry:
        return
    with open(os.path.join(out, "qc_params_used.yaml"), "w") as fh:
        yaml.safe_dump({"network": cfg["network"], "label": cfg["label"], "qc": q}, fh,
                       sort_keys=False)
    cur = os.path.join(tree, "qc_current")
    if os.path.islink(cur):
        os.unlink(cur)
    os.symlink("qc_" + cfg["label"], cur)
    print("[qc] qc_current -> qc_%s" % cfg["label"])


def stage_export(cfg, dry):
    e = cfg.get("export", {})
    src = os.path.join(cfg["paths"]["picks_tree"], "qc_current", "picks_unified_QCd.csv")
    outdir = cfg["paths"]["export_dir"]
    cmd = [sys.executable, TOOLS["export"], "--net", cfg["network"], "--src", src,
           "--outdir", outdir]
    for k, f in EXPORT_FLAGS.items():
        if k in e and str(e[k]) != "":
            cmd += [f, str(e[k])]
    print("[export] %s -> %s" % (src, outdir))
    sh(cmd, dry=dry)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="param_files/pipeline_<network>.yaml")
    ap.add_argument("--stage", required=True, choices=("pick", "qc", "export", "all"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    with open(a.config) as fh:
        cfg = yaml.safe_load(fh)
    stages = ("pick", "qc", "export") if a.stage == "all" else (a.stage,)
    for s in stages:
        {"pick": stage_pick, "qc": stage_qc, "export": stage_export}[s](cfg, a.dry_run)


if __name__ == "__main__":
    main()
