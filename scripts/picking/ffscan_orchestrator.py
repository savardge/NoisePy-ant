"""ffscan step 4: run the whole r/lambda-scan tomography campaign from the manifest.

Per arm (net, measure, dx, ff): three sequential stages
  iso     run_production.py --fast per wave  -> {output_root}/production/{wave}/map_T*.npz
  aniso   selection CSV from the arm's cache -> swtomotv produce --topo -> swtomotv aniso
          (-> vg-maps-final-{wave}-topo[-aniso]/ under the same output_root)
  styled  ffscan_styled_maps.py (base env: geopandas)  -> {output_root}/styled_maps/

Arms run in parallel (default 4 workers x 2 BLAS threads = the half-machine budget);
stages within an arm are sequential. Completion state in ffscan_logs/ffscan_state.json —
re-running skips done stages (delete a key or pass --reset-failed to retry failures).
Coarse grids are scheduled before fine, group before phase, so the cheap/most
informative arms land first.

Usage (das-ambient-noise env):
  python ffscan_orchestrator.py [--workers 4] [--nets riehen,aargau] [--ff 1.0,2.0]
                                [--stages iso,aniso,styled] [--reset-failed] [--dry-run]
"""
import argparse
import glob
import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
LOGD = os.path.normpath(os.path.join(EHM, "..", "ffscan_logs"))
STATE = os.path.join(LOGD, "ffscan_state.json")
DAS = "/opt/anaconda3/envs/das-ambient-noise/bin/python"
BASE = "/opt/anaconda3/bin/python3"
SWT = os.path.expanduser("~/Codes/swtomotv/src")
NPA = os.path.expanduser("~/Codes/NoisePy-ant")

_lock = threading.Lock()


def load_state():
    if os.path.exists(STATE):
        with open(STATE) as fh:
            return json.load(fh)
    return {}


def mark(state, key, status):
    with _lock:
        state[key] = {"status": status, "t": time.strftime("%Y-%m-%d %H:%M:%S")}
        tmp = STATE + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(state, fh, indent=1)
        os.replace(tmp, STATE)


def sh(cmd, log, env_extra=None, timeout=12 * 3600):
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2", "VECLIB_MAXIMUM_THREADS": "2",
                "PYTHONPATH": f"{NPA}:{SWT}"})
    env.update(env_extra or {})
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "a") as fh:
        fh.write(f"\n==== {time.strftime('%F %T')} $ {' '.join(cmd)}\n")
        fh.flush()
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                           timeout=timeout)
    return r.returncode


def write_selection(job, wave):
    """Selection CSV for `swtomotv produce`: the periods the iso stage actually mapped
    (run_production skips sparse periods the cache still holds), sigma_eff/LC = the
    production constants of this arm."""
    root = job["output_root"]
    pat = os.path.join(root, "production", wave, "map_T*.npz")
    Ts = sorted(float(re.search(r"map_T([0-9.]+)\.npz", os.path.basename(f)).group(1))
                for f in glob.glob(pat))
    if not Ts:
        return None
    sel = os.path.join(root, f"selection_{wave}_ffscan.csv")
    with open(sel, "w") as fh:
        fh.write("T,feasible,sigma_eff,LC\n")
        for T in Ts:
            fh.write(f"{T:g},1,0.025,{job['lc']}\n")
    return sel


def run_arm(job, state, stages, dry):
    name = f"{job['net']}_{job['run']}"
    jlog = os.path.join(LOGD, "jobs")
    ok = True
    # ---- stage: iso ----
    if "iso" in stages:
        for wave in job["waves"]:
            key = f"{name}:iso:{wave}"
            if state.get(key, {}).get("status") == "done":
                continue
            cmd = [DAS, os.path.join(HERE, "run_production.py"),
                   "--config", job["yaml"], "--wave", wave,
                   "--lc", str(job["lc"]), "--se", "0.025", "--fast"]
            print(f"[{time.strftime('%T')}] {key}", flush=True)
            if dry:
                print("   ", " ".join(cmd)); continue
            rc = sh(cmd, os.path.join(jlog, f"{name}_iso_{wave}.log"))
            mark(state, key, "done" if rc == 0 else "failed")
            ok &= (rc == 0)
    # ---- stage: topo + aniso ----
    if "aniso" in stages and ok:
        for wave in job["waves"]:
            key = f"{name}:aniso:{wave}"
            if state.get(key, {}).get("status") == "done":
                continue
            if dry:
                print(f"    (aniso) {key}"); continue
            sel = write_selection(job, wave)
            if sel is None:
                mark(state, key, "skipped-nocache")
                continue
            log = os.path.join(jlog, f"{name}_aniso_{wave}.log")
            rc = sh([DAS, "-m", "swtomotv.cli", "produce", "--config", job["yaml"],
                     "--wave", wave, "--selection", sel, "--topo"], log)
            if rc == 0:
                rc = sh([DAS, "-m", "swtomotv.cli", "aniso", "--config", job["yaml"],
                         "--wave", wave], log)
            mark(state, key, "done" if rc == 0 else "failed")
            ok &= (rc == 0)
    # ---- stage: styled (+ the pick-distribution QC figure, cheap) ----
    if "styled" in stages and ok:
        key = f"{name}:styled"
        if state.get(key, {}).get("status") != "done":
            sh([DAS, os.path.join(HERE, "ffscan_pick_hists.py"),
                "--yaml", job["yaml"], "--ff", str(job["ff"])],
               os.path.join(jlog, f"{name}_pickhist.log"))
            tag = f"r/lam>={job['ff']:.1f} dx{job['dx']:g}"
            cmd = [BASE, os.path.join(HERE, "ffscan_styled_maps.py"),
                   "--net", job["net"], "--yaml", job["yaml"], "--tag", tag,
                   "--measure", job["measure"]]
            if dry:
                print(f"    (styled) {key}")
            else:
                rc = sh(cmd, os.path.join(LOGD, "jobs", f"{name}_styled.log"),
                        env_extra={"PYTHONPATH": f"{NPA}:{HERE}:{SWT}"})
                mark(state, key, "done" if rc == 0 else "failed")
                ok &= (rc == 0)
    return name, ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default=os.path.join(LOGD, "ffscan_manifest.json"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--nets", default=None)
    ap.add_argument("--ff", default=None)
    ap.add_argument("--measures", default=None)
    ap.add_argument("--grids", default=None, help="coarse|fine|both (default both)")
    ap.add_argument("--stages", default="iso,aniso,styled")
    ap.add_argument("--reset-failed", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(args.manifest) as fh:
        jobs = json.load(fh)
    if args.nets:
        jobs = [j for j in jobs if j["net"] in args.nets.split(",")]
    if args.ff:
        keep = {float(x) for x in args.ff.split(",")}
        jobs = [j for j in jobs if j["ff"] in keep]
    if args.measures:
        jobs = [j for j in jobs if j["measure"] in args.measures.split(",")]
    if args.grids == "coarse":
        jobs = [j for j in jobs if not j["fine"]]
    elif args.grids == "fine":
        jobs = [j for j in jobs if j["fine"]]
    stages = args.stages.split(",")

    state = load_state()
    if args.reset_failed:
        state = {k: v for k, v in state.items() if v.get("status") == "done"}

    # cheap/informative first: coarse grids, group, ascending ff; nets interleave by size
    order = {"riehen": 0, "aargau": 1, "hautesorne": 2}
    jobs.sort(key=lambda j: (j["fine"], j["measure"] == "phase", j["ff"], order[j["net"]]))
    print(f"{len(jobs)} arms, stages={stages}, workers={args.workers}")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_arm, j, state, stages, args.dry_run) for j in jobs]
        for f in futs:
            name, ok = f.result()
            print(f"[{time.strftime('%T')}] ARM {'DONE' if ok else 'FAILED'} {name} "
                  f"({(time.time() - t0) / 3600:.1f} h elapsed)", flush=True)
    n_fail = sum(1 for v in state.values() if v.get("status") == "failed")
    print(f"campaign finished, {n_fail} failed stage keys (see {STATE})")


if __name__ == "__main__":
    main()
