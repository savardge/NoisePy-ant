"""ffscan step 1->4 driver: QC-wait -> base exports -> threshold tables -> YAMLs -> campaign.

Idempotent: every step checks its outputs first, so re-running resumes. Run it in the
das-ambient-noise env; it shells out with the right PYTHONPATH itself.

  nohup $DAS ffscan_driver.py > ffscan_logs/driver.log 2>&1 &
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
LOGD = os.path.normpath(os.path.join(EHM, "..", "ffscan_logs"))
DAS = "/opt/anaconda3/envs/das-ambient-noise/bin/python"
NETS = ("riehen", "aargau", "hautesorne")
WAVES = {"riehen": ("fund", "overtone", "love"),
         "aargau": ("fund", "overtone", "love"),
         "hautesorne": ("fund", "overtone", "love", "love_ot")}

env = dict(os.environ)
env["PYTHONPATH"] = os.path.expanduser("~/Codes/NoisePy-ant")


def run(cmd, log):
    with open(log, "a") as fh:
        fh.write(f"\n==== {time.strftime('%F %T')} $ {' '.join(cmd)}\n")
        fh.flush()
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    if rc:
        raise SystemExit(f"FAILED (rc={rc}): {' '.join(cmd)}  [log: {log}]")


def qc_done(net):
    qc = f"{EHM}/{net}/tomo/1_velocity_maps/inputs/ffscan/qc"
    return os.path.exists(f"{qc}/qc_before_after.png")


def prepare(net):
    inputs = f"{EHM}/{net}/tomo/1_velocity_maps/inputs"
    ffdir = f"{inputs}/ffscan"
    qcsv = f"{ffdir}/qc/picks_unified_QCd.csv"
    log = f"{LOGD}/prepare_{net}.log"
    for measure, suffix in (("group", "_nf"), ("phase", "_nf_phase")):
        probe = f"{ffdir}/picks_fund_uni{suffix}.csv"
        if not os.path.exists(probe):
            print(f"[{time.strftime('%T')}] {net}: export {measure}", flush=True)
            run([DAS, f"{HERE}/export_unified_tomo_picks.py", "--net", net,
                 "--measure", measure, "--src", qcsv, "--outdir", ffdir,
                 "--out-suffix", suffix], log)
    probe = f"{ffdir}/picks_fund_uni_ff1.0.csv"
    if not os.path.exists(probe):
        print(f"[{time.strftime('%T')}] {net}: threshold tables", flush=True)
        run([DAS, f"{HERE}/ffscan_filter_picks.py", "--net", net], log)


def main():
    for net in NETS:
        while not qc_done(net):
            print(f"[{time.strftime('%T')}] waiting on {net} QC re-run ...", flush=True)
            time.sleep(300)
        prepare(net)
    run([DAS, f"{HERE}/ffscan_make_yamls.py"], f"{LOGD}/driver_steps.log")
    print(f"[{time.strftime('%T')}] launching orchestrator", flush=True)
    rc = subprocess.run([DAS, f"{HERE}/ffscan_orchestrator.py", "--workers", "4"],
                        env=env).returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()
