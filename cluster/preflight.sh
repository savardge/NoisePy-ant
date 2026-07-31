#!/bin/sh
# Readiness check for the yggdrasil ts-PWS chain. Read-only: submits nothing, writes only
# a probe file under $PROJ to prove the output tree is writable. Run it on a login node
# from this directory BEFORE the first sbatch.
#
#   sh preflight.sh            # all three networks
#   sh preflight.sh riehen     # just one
#
# Section 1 answers "what am I allowed to request"; sections 2-5 answer "will the jobs run";
# section 6 measures the real per-pair cost so the array widths in README.md can be checked
# against this machine rather than against the laptop timings they were derived from.
. "$(dirname "$0")/env.sh"
NETS="${*:-riehen aargau hautesorne}"
FAIL=0
ok()   { echo "  OK    $*"; }
bad()  { echo "  FAIL  $*"; FAIL=$((FAIL+1)); }
warn() { echo "  WARN  $*"; }

echo "============ 1. Slurm limits for $USER ============"
echo "-- association limits (GrpTRES/MaxTRES = the CPU ceiling; MaxSubmit = queue depth) --"
sacctmgr -p show assoc user="$USER" \
  format=Cluster,Account,Partition,QOS,GrpTRES,MaxTRES,MaxJobs,MaxSubmit,MaxWall 2>/dev/null \
  || warn "sacctmgr unavailable"
echo "-- QOS limits (PU = per user) --"
sacctmgr -p show qos format=Name,Priority,MaxWall,MaxTRESPU,MaxJobsPU,MaxSubmitJobsPU 2>/dev/null \
  | head -20 || warn "sacctmgr qos unavailable"
echo "-- array + job-count caps --"
scontrol show config 2>/dev/null | grep -Ei 'MaxArraySize|MaxJobCount' || warn "scontrol unavailable"
echo "-- partitions used here --"
for p in shared-cpu public-bigmem public-cpu; do
    scontrol show partition "$p" 2>/dev/null \
      | tr ' ' '\n' | grep -E '^(PartitionName|MaxTime|MaxNodes|TotalCPUs|State)=' \
      | tr '\n' ' ' | sed 's/^/  /'; echo
done
echo "-- your current usage --"
echo "  queued/running jobs: $(squeue -h -u "$USER" 2>/dev/null | wc -l)"
sshare -U -u "$USER" 2>/dev/null | head -3
echo
echo "  >> Set the array throttle (%N in --array=0-49%N) so N x cpus-per-task stays under"
echo "     the CPU ceiling above. With --cpus-per-task=8, %25 asks for 200 CPUs."

echo
echo "============ 2. Code, environment, paths ============"
[ -d "$NOISEPY" ] && ok "NOISEPY $NOISEPY" || bad "NOISEPY missing: $NOISEPY"
for f in scripts/picking/build_tspws_stacks.py scripts/picking/substack_jackknife.py \
         scripts/picking/dispersion_unified.py scripts/picking/qc_unified_picks.py \
         scripts/picking/export_unified_tomo_picks.py scripts/picking/run_pipeline.py \
         scripts/picking/attach_substack_sigma.py scripts/picking/plot_sigma_census.py \
         scripts/picking/plot_exported_picks_hist2d.py noisepy/dispersion.py \
         noisepy/stacking.py noisepy/unified_picking.py; do
    [ -f "$NOISEPY/$f" ] && ok "$f" || bad "missing $f"
done
activate_env
echo "  python: $(command -v python)"
python - <<'EOF' || echo "  FAIL  import check"
import importlib, sys
miss = [m for m in ("numpy", "scipy", "h5py", "pycwt", "pandas", "matplotlib", "yaml")
        if not importlib.util.find_spec(m)]
print("  %s  imports: %s" % ("FAIL " if miss else "OK   ",
                             "missing " + ", ".join(miss) if miss else "all present"))
sys.exit(1 if miss else 0)
EOF
[ $? -ne 0 ] && FAIL=$((FAIL+1))
python -c "import matplotlib; matplotlib.use('Agg')" 2>/dev/null \
  && ok "matplotlib Agg backend (headless figures)" || bad "matplotlib Agg unavailable"

echo
echo "============ 3. Output tree writable ============"
if mkdir -p "$PROJ" 2>/dev/null && touch "$PROJ/.preflight" 2>/dev/null; then
    rm -f "$PROJ/.preflight"; ok "PROJ writable: $PROJ"
    df -h "$PROJ" | tail -1 | sed 's/^/        /'
else
    bad "PROJ not writable: $PROJ"
fi

echo
echo "============ 4. Configs agree with env.sh ============"
for net in $NETS; do
    for c in modesep pipeline; do
        f="$NOISEPY/param_files/cluster/${c}_${net}_tspws.yaml"
        [ -f "$f" ] && ok "$(basename "$f")" || bad "missing $(basename "$f") (run make_cluster_configs.py)"
    done
    m="$NOISEPY/param_files/cluster/modesep_${net}_tspws.yaml"
    if [ -f "$m" ]; then
        want="$PROJ/$net/stacks_tspws"
        got=$(grep -E '^\s*stack_root:' "$m" | head -1 | sed 's/.*"\(.*\)".*/\1/')
        [ "$got" = "$want" ] && ok "  stack_root matches env.sh" \
                             || bad "  stack_root is '$got', env.sh implies '$want'"
    fi
    if grep -qE '/Users/|/Volumes/' "$NOISEPY/param_files/cluster/"*_"$net"_tspws.yaml 2>/dev/null; then
        bad "  leftover laptop path in $net configs (re-run make_cluster_configs.py)"
    fi
done

echo
echo "============ 5. Input substacks ============"
for net in $NETS; do
    src=$(src_root "$net"); code=$(net_code "$net")
    if [ -d "$src" ]; then
        n=$(find "$src" -mindepth 2 -name "$code.*_$code.*.h5" 2>/dev/null | wc -l)
        [ "$n" -gt 0 ] && ok "$net: $n pairs under $src" \
                       || bad "$net: 0 pairs matching $code.*_$code.*.h5 under $src"
        built=$(find "$PROJ/$net/stacks_tspws" -name '*.h5' 2>/dev/null | wc -l)
        echo "        already built: $built  ->  $((n - built)) to do"
    else
        bad "$net: source missing: $src"
    fi
done

echo
echo "============ 6. Per-pair cost probe (3 pairs/network) ============"
echo "Times one ts-PWS build per network on THIS hardware, so the array widths in"
echo "README.md can be re-derived instead of trusted."
for net in $NETS; do
    src=$(src_root "$net"); code=$(net_code "$net")
    [ -d "$src" ] || continue
    python - "$src" "$code" "$net" <<'EOF'
import glob, os, sys, time
import h5py, numpy as np
sys.path.insert(0, os.environ["NOISEPY"])
from noisepy import dispersion
src, code, net = sys.argv[1:4]
files = sorted(glob.glob(os.path.join(src, "%s.*" % code, "%s.*_%s.*.h5" % (code, code))))[:3]
if not files:
    print("  %-11s no pairs" % net); raise SystemExit
tot, nw = 0.0, []
for fn in files:
    with h5py.File(fn, "r") as f:
        aux = f["AuxiliaryData"]
        tg = sorted(k for k in aux if k.startswith("T"))
        at = next((aux[k]["ZZ"].attrs for k in tg if "ZZ" in aux[k]), None)
        if at is None or not tg:
            continue
        nw.append(len(tg))
        dist, dt = float(at["dist"]), float(at["dt"])
        npts = aux[tg[0]][sorted(aux[tg[0]].keys())[0]].shape[0]
        mid = npts // 2
        L = min(int(dist / 0.2 / dt) + 64, mid)
        blocks = []
        for i in range(0, len(tg), 2):
            acc, cnt = 0.0, 0
            for k in tg[i:i + 2]:
                if "ZZ" in aux[k]:
                    acc = acc + aux[k]["ZZ"][mid - L:mid + L + 1].astype(np.float64); cnt += 1
            if cnt:
                blocks.append(acc / cnt)
    if len(blocks) < 6:
        continue
    t0 = time.time(); dispersion.tf_pws(np.asarray(blocks), dt); tot += time.time() - t0
n = max(len(nw), 1)
per_pair = tot / n * 9.0          # one component timed; the build does 9
print("  %-11s windows/pair ~%d | lag samples %d | %.2f s/pair/core (9 comps)"
      % (net, int(np.median(nw)) if nw else 0, 2 * L + 1, per_pair))
EOF
done

echo
echo "============ RESULT ============"
[ "$FAIL" -eq 0 ] && echo "  no blocking failures -- safe to submit" \
                  || echo "  $FAIL blocking failure(s) above -- fix before submitting"
exit $FAIL
