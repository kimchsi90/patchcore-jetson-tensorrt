#!/bin/sh
# Runs ALL timed e2e configs sequentially (root-free). One log + one json per config in ~/e2e/results/<tag>/.
# Usage:  sh ~/e2e/run_suite.sh [TAG] [CONFIG ...]     TAG labels the result folder (default: run_<date>).
#         DURATION (s, default 20), WARMUP (s, default 5) and REPS (repeats per config, default 1) can be set in the environment.
#         In the 10W default mode the mixed GPU+DLA configs jumped between plateaus from run to run (67 / 73-86 / 104 FPS); use REPS=3 to see it.
# Run tegrastats separately if wanted, e.g.  tegrastats --interval 500 --logfile ~/e2e/results/<TAG>/tegra.log &
# Safety: bench_e2e.py aborts a config when CPU/GPU thermal zone >= 82 C; a shell watchdog kills it as a backstop.
H=$(cd "$(dirname "$0")" && pwd)   # script dir, so it also works under sudo ($HOME may be /root)
cd $H || exit 1
TAG=${1:-run_$(date +%m%d_%H%M)}; [ $# -gt 0 ] && shift
R=$H/results/$TAG; mkdir -p $R
ONLY="$*"
DUR=${DURATION:-20}; WARM=${WARMUP:-5}
GP0=$H/engines/p0_gpu_mixed_name.engine      # engine without post (m, xproj): used by P0
G1=$H/engines/p1_gpu_post.engine            # GPU engine, post inside
D0=$H/engines/p3_dla0_post.engine           # DLA backbone engine (built for core 0; runtime.DLA_core selects the core), post falls back to GPU
G2=$H/engines/p2_gpu_full900.engine         # 900x900 input, resize + post inside
for e in $GP0 $G1 $D0 $G2; do [ -s $e ] || { echo "missing engine $e"; exit 2; }; done

temp() { m=0; for z in /sys/class/thermal/thermal_zone*; do
    case "$(cat $z/type 2>/dev/null)" in CPU-therm|GPU-therm) t=$(cat $z/temp 2>/dev/null); [ -n "$t" ] && [ "$t" -gt "$m" ] && m=$t ;; esac
  done; echo $m; }
watchdog() { while true; do t=$(temp); if [ "$t" -ge 82000 ]; then echo "WATCHDOG over temp $t $(date +%T)" >> $R/overtemp.log; pkill -f bench_e2e.py; fi; sleep 1; done; }
watchdog & WD=$!
trap 'kill $WD 2>/dev/null' EXIT INT TERM

{ echo "suite $TAG start $(date)"; uname -a; nvpmodel -q 2>&1 | head -3; cat /sys/devices/system/cpu/online; } > $R/suite_info.txt 2>&1

cool() { n=0; while [ "$(temp)" -ge 70000 ] && [ $n -lt 60 ]; do sleep 2; n=$((n+1)); done; }

# name | pipe | interp | workers
run() {
  r=1; while [ $r -le ${REPS:-1} ]; do run1 "$1" "$2" "$3" "$4" "$5" $r; r=$((r+1)); done
}
run1() {
  name=$1; pipe=$2; interp=$3; workers=$4; extra=$5; rep=$6
  if [ -n "$ONLY" ] && ! echo " $ONLY " | grep -q " $name "; then return; fi
  [ ${REPS:-1} -gt 1 ] && name=${name}_r$rep
  cool
  echo "== $name pipe=$pipe interp=$interp workers=$workers start=$(date +%T) temp=$(temp)" | tee -a $R/suite.log
  python3 bench_e2e.py --mode bench --name $name --pipe $pipe --interp $interp --workers "$workers" $extra \
      --duration $DUR --warmup $WARM --out $R/$name.json > $R/$name.log 2>&1
  echo "   rc=$? $(grep -m1 '"fps"' $R/$name.json 2>/dev/null)" | tee -a $R/suite.log
}

python3 bench_e2e.py --mode micro --out $R/micro.json > $R/micro.log 2>&1
run P0_gpu1            P0 area   "gpu=$GP0"
run P0_gpu1_linear     P0 linear "gpu=$GP0"
run P1_gpu1            P1 area   "gpu=$G1"
run P1_gpu1_linear     P1 linear "gpu=$G1"
run P1_gpu2            P1 area   "gpu=$G1,gpu2=$G1"
run P1_gpu2_linear     P1 linear "gpu=$G1,gpu2=$G1"
run P2_gpu1            P2 area   "gpu=$G2"
run P2_gpu2            P2 area   "gpu=$G2,gpu2=$G2"
run P3_dla0            P1 area   "dla0=$D0"
run P3_dla1            P1 area   "dla1=$D0"
run P3_dla0_dla1       P1 area   "dla0=$D0,dla1=$D0"
run P3_gpu_dla0_dla1   P1 area   "gpu=$G1,dla0=$D0,dla1=$D0"
run P3_dla0_linear     P1 linear "dla0=$D0"
run P3_dla0_dla1_linear P1 linear "dla0=$D0,dla1=$D0"
run P3_gpu_dla0_dla1_blockingstream P1 linear "gpu=$G1,dla0=$D0,dla1=$D0" "--blocking-stream"
run P3_gpu_dla0_dla1_linear P1 linear "gpu=$G1,dla0=$D0,dla1=$D0"
echo "suite $TAG done $(date)" | tee -a $R/suite.log
