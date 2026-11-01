#!/bin/sh
# Stage-split configs only: P4 (CPU resize + DLA backbone + CPU post) and P5 (CPU resize + DLA backbone + small GPU post engine).
# Root-free and independent of ~ (all paths come from the script directory), so `sudo sh run_suite_split.sh TAG` works too.
# Usage:  sh ~/e2e/run_suite_split.sh [TAG] [CONFIG ...]    TAG labels the result folder ~/e2e/results/<TAG>/ (default: split_<date>).
#         DURATION (s, default 20), WARMUP (s, default 5), REPS (repeats per config, default 1) are read from the environment.
# Safety: bench_e2e.py aborts a config at >= 82 C (CPU/GPU thermal zone); a shell watchdog kills it as a backstop; each config waits to cool < 70 C.
H=$(cd "$(dirname "$0")" && pwd)
cd $H || exit 1
TAG=${1:-split_$(date +%m%d_%H%M)}; [ $# -gt 0 ] && shift
R=$H/results/$TAG; mkdir -p $R
ONLY="$*"
DUR=${DURATION:-20}; WARM=${WARMUP:-5}
G1=$H/engines/p1_gpu_post.engine             # GPU engine with post inside (the gpu worker of the mixed sets, pipe P1)
DB=$H/engines/int8mix_dla0.engine            # DLA backbone WITHOUT post (m, xproj); runtime.DLA_core selects the core
PE=$H/engines/p5_gpu_post_only.engine        # GPU post-only engine (m, xproj -> score_max, score_top10, map), fp32
for e in $G1 $DB $PE $H/test_labels.csv; do [ -s $e ] || { echo "missing $e"; exit 2; }; done
[ -d $H/test900 ] || { echo "missing $H/test900"; exit 2; }
COMMON="--test-dir $H/test900 --labels $H/test_labels.csv --post-engine $PE"

temp() { m=0; for z in /sys/class/thermal/thermal_zone*; do
    case "$(cat $z/type 2>/dev/null)" in CPU-therm|GPU-therm) t=$(cat $z/temp 2>/dev/null); [ -n "$t" ] && [ "$t" -gt "$m" ] && m=$t ;; esac
  done; echo $m; }
watchdog() { while true; do t=$(temp); if [ "$t" -ge 82000 ]; then echo "WATCHDOG over temp $t $(date +%T)" >> $R/overtemp.log; pkill -f bench_e2e.py; fi; sleep 1; done; }
watchdog & WD=$!
trap 'kill $WD 2>/dev/null' EXIT INT TERM

{ echo "suite $TAG start $(date)"; uname -a; nvpmodel -q 2>&1 | head -3; cat /sys/devices/system/cpu/online; } > $R/suite_info.txt 2>&1

cool() { n=0; while [ "$(temp)" -ge 70000 ] && [ $n -lt 60 ]; do sleep 2; n=$((n+1)); done; }

# name | pipe | interp | workers | extra args
run() {
  r=1; while [ $r -le ${REPS:-1} ]; do run1 "$1" "$2" "$3" "$4" "$5" $r; r=$((r+1)); done
}
run1() {
  name=$1; pipe=$2; interp=$3; workers=$4; extra=$5; rep=$6
  if [ -n "$ONLY" ] && ! echo " $ONLY " | grep -q " $name "; then return; fi
  [ ${REPS:-1} -gt 1 ] && name=${name}_r$rep
  cool
  echo "== $name pipe=$pipe interp=$interp workers=$workers $extra start=$(date +%T) temp=$(temp)" | tee -a $R/suite.log
  python3 bench_e2e.py --mode bench --name $name --pipe $pipe --interp $interp --workers "$workers" $extra $COMMON \
      --duration $DUR --warmup $WARM --out $R/$name.json > $R/$name.log 2>&1
  echo "   rc=$? $(grep -m1 '"fps"' $R/$name.json 2>/dev/null)" | tee -a $R/suite.log
}

python3 bench_e2e.py --mode check --name post_engine_check --pipe P5 --workers dla0=$DB $COMMON --out $R/post_engine_check.json > $R/post_engine_check.log 2>&1
echo "post engine check rc=$?" | tee -a $R/suite.log
# gpu worker of the mixed sets is the P1 engine (post inside); the DLA workers use the split pipeline
run P4_dla0            P4 area   "dla0=$DB"
run P4_dla1            P4 area   "dla1=$DB"
run P4_dla0_dla1       P4 area   "dla0=$DB,dla1=$DB"
run P4_gpu_dla0_dla1   P4 area   "gpu=$G1@P1,dla0=$DB,dla1=$DB"
run P5_dla0            P5 area   "dla0=$DB"
run P5_dla1            P5 area   "dla1=$DB"
run P5_dla0_dla1       P5 area   "dla0=$DB,dla1=$DB"
run P5_gpu_dla0_dla1   P5 area   "gpu=$G1@P1,dla0=$DB,dla1=$DB"
run P4_dla0_dla1_linear P4 linear "dla0=$DB,dla1=$DB"
run P5_dla0_dla1_linear P5 linear "dla0=$DB,dla1=$DB"
# hand-over comparison: device buffers (default above) vs D2H + copy + H2D
run P5host_dla0        P5 area   "dla0=$DB" "--p5-link host"
run P5host_dla0_dla1   P5 area   "dla0=$DB,dla1=$DB" "--p5-link host"
echo "suite $TAG done $(date)" | tee -a $R/suite.log
