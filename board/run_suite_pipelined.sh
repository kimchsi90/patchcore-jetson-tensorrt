#!/bin/sh
# Pipelined (stage-decoupled) end-to-end configs: P1 (post in the engine), P2 (resize+post in the engine), P4 (DLA backbone + CPU post).
# Root-free and independent of ~ (all paths come from the script directory), so `sudo sh run_suite_pipelined.sh TAG` works too.
# Usage:  sh ~/e2e/run_suite_pipelined.sh [TAG] [CONFIG ...]   TAG labels the result folder ~/e2e/results/<TAG>/ (default: pipe_<date>).
#         Env: DURATION (s, default 20), WARMUP (s, default 5), REPS (repeats per config, default 1),
#              RT (resize threads, default 2), QD (queue depth, default 4), SEQ=1 also runs the sequential counterpart of every config
#              (names SEQ_*; same engines, same flags minus --pipelined) for a same-session comparison.
# Safety: bench_e2e.py aborts a config at >= 82 C (CPU/GPU thermal zone); a shell watchdog kills it as a backstop; each config waits to cool < 70 C.
H=$(cd "$(dirname "$0")" && pwd)
cd $H || exit 1
TAG=${1:-pipe_$(date +%m%d_%H%M)}; [ $# -gt 0 ] && shift
R=$H/results/$TAG; mkdir -p $R
ONLY="$*"
DUR=${DURATION:-20}; WARM=${WARMUP:-5}; RT=${RT:-2}; QD=${QD:-4}
G1=$H/engines/p1_gpu_post.engine             # GPU engine with post inside (P1)
G2=$H/engines/p2_gpu_full900.engine          # GPU engine with resize + post inside, 900x900 input (P2)
DB=$H/engines/int8mix_dla0.engine            # DLA backbone WITHOUT post (m, xproj); runtime.DLA_core selects the core (P4)
for e in $G1 $G2 $DB $H/test_labels.csv; do [ -s $e ] || { echo "missing $e"; exit 2; }; done
[ -d $H/test900 ] || { echo "missing $H/test900"; exit 2; }
COMMON="--test-dir $H/test900 --labels $H/test_labels.csv"

temp() { m=0; for z in /sys/class/thermal/thermal_zone*; do
    case "$(cat $z/type 2>/dev/null)" in CPU-therm|GPU-therm) t=$(cat $z/temp 2>/dev/null); [ -n "$t" ] && [ "$t" -gt "$m" ] && m=$t ;; esac
  done; echo $m; }
watchdog() { while true; do t=$(temp); if [ "$t" -ge 82000 ]; then echo "WATCHDOG over temp $t $(date +%T)" >> $R/overtemp.log; pkill -f bench_e2e.py; fi; sleep 1; done; }
watchdog & WD=$!
trap 'kill $WD 2>/dev/null' EXIT INT TERM

{ echo "suite $TAG start $(date)"; uname -a; nvpmodel -q 2>&1 | head -3; cat /sys/devices/system/cpu/online; } > $R/suite_info.txt 2>&1

cool() { n=0; while [ "$(temp)" -ge 70000 ] && [ $n -lt 60 ]; do sleep 2; n=$((n+1)); done; }

# name | pipe | interp | workers | post threads (P4 only)
run() {
  r=1; while [ $r -le ${REPS:-1} ]; do
    run1 "$1" "$2" "$3" "$4" "$5" $r pipe
    [ "${SEQ:-0}" = 1 ] && run1 "SEQ_$1" "$2" "$3" "$4" "$5" $r seq
    r=$((r+1)); done
}
run1() {
  name=$1; pipe=$2; interp=$3; workers=$4; pt=$5; rep=$6; kind=$7
  base=${name#SEQ_}
  if [ -n "$ONLY" ] && ! echo " $ONLY " | grep -q " $base "; then return; fi
  [ ${REPS:-1} -gt 1 ] && name=${name}_r$rep
  if [ $kind = pipe ]; then extra="--pipelined --resize-threads $RT --queue-depth $QD --post-threads ${pt:-0}"; else extra=""; fi
  cool
  echo "== $name pipe=$pipe interp=$interp workers=$workers $extra start=$(date +%T) temp=$(temp)" | tee -a $R/suite.log
  python3 bench_e2e.py --mode bench --name $name --pipe $pipe --interp $interp --workers "$workers" $extra $COMMON \
      --duration $DUR --warmup $WARM --out $R/$name.json > $R/$name.log 2>&1
  echo "   rc=$? $(grep -m1 '"fps"' $R/$name.json 2>/dev/null)" | tee -a $R/suite.log
}

# P1: post inside the GPU engine, one or two GPU workers; resize on the feeder threads
run PIPE_P1_gpu            P1 area   "gpu=$G1"
run PIPE_P1_gpu_gpu2       P1 area   "gpu=$G1,gpu2=$G1"
# P2: resize + post inside the engine; feeders only convert uint8 -> fp32 into pinned memory
run PIPE_P2_gpu            P2 area   "gpu=$G2"
run PIPE_P2_gpu_gpu2       P2 area   "gpu=$G2,gpu2=$G2"
# P4: DLA backbone, CPU post inline in the accelerator thread (pt 0) or in a post thread pool (pt 2)
run PIPE_P4_dla0           P4 area   "dla0=$DB" 0
run PIPE_P4_dla0_pt2       P4 area   "dla0=$DB" 2
run PIPE_P4_dla0_dla1      P4 area   "dla0=$DB,dla1=$DB" 0
run PIPE_P4_dla0_dla1_pt2  P4 area   "dla0=$DB,dla1=$DB" 2
# mixed: the gpu worker is P1, the DLA workers P4
run PIPE_mix               P4 area   "gpu=$G1@P1,dla0=$DB,dla1=$DB" 0
run PIPE_mix_pt2           P4 area   "gpu=$G1@P1,dla0=$DB,dla1=$DB" 2
# INTER_LINEAR
run PIPE_P1_gpu_gpu2_linear      P1 linear "gpu=$G1,gpu2=$G1"
run PIPE_P4_dla0_dla1_pt2_linear P4 linear "dla0=$DB,dla1=$DB" 2
run PIPE_mix_pt2_linear          P4 linear "gpu=$G1@P1,dla0=$DB,dla1=$DB" 2
echo "suite $TAG done $(date)" | tee -a $R/suite.log
