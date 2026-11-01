#!/bin/sh
# Throughput and energy of GPU + DLA combinations with the accuracy-validated engines (run as root, detached).
#   DLA engine : calibrated int8 mixed (stem + layer1.0 + head fp16) or fp16, built for core 0, run on core 0 and core 1
#   GPU engine : mixed engine (mixed_name) or fp16
# Max performance mode and forced-fan request via perf_mode.sh (the kernel may cap the fan). Always restores the normal mode.
H=/home/<user>
S=$H/trt_run/combo
T=/usr/src/tensorrt/bin/trtexec
DLA_INT8=$H/trt_dla/bottle/int8mix_dla0.engine
DLA_FP16=$H/trt_dla/bottle/fp16_dla0.engine
GPU_MIX=$H/trt_run/mixed_name.engine
GPU_FP16=$H/trt_run/fp16.engine
mkdir -p "$S"; rm -f "$S/combo_done.flag" "$S/overtemp.log"
echo "start $(date +%T)" > "$S/combo.log"

( while true; do
    for z in /sys/class/thermal/thermal_zone*; do
      case "$(cat $z/type 2>/dev/null)" in CPU-therm|GPU-therm|AUX-therm)
        t=$(cat $z/temp 2>/dev/null)
        case "$t" in ''|*[!0-9]*) ;; *) if [ "$t" -ge 85000 ]; then echo "OVERTEMP $(cat $z/type) $t $(date +%T)" >> "$S/overtemp.log"; pkill trtexec; fi ;; esac ;;
      esac
    done
    sleep 2
  done ) &
WD=$!
cleanup() { kill $WD 2>/dev/null; pkill tegrastats 2>/dev/null; sh $H/perf_mode.sh off > "$S/perf_off.log" 2>&1; echo "end $(date +%T)" >> "$S/combo.log"; echo DONE > "$S/combo_done.flag"; }
trap cleanup EXIT

sh $H/perf_mode.sh on > "$S/perf_on.log" 2>&1
OPTS="--duration=25 --warmUp=1500 --avgRuns=100 --useSpinWait"

# one process or several concurrent ones, each "label|engine|dlacore(-1 = GPU)"; tegrastats covers the whole group
group() {
  name=$1; shift
  tegrastats --interval 200 > "$S/$name.tegra" 2>&1 &
  TP=$!
  pids=""
  for spec in "$@"; do
    lab=${spec%%|*}; rest=${spec#*|}; eng=${rest%%|*}; core=${rest#*|}
    if [ "$core" = "-1" ]; then
      $T --loadEngine=$eng $OPTS > "$S/${name}_$lab.log" 2>&1 &
    else
      $T --loadEngine=$eng --useDLACore=$core $OPTS > "$S/${name}_$lab.log" 2>&1 &
    fi
    pids="$pids $!"
  done
  wait $pids
  kill $TP 2>/dev/null
  echo "$(date +%T) done $name" >> "$S/combo.log"
}

group solo_gpu_mix       "gpu|$GPU_MIX|-1"
group solo_dla0_int8mix  "dla0|$DLA_INT8|0"
group solo_dla1_int8mix  "dla1|$DLA_INT8|1"
group dla0dla1_int8mix   "dla0|$DLA_INT8|0" "dla1|$DLA_INT8|1"
group gpu_dla0_int8mix   "gpu|$GPU_MIX|-1" "dla0|$DLA_INT8|0"
group gpu_dla0dla1_int8mix "gpu|$GPU_MIX|-1" "dla0|$DLA_INT8|0" "dla1|$DLA_INT8|1"
group solo_dla0_fp16     "dla0|$DLA_FP16|0"
group gpu_dla0dla1_fp16  "gpu|$GPU_FP16|-1" "dla0|$DLA_FP16|0" "dla1|$DLA_FP16|1"
