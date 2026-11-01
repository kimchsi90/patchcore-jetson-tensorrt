#!/bin/sh
# Jetson-native speed suite (run as root, detached). Results go to $S, parsed on the PC by parse_suite.py.
#   Phase A (default mode): build DLA engines (speed only; int8 engines are uncalibrated).
#   Phase B (default mode): trtexec matrix per engine: base, CUDA graph, no data transfers, 2/3 streams. tegrastats per run.
#   Phase C (max mode + fan forced): the same matrix, DLA0+DLA1+GPU concurrent runs, 10 minute soak. Always restores normal mode.
# Watchdog: any CPU/GPU/AUX/AO thermal zone >= 85 C kills trtexec.
H=/home/<user>
R=$H/trt_run
S=$H/trt_run/suite
# engines built by revalidate.sh on this JetPack: fp16, mixed (stem + layer1.0 fp16, rest int8), int8 minmax
E_FP16=$R/fp16.engine
E_MIX=$R/mixed_name.engine
E_INT8=$R/int8mm.engine
T=/usr/src/tensorrt/bin/trtexec
ONNX=$R/fused_random384_c0.01_rmax_pf_pre.onnx
mkdir -p "$S"
rm -f "$S/suite_done.flag" "$S/overtemp.log"
echo "start $(date)" > "$S/suite.log"

( while true; do
    for z in /sys/devices/virtual/thermal/thermal_zone*; do
      case "$(cat $z/type)" in CPU-therm|GPU-therm|AUX-therm|AO-therm)
        t=$(cat $z/temp)
        if [ "$t" -ge 85000 ]; then echo "OVERTEMP $(cat $z/type) $t $(date +%T)" >> "$S/overtemp.log"; pkill trtexec; fi ;;
      esac
    done
    sleep 2
  done ) &
WD=$!

cleanup() { kill $WD 2>/dev/null; pkill tegrastats 2>/dev/null; sh $H/perf_mode.sh off > "$S/perf_off.log" 2>&1; echo "end $(date)" >> "$S/suite.log"; echo DONE > "$S/suite_done.flag"; }
trap cleanup EXIT

# run <label> <trtexec args...>: one trtexec run with a tegrastats log next to it
run() {
  label=$1; shift
  tegrastats --interval 200 > "$S/$label.tegra" 2>&1 &
  TP=$!
  $T "$@" > "$S/$label.log" 2>&1
  kill $TP 2>/dev/null
  echo "$(date +%T) done $label" >> "$S/suite.log"
}

matrix() {
  mode=$1
  for e in fp16:$E_FP16 mixed:$E_MIX int8mm:$E_INT8; do
    name=${e%%:*}; path=${e#*:}
    base="--loadEngine=$path --duration=15 --warmUp=1000 --avgRuns=100 --useSpinWait"
    run ${mode}_${name}_base      $base
    run ${mode}_${name}_graph     $base --useCudaGraph
    run ${mode}_${name}_nodt      $base --noDataTransfers
    run ${mode}_${name}_streams2  $base --streams=2 --threads
    run ${mode}_${name}_streams3  $base --streams=3 --threads
  done
}

# ---- Phase A: DLA engines (default mode)
for core in 0 1; do
  $T --onnx=$ONNX --useDLACore=$core --allowGPUFallback --fp16 --workspace=1024 --saveEngine=$S/dla${core}_fp16.engine > "$S/build_dla${core}_fp16.log" 2>&1
  echo "$(date +%T) built dla${core}_fp16" >> "$S/suite.log"
  $T --onnx=$ONNX --useDLACore=$core --allowGPUFallback --int8 --fp16 --workspace=1024 --saveEngine=$S/dla${core}_int8.engine > "$S/build_dla${core}_int8.log" 2>&1
  echo "$(date +%T) built dla${core}_int8" >> "$S/suite.log"
done

# ---- Phase B: default mode
nvpmodel -q > "$S/mode_default.txt" 2>&1
jetson_clocks --show >> "$S/mode_default.txt" 2>&1
matrix default

# ---- Phase C: max mode
sh $H/perf_mode.sh on > "$S/perf_on.log" 2>&1
nvpmodel -q > "$S/mode_max.txt" 2>&1
jetson_clocks --show >> "$S/mode_max.txt" 2>&1
matrix max

# DLA runs alone, then DLA0 + DLA1 + GPU concurrently
for k in fp16 int8; do
  for core in 0 1; do
    run max_dla${core}_${k}_alone --loadEngine=$S/dla${core}_${k}.engine --useDLACore=$core --duration=15 --warmUp=1000 --avgRuns=100 --useSpinWait
  done
done
for k in fp16 int8; do
  if [ "$k" = fp16 ]; then g=$E_FP16; else g=$E_MIX; fi
  tegrastats --interval 200 > "$S/max_concurrent_${k}.tegra" 2>&1 &
  TP=$!
  $T --loadEngine=$S/dla0_${k}.engine --useDLACore=0 --duration=20 --warmUp=1000 --avgRuns=100 --useSpinWait > "$S/max_concurrent_${k}_dla0.log" 2>&1 &
  P1=$!
  $T --loadEngine=$S/dla1_${k}.engine --useDLACore=1 --duration=20 --warmUp=1000 --avgRuns=100 --useSpinWait > "$S/max_concurrent_${k}_dla1.log" 2>&1 &
  P2=$!
  $T --loadEngine=$g --duration=20 --warmUp=1000 --avgRuns=100 --useSpinWait > "$S/max_concurrent_${k}_gpu.log" 2>&1 &
  P3=$!
  wait $P1 $P2 $P3
  kill $TP 2>/dev/null
  echo "$(date +%T) done concurrent $k" >> "$S/suite.log"
done

# 10 minute soak on the best mixed engine, 2 streams
tegrastats --interval 1000 > "$S/max_soak.tegra" 2>&1 &
TP=$!
$T --loadEngine=$E_MIX --duration=600 --warmUp=1000 --avgRuns=1000 --useSpinWait --streams=2 --threads > "$S/max_soak.log" 2>&1
kill $TP 2>/dev/null
echo "$(date +%T) done soak" >> "$S/suite.log"
