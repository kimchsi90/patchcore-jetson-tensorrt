#!/bin/sh
# Pure-CPU llama.cpp (CUDA hidden) and GPU runs with temperature/fan logging (called by limits_all.sh as root).
L=$HOME/llm; R=$L/results_limits; mkdir -p $R
export LD_LIBRARY_PATH=$L/llama.cpp/build/bin:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
B=$L/llama.cpp/build/bin/llama-bench; M=$L/models/qwen2.5-1.5b-instruct-q4_k_m.gguf
fan() { for d in /sys/class/hwmon/hwmon*; do [ "$(cat $d/name 2>/dev/null)" = pwmfan ] && cat $d/pwm1; done; }
run() { tag=$1; shift
  tegrastats --interval 500 --logfile $R/$tag.tegra > /dev/null 2>&1 &
  TP=$!
  ( while kill -0 $TP 2>/dev/null; do echo "$(date +%T) pwm=$(fan)"; sleep 2; done ) > $R/$tag.fan &
  FP=$!
  "$@" > $R/$tag.txt 2>&1
  kill $TP $FP 2>/dev/null
}
MODE=$1; T=$2
run ${MODE}_gpu  $B -m $M -ngl 99 -p 128,512 -n 128 -r 3
run ${MODE}_cpuonly env CUDA_VISIBLE_DEVICES= $B -m $M -ngl 0 -t $T -p 128,512 -n 32 -r 2
