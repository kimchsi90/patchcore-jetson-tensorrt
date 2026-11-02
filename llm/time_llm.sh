#!/bin/sh
# Final max-mode LLM timing (prepared before the build exists; paths follow the plan below).
# Runs llama-bench for the Q4_K_M model with all layers on the GPU, with a tegrastats log next to each result. No sudo inside:
# run it as root wrapped by perf_mode.sh, e.g.  sudo sh -c 'sh /home/<user>/perf_mode.sh on; sh /home/<user>/llm/time_llm.sh; sh /home/<user>/perf_mode.sh off'
H=/home/<user>
L=$H/llm
export LD_LIBRARY_PATH=$L/llama.cpp/build/bin:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
BIN=$L/llama.cpp/build/bin/llama-bench
MODEL=$L/models/qwen2.5-1.5b-instruct-q4_k_m.gguf
R=$L/results
mkdir -p $R
[ -x $BIN ] || { echo "missing $BIN"; exit 2; }
[ -s $MODEL ] || { echo "missing $MODEL"; exit 2; }
run() {  # run <name> <llama-bench args...>
  name=$1; shift
  tegrastats --interval 200 > $R/$name.tegra 2>&1 &
  TP=$!
  $BIN -m $MODEL "$@" > $R/$name.txt 2>&1
  kill $TP 2>/dev/null
  echo "$(date +%T) done $name" >> $R/time_llm.log
}
run gpu_pp512_tg128 -ngl 99 -p 512 -n 128 -r 5
run gpu_pp128_tg128 -ngl 99 -p 128 -n 128 -r 5
run cpu_pp128_tg64  -ngl 0 -t 4 -p 128 -n 64 -r 3
