#!/bin/sh
# Normal-mode (10W) llama-bench: GPU all layers and CPU only. Detached; writes bench10w.flag.
L=$HOME/llm
B=$L/llama.cpp/build/bin/llama-bench
M=$L/models/qwen2.5-1.5b-instruct-q4_k_m.gguf
R=$L/results10w
mkdir -p $R; rm -f $L/bench10w.flag
export LD_LIBRARY_PATH=$L/llama.cpp/build/bin:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
nvpmodel -q > $R/nvpmodel.txt 2>&1
free -m > $R/free_before.txt
$B -m $M -ngl 99 -p 128,512 -n 128 -r 3 > $R/gpu.txt 2>&1
free -m > $R/free_after.txt
$B -m $M -ngl 99 -fa 1 -p 128,512 -n 128 -r 3 > $R/gpu_fa.txt 2>&1
$B -m $M -ngl 0 -t 4 -p 128 -n 32 -r 2 > $R/cpu.txt 2>&1
echo DONE > $L/bench10w.flag
