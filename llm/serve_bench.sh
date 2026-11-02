#!/bin/sh
# Start llama-server (Qwen2.5-1.5B Q4_K_M, GPU, 4 slots), run the load client, stop the server. Usage: serve_bench.sh <tag>
TAG=${1:-10w}
L=$HOME/llm
R=$L/results_serve; mkdir -p $R
export LD_LIBRARY_PATH=$L/llama.cpp/build/bin:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
$L/llama.cpp/build/bin/llama-server -m $L/models/qwen2.5-1.5b-instruct-q4_k_m.gguf -ngl 99 -c 4096 -np 4 --host 127.0.0.1 --port 8080 > $R/server_$TAG.log 2>&1 &
SP=$!
n=0; until curl -fs http://127.0.0.1:8080/health > /dev/null 2>&1; do n=$((n+1)); [ $n -gt 120 ] && { echo "server did not start" > $R/$TAG.txt; kill $SP; exit 1; }; sleep 1; done
free -m | sed -n 2p > $R/mem_$TAG.txt
tegrastats --interval 500 --logfile $R/$TAG.tegra > /dev/null 2>&1 &
TP=$!
python3 $L/serve_client.py 128 > $R/$TAG.txt 2>&1
free -m | sed -n 2p >> $R/mem_$TAG.txt
kill $TP 2>/dev/null; kill $SP 2>/dev/null; wait $SP 2>/dev/null
echo DONE >> $R/$TAG.txt
