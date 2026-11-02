#!/bin/sh
# Root wrapper: vLLM smoke/timing in max performance mode with tegrastats, always restores normal mode.
export HOME=/home/<user>
H=/home/<user>
L=$H/llm
R=$L/results_vllm
mkdir -p $R; rm -f $L/vllm_max.flag
cleanup() { pkill tegrastats 2>/dev/null; sh $H/perf_mode.sh off > $R/perf_off.log 2>&1; echo DONE > $L/vllm_max.flag; chown -R kimchsi90:kimchsi90 $R; }
trap cleanup EXIT
sh $H/perf_mode.sh on > $R/perf_on.log 2>&1
sleep 2
. $L/vllm_env/bin/activate
export PATH=/usr/local/cuda/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
for mode in max; do
  tegrastats --interval 500 --logfile $R/$mode.tegra > /dev/null 2>&1 &
  TP=$!
  cd $L && python vllm_smoke.py $L/models/tinyllama 128 > $R/$mode.log 2>&1
  kill $TP 2>/dev/null
done
