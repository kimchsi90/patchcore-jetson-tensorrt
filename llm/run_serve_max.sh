#!/bin/sh
# Root wrapper: llama-server load test in max performance mode, always restores normal mode.
export HOME=/home/<user>
H=/home/<user>
cleanup() { pkill tegrastats 2>/dev/null; sh $H/perf_mode.sh off > $H/llm/results_serve/perf_off.log 2>&1; echo DONE > $H/llm/serve_max.flag; chown -R kimchsi90:kimchsi90 $H/llm/results_serve; }
trap cleanup EXIT
mkdir -p $H/llm/results_serve; rm -f $H/llm/serve_max.flag
sh $H/perf_mode.sh on > $H/llm/results_serve/perf_on.log 2>&1
sleep 2
sh $H/llm/serve_bench.sh max
