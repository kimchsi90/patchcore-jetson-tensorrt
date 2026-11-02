#!/bin/sh
# Root wrapper: max performance mode, LLM timing, always restore normal mode. Writes llm_max.flag when finished.
export HOME=/home/<user>
H=/home/<user>
rm -f $H/llm/llm_max.flag $H/llm/results/time_llm.log
cleanup() { pkill tegrastats 2>/dev/null; sh $H/perf_mode.sh off > $H/llm/perf_off.log 2>&1; echo DONE > $H/llm/llm_max.flag; chown -R kimchsi90:kimchsi90 $H/llm/results; }
trap cleanup EXIT
sh $H/perf_mode.sh on > $H/llm/perf_on.log 2>&1
sleep 2
sh $H/llm/time_llm.sh > $H/llm/time_llm.out 2>&1
