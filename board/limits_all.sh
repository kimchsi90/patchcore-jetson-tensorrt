#!/bin/sh
# Root wrapper for the limitation follow-ups. Order: LLM (default 10W, then max mode), power-mode sweep, per-category e2e (max mode). Always restores MODE_10W_DESKTOP.
export HOME=/home/<user>
H=/home/<user>; L=$H/llm; E=$H/e2e
O=$E/results/limits; mkdir -p $O; rm -f $O/limits_done.flag
cleanup() { pkill tegrastats 2>/dev/null; sh $H/perf_mode.sh off > $O/perf_off.log 2>&1; nvpmodel -m 5 > $O/restore.log 2>&1; echo DONE > $O/limits_done.flag; chown -R kimchsi90:kimchsi90 $O $E/results $L/results_limits; }
trap cleanup EXIT
echo "start $(date +%T)" > $O/limits.log
sh $L/limits_llm.sh 10w 4 ;               echo "llm 10w done $(date +%T)" >> $O/limits.log
sh $H/perf_mode.sh on > $O/perf_on1.log 2>&1; sleep 2
sh $L/limits_llm.sh max 6 ;               echo "llm max done $(date +%T)" >> $O/limits.log
sh $H/perf_mode.sh off > $O/perf_off1.log 2>&1; sleep 3
sh $E/limits_modes.sh ;                   echo "modes done $(date +%T)" >> $O/limits.log
sh $H/perf_mode.sh on > $O/perf_on2.log 2>&1; sleep 2
sh $E/limits_cat.sh ;                     echo "cat done $(date +%T)" >> $O/limits.log
