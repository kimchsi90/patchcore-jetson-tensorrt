#!/bin/sh
# Queue-depth sweep (1 and 2) of the pipelined end-to-end suite under max performance mode (run as root, detached).
# Depth 4 results already exist (max_pipelined). Same structure as e2e_final.sh; HOME is set explicitly because sudo changes it.
export HOME=/home/<user>
H=/home/<user>
E=$H/e2e
TAG=max_qd
mkdir -p $E/results/$TAG
rm -f $E/results/$TAG/e2e_done.flag
cleanup() { pkill tegrastats 2>/dev/null; sh $H/perf_mode.sh off > $E/results/$TAG/perf_off.log 2>&1; echo "end $(date +%T)" >> $E/results/$TAG/e2e_final.log; echo DONE > $E/results/$TAG/e2e_done.flag; }
trap cleanup EXIT
echo "start $(date +%T)" > $E/results/$TAG/e2e_final.log
sh $H/perf_mode.sh on > $E/results/$TAG/perf_on.log 2>&1
tegrastats --interval 500 --logfile $E/results/$TAG/tegra.log > /dev/null 2>&1 &
sleep 2
cd $E
for qd in 1 2; do
  REPS=3 QD=$qd sh $E/run_suite_pipelined.sh ${TAG}_qd$qd PIPE_P1_gpu PIPE_P4_dla0_dla1_pt2 PIPE_mix_pt2 > $E/results/$TAG/run_suite_qd$qd.out 2>&1
  echo "qd $qd finished $(date +%T)" >> $E/results/$TAG/e2e_final.log
done
