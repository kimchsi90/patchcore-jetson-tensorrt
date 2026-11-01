#!/bin/sh
# Final end-to-end run under max performance mode (run as root, detached).
# One tegrastats log covers the whole suite; per-config power is computed later from the start times in suite.log.
# sudo sets HOME=/root, but bench_e2e.py resolves its default paths under ~/e2e, so HOME is set explicitly
export HOME=/home/<user>
H=/home/<user>
E=$H/e2e
TAG=max_pw2
mkdir -p $E/results/$TAG
rm -f $E/results/$TAG/e2e_done.flag
cleanup() { pkill tegrastats 2>/dev/null; sh $H/perf_mode.sh off > $E/results/$TAG/perf_off.log 2>&1; echo "end $(date +%T)" >> $E/results/$TAG/e2e_final.log; echo DONE > $E/results/$TAG/e2e_done.flag; }
trap cleanup EXIT
echo "start $(date +%T)" > $E/results/$TAG/e2e_final.log
sh $H/perf_mode.sh on > $E/results/$TAG/perf_on.log 2>&1
tegrastats --interval 500 --logfile $E/results/$TAG/tegra.log > /dev/null 2>&1 &
sleep 2
cd $E && REPS=3 sh $E/run_suite.sh $TAG > $E/results/$TAG/run_suite.out 2>&1
echo "suite finished $(date +%T)" >> $E/results/$TAG/e2e_final.log
