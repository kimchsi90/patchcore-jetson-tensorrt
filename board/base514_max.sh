#!/bin/sh
# Max-mode timing of the anomalib baseline engines (run as root, detached). Always restores the normal mode.
export HOME=/home/<user>
H=/home/<user>
D=$H/base514/time
mkdir -p $D
rm -f $D/max_done.flag
cleanup() { pkill tegrastats 2>/dev/null; sh $H/perf_mode.sh off > $D/perf_off.log 2>&1; echo DONE > $D/max_done.flag; }
trap cleanup EXIT
sh $H/perf_mode.sh on > $D/perf_on.log 2>&1
sh $H/base514/time_baseline.sh > $D/time_baseline.out 2>&1
