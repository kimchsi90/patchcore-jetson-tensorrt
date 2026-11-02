#!/bin/sh
# Power-mode and CPU-core sweep of the pipelined end-to-end configs (root, detached). No jetson_clocks: only the nvpmodel mode changes.
# Restores MODE_10W_DESKTOP (id 5) on exit. HOME is set explicitly because sudo changes it.
export HOME=/home/<user>
H=/home/<user>
E=$H/e2e
TAG=modes
R=$E/results/$TAG
mkdir -p $R; rm -f $R/modes_done.flag
cleanup() { pkill tegrastats 2>/dev/null; nvpmodel -m 5 > $R/restore.log 2>&1; echo "end $(date +%T)" >> $R/modes.log; echo DONE > $R/modes_done.flag; chown -R kimchsi90:kimchsi90 $R; }
trap cleanup EXIT
echo "start $(date +%T)" > $R/modes.log
for id in 8 7 6 2 1 0 5 4 3; do
  nvpmodel -m $id > $R/nvpmodel_$id.log 2>&1
  sleep 8
  { echo "== mode $id $(nvpmodel -q 2>/dev/null | head -1) online=$(cat /sys/devices/system/cpu/online) gpu_max=$(cat /sys/devices/17000000.gv11b/devfreq/17000000.gv11b/max_freq) cpu0_max=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)"; } >> $R/modes.log
  tegrastats --interval 500 --logfile $R/tegra_$id.log > /dev/null 2>&1 &
  TP=$!
  cd $E && REPS=2 sh $E/run_suite_pipelined.sh modes_$id PIPE_P1_gpu PIPE_P4_dla0_dla1_pt2 PIPE_mix_pt2 > $R/suite_$id.out 2>&1
  kill $TP 2>/dev/null
  echo "mode $id finished $(date +%T)" >> $R/modes.log
done
