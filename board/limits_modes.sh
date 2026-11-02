#!/bin/sh
# Power-mode sweep v2 (called by limits_all.sh as root): per mode a discarded warm-up run, then 3 repetitions per config, tegrastats around every run.
E=$HOME/e2e; R=$E/results/modes2; mkdir -p $R
G1=$E/engines/p1_gpu_post.engine; DB=$E/engines/int8mix_dla0.engine
COMMON="--mode bench --interp area --pipelined --resize-threads 2 --queue-depth 4 --test-dir $E/test900 --labels $E/test_labels.csv"
cd $E
bench() { name=$1; dur=$2; wu=$3; pipe=$4; pt=$5; w=$6
  python3 bench_e2e.py $COMMON --pipe $pipe --post-threads $pt --duration $dur --warmup $wu --name $name --workers "$w" --out $R/$name.json > $R/$name.log 2>&1; }
for id in 8 7 6 2 1 0 5 4 3; do
  nvpmodel -m $id > $R/nvpmodel_$id.log 2>&1
  sleep 8
  echo "== mode $id $(nvpmodel -q 2>/dev/null | head -1) online=$(cat /sys/devices/system/cpu/online) gpu_max=$(cat /sys/devices/17000000.gv11b/devfreq/17000000.gv11b/max_freq)" >> $R/modes.log
  bench warm_$id 15 3 P1 0 "gpu=$G1"
  for cfg in p1gpu dla2 mix; do
    case $cfg in
      p1gpu) pipe=P1; pt=0; w="gpu=$G1" ;;
      dla2)  pipe=P4; pt=2; w="dla0=$DB,dla1=$DB" ;;
      mix)   pipe=P4; pt=2; w="gpu=$G1@P1,dla0=$DB,dla1=$DB" ;;
    esac
    for rep in 1 2 3; do
      n=0; while [ "$(cat /sys/class/thermal/thermal_zone1/temp)" -ge 70000 ] && [ $n -lt 60 ]; do sleep 2; n=$((n+1)); done
      tegrastats --interval 500 --logfile $R/m${id}_${cfg}_r$rep.tegra > /dev/null 2>&1 &
      TP=$!
      bench m${id}_${cfg}_r$rep 12 4 $pipe $pt "$w"
      kill $TP 2>/dev/null
    done
  done
  echo "mode $id finished $(date +%T)" >> $R/modes.log
done
nvpmodel -m 5 > $R/restore.log 2>&1
