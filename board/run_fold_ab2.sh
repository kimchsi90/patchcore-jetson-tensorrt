#!/bin/sh
# A/B in max mode (root, detached): DLA backbone engine with the GPU Sub/Div fallback (int8mix_dla0) vs the folded NCHW engine with a mean-filled border
# (int8mix_foldnb_dla0, --border 3 --nchw). 2 DLA cores, 2 resize threads, queue depth 4, 2 post threads. 5 alternating repetitions, tegrastats per run.
export HOME=/home/<user>
H=/home/<user>
E=$H/e2e
R=$E/results/fold_ab2
mkdir -p $R; rm -f $R/ab_done.flag
cleanup() { pkill tegrastats 2>/dev/null; sh $H/perf_mode.sh off > $R/perf_off.log 2>&1; echo DONE > $R/ab_done.flag; chown -R kimchsi90:kimchsi90 $R; }
trap cleanup EXIT
sh $H/perf_mode.sh on > $R/perf_on.log 2>&1
sleep 2
cd $E
COMMON="--mode bench --pipe P4 --interp area --pipelined --resize-threads 2 --queue-depth 4 --post-threads 2 --test-dir $E/test900 --labels $E/test_labels.csv --duration 20 --warmup 5"
for rep in 1 2 3 4 5; do
  for v in orig fold; do
    if [ $v = orig ]; then ENG=$E/engines/int8mix_dla0.engine; EXTRA=""; else ENG=$E/engines/int8mix_foldnb_dla0.engine; EXTRA="--border 3 --nchw"; fi
    n=$((0)); while [ "$(cat /sys/class/thermal/thermal_zone1/temp)" -ge 70000 ] && [ $n -lt 60 ]; do sleep 2; n=$((n+1)); done
    tegrastats --interval 500 --logfile $R/${v}_r$rep.tegra > /dev/null 2>&1 &
    TP=$!
    python3 bench_e2e.py $COMMON $EXTRA --name ab_${v}_r$rep --workers "dla0=$ENG,dla1=$ENG" --out $R/ab_${v}_r$rep.json > $R/ab_${v}_r$rep.log 2>&1
    kill $TP 2>/dev/null
    echo "$v r$rep rc=$? $(date +%T)" >> $R/ab.log
  done
done
