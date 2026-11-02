#!/bin/sh
# Per-category end-to-end benchmark (root, detached, max mode): frames at the original MVTec resolution, CPU resize to 256 (INTER_AREA),
# GPU mixed engine (no post) and/or the DLA int8mix engine, CPU post. Configs per category: dla2 (2 DLA), gpu (GPU only), mix (GPU + 2 DLA).
# 2 resize threads, queue depth 4, 2 post threads. One repetition per config, tegrastats around each. CATS env overrides the category list.
export HOME=/home/<user>
H=/home/<user>
E=$H/e2e
R=$E/results/cat_e2e
CATS=${CATS:-"bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"}
mkdir -p $R; rm -f $R/cat_done.flag
cleanup() { pkill tegrastats 2>/dev/null; sh $H/perf_mode.sh off > $R/perf_off.log 2>&1; echo DONE > $R/cat_done.flag; chown -R kimchsi90:kimchsi90 $R; }
trap cleanup EXIT
sh $H/perf_mode.sh on > $R/perf_on.log 2>&1
sleep 2
temp() { m=0; for z in /sys/class/thermal/thermal_zone*; do case "$(cat $z/type 2>/dev/null)" in CPU-therm|GPU-therm) t=$(cat $z/temp 2>/dev/null); [ -n "$t" ] && [ "$t" -gt "$m" ] && m=$t ;; esac; done; echo $m; }
cd $E
for c in $CATS; do
  GE=$H/trt_cat/$c/mixed.engine
  DE=$H/trt_dla/$c/int8mix_dla0.engine
  for f in $GE $DE $E/e2e_frames/$c/test_labels.csv; do [ -s $f ] || { echo "missing $f" >> $R/cat.log; continue 2; }; done
  for cfg in dla2 gpu mix; do
    case $cfg in
      dla2) W="dla0=$DE,dla1=$DE" ;;
      gpu)  W="gpu=$GE@P4" ;;
      mix)  W="gpu=$GE@P4,dla0=$DE,dla1=$DE" ;;
    esac
    n=0; while [ "$(temp)" -ge 70000 ] && [ $n -lt 60 ]; do sleep 2; n=$((n+1)); done
    tegrastats --interval 500 --logfile $R/${c}_$cfg.tegra > /dev/null 2>&1 &
    TP=$!
    python3 bench_e2e.py --mode bench --pipe P4 --interp area --pipelined --resize-threads 2 --queue-depth 4 --post-threads 2 \
      --test-dir $E/e2e_frames/$c --labels $E/e2e_frames/$c/test_labels.csv --duration 20 --warmup 5 \
      --name ${c}_$cfg --workers "$W" --out $R/${c}_$cfg.json > $R/${c}_$cfg.log 2>&1
    rc=$?
    kill $TP 2>/dev/null
    echo "$c $cfg rc=$rc temp=$(temp) $(date +%T)" >> $R/cat.log
  done
done
