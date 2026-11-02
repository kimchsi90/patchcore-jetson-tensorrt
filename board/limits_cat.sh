#!/bin/sh
# Per-category e2e v2 (called by limits_all.sh as root, max mode on): 4 configs x 3 repetitions, 12 s runs, tegrastats around every run.
E=$HOME/e2e; H=$HOME; R=$E/results/cat_e2e2; mkdir -p $R
CATS="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
cd $E
for c in $CATS; do
  GE=$H/trt_cat/$c/mixed.engine; DE=$H/trt_dla/$c/int8mix_dla0.engine
  for f in $GE $DE $E/e2e_frames/$c/test_labels.csv; do [ -s $f ] || { echo "missing $f" >> $R/cat.log; continue 2; }; done
  for cfg in dla2 gpu gpudla0 mix; do
    case $cfg in
      dla2)    W="dla0=$DE,dla1=$DE" ;;
      gpu)     W="gpu=$GE@P4" ;;
      gpudla0) W="gpu=$GE@P4,dla0=$DE" ;;
      mix)     W="gpu=$GE@P4,dla0=$DE,dla1=$DE" ;;
    esac
    for rep in 1 2 3; do
      n=0; while [ "$(cat /sys/class/thermal/thermal_zone1/temp)" -ge 70000 ] && [ $n -lt 60 ]; do sleep 2; n=$((n+1)); done
      tegrastats --interval 500 --logfile $R/${c}_${cfg}_r$rep.tegra > /dev/null 2>&1 &
      TP=$!
      python3 bench_e2e.py --mode bench --pipe P4 --interp area --pipelined --resize-threads 2 --queue-depth 4 --post-threads 2 \
        --test-dir $E/e2e_frames/$c --labels $E/e2e_frames/$c/test_labels.csv --duration 12 --warmup 4 \
        --name ${c}_${cfg}_r$rep --workers "$W" --out $R/${c}_${cfg}_r$rep.json > $R/${c}_${cfg}_r$rep.log 2>&1
      kill $TP 2>/dev/null
    done
    echo "$c $cfg $(date +%T)" >> $R/cat.log
  done
done
