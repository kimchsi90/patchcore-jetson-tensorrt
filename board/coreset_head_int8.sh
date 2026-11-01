#!/bin/sh
# Does an int8 kNN head cut the cost of a larger coreset? Builds the screw engine for coreset 5% and 10% with the stem + layer1.0 pinned to fp16 but
# WITHOUT the "head" token (so the projection / kNN Conv / reduce are quantized to int8 by the entropy calibrator), dumps the test images for scoring
# on the PC, and times the engines as root with max clocks. Compare with coreset_speed.sh (head in fp16).
# Usage:  sh coreset_head_int8.sh build   (normal user)      sh coreset_head_int8.sh time   (root; restores the normal mode)
H=/home/<user>
D=$H/trt_coreset
NAMEPIN_NOHEAD="name:pre_,name:feature_extractor/conv1/,name:feature_extractor/act1,name:feature_extractor/maxpool,name:layer1/layer1.0/"
cd $H/trt_run || exit 1

case "$1" in
build)
  for r in 0.05 0.1; do
    t0=$(date +%s)
    python3 -u trt_runner.py build --onnx $D/screw_c$r.onnx --engine $D/screw_c${r}_h8.engine --precision int8 \
        --calib-dir $H/trt_cat/screw/calib --calib-cache $D/calib_c${r}_h8.cache --fp16-layers "$NAMEPIN_NOHEAD" > $D/build_c${r}_h8.log 2>&1
    echo "$(date +%T) built c$r h8 rc=$? in $(( $(date +%s) - t0 )) s" >> $D/coreset_h8.log
    python3 -u dump_trt.py --backend trt --model $D/screw_c${r}_h8.engine --images $H/trt_cat/screw/test --labels $H/trt_cat/screw/test_labels.csv \
        --out $D/out_c${r}_h8 > $D/dump_c${r}_h8.log 2>&1
    echo "$(date +%T) dumped c$r h8 rc=$?" >> $D/coreset_h8.log
  done
  echo DONE > $D/build_h8.flag
  ;;
time)
  rm -f $D/time_h8.flag
  trap 'pkill tegrastats 2>/dev/null; sh $H/perf_mode.sh off > $D/perf_off_h8.log 2>&1; echo DONE > $D/time_h8.flag' EXIT
  sh $H/perf_mode.sh on > $D/perf_on_h8.log 2>&1
  for r in 0.05 0.1; do
    tegrastats --interval 200 > $D/time_c${r}_h8.tegra 2>&1 &
    TP=$!
    /usr/src/tensorrt/bin/trtexec --loadEngine=$D/screw_c${r}_h8.engine --duration=20 --warmUp=1500 --avgRuns=100 --useSpinWait > $D/time_c${r}_h8.log 2>&1
    kill $TP 2>/dev/null
    echo "$(date +%T) timed c$r h8" >> $D/coreset_h8.log
  done
  ;;
*) echo "usage: $0 build|time"; exit 2 ;;
esac
