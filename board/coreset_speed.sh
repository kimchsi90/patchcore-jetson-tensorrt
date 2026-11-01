#!/bin/sh
# Speed cost of larger coresets on the board: build the mixed engine for each screw ONNX (c0.01 / c0.05 / c0.1), dump the test images
# (accuracy is scored later on the PC) and, as root with max clocks, time each engine with trtexec.
# Usage:  sh coreset_speed.sh build        (normal user, builds + dumps)
#         sh coreset_speed.sh time         (root, max mode, trtexec timing; restores the normal mode)
H=/home/<user>
D=$H/trt_coreset
NAMEPIN="name:pre_,name:feature_extractor/conv1/,name:feature_extractor/act1,name:feature_extractor/maxpool,name:layer1/layer1.0/,head"
cd $H/trt_run || exit 1

case "$1" in
build)
  for r in 0.01 0.05 0.1; do
    t0=$(date +%s)
    python3 -u trt_runner.py build --onnx $D/screw_c$r.onnx --engine $D/screw_c$r.engine --precision int8 \
        --calib-dir $H/trt_cat/screw/calib --calib-cache $D/calib_c$r.cache --fp16-layers "$NAMEPIN" > $D/build_c$r.log 2>&1
    echo "$(date +%T) built c$r rc=$? in $(( $(date +%s) - t0 )) s" >> $D/coreset.log
    python3 -u dump_trt.py --backend trt --model $D/screw_c$r.engine --images $H/trt_cat/screw/test --labels $H/trt_cat/screw/test_labels.csv \
        --out $D/out_c$r > $D/dump_c$r.log 2>&1
    echo "$(date +%T) dumped c$r rc=$?" >> $D/coreset.log
  done
  echo DONE > $D/build.flag
  ;;
time)
  rm -f $D/time.flag
  trap 'pkill tegrastats 2>/dev/null; sh $H/perf_mode.sh off > $D/perf_off.log 2>&1; echo DONE > $D/time.flag' EXIT
  sh $H/perf_mode.sh on > $D/perf_on.log 2>&1
  for r in 0.01 0.05 0.1; do
    tegrastats --interval 200 > $D/time_c$r.tegra 2>&1 &
    TP=$!
    /usr/src/tensorrt/bin/trtexec --loadEngine=$D/screw_c$r.engine --duration=20 --warmUp=1500 --avgRuns=100 --useSpinWait > $D/time_c$r.log 2>&1
    kill $TP 2>/dev/null
    echo "$(date +%T) timed c$r" >> $D/coreset.log
  done
  ;;
*) echo "usage: $0 build|time"; exit 2 ;;
esac
