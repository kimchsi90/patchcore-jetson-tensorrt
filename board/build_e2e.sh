#!/bin/sh
# Builds the e2e engines under ~/e2e/engines (one at a time). No root needed. Usage: sh build_e2e.sh [p1 p3 p2 ...]
# All engines: calibrated int8 (entropy2) with stem/layer1.0/head(+post) pinned fp16 = the same name spec as mixed_name.engine.
E=$HOME/e2e
T=$HOME/trt_run/trt_runner.py
mkdir -p $E/engines $E/cache $E/logs
NAMEPIN="name:pre_,name:feature_extractor/conv1/,name:feature_extractor/act1,name:feature_extractor/maxpool,name:layer1/layer1.0/,head"
POST32="name:post_,name:score_max,name:score_top10,name:map_n"
cd $E || exit 1
want=${*:-"p1 p3 p2"}
for w in $want; do
  t0=$(date +%s)
  case $w in
    p1) python3 -u build_e2e.py build --onnx onnx/bottle_post.onnx --engine engines/p1_gpu_post.engine --precision int8 \
          --calib-dir $HOME/trt_run/calib --calib-cache cache/calib_post256.cache --calib-algo entropy2 \
          --fp16-layers "$NAMEPIN" --fp32-layers "$POST32" --inspect logs/p1_gpu_post.json > logs/build_p1.log 2>&1 ;;
    p3) python3 -u build_e2e.py build --onnx onnx/bottle_post.onnx --engine engines/p3_dla0_post.engine --precision int8 \
          --calib-dir $HOME/trt_run/calib --calib-cache cache/calib_post256.cache --calib-algo entropy2 \
          --fp16-layers "$NAMEPIN" --fp32-layers "$POST32" --dla-core 0 --inspect logs/p3_dla0_post.json > logs/build_p3.log 2>&1 ;;
    p2) python3 -u build_e2e.py build --onnx onnx/bottle_post_fullres900_matmul.onnx --engine engines/p2_gpu_full900.engine --precision int8 \
          --calib-dir calib900 --calib-cache cache/calib_full900.cache --calib-algo entropy2 \
          --fp16-layers "$NAMEPIN" --fp32-layers "name:fr_,$POST32" --inspect logs/p2_gpu_full900.json > logs/build_p2.log 2>&1 ;;
    *) echo "bad target $w" ;;
  esac
  echo "$w rc=$? $(( $(date +%s) - t0 )) s temp=$(cat /sys/class/thermal/thermal_zone1/temp)" >> logs/build_status.txt
done
echo ALLDONE >> logs/build_status.txt
