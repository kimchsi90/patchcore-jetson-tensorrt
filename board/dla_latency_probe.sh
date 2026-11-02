#!/bin/sh
# Single-core DLA latency of the original and folded backbone engines with trtexec (normal power mode, relative numbers). Builds the NCHW variant.
T=$HOME/trt_dla; D=$T/bottle; S=$HOME/trt_cat/bottle
NAMEPIN="name:feature_extractor/conv1/,name:feature_extractor/act1,name:feature_extractor/maxpool,name:layer1/layer1.0/,head"
cp $D/calib_ent_fold.cache $D/calib_ent_foldn.cache
cd $T && python3 -u trt_runner.py build --onnx $S/foldn.onnx --engine $D/int8mix_foldn_dla0.engine --precision int8 --calib-cache $D/calib_ent_foldn.cache \
  --calib-algo entropy2 --fp16-layers $NAMEPIN --dla-core 0 --inspect $D/int8mix_foldn_dla0.json > $D/build_foldn.log 2>&1
echo "build foldn rc=$?"
for round in 1; do
for v in int8mix_dla0 int8mix_fold_dla0 int8mix_foldb_dla0 int8mix_foldn_dla0; do
  for nd in "" "--noDataTransfers"; do
    r=$(/usr/src/tensorrt/bin/trtexec --loadEngine=$D/$v.engine --useDLACore=0 --iterations=300 --warmUp=500 --avgRuns=100 --useSpinWait $nd 2>&1 | grep -m1 "GPU Compute Time" | sed 's/.*mean = \([0-9.]*\) ms.*/\1/')
    echo "round $round $v ${nd:-with_transfers} gpu_compute_mean_ms=$r"
  done
done
done
