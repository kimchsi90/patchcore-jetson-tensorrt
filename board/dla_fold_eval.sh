#!/bin/sh
# DLA int8mix build and accuracy dump from the folded graph (input = RGB - mean255, no Sub/Div on the GPU): dla_fold_eval.sh <category> [core]
# Needs ~/trt_cat/<cat>/fold.onnx (trt/fold_norm.py --border). CENTER_INPUT=1 makes the calibrator and the dump loader subtract mean255.
C=$1; K=${2:-0}
S=/home/<user>/trt_cat/$C
D=/home/<user>/trt_dla/$C
T=/home/<user>/trt_dla
NAMEPIN="name:pre_,name:feature_extractor/conv1/,name:feature_extractor/act1,name:feature_extractor/maxpool,name:layer1/layer1.0/,head"
export BORDER_INPUT=3
mkdir -p $D
E=$D/int8mix_foldb_dla$K.engine
rm -f $D/foldb.done
t0=$(date +%s)
cd $T || exit 1
python3 -u trt_runner.py build --onnx $S/foldb.onnx --engine $E --precision int8 --calib-dir $S/calib --calib-cache $D/calib_ent_foldb.cache \
  --calib-algo entropy2 --fp16-layers $NAMEPIN --dla-core $K --inspect $D/int8mix_foldb_dla$K.json > $D/build_foldb.log 2>&1
echo "build rc=$? $(( $(date +%s) - t0 )) s" > $D/foldb.status
if [ -s $E ]; then
  python3 -u dump_trt.py --backend trt --model $E --dla-core $K --images $S/test --labels $S/test_labels.csv --out $D/out_foldb > $D/dump_foldb.log 2>&1
  echo "dump rc=$?" >> $D/foldb.status
fi
echo DONE > $D/foldb.done
