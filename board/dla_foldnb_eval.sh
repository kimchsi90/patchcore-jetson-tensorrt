#!/bin/sh
# DLA int8mix build and accuracy dump from the folded NCHW graph with a mean-filled border (fold_norm.py --border --nchw): dla_foldnb_eval.sh <category> [core]
# Needs ~/trt_cat/<cat>/foldnb.onnx. BORDER_INPUT=3 and NCHW_INPUT=1 make the calibrator and the dump loader build the same input.
C=$1; K=${2:-0}
S=/home/<user>/trt_cat/$C
D=/home/<user>/trt_dla/$C
T=/home/<user>/trt_dla
NAMEPIN="name:feature_extractor/conv1/,name:feature_extractor/act1,name:feature_extractor/maxpool,name:layer1/layer1.0/,head"
export BORDER_INPUT=3 NCHW_INPUT=1
mkdir -p $D
E=$D/int8mix_foldnb_dla$K.engine
rm -f $D/foldnb.done
t0=$(date +%s)
cd $T || exit 1
python3 -u trt_runner.py build --onnx $S/foldnb.onnx --engine $E --precision int8 --calib-dir $S/calib --calib-cache $D/calib_ent_foldnb.cache \
  --calib-algo entropy2 --fp16-layers $NAMEPIN --dla-core $K --inspect $D/int8mix_foldnb_dla$K.json > $D/build_foldnb.log 2>&1
echo "build rc=$? $(( $(date +%s) - t0 )) s" > $D/foldnb.status
if [ -s $E ]; then
  python3 -u dump_trt.py --backend trt --model $E --dla-core $K --images $S/test --labels $S/test_labels.csv --out $D/out_foldnb > $D/dump_foldnb.log 2>&1
  echo "dump rc=$?" >> $D/foldnb.status
fi
echo DONE > $D/foldnb.done
