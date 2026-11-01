#!/bin/sh
# Per-category TensorRT accuracy run on the board (run as the normal user).
# Usage: cat_eval.sh <category> [fp16]
#   builds the mixed engine (stem + layer1.0 fp16, the rest int8, entropy calibrator) from ~/trt_cat/<cat>/pre.onnx, dumps all test images.
#   With the second argument "fp16" it also builds and dumps an fp16 engine (use it for the hard categories).
# Layer spec by name, because layer indices can shift between TensorRT versions.
C=$1
D=/home/<user>/trt_cat/$C
cd /home/<user>/trt_run || exit 1
NAMEPIN="name:pre_,name:feature_extractor/conv1/,name:feature_extractor/act1,name:feature_extractor/maxpool,name:layer1/layer1.0/,head"
rm -f $D/done.flag
echo "start $(date +%T)" > $D/eval.log

t0=$(date +%s)
python3 -u trt_runner.py build --onnx $D/pre.onnx --engine $D/mixed.engine --precision int8 \
    --calib-dir $D/calib --calib-cache $D/calib.cache --fp16-layers "$NAMEPIN" > $D/build_mixed.log 2>&1
echo "$(date +%T) mixed built rc=$? in $(( $(date +%s) - t0 )) s" >> $D/eval.log
python3 -u dump_trt.py --backend trt --model $D/mixed.engine --images $D/test --labels $D/test_labels.csv --out $D/out_mixed > $D/dump_mixed.log 2>&1
echo "$(date +%T) mixed dumped rc=$?" >> $D/eval.log

if [ "$2" = "fp16" ]; then
  t0=$(date +%s)
  python3 -u trt_runner.py build --onnx $D/pre.onnx --engine $D/fp16.engine --precision fp16 > $D/build_fp16.log 2>&1
  echo "$(date +%T) fp16 built rc=$? in $(( $(date +%s) - t0 )) s" >> $D/eval.log
  python3 -u dump_trt.py --backend trt --model $D/fp16.engine --images $D/test --labels $D/test_labels.csv --out $D/out_fp16 > $D/dump_fp16.log 2>&1
  echo "$(date +%T) fp16 dumped rc=$?" >> $D/eval.log
fi
echo "end $(date +%T)" >> $D/eval.log
echo DONE > $D/done.flag
