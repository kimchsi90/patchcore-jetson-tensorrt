#!/bin/sh
# DLA accuracy run on the board: dla_eval.sh <category> <variant> [core]
#   variants: fp16 | int8mm (minmax) | int8ent (entropy2) | int8mix (entropy2 + stem/layer1.0/head pinned fp16)
#   builds ~/trt_dla/<cat>/<variant>_dla<core>.engine from ~/trt_cat/<cat>/pre.onnx, dumps every test image on that core.
C=$1; V=$2; K=${3:-0}
S=/home/<user>/trt_cat/$C
D=/home/<user>/trt_dla/$C
T=/home/<user>/trt_dla
NAMEPIN="name:pre_,name:feature_extractor/conv1/,name:feature_extractor/act1,name:feature_extractor/maxpool,name:layer1/layer1.0/,head"
mkdir -p $D
E=$D/${V}_dla$K.engine
rm -f $D/$V.done
t0=$(date +%s)
case $V in
  fp16)   ARGS="--precision fp16" ;;
  int8mm) ARGS="--precision int8 --calib-dir $S/calib --calib-cache $D/calib_mm.cache --calib-algo minmax" ;;
  int8ent) ARGS="--precision int8 --calib-dir $S/calib --calib-cache $D/calib_ent.cache --calib-algo entropy2" ;;
  int8mix) ARGS="--precision int8 --calib-dir $S/calib --calib-cache $D/calib_ent.cache --calib-algo entropy2 --fp16-layers $NAMEPIN" ;;
  *) echo "bad variant $V"; exit 2 ;;
esac
cd $T || exit 1
python3 -u trt_runner.py build --onnx $S/pre.onnx --engine $E $ARGS --dla-core $K --inspect $D/${V}_dla$K.json > $D/build_$V.log 2>&1
echo "build rc=$? $(( $(date +%s) - t0 )) s" > $D/$V.status
if [ -s $E ]; then
  python3 -u dump_trt.py --backend trt --model $E --dla-core $K --images $S/test --labels $S/test_labels.csv --out $D/out_$V > $D/dump_$V.log 2>&1
  echo "dump rc=$?" >> $D/$V.status
fi
echo DONE > $D/$V.done
