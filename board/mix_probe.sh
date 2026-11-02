#!/bin/sh
# Probe why the GPU+2DLA mix (GPU engine without post, CPU post) is slower than GPU alone. Root-free, normal power mode, relative numbers only.
E=$HOME/e2e; R=$E/results/mix_probe; mkdir -p $R
GM=$HOME/trt_cat/bottle/mixed.engine; G1=$E/engines/p1_gpu_post.engine; DE=$HOME/trt_dla/bottle/int8mix_dla0.engine
cd $E
run() { name=$1; pt=$2; shift 2
  python3 bench_e2e.py --mode bench --pipe P4 --interp area --pipelined --resize-threads 2 --queue-depth 4 --post-threads $pt \
    --test-dir $E/e2e_frames/bottle --labels $E/e2e_frames/bottle/test_labels.csv --duration 12 --warmup 4 --name $name --workers "$*" --out $R/$name.json > $R/$name.log 2>&1
  python3 - $R/$name.json $name <<PY
import json,sys
d=json.load(open(sys.argv[1])); print(sys.argv[2], "fps", d["fps"], "workers", d["per_worker_fps"], "accel_ms", d["stage_ms_mean_per_frame"]["accel_h2d_exec_d2h"])
PY
}
run a_mix_mixedeng_pt2 2 "gpu=$GM@P4,dla0=$DE,dla1=$DE"
run b_mix_posteng_pt2  2 "gpu=$G1@P1,dla0=$DE@P4,dla1=$DE@P4"
run c_mix_mixedeng_pt4 4 "gpu=$GM@P4,dla0=$DE,dla1=$DE"
run d_gpu_mixedeng_pt2 2 "gpu=$GM@P4"
run e_gpu_mixed_dla0   2 "gpu=$GM@P4,dla0=$DE"
