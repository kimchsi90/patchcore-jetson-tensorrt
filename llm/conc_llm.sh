#!/bin/sh
# Root wrapper: interference between llama.cpp decoding (GPU) and the PatchCore pipeline, max performance mode.
# Case dla: PIPE_P4_dla0_dla1_pt2 (2 DLA + CPU) ; case gpu: PIPE_P1_gpu. llama-bench starts first, the pipeline starts 15 s later.
export HOME=/home/<user>
H=/home/<user>
L=$H/llm
R=$L/results_conc
export LD_LIBRARY_PATH=$L/llama.cpp/build/bin:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
BIN=$L/llama.cpp/build/bin/llama-bench
MODEL=$L/models/qwen2.5-1.5b-instruct-q4_k_m.gguf
mkdir -p $R; rm -f $L/conc.flag $R/*
cleanup() { pkill tegrastats 2>/dev/null; pkill llama-bench 2>/dev/null; sh $H/perf_mode.sh off > $R/perf_off.log 2>&1; echo DONE > $L/conc.flag; chown -R kimchsi90:kimchsi90 $R $H/e2e/results; }
trap cleanup EXIT
sh $H/perf_mode.sh on > $R/perf_on.log 2>&1
sleep 2
for c in dla:PIPE_P4_dla0_dla1_pt2 gpu:PIPE_P1_gpu; do
  name=${c%%:*}; cfg=${c#*:}
  tegrastats --interval 500 --logfile $R/$name.tegra > /dev/null 2>&1 &
  TP=$!
  date +%s > $R/$name.t_bench_start
  $BIN -m $MODEL -ngl 99 -p 0 -n 128 -r 30 -o jsonl > $R/$name.llm.jsonl 2> $R/$name.llm.err &
  LP=$!
  sleep 15
  date +%s > $R/$name.t_pipe_start
  (cd $H/e2e && DURATION=30 WARMUP=5 REPS=1 sh run_suite_pipelined.sh conc_$name $cfg > $R/$name.suite.out 2>&1)
  date +%s > $R/$name.t_pipe_end
  wait $LP
  date +%s > $R/$name.t_bench_end
  kill $TP 2>/dev/null
  sleep 20
done
