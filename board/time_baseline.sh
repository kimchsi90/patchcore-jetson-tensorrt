#!/bin/sh
# Time the anomalib-default baseline engines (TensorRT 8.5.2, fp16) for 20 s each with a tegrastats log next to each.
# Root-free: no nvpmodel / jetson_clocks inside. The caller sets the power mode before running it (the user runs it as root in max mode).
# Usage: sh time_baseline.sh          (engines b1_fp16.engine and, if present, b2_fp16.engine in /home/<user>/base514)
# Output: /home/<user>/base514/time/{b1,b2}.log (trtexec), {b1,b2}.tegra (tegrastats, 200 ms), time.flag (DONE)
D=/home/<user>/base514
O=$D/time
mkdir -p $O
rm -f $O/time.flag
trap 'pkill tegrastats 2>/dev/null; echo DONE > $O/time.flag' EXIT
cat /var/lib/nvpmodel/status > $O/mode.txt 2>&1
for b in b1 b2; do
  E=$D/${b}_fp16.engine
  [ -f "$E" ] || { echo "$(date +%T) skip $b: $E missing" >> $O/time.log; continue; }
  tegrastats --interval 200 > $O/$b.tegra 2>&1 &
  TP=$!
  /usr/src/tensorrt/bin/trtexec --loadEngine=$E --duration=20 --warmUp=1500 --avgRuns=100 --useSpinWait > $O/$b.log 2>&1
  kill $TP 2>/dev/null
  wait $TP 2>/dev/null
  echo "$(date +%T) timed $b rc done" >> $O/time.log
done
