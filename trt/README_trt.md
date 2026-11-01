# TensorRT port (Jetson Xavier NX, JetPack 4.x, TRT 8.2.1, Python 3.6) - commands

Files: trt_runner.py (ctypes CUDA + TRT runner/builder/CLI), dump_trt.py (--backend ort|trt),
img_checksum.py, compare_dump.py, score_dump.py. Board dir: ~/trt_early/run/ (test/, calib/, out/).

PC (PYTHONUTF8=1, c:\dev_venvs\PatchcoreJetson\Scripts\python.exe):
  python img_checksum.py pil ..\patchcore\work\bottle_256\test 3     # board: python3 img_checksum.py cv2 test 3 (identical sha1)
  scp trt_runner.py dump_trt.py img_checksum.py test_labels.csv <pre.onnx> test/*.png calib/*.png -> board run/

Board (cd ~/trt_early/run), ONNX = fused_random384_c0.01_rmax_pf_pre.onnx:
  python3 trt_runner.py build --onnx $ONNX --engine pre_fp16.engine --precision fp16
  python3 trt_runner.py build --onnx $ONNX --engine pre_int8.engine --precision int8 --calib-dir calib --calib-cache calib_pre.cache            # entropy2
  python3 trt_runner.py build --onnx $ONNX --engine pre_int8_minmax.engine --precision int8 --calib-dir calib --calib-cache calib_pre_minmax.cache --calib-algo minmax
  python3 dump_trt.py --backend trt --model pre_fp16.engine --images test --labels test_labels.csv --out out/bottle_256_trt_fp16   # same for int8, int8mm
  /usr/src/tensorrt/bin/trtexec --loadEngine=X.engine --iterations=300 --avgRuns=100 --useSpinWait
  python3 trt_runner.py bench --engine X.engine --images test --n 200

PC scoring:
  python compare_dump.py out\bottle_256_ort_fp32 out\bottle_256_trt_<e>
  python score_dump.py --dump out\bottle_256_trt_<e> --labels <bottle_256>\test_labels.csv --masks <bottle_256>\test_masks
Notes: int8 builds also enable the FP16 flag (fallback layers). Power mode stayed MODE_10W_DESKTOP.

## Mixed precision (backbone INT8, head FP16) - added 2026-11-01
New in trt_runner.py build: --fp16-layers SPEC, --fp32-layers SPEC (tokens: head | idx:A-B | name:SUBSTR, comma separated),
--calib-algo legacy (+ --quantile), --inspect J.json (per-layer engine info, needs ProfilingVerbosity.DETAILED).
Pinning sets BuilderFlag.OBEY_PRECISION_CONSTRAINTS and layer.precision + set_output_type. Layer indices of the pre.onnx
network: 0-4 preprocessing, 5-7 conv1/relu/maxpool, 8-29 layer1, 30-58 layer2, 59-101 layer3, 102-111 head
(pool, pool_1, proj2, proj3, resize, add=xproj, dk0 conv, ReduceMax, m). 'head' = 102-111.
Reuse the existing calib caches (calib_pre.cache entropy2, calib_pre_minmax.cache): the cache is keyed by tensor name,
so pinned variants need no new calibration. Build ~260 s each on the board (10 W mode).

Board, per variant NAME (ev.sh = dump + trtexec 300 it + python bench; queue lines "name|algo|cache|extra args" via q.sh):
  python3 trt_runner.py build --onnx $ONNX --engine NAME.engine --precision int8 --calib-dir calib --calib-cache CACHE \
      --calib-algo ALGO --inspect out/insp_NAME.json <extra>
  bash ev.sh NAME            # writes out/bottle_256_trt_mixed_NAME_{m,x}.npy, lat_NAME.log, bench_NAME.log
Variants (extra args):
  mix_head16_ent2 / mix_head16_mm : --fp16-layers head                (cache calib_pre.cache / calib_pre_minmax.cache)
  e_pin0_29 / m_pin0_29           : --fp16-layers idx:0-29,head       (stem + layer1 + head FP16; layer2/3 INT8)
  e_pin0_15                       : --fp16-layers idx:0-15,head       (stem + layer1.0 FP16)
  e_pin16_29, e_pin0_5            : --fp16-layers idx:16-29,head / idx:0-5,head
  s_l1only idx:30-111, s_l12 idx:59-111, s_l3only idx:0-58,idx:102-111  (all with head FP16 inside the range)
  leg_all : --calib-algo legacy --quantile 0.9999 --calib-cache calib_pre_legacy.cache (collapsed; legacy + pinning failed to build in 8.2)
PC (PYTHONUTF8=1): bash score_variant.sh NAME   # scp dump, score_dump.py, compare_dump.py vs bottle_256_ort_fp32
  python sim_head_quant.py --onnx <pre.onnx> --ref out\bottle_256_ort_fp32 --out out\bottle_256_sim   # head-only int8 simulation
Latency: trtexec --loadEngine=NAME.engine --iterations=300 --avgRuns=100 --useSpinWait ; python3 trt_runner.py bench --engine NAME.engine --images test --n 200

Results (bottle/256, 83 test images; GPU = trtexec GPU Compute mean, 10 W mode; py = Python infer H2D+exec+D2H / +cv2 read):
  fp16                     0.9848 0.9429 | 16.09 ms | py 17.48 / 21.37
  int8 minmax all          0.9832 0.9328 |  9.10 ms | py 10.53 / 14.46
  mix head16 entropy2      0.7654 0.2846 |  9.15 ms   (head FP16 does NOT fix it)
  mix head16 minmax        0.9832 0.9331 |  9.16 ms
  stem+L1.0 FP16, ent2     0.9849 0.9431 | 10.16 ms | py 11.56 / 15.54
  stem+L1 FP16, ent2       0.9849 0.9431 | 10.99 ms | py 12.51 / 16.33
  stem+L1 FP16, minmax     0.9849 0.9431 | 10.97 ms | py 12.52 / 16.48
  only layer3 INT8 (ent2)  0.9849 0.9433 | 12.85 ms
Findings (entropy2, head FP16 always): collapse is not the head. Pinning idx 0-5 alone (0.7596) or idx 8-15 alone (0.7616, e_pin8_15,
9.80 ms) or idx 16-29 (0.7291) does not recover; idx 0-15 (stem + layer1.0) does (0.9849). Neither piece alone suffices, idx 6-7
(relu/maxpool) were not pinned alone. Calib cache amax (scale*127): entropy2 clips layer1.0 Add 28.5 vs minmax 60.4, downsample
32.7 vs 61.0, conv1 41.7 vs 65.5 (and many layer3 tensors to ~0.43-0.5). Mechanism [Reasoned], not isolated per tensor.
Extra: board/ has ev.sh q.sh q1-q4.txt (copies of what ran on the board); out/calib_pre*.cache are the 100-image caches (TRT 8.2.1 only).
TensorRT 8.5 differences [from memory, not verified on 8.5]: OBEY_PRECISION_CONSTRAINTS is kept (STRICT_TYPES is deprecated);
workspace via set_memory_pool_limit (runner already switches); get_binding_*/execute_async_v2 are deprecated in favor of the
name-based tensor API/execute_async_v3 (trt_runner still uses the old calls); calibrator caches and engines are TRT-version
specific, so rebuild (and recalibrate or re-validate the cache) on JetPack 5.1.4; layer indices must be re-checked there
(parser may fold/insert layers), prefer name: specs; legacy calibrator + pinning failed to build in 8.2 (AveragePool: no
implementation obeys the precision constraints), one setting only (quantile 0.9999, cutoff equal), not retested.

## anomalib-default baseline on TensorRT 8.2.1 (bottle/256, fp16) - added 2026-11-01
Baseline = anomalib 2.7.0 Lightning forward (pre_processor + PatchcoreModel + post_processor, 21401x1536 bank, k=9 re-weighting), NCHW float 0..1 input.
Board dir ~/trt_early/base/. Files: export_baseline_legacy.py, fold_baseline.py, verify_baseline_onnx.py, score_path_nodes.py (PC);
dump_baseline.py, build_baseline.py, head_debug.py (board); score_baseline.py (PC scorer, same aupro() as score_dump.py, no extra blur).
trt_runner.py got one additive change: bool bindings map to np.bool_ (pred_label / pred_mask outputs).

PC (PYTHONUTF8=1; PE=c:\dev_venvs\PatchcoreExport\Scripts\python.exe, PJ=c:\dev_venvs\PatchcoreJetson\Scripts\python.exe), B=..\patchcore\work\bottle_256\baseline:
  PE export_baseline_legacy.py --batch 1 --opset 17      # TorchScript exporter (dynamo=False), also --batch 2; prints torch-vs-ORT diff
  PE fold_baseline.py --src B\anomalib_legacy_b1_op17.onnx --dst B\anomalib_b1_trt.onnx   # same for b2
  PE verify_baseline_onnx.py B\anomalib_legacy_b1_op17.onnx B\anomalib_b1_trt.onnx B\anomalib_b2_trt.onnx ..\patchcore\work\bottle_256\weights\onnx\anomalib_default.onnx
  PJ dump_baseline.py --backend ort --model B\anomalib_b1_trt.onnx --images ..\patchcore\work\bottle_256\test --labels ..\patchcore\work\bottle_256\test_labels.csv --out out\bottle_256_baseline_ort_fp32
  PJ score_baseline.py --dump out\bottle_256_baseline_trt_fp16_b1 --ref out\bottle_256_baseline_ort_fp32 --labels <bottle_256>\test_labels.csv --masks <bottle_256>\test_masks
Board (cd ~/trt_early/base; scp anomalib_b1_trt.onnx trt_runner.py dump_baseline.py build_baseline.py test_labels.csv test/*.png):
  python3 build_baseline.py --onnx anomalib_b1_trt.onnx --engine b1t_fp16.engine --precision fp16          # 487.9 s, engine 383.6 MB
  /usr/src/tensorrt/bin/trtexec --loadEngine=b1t_fp16.engine --iterations=100 --avgRuns=20 --useSpinWait
  python3 dump_baseline.py --backend trt --model b1t_fp16.engine --images test --labels test_labels.csv --out out/bottle_256_baseline_trt_fp16_b1
  (batch 2: same with anomalib_b2_trt.onnx / b2t_fp16.engine / dump_baseline.py --batch 2 --out out/bottle_256_baseline_trt_fp16_b2)
  /usr/src/tensorrt/bin/trtexec --onnx=anomalib_legacy_b1_op17.onnx --fp16 --workspace=1024   # parse check of the raw legacy export: fails, see below

ONNX changes (all checked bit-exact against the unmodified legacy export with ORT fp32, max abs diff 0.0 on 8 images and 0.0 on all 83 for b1):
 1. Exporter: anomalib 2.7.0 only offers the dynamo exporter (opset 18 here); torch.onnx.export(dynamo=False, opset 17) on lit(x) instead.
    Legacy vs torch: pred_score 1.5e-6, anomaly_map 8.4e-6. Legacy vs anomalib_default.onnx: 7.7e-7 / 3.0e-6.
 2. Constant folding with plain onnx + ORT per node (small constants only, Shape of static tensors, Where with constant scalar bool
    condition): needed because TRT 8.2.1 rejects the zero-element Identity/ConstantOfShape input of the post_processor ("constant weights has count 0").
    This also removes IsNaN/And/Not from the post_processor (thresholds and min/max are constants).
 3. Flatten(axis == rank) -> Reshape([B*N,1]): TRT 8.2.1 mis-shapes it ("reshape changes volume [1024] to [1]") in the pred_score indexing.
 4. TopK(largest=0, k=9) -> Neg / TopK(largest=1) / Neg on the values: measured on the board, TRT 8.2.1 returns the 9 LARGEST values
    (indices and values equal to the top-9 of Sqrt_1) and ignores largest=0. Without this fix pred_score is garbage (image AUROC 0.6266 in fp16;
    0.631 with the score-path layers pinned to FP32; a head-only fp32 engine showed the same TopK mismatch while all tensors before TopK matched ORT).
 No change to ArgMin/ArgMax/GatherND(Gather)/Resize/Pad/Softmax/Pow; they parse and run. Outputs pred_label / pred_mask (bool) are kept.
Untested: TRT 8.5 (JetPack 5.1.4). Whether the three parser problems exist there is not known; rerun the same trtexec parse and the head check.
Debug helpers: score_path_nodes.py (nodes only on the pred_score path), head_debug.py + head_b1.onnx (extract_model of the head from the embedding
'/model/Reshape_output_0'), build_baseline.py --pin-fp32-json (FP32 pinning of those nodes; did not help, kept for reference).
Result files: out\bottle_256_baseline_{ort_fp32,trt_fp16_b1,trt_fp16_b2}_{score,map,time}.npy + .json, *_prefix_pin_* = engine before the TopK fix.
Results (83 test images, board MODE_10W_DESKTOP; GPU = trtexec GPU Compute mean over 100 it; py = dump_baseline.py infer H2D+exec+D2H per image, image read excluded):
                      image AUROC  pixel AUROC  AUPRO   GPU ms   py ms/img  engine   build
  ORT fp32 (PC CPU)   1.0000       0.9853       0.9437   -        -          -        -
  TRT fp16 b1         1.0000       0.9854       0.9447   64.33    65.06      383.6 MB 487.9 s
  TRT fp16 b2         1.0000       0.9854       0.9448   120.89 (60.4/img)  60.85  909.5 MB 522.6 s
  TRT fp16 b1 before the TopK fix: 0.6266 / 0.9854 / 0.9446 (64.5 ms); with score-path layers pinned FP32: 0.6310 (69.2 ms, 712 MB engine).
  RKNN board fp16 reference: 1.0000 / 0.9854, 977 ms. TRT fp16 vs ORT fp32: pred_score max abs 0.041, anomaly_map max abs 0.075 (mean 0.022).
Build log warnings: INT64 weights cast to INT32; "Converted FP32 value in weights ... to FP16 infinity" on 4 constant layers (not identified; scores are fine);
no memory warnings; "Some tactics do not have sufficient workspace memory" at --workspace 1024. Free RAM during the build dropped to ~100-300 MB (buff/cache 3 GB).
Separate fp16 finding (head-only engine, 7 images): in fp16 the nearest-neighbor index differs for ~20-25% of patches and patch scores differ by up to ~7 (fp32 engine: 0 / 0.003);
the image score still passed because the max-patch neighbor matched or was close enough on these data. Not analyzed further.


## DLA (Xavier NX, TensorRT 8.5.2, JetPack 5.1.4) - bottle only, hard categories NOT run
Code: `trt_runner.py build --dla-core N` (default_device_type=DLA, DLA_core=N, GPU_FALLBACK; int8 calibrator and `--fp16-layers` as before),
`TrtRunner(engine, dla_core)` / `bench --dla-core N` (runtime.DLA_core set before deserialize), `dump_trt.py --dla-core N`.
Board: `~/trt_dla/` (dla_eval.sh <cat> <fp16|int8mm|int8ent|int8mix> [core]); PC driver `pc\p9_dla.ps1`, summary `p9_dla_summary.csv`,
build logs and per-layer engine JSON in `out\dla_logs\`, scores `out\dla_bottle_*.json`.
Board power mode: /var/lib/nvpmodel/status = pmode:0005 (10W desktop, unchanged); `nvpmodel -q` itself fails without root (curr1_crit permission).
Placement (same for fp16, int8ent, int8mix; see build logs): DLA subgraph 1 = pre_transpose..scale_operand_of_pre_sub (only the NHWC->NCHW transpose);
GPU = pre_sub/pre_div (PWN; DLA rejects Div and Sub with a [1,3,1,1] operand), then DLA subgraph 2 = whole backbone + projection
(conv1 ... node_of_dk0, one ForeignNode); GPU = head ReduceMax (node_of_gk0) + output copy. 3 DLA<->GPU transitions per image.
DLA int8 needs the entropy2 calibrator: `--calib-algo minmax` fails at build ("Network built for DLA requires kENTROPY_CALIBRATION_2", DLA validation failed).
Per-layer precision pinning (`--fp16-layers`, OBEY_PRECISION_CONSTRAINTS) is accepted on DLA (26 layers pinned, engine builds and runs).
One engine per core is NOT required: int8mix_dla0.engine deserialized with runtime.DLA_core=1 runs; outputs bit-identical to core 0 (max abs diff 0.0, one random image).
Bottle (83 test images, fp32 ref 1.0000 / 0.9848 / 0.9430; GPU mixed engine 1 / 0.9849 / 0.943):
                       img AUROC  pix AUROC  AUPRO   trtexec latency (1 core, 10W)  engine   build
  DLA fp16             1.0000     0.9848     0.9430  29.72 ms (34.5 qps)            75.5 MB  88-95 s
  DLA int8 minmax      build fails (see above)
  DLA int8 entropy2    0.5000     0.7702     0.2958  17.75 ms (55.1 qps)            27.4 MB  262 s    (unusable)
  DLA int8 mixed       1.0000     0.9850     0.9430  19.92 ms (52.3 qps)            29.3 MB  74 s     (stem+layer1.0+head fp16, entropy2)
Not measured: screw/wood/toothbrush/tile (the PC driver was stopped by the host for low memory after screw fp16 was built and dumped on the board
(~/trt_dla/screw/out_fp16_*.npy, not scored)); two-core + GPU concurrent throughput; whether int8mix really executes int8 convs inside the
DLA ForeignNode (the inspector only shows the ForeignNode; the 29 MB engine size vs 75 MB fp16 suggests int8 weights).


## End-to-end benchmark (CPU resize + GPU/DLA engine + post), bottle, Xavier NX - added 2026-11-01
Code: `board\bench_e2e.py` (modes bench | acc | micro), `board\build_e2e.py` + `board\build_e2e.sh` (engines), `board\run_suite.sh` (all timed configs),
`pc\e2e_score.py` (accuracy on the PC). Board dir `~/e2e/` (engines/, onnx/, test900/ = the 83 original 900x900 frames, calib900/, results/, acc/, logs/).
Frames are decoded once (cv2.imread, BGR->RGB) and kept as uint8 HWC in RAM; PNG decode is not timed. Timed loop = cv2.resize -> uint8->float32 copy into the
pinned input buffer -> H2D -> execute_async_v2 -> D2H -> stream sync -> (P0 only) numpy/cv2 post. One thread, one engine/context/stream per worker.
Pipelines: P0 = CPU resize + engine without post (`mixed_name.engine`: m, xproj) + CPU post (float32 numpy sqrt/top10, cv2 nearest x8, cv2.GaussianBlur 33x33 sigma 4
reflect-101; matches graph_opt.host_post to 2e-5 score / 1.7e-4 map). P1 = CPU resize + engine with post inside (score_max, score_top10, map 256x256).
P2 = 900x900 float32 input (9.72 MB), matmul (Pillow-weight) resize + post inside the engine. P3 = P1 whose backbone runs on the DLA (GPU fallback for the rest).
Resize: `cv2.INTER_AREA` (default, closest to the Pillow antialias used for the 256 PNGs, NOT identical) or `cv2.INTER_LINEAR` (no antialias, ~ same error as
graph_opt fullres_bilinear). Mean |uint8 diff| vs Pillow-resized calib image 000: AREA 0.40, LINEAR 1.30.
Engines (int8 entropy2; stem+layer1.0+head fp16 as mixed_name; calibration = the same 100 train images, 256 px for P1/P3, original 900 px for P2):
post layers (post_*, score_*, map) pinned fp32, P2 resize layers (fr_*) pinned fp32. `build_e2e.py` wraps trt_runner.pin_layers to skip Identity/int32 layers
(the shared pin_layers made the post graph fail with "No supported formats for node_of_m").
  p1_gpu_post.engine 28.9 MB, 230 s | p3_dla0_post.engine 29.3 MB, 89 s (calibration cache reused from P1) | p2_gpu_full900.engine 30.8 MB, 246 s.
P3 placement (build log): 2 DLA subgraphs (pre_transpose..scale_operand_of_pre_sub; conv1..node_of_dk0); GPU = pre_sub/pre_div, node_of_gk0 (head ReduceMax),
the whole post (TopK, Reduce, nearest Resize, blur conv; the 33-tap kernel exceeds the DLA limit of 32). One engine file runs on both cores via runtime.DLA_core.
Accuracy, 83 bottle test frames at 900x900 (fp32 reference at 256 Pillow: 1.0000 / 0.9848 / 0.9430; scores in out\e2e_accuracy.json):
                         img AUROC (max, top10)   pixel AUROC   AUPRO
  P0 / P1 resize AREA     1.0000, 1.0000          0.9845        0.9428   (P0 and P1 agree: engine post == host post)
  P0 / P1 resize LINEAR   1.0000, 1.0000          0.9830        0.9394
  P2 matmul in-engine     1.0000, 1.0000          0.9849        0.9426
  P3 DLA0 AREA            1.0000, 1.0000          0.9846        0.9428
  P3 DLA0 LINEAR          1.0000, 1.0000          0.9831        0.9398
Functional run in the default 10W mode, NOT final (4 cores online, DVFS: CPU 1.2-1.9 GHz, GPU 0.1-0.5 GHz; 20 s measured after 5 s warm-up; non-blocking CUDA streams),
`out\e2e_functional_10W.json`, `out\e2eunc_10W_nb\*.json` (each has a `placement` field). An earlier identical run with blocking streams is in `out\e2eunc_10W_blockingstream_run1`.
  config                    FPS    p50/p99 ms   CPU % (4 cores, whole system)
  P0 gpu1 AREA              51.4   19.3/21.2    16     (first run: 43.2)
  P0 gpu1 LINEAR            69.8   14.2/15.7    11
  P1 gpu1 AREA              55.7   17.8/19.5    14
  P1 gpu1 LINEAR            84.0   11.8/13.0    12
  P1 gpu2 AREA              94.9   20.9/23.3    27
  P1 gpu2 LINEAR            96.8   21.0/21.1    11
  P2 gpu1                   58.1   17.0/19.2    12
  P2 gpu2                   66.9   30.2/30.5     9
  P3 dla0 AREA              33.4   30.0/31.9    12
  P3 dla1 AREA              29.9   33.9/35.6    12
  P3 dla0+dla1 AREA         55.8   35.7/35.9    22
  P3 gpu+dla0+dla1 AREA     67.1   44.7/44.9    20
  LINEAR: P3 dla0 46.3 | dla0+dla1 58.7 | gpu+dla0+dla1 68.3
Stage means (ms/frame, 10W): CPU resize AREA 6.6-7.0 here (micro-benchmark 8.8), LINEAR 0.85-0.9; uint8->float32 copy 0.10 (256 px) / 0.86-1.0 (900 px);
engine incl. H2D+D2H+sync 10.8-11.2 (P1 gpu), 16.2 (P2 gpu), 20.0 (P3 dla0); P0 host post 2.0-2.5 ms.
Micro (single thread, `results\micro.json`): pinned H2D 9.72 MB = 2.5 ms (3.9 GB/s), 0.79 MB = 0.17 ms (the 1.7 ms of the first micro run was the idle-clock first copy).
Findings:
(1) cv2 INTER_AREA at 900->256 costs 7-9 ms on this CPU, about as much as the GPU exec; INTER_LINEAR is ~10x cheaper but loses 0.0015 pixel AUROC / 0.0034 AUPRO.
(2) In-engine post saves only 2-2.5 ms per frame (P0 -> P1) because the host post is small.
(3) Two GPU worker threads reach 95-97 FPS (0.36-0.84 cores busy per process), i.e. the GPU is saturated at ~21 ms per frame per worker; no GIL limit was visible for the GPU x2 case.
(4) P2 (900x900 input, resize in the engine) is slower than P1+LINEAR at 10W (58.1 vs 84.0 FPS): the in-engine resize adds ~5 ms GPU time and 9.7 MB of input conversion/H2D; it saves the CPU resize only.
(5) DLA workers: 33.4 / 29.9 FPS per core, both cores 55.8-58.7 FPS. The mixed GPU+DLA0+DLA1 threaded config is NOT reproducible in 10W mode: the same command
gave 67.1 FPS (22.4 per worker, repeated 5 times), but also 104.3, 86.3 and 72.9 FPS in other runs (diag runs `out\e2e\diag`, 15 s each). This is independent of the stream
type (blocking: 67.1 / 86.3 / 72.9, non-blocking: 67.1 / 104.3 / 67.2). Runs that started with the GPU at 114 MHz gave 67, runs that started at 204 MHz gave 73-104 (correlation only, 8 runs).
A 3-process control (one gpu, one dla0, one dla1 process, 15 s) gave 27.0 + 35.0 + 34.8 = 96.8 FPS. Cause not established (DVFS / EMC / power cap are suspects); judge GPU+DLA only
under fixed max clocks, with REPS=3 (`REPS=3 sh run_suite.sh`).
Not verified: output correctness of concurrent workers (single-worker acc dumps only), the temperature guard under real heat (only the abort path, tested by lowering the limit),
max power mode, power. The shared pin_layers failure was worked around in `build_e2e.py`, trt_runner.py was not changed.
P0 engine: `~/e2e/engines/p0_gpu_mixed_name.engine` is a copy of the earlier `~/trt_run/mixed_name.engine` (27.5 MB, built 220 s, no post). Loading it prints the TensorRT warning
"Using an engine plan file across different models of devices is not recommended"; the three engines built here do not print it in the acc runs (cause not investigated; P0 numbers are from that engine as is).
Run: `sh ~/e2e/run_suite.sh <TAG>` (results in `~/e2e/results/<TAG>/`: one .log/.json per config, micro.json, suite.log; optional REPS=3, DURATION, WARMUP, config-name filter after TAG).


## Stage-split pipelines P4 / P5 (DLA backbone without post) - added 2026-11-01, 10W functional run, NOT final
Idea: keep the post graph out of the DLA engine (in P3 it falls back to the GPU and the 33-tap blur exceeds the DLA limit) and run the post separately.
Code: `board\bench_e2e.py` (new pipes P4/P5, `--post-engine`, `--p5-link device|host`, `--mode check`, worker spec `label=ENGINE[@PIPE]`; P0..P3 code paths unchanged,
re-run after the edit: P0 / P1 / P3 start and run), `board\run_suite_split.sh` (root-free, no `~`), `pc\build_post_only.py` (builds + checks the post-only ONNX), `pc\e2e_score.py` (unchanged).
- P4 = CPU resize (INTER_AREA from 900) + DLA engine `int8mix_dla0.engine` (m, xproj; copied to `~/e2e/engines/`) + CPU numpy/cv2 post (the P0 code).
- P5 = same front end + DLA engine + GPU post-only engine `~/e2e/engines/p5_gpu_post_only.engine`, both on one stream in one worker thread.
  Link `device` (default): the post engine reads the backbone's output device buffers (no copy). Link `host`: D2H m/xproj, CPU copy, H2D (two-process style).
- Post-only ONNX `out\bottle_post_only.onnx` (1.8 kB, opset 12; `graph_opt.add_post(mode="conv")` on a dummy m/xproj graph): inputs m [1,1,32,32], xproj [1,384,32,32] fp32,
  outputs score_max [1], score_top10 [1], map [1,1,256,256]. ORT vs `graph_opt.host_post` (float64 + scipy), 83 frames of the fp32 dumps: max |diff| score_max 2.3e-5, top10 1.2e-5,
  map 2.9e-4 (map range 2.9..77), `out\e2e_split_post_only_check.json`. Engine: pure fp32 (no int8/fp16 anywhere, so ||x||^2 - m is fp32), 30 kB, built in 22 s.
  trtexec on the board (10W): GPU compute 0.64 ms mean (0.62 with CUDA Graph), H2D 0.16 ms, D2H 0.04 ms, 1558 qps. TRT engine vs host_post on the DLA engine's own outputs (83 frames, both links):
  max |diff| score_max 7.6e-6, top10 1.5e-5, map 3.8e-5.
- Raw DLA engine in trtexec at 10W: 19.1 ms/core (52 qps). The 73 FPS/core quoted for max mode is NOT reproducible in 10W; the DLA runs slower in this mode (cause: power mode clocks, not measured).

10W functional run (15 s measured after 5 s warm-up, non-blocking streams; `out\e2e_split_functional_10W.json`, `out\e2e_split_func_10W\`). P3 column = the earlier 20 s 10W run above.
  worker set            P3 (post in engine)   P4 (CPU post)  p50/p99 ms   CPU%   P5 device link  p50/p99 ms   CPU%   P5 host link
  dla0                  33.4                  30.5           32.8/34.5    12     28.6            35.3/37.6    15     32.0 (3 more runs: 29.0, 30.2, 29.4)
  dla1                  29.9                  30.6           32.7/34.4    12     29.0            34.6/36.7    15     -
  dla0+dla1             55.8                  61.0           32.5/36.8    22     57.1            34.7/35.9    19     60.6
  gpu(P1)+dla0+dla1     67.1                  106.6          27.8/30.9    32     67.0            44.7/45.4    22     -
  INTER_LINEAR dla0+dla1 58.7                 83.4           23.6/27.4    18     58.6            34.1/34.3     7     -
  P5 device link, 3 repeats of dla0: 29.95 / 29.36 / 29.09 FPS; host link: 28.96 / 30.16 / 29.36. No measurable difference; device link needs no copies and is the default.
  Stage means (ms/frame, P4 dla0): resize 9.2, uint8->fp32 0.12, H2D+DLA exec+D2H 20.8, host post 2.6. P5 device: resize 10.5, engine phase (backbone + post engine) 24.1.
  P5 host link: backbone incl. its D2H 18.8-20.8, copies + post engine about 2.4-3.6.
Accuracy, 83 frames at 900x900, dla0 engine (`out\e2e_split_accuracy.json`; fp32 reference 1.0000 / 1.0000 / 0.9848 / 0.9430):
                        img AUROC (max, top10)  pixel AUROC  AUPRO
  P3 dla0 AREA          1.0000, 1.0000          0.9846       0.9428
  P4 dla0 AREA          1.0000, 1.0000          0.9846       0.9428
  P5 dla0 AREA          1.0000, 1.0000          0.9846       0.9428
  P4 dla0 LINEAR        1.0000, 1.0000          0.9831       0.9398
  P5 dla0 LINEAR        1.0000, 1.0000          0.9831       0.9398
  P4 and P5 maps differ by 4e-5 at most from each other, so the split does not change accuracy; the DLA backbone is the same engine as before.
Findings: (1) In 10W mode the split does NOT help: the DLA backbone alone takes ~19-21 ms per frame, so one worker is DLA-bound (and resize-bound) and P3's in-engine post cost only
~1 ms there. P4 dla0 30.5 vs P3 33.4 FPS, P4 dla0+dla1 61.0 vs 55.8 (+9%, within the run-to-run spread seen for P3). P5 is not faster than P4: the GPU post saves the 2.6 ms CPU post but the
extra engine phase costs about as much. (2) The mixed gpu+dla0+dla1 numbers (106.6 vs 67.0 vs 67.1) are the known 10W plateau lottery (see P3 note above), not a P4/P5 effect.
(3) Whether the split pays off in max mode (where P3 reached only 42.6 FPS/core against 73 in trtexec) is NOT measured here; run the suite below. (4) INTER_LINEAR P4 dla0+dla1 reaches 83 FPS because the resize drops 9 -> 1 ms.
Limits: P4/P5 timings use one run per config (15 s) except the link comparison; the accuracy runs use the dla0 engine on core 0 only; the post-only engine was not built with int8 (not needed, 0.64 ms).
`bench_e2e.py --mode check` compares the post engine with host_post on the DLA outputs; PostRunner in bench_e2e.py is needed because trt_runner.TrtRunner accepts only one input binding.
Final run (as root, max mode): `export HOME=/home/<user>; REPS=3 sh /home/<user>/e2e/run_suite_split.sh max_split` (results in `~/e2e/results/max_split/`; DURATION=20 default; config names as filter, e.g. `... max_split P4_dla0_dla1`).

## Pipelined (stage-decoupled) mode - added 2026-11-01, 10W functional run, NOT final
Idea: the accelerator threads (GPU, DLA0, DLA1) only do H2D + execute + D2H. Resize (and for P4 optionally the CPU post) run on separate CPU threads, so the DLA no longer idles during the resize.
Code: `board\bench_e2e.py --pipelined` (default off; sequential modes P0..P5 unchanged, the SEQ_ runs below use them), `board\run_suite_pipelined.sh` (root-free, no `~`), `pc\e2e_pipe_acc.ps1`.
- Flow: R feeder threads (`--resize-threads`, default 2) take a free pinned buffer, read a frame from the in-RAM list, cv2.resize (INTER_AREA/LINEAR) and write float32 NHWC straight into the pinned buffer
  (P2: only the uint8 -> fp32 copy of the 900x900 frame) -> one shared bounded ready queue (`--queue-depth`, default 4) -> accelerator threads (one engine/context/non-blocking stream each; H2D reads the pinned buffer directly,
  the buffer is returned after the stream sync) -> P4: host_post inline (`--post-threads 0`) or in a post thread pool (`--post-threads N`, bounded queue). Frame ids travel with each item; `--mode acc --pipelined` stores outputs by id.
- Metrics: fps over the measured window (completions inside it), latency = feeder starts on a frame -> result ready (includes ready-queue wait and back-pressure wait), system CPU %, queue occupancy sampled every 50 ms,
  fractions of the window the accelerators waited on an empty queue (`accel_wait_frac`), the feeders waited on a full queue (`feeder_put_wait_frac`), the accelerator busy (`accel_busy_frac`), post threads busy; `bottleneck_heuristic` is derived from them.
  JSON field names are those of the sequential mode (parse_e2e.py works unchanged; verified on `out\e2e_pipe_func_10W`), plus `pipelined` and `queue`.
- The 4-CPU 10W mode (cpus online 4) limits the CPU side; max mode has 6 cores.
10W functional results (15 s measured, 5 s warm-up, 1 run each, `out\e2e_pipe_functional_10W.json`, per-run files in `out\e2e_pipe_func_10W\`). SEQ = same session, sequential mode, same engines. Latency p50/p99 at queue depth 4.
  config                       SEQ FPS (p50 ms)   PIPE FPS   PIPE p50/p99 ms   CPU% (PIPE)   bottleneck
  P1 gpu                       54.8 (18.1)        91.0       76.7/77.9         24           accelerator (feeders wait on full queue)
  P1 gpu+gpu2                  95.4 (20.9)        95.4       83.6/83.9         25           accelerator (GPU)
  P2 gpu                       58.9 (17.0)        61.8       113/114           8            accelerator
  P2 gpu+gpu2                  65.9 (30.2)        65.8       121/122           9            accelerator
  P4 dla0                      30.1 (32.6)        45.6       151/217           21           accelerator (DLA exec 19.1 ms)
  P4 dla0+dla1                 65.9 (29.9)        87.5       88/207            41           accelerator
  P4 dla0+dla1, post pool 2    68.3 (28.4)        101.8      78/210            42           accelerator (DLA exec 19.1 ms/core -> 51 FPS/core)
  mixed gpu(P1)+dla0+dla1      105.8 (27.8)       106.4      84/122            39           accelerators (35 FPS each, GPU-side contention)
  mixed, post pool 2           102.1 (28.6)       105.2      87/157            42           accelerators
  INTER_LINEAR P1 gpu+gpu2     94.7               94.1       84/140            17
  INTER_LINEAR P4 dla0+dla1 pool 2   82.5         97.2       83/166            28
  INTER_LINEAR mixed pool 2    106.5              105.5      89/187            28
In all 13 pipelined configs the accelerators were never starved (accel_wait_frac <= 0.01, ready queue full), so the feeders (resize 7.4-8.8 ms per frame, 2 threads) kept up in 10W.
Queue depth vs latency (P1 gpu / P4 dla0 / P4 dla0+dla1 pool 2 / mixed pool 2): depth 1: 91.2 FPS p50 44 ms / 48.8 FPS 82 ms / 105.5 FPS 50 ms / 107.0 FPS 58 ms;
depth 2: 91.2 / 54.6 ms, 46.4 / 107.5, 100.3 / 61.3, 106.8 / 65.3 (depth 4 above). Throughput does not depend on the depth in these runs, latency does (back-pressure: a saturated pipeline holds about 4-7 frames).
Accuracy (83 frames at 900x900, `out\e2e_pipe_accuracy.json`; fp32 reference 1.0000 / 0.9848 / 0.9430), outputs also compared bit for bit with the sequential npz files:
  pipelined config                    img AUROC (max, top10)   pixel AUROC   AUPRO   vs sequential npz
  P4 dla0 AREA                        1.0000, 1.0000           0.9846        0.9428  max |map diff| 0.0 (vs split_P4_dla0_area)
  P4 dla0 LINEAR                      1.0000, 1.0000           0.9831        0.9398  0.0
  P4 dla0+dla1, pool 2, AREA          1.0000, 1.0000           0.9846        0.9428  0.0 (frames spread over both cores, ids mapped correctly)
  P1 gpu AREA                         1.0000, 1.0000           0.9845        0.9428  0.0 (vs P1_area)
  P1 gpu+gpu2 AREA                    1.0000, 1.0000           0.9845        0.9428  0.0 (identical to P1 gpu)
Findings: (1) Pipelining helps where a single sequential thread leaves the accelerator idle: P1 one GPU worker +66%, P4 dla0 +51%, P4 dla0+dla1 +33% (+49% with the post pool). (2) It does not help where the sequential run already
saturated the device: P1 gpu+gpu2, P2, and the mixed gpu+dla0+dla1 set (106 FPS both ways in this session; per-worker 35 FPS with an accelerator time of 28 ms vs 10.8 ms for the GPU engine alone, i.e. the three workers slow each other down;
the cause is not measured). (3) The post pool adds about +16% for dla0+dla1 (87.5 -> 101.8) and nothing for dla0 alone or the mixed set. (4) Per-frame latency under saturation is 2.7-6.6x the sequential latency (queue wait).
(5) P4 dla0+dla1 pipelined reaches 51 FPS per core = 1 / 19.1 ms, the trtexec DLA rate of this 10W mode, so there is no head room left for the pipeline on the DLA at 10W.
Limits and notes: GIL contention was not isolated (process CPU was 1.25 cores for the mixed set, accelerator threads never starved); no per-worker queues (one shared queue, so a slow DLA worker cannot block a fast GPU worker);
pinned buffers: depth + workers + R x 0.79 MB (9.7 MB each for P2); P2 cannot be mixed with P1/P4 workers in one run (one buffer pool); ≤ 2 DLA contexts as before; latency is back-pressure dominated when saturated (see depth runs);
all numbers are single 15 s runs at 10W; the max-mode numbers are NOT measured.
Final run (as root, max mode): `export HOME=/home/<user>; REPS=3 SEQ=1 sh /home/<user>/e2e/run_suite_pipelined.sh max_pipe` (results in `~/e2e/results/max_pipe/`; DURATION=20 default; env RT=resize threads, QD=queue depth; config names as filter, e.g. `... max_pipe PIPE_P4_dla0_dla1_pt2`; SEQ=1 adds the sequential counterparts `SEQ_*`). Aggregate with `python parse_e2e.py <dir>`.

## Resize study (train/inference resize alignment, PC fp32, onnxruntime CPU)
Scripts: `pc\resize_study.py` (driver), `pc\make_knn_head_resize.py` (make_knn_head.py + RESIZE_METHOD switch only), `pc\run_all.sh`, `pc\run_c05.sh`.
Results: `out\resize_study\results.csv` and one json per run. New banks: `patchcore\work\<cat>_256\resize_<tag>\`.
Protocol: bank (dim 384, coreset 1%, seed 1) built from train images resized by M; test = originals resized in-pipeline by M (uint8), fused `_rmax_pf_pre` graph, host_post, metrics.
Checks: pil_aa aligned reproduces p6_ort_summary exactly; rebuilding pil_aa on screw gives identical metrics (noise floor 0 at 4 decimals).
cv2 linear vs torchvision bilinear antialias=False: mean abs 0.37, max 1 gray level, so tv_bilinear_noaa was dropped.
Table columns: aligned image max / pixel / AUPRO ; mismatched (Pillow bank) pixel / AUPRO.
| cat | M | aligned img/pix/AUPRO | mismatch pix/AUPRO |
|---|---|---|---|
| screw | pil_aa | .9625/.9801/.9127 | .9801/.9127 |
| screw | area | .9535/.9852/.9258 | .9854/.9348 |
| screw | linear | .9836/.9877/.9418 | .9862/.9383 |
| bottle | pil_aa | 1/.9848/.9430 | same |
| bottle | area | 1/.9847/.9431 | .9845/.9431 |
| bottle | linear | 1/.9836/.9355 | .9829/.9395 |
| wood | pil_aa | .9851/.9318/.8790 | same |
| wood | area | .9868/.9339/.8854 | .9342/.8846 |
| wood | linear | .9868/.9350/.8852 | .9362/.8837 |
| toothbrush | pil_aa | .9111/.9880/.8690 | same |
| toothbrush | area | .9000/.9873/.8610 | .9879/.8688 |
| toothbrush | linear | .9056/.9877/.8661 | .9878/.8687 |
Screw coreset 5% aligned: pil_aa .9664/.9889/.9409 ; linear .9779/.9920/.9572.


## anomalib-default baseline re-measured on TensorRT 8.5.2 (JetPack 5.1.4) - bottle/256, fp16, 10W - added 2026-11-01
Board dir ~/base514/ (parse_check.py, build_baseline.py, dump_baseline.py, head_debug.py, trt_runner.py, full_514.sh, run_all_514.sh, heads_514.sh, time_baseline.sh).
PC helpers: fold_variants_514.py (copy of fold_baseline.py with --no-topk-fix / --no-flatten-fix), head_variant_514.py, compare_head_514.py.
New ONNX (in patchcore\work\bottle_256\baseline\): anomalib_b1_foldonly.onnx (op17 + constant folding, Flatten and TopK left as exported), head_foldonly_b1.onnx (+_ref.npz), head_folded_b1.onnx.
Logs/results: out\baseline514_*.json/.npy, out\baseline514_variants.json, out\b514\ (build, trtexec, dump logs, free -m samples every 3 s).
Parse check = TRT python OnnxParser (parse_check.py), not trtexec: `trtexec --skipInference` also builds the engine (about 12 min here), so the build step was the build_baseline.py run.

Which workarounds are still needed on 8.5.2 (all measured):
  variant                                        parse                         fp16 build + 83 images (image / pixel / AUPRO)
  legacy op17, no fixes                          FAILS (post_processor/ConstantOfShape: INVALID_NODE, "Weights of same values but of different types")  -
  anomalib dynamo export (opset 18, batch 2)     FAILS (node_cat_1: error while computing output extent)                                              -
  op17 + constant folding only (foldonly)        OK                            built 721.8 s; 1.0000 / 0.9854 / 0.9447, trtexec 63.66 ms
  op17 + folding + Flatten->Reshape + TopK fix   OK                            built 724.3 s; 1.0000 / 0.9854 / 0.9446, trtexec 63.61 ms
  => (1) TorchScript op17 export: still needed in the sense that the dynamo export does not parse (cause not analyzed; folding was not tried on it).
     (2) constant folding of the zero-element nodes: still needed (the unfolded op17 export does not parse).
     (3) Flatten(axis==rank)->Reshape: NOT needed (foldonly engine scores equal; head-only fp32 pred_score max abs diff 6e-6 vs ORT).
     (4) TopK largest=0 -> Neg/TopK/Neg: NOT needed (foldonly engine scores equal; head-only fp32: TopK indices equal 1.0000, so largest=0 is honored).
  The "folded, Flatten fixed, TopK unfixed" ONNX (anomalib_b1_folded.onnx) was only parsed (OK), not built: its head-only engines were stopped to save board time; foldonly covers both changes at once.
Head-only checks (7 embeddings, ORT fp32 reference): fp32 engine all outputs match (anomaly_map max abs 1.8e-5); fp16 engine shows the known fp16 nearest-neighbor
deviation (NN index equal for 71% of patches, pred_score max abs 0.028, anomaly_map 0.071), as on 8.2.1.

Results (83 bottle frames, 10W mode, GPU = trtexec GPU Compute mean of 100 it, py = dump_baseline.py H2D+exec+D2H per image):
                      image   pixel   AUPRO   GPU ms           py ms/img  engine    build    RAM low-water during build (free / available)
  ORT fp32 (PC)       1.0000  0.9853  0.9437  -                -          -         -        -
  TRT 8.2.1 b1 fp16   1.0000  0.9854  0.9447  64.33            65.06      383.6 MB  487.9 s  ~100-300 MB free
  TRT 8.5.2 b1 fp16   1.0000  0.9854  0.9446  63.61            64.95      313.5 MB  724.3 s  157 / 2433 MB
  TRT 8.2.1 b2 fp16   1.0000  0.9854  0.9448  120.89 (60.4/i)  60.85      909.5 MB  522.6 s
  TRT 8.5.2 b2 fp16   1.0000  0.9854  0.9447  115.29 (57.6/i)  58.43      313.5 MB  763.4 s  215 / 2129 MB
  RKNN RK3588 fp16    1.0000  0.9854  -       977 ms
Build warnings (8.5.2): INT64 weights cast to INT32; FP16 conversion of 48 subnormal / 38 below-minimum / 2 overflowing weights; "Some tactics do not have sufficient workspace memory" (workspace 1024 MB, same as before).
GPU temperature stayed at 44.5-46 C at the end of each run; swap use after the builds 307-378 MB.

Max-mode timing (root, run by the user): see board\time_baseline.sh.
  ssh <user>@<board-ip>
  sudo sh -c 'sh /home/<user>/perf_mode.sh on; sh /home/<user>/base514/time_baseline.sh; sh /home/<user>/perf_mode.sh off'
  -> ~/base514/time/{b1,b2}.log (trtexec), {b1,b2}.tegra (tegrastats 200 ms), mode.txt, time.log, time.flag
