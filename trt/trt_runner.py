"""Self-contained TensorRT runner for Jetson (python3 >= 3.6, numpy >= 1.13, cv2, tensorrt).

No pycuda / polygraphy: the CUDA runtime is loaded through ctypes (libcudart.so).
Uses only the TensorRT API that exists in both 8.2 and 8.5 (execute_async_v2, get_binding_*).

CLI:
  python3 trt_runner.py build --onnx m.onnx --engine m.engine --precision fp16|int8 [--calib-dir D] [--calib-cache C]
         [--calib-algo entropy2|minmax|entropy|legacy] [--fp16-layers SPEC] [--fp32-layers SPEC] [--inspect J.json]
         [--dla-core N]   (DLA default device + GPU fallback; the engine is tied to that core at build time)
  python3 trt_runner.py bench --engine m.engine --images DIR [--n 200] [--dla-core N]
"""
import argparse
import ctypes
import glob
import os
import sys
import time

import numpy as np

_CUDA_MEMCPY_H2D = 1
_CUDA_MEMCPY_D2H = 2


class CudaError(RuntimeError):
    pass


class Cudart(object):
    """Minimal ctypes wrapper of the CUDA runtime. Every call checks its return code."""

    def __init__(self):
        last = None
        self.lib = None
        for name in ("libcudart.so", "libcudart.so.10.2", "libcudart.so.11.0", "/usr/local/cuda/lib64/libcudart.so"):
            try:
                self.lib = ctypes.CDLL(name)
                break
            except OSError as e:
                last = e
        if self.lib is None:
            raise CudaError("cannot load libcudart: %s" % last)
        L = self.lib
        L.cudaGetErrorString.restype = ctypes.c_char_p
        L.cudaGetErrorString.argtypes = [ctypes.c_int]
        L.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        L.cudaFree.argtypes = [ctypes.c_void_p]
        L.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        L.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
        L.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        L.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        L.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        L.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
        L.cudaFreeHost.argtypes = [ctypes.c_void_p]

    def check(self, code, what):
        if code != 0:
            msg = self.lib.cudaGetErrorString(code)
            raise CudaError("%s failed: code=%d (%s)" % (what, code, msg.decode() if msg else "?"))

    def malloc(self, nbytes):
        p = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(p), nbytes), "cudaMalloc(%d bytes)" % nbytes)
        return p.value

    def free(self, ptr):
        self.check(self.lib.cudaFree(ctypes.c_void_p(ptr)), "cudaFree")

    def host_alloc(self, nbytes):
        p = ctypes.c_void_p()
        self.check(self.lib.cudaHostAlloc(ctypes.byref(p), nbytes, 0), "cudaHostAlloc(%d bytes)" % nbytes)
        return p.value

    def free_host(self, ptr):
        self.check(self.lib.cudaFreeHost(ctypes.c_void_p(ptr)), "cudaFreeHost")

    def memcpy_h2d_async(self, dst, src, nbytes, stream):
        self.check(self.lib.cudaMemcpyAsync(ctypes.c_void_p(dst), ctypes.c_void_p(src), nbytes,
                                            _CUDA_MEMCPY_H2D, ctypes.c_void_p(stream)), "cudaMemcpyAsync H2D")

    def memcpy_d2h_async(self, dst, src, nbytes, stream):
        self.check(self.lib.cudaMemcpyAsync(ctypes.c_void_p(dst), ctypes.c_void_p(src), nbytes,
                                            _CUDA_MEMCPY_D2H, ctypes.c_void_p(stream)), "cudaMemcpyAsync D2H")

    def memcpy_h2d(self, dst, src, nbytes):
        self.check(self.lib.cudaMemcpy(ctypes.c_void_p(dst), ctypes.c_void_p(src), nbytes, _CUDA_MEMCPY_H2D),
                   "cudaMemcpy H2D")

    def stream_create(self):
        s = ctypes.c_void_p()
        self.check(self.lib.cudaStreamCreate(ctypes.byref(s)), "cudaStreamCreate")
        return s.value

    def stream_sync(self, stream):
        self.check(self.lib.cudaStreamSynchronize(ctypes.c_void_p(stream)), "cudaStreamSynchronize")

    def stream_destroy(self, stream):
        self.check(self.lib.cudaStreamDestroy(ctypes.c_void_p(stream)), "cudaStreamDestroy")


def _np_dtype(trt, d):
    m = {trt.float32: np.float32, trt.float16: np.float16, trt.int32: np.int32, trt.int8: np.int8}
    if hasattr(trt, "bool"):  # bool bindings (anomalib pred_label / pred_mask)
        m[trt.bool] = np.bool_
    if d not in m:
        raise ValueError("unsupported TensorRT dtype %s" % d)
    return m[d]


def _host_array(cuda, nbytes, dtype, shape):
    """Pinned host buffer exposed as a numpy array (the buffer is freed by TrtRunner.close)."""
    ptr = cuda.host_alloc(nbytes)
    buf = (ctypes.c_char * nbytes).from_address(ptr)
    arr = np.frombuffer(buf, dtype=dtype).reshape(shape)
    return ptr, arr


class TrtRunner(object):
    def __init__(self, engine_path, dla_core=None):
        import tensorrt as trt
        self.trt = trt
        if not os.path.isfile(engine_path):
            raise IOError("engine file not found: %s" % engine_path)
        self.cuda = Cudart()
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            data = f.read()
        self.runtime = trt.Runtime(logger)
        if dla_core is not None:
            self.runtime.DLA_core = int(dla_core)  # must be set before deserialize
        self.engine = self.runtime.deserialize_cuda_engine(data)
        if self.engine is None:
            raise RuntimeError("deserialize_cuda_engine failed for %s (%d bytes, dla_core=%s); engine must be built with "
                               "the same TensorRT version on this device" % (engine_path, len(data), dla_core))
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("create_execution_context failed (GPU memory?)")
        self.stream = self.cuda.stream_create()
        self.names, self.is_input, self.dev, self.host, self.host_ptr, self.shapes = [], [], [], [], [], []
        self._freed = False
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = tuple(self.engine.get_binding_shape(i))
            if any(s < 0 for s in shape):
                raise ValueError("dynamic shape %s for binding %s is not supported" % (shape, name))
            dt = _np_dtype(trt, self.engine.get_binding_dtype(i))
            nbytes = int(np.prod(shape)) * np.dtype(dt).itemsize
            ptr, arr = _host_array(self.cuda, nbytes, dt, shape)
            self.names.append(name)
            self.is_input.append(self.engine.binding_is_input(i))
            self.dev.append(self.cuda.malloc(nbytes))
            self.host.append(arr)
            self.host_ptr.append(ptr)
            self.shapes.append(shape)
        self.bindings = list(self.dev)
        ins = [i for i, f in enumerate(self.is_input) if f]
        if len(ins) != 1:
            raise ValueError("expected exactly 1 input, got %s" % [self.names[i] for i in ins])
        self.in_idx = ins[0]
        self.in_shape = self.shapes[self.in_idx]
        self.in_dtype = self.host[self.in_idx].dtype

    def infer(self, nhwc):
        """nhwc float32 array of the engine input shape -> dict(output name -> new ndarray)."""
        a = np.asarray(nhwc)
        if tuple(a.shape) != tuple(self.in_shape) or a.dtype != self.in_dtype:
            raise ValueError("input mismatch: got shape=%s dtype=%s, engine wants shape=%s dtype=%s (device gpu)" %
                             (a.shape, a.dtype, self.in_shape, self.in_dtype))
        c = self.cuda
        np.copyto(self.host[self.in_idx], a)
        c.memcpy_h2d_async(self.dev[self.in_idx], self.host_ptr[self.in_idx], self.host[self.in_idx].nbytes,
                           self.stream)
        ok = self.context.execute_async_v2(self.bindings, self.stream)
        if not ok:
            raise RuntimeError("execute_async_v2 returned False (input shape=%s dtype=%s)" % (a.shape, a.dtype))
        for i, f in enumerate(self.is_input):
            if not f:
                c.memcpy_d2h_async(self.host_ptr[i], self.dev[i], self.host[i].nbytes, self.stream)
        c.stream_sync(self.stream)
        return dict((self.names[i], self.host[i].copy()) for i, f in enumerate(self.is_input) if not f)

    def close(self):
        if self._freed:
            return
        self._freed = True
        for p in self.dev:
            self.cuda.free(p)
        for p in self.host_ptr:
            self.cuda.free_host(p)
        self.cuda.stream_destroy(self.stream)
        del self.context

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load_rgb_nhwc(path):
    """cv2.imread (BGR) -> RGB -> float32 NHWC [1,H,W,3], values 0..255."""
    import cv2
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError("cannot read image: %s" % path)
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    x = rgb.astype(np.float32)[None]
    pad = int(os.environ.get("BORDER_INPUT", "0"))  # engines built from fold_norm.py --border graphs expect a mean-filled frame
    if pad:
        mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 1, 3) * 255.0
        xb = np.empty((1, x.shape[1] + 2 * pad, x.shape[2] + 2 * pad, 3), np.float32)
        xb[:] = mean
        xb[:, pad:-pad, pad:-pad] = x
        x = xb
    if os.environ.get("NCHW_INPUT") == "1":  # engines built from fold_norm.py --nchw graphs take NCHW
        x = np.ascontiguousarray(x.transpose(0, 3, 1, 2))
    return x


def _make_calibrator(trt, files, input_shape, cache_path, cuda, algo="entropy2", quantile=0.9999):
    bases = {"entropy2": trt.IInt8EntropyCalibrator2, "minmax": trt.IInt8MinMaxCalibrator,
             "entropy": trt.IInt8EntropyCalibrator, "legacy": trt.IInt8LegacyCalibrator}
    if algo not in bases:
        raise ValueError("calib algo must be one of %s, got %s" % (sorted(bases), algo))

    class Calib(bases[algo]):
        def __init__(self):
            bases[algo].__init__(self)
            self.files, self.idx, self.cache_path = files, 0, cache_path
            self.nbytes = int(np.prod(input_shape)) * 4
            self.dev = cuda.malloc(self.nbytes)

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            if self.idx >= len(self.files):
                return None
            x = load_rgb_nhwc(self.files[self.idx])
            if tuple(x.shape) != tuple(input_shape):
                raise ValueError("calib image %s has shape %s dtype %s, network input is %s" %
                                 (self.files[self.idx], x.shape, x.dtype, input_shape))
            self.idx += 1
            x = np.ascontiguousarray(x)
            cuda.memcpy_h2d(self.dev, x.ctypes.data, self.nbytes)
            return [int(self.dev)]

        def read_calibration_cache(self):
            if self.cache_path and os.path.isfile(self.cache_path):
                with open(self.cache_path, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            if self.cache_path:
                with open(self.cache_path, "wb") as f:
                    f.write(cache)

        # IInt8LegacyCalibrator (percentile) only: quantile of the activation histogram kept, plus a histogram cache.
        def get_quantile(self):
            return quantile

        def get_regression_cutoff(self):
            return quantile

        def read_histogram_cache(self, length):
            return None

        def write_histogram_cache(self, data, length):
            return None

        def release(self):
            cuda.free(self.dev)
            self.dev = None

    return Calib()


def select_layers(network, spec):
    """Layer indices selected by a comma separated spec.

    Tokens: head (from the first '/pool/' layer to the end), idx:A-B (inclusive), name:SUBSTR (substring of the layer name).
    """
    n = network.num_layers
    chosen = set()
    for tok in [t.strip() for t in (spec or "").split(",") if t.strip()]:
        if tok == "head":
            start = [i for i in range(n) if network.get_layer(i).name.startswith("/pool/")]
            if not start:
                raise ValueError("no layer named '/pool/*' in the network, cannot locate the head")
            chosen.update(range(start[0], n))
        elif tok.startswith("idx:"):
            lo, _, hi = tok[4:].partition("-")
            lo, hi = int(lo), int(hi or lo)
            if not (0 <= lo <= hi < n):
                raise ValueError("layer range %s outside 0..%d" % (tok, n - 1))
            chosen.update(range(lo, hi + 1))
        elif tok.startswith("name:"):
            hits = [i for i in range(n) if tok[5:] in network.get_layer(i).name]
            if not hits:
                raise ValueError("no layer name contains '%s'" % tok[5:])
            chosen.update(hits)
        else:
            raise ValueError("bad layer token '%s' (use head | idx:A-B | name:SUBSTR)" % tok)
    return sorted(chosen)


def pin_layers(trt, network, indices, dtype):
    """Pin the compute precision and the output type of the given layers (needs OBEY_PRECISION_CONSTRAINTS)."""
    for i in indices:
        layer = network.get_layer(i)
        if layer.type == trt.LayerType.CONSTANT:
            continue
        layer.precision = dtype
        for j in range(layer.num_outputs):
            layer.set_output_type(j, dtype)


def build_engine(onnx_path, engine_path, precision, calib_dir=None, calib_cache=None, workspace_mb=1024, calib_algo="entropy2",
                 fp16_layers=None, fp32_layers=None, inspect_path=None, quantile=0.9999, dla_core=None):
    """Build and save an engine with the TensorRT python builder. Returns build seconds.

    fp16_layers / fp32_layers: layer specs (see select_layers) pinned to that precision. fp32 is applied last.
    Pinning sets OBEY_PRECISION_CONSTRAINTS (TensorRT >= 8.2). inspect_path writes the per-layer engine JSON.
    dla_core: None = GPU only. N = DLA core N as default device with GPU_FALLBACK (the build log lists the layers per device).
    """
    import tensorrt as trt
    if precision not in ("fp16", "int8", "fp32"):
        raise ValueError("precision must be fp16|int8|fp32, got %s" % precision)
    if not os.path.isfile(onnx_path):
        raise IOError("onnx not found: %s" % onnx_path)
    t0 = time.time()
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        ok = parser.parse(f.read())
    if not ok:
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed for %s: %s" % (onnx_path, errs))
    in_shape = tuple(network.get_input(0).shape)
    config = builder.create_builder_config()
    ws = int(workspace_mb) << 20
    if hasattr(config, "set_memory_pool_limit"):  # TensorRT >= 8.4
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ws)
    else:
        config.max_workspace_size = ws
    if dla_core is not None:
        if int(dla_core) < 0 or int(dla_core) >= builder.num_DLA_cores:
            raise ValueError("dla_core=%s but the builder reports %d DLA cores" % (dla_core, builder.num_DLA_cores))
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = int(dla_core)
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
    pinned = {}
    if fp16_layers or fp32_layers:
        if precision == "fp32":
            raise ValueError("layer pinning needs --precision fp16 or int8")
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        for spec, dt, tag in ((fp16_layers, trt.float16, "fp16"), (fp32_layers, trt.float32, "fp32")):
            if spec:
                idx = select_layers(network, spec)
                pin_layers(trt, network, idx, dt)
                for i in idx:
                    pinned[i] = tag
        print("pinned %d layers: %d fp16, %d fp32" % (len(pinned), list(pinned.values()).count("fp16"),
                                                        list(pinned.values()).count("fp32")))
    if inspect_path:
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    cuda = None
    calib = None
    if precision in ("fp16", "int8"):
        config.set_flag(trt.BuilderFlag.FP16)  # int8 also allows fp16 fallback layers
    if precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        files = []
        if calib_dir:
            files = sorted(glob.glob(os.path.join(calib_dir, "*.png")))
        have_cache = bool(calib_cache) and os.path.isfile(calib_cache)
        if not files and not have_cache:
            raise ValueError("int8 needs --calib-dir with PNGs or an existing --calib-cache")
        cuda = Cudart()
        calib = _make_calibrator(trt, files, in_shape, calib_cache, cuda, calib_algo, quantile)
        config.int8_calibrator = calib
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("build_serialized_network returned None (onnx=%s precision=%s input=%s)" %
                           (onnx_path, precision, in_shape))
    with open(engine_path, "wb") as f:
        f.write(bytearray(blob))
    if calib is not None:
        calib.release()
    secs = time.time() - t0
    if inspect_path:
        runtime = trt.Runtime(logger)
        if dla_core is not None:
            runtime.DLA_core = int(dla_core)
        eng = runtime.deserialize_cuda_engine(bytes(blob))
        if eng is None:
            raise RuntimeError("cannot deserialize %s for inspection" % engine_path)
        with open(inspect_path, "w") as f:
            f.write(eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    return secs


def _cli():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("build")
    b.add_argument("--onnx", required=True)
    b.add_argument("--engine", required=True)
    b.add_argument("--precision", required=True, choices=["fp16", "int8", "fp32"])
    b.add_argument("--calib-dir")
    b.add_argument("--calib-cache")
    b.add_argument("--workspace-mb", type=int, default=1024)
    b.add_argument("--calib-algo", default="entropy2", choices=["entropy2", "minmax", "entropy", "legacy"])
    b.add_argument("--quantile", type=float, default=0.9999, help="legacy (percentile) calibrator only")
    b.add_argument("--fp16-layers", help="e.g. head,name:/layer3/layer3.5/  (tokens: head | idx:A-B | name:SUBSTR)")
    b.add_argument("--fp32-layers", help="same grammar; applied after --fp16-layers")
    b.add_argument("--inspect", help="write per-layer engine info (JSON) to this path")
    b.add_argument("--dla-core", type=int, default=None, help="build for this DLA core (GPU fallback enabled)")
    r = sub.add_parser("bench")
    r.add_argument("--engine", required=True)
    r.add_argument("--images", required=True)
    r.add_argument("--n", type=int, default=200)
    r.add_argument("--dla-core", type=int, default=None, help="execute on this DLA core (runtime.DLA_core)")
    a = ap.parse_args()
    if a.cmd == "build":
        s = build_engine(a.onnx, a.engine, a.precision, a.calib_dir, a.calib_cache, a.workspace_mb, a.calib_algo,
                         a.fp16_layers, a.fp32_layers, a.inspect, a.quantile, a.dla_core)
        print("built %s (%s) in %.1f s" % (a.engine, a.precision, s))
        return 0
    if a.cmd == "bench":
        files = sorted(glob.glob(os.path.join(a.images, "*.png")))
        if not files:
            print("no png in %s" % a.images)
            return 1
        with TrtRunner(a.engine, a.dla_core) as rn:
            rn.infer(load_rgb_nhwc(files[0]))  # warm-up
            tp = ti = 0.0
            for k in range(a.n):
                t0 = time.time()
                x = load_rgb_nhwc(files[k % len(files)])
                t1 = time.time()
                rn.infer(x)
                t2 = time.time()
                tp += t1 - t0
                ti += t2 - t1
        print("n=%d preprocess(read+convert) %.2f ms/img, infer(H2D+exec+D2H) %.2f ms/img, total %.2f ms/img" %
              (a.n, 1e3 * tp / a.n, 1e3 * ti / a.n, 1e3 * (tp + ti) / a.n))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(_cli())
