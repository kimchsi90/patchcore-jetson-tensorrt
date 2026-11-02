#!/usr/bin/env python3
"""End-to-end PatchCore benchmark for Jetson (python3.8, numpy 1.17, cv2 4.5, TensorRT 8.5). Reuses trt_runner.py.

Pipelines (frames are uint8 RGB HWC 900x900 held in RAM, like a camera buffer; PNG decode is not timed):
  P0  CPU cv2.resize 900->256  | engine -> (m, xproj)                   | CPU numpy/cv2 post (score, top10, upsample, gaussian)
  P1  CPU cv2.resize 900->256  | engine -> (score_max, score_top10, map) post inside the engine
  P2  engine with 900x900 input: resize + post inside the engine (matmul resize)
  P3  = P1 with the backbone engine built for the DLA (post layers fall back to the GPU); selected by passing a DLA engine
  P4  CPU cv2.resize 900->256  | DLA backbone engine WITHOUT post -> (m, xproj) | CPU numpy/cv2 post (as P0)
  P5  CPU cv2.resize 900->256  | DLA backbone engine WITHOUT post -> (m, xproj) | small GPU post engine -> (score_max, score_top10, map),
      both engines on one stream in one worker thread. --p5-link device: the post engine reads the backbone's device buffers (no copy);
      --p5-link host: m/xproj go D2H into pinned memory, are copied, and go H2D into the post engine (what two separate processes would do)
  Worker spec: label=ENGINE[@PIPE]; @PIPE overrides --pipe for that worker (e.g. gpu=p1.engine@P1 next to P4/P5 DLA workers).

Modes:
  bench : timed run, N worker threads (one engine/context/stream each), --workers gpu=ENG,dla0=ENG,dla1=ENG
  acc   : run the labelled test frames once through the first worker and save outputs (npz) for scoring on the PC
  check : P5 engines only: compare the post engine against host_post() on the DLA engine's own (m, xproj) for all frames
  micro : single-thread cost of resize / uint8->float32 copy / pinned H2D (no engine needed except --engine for the H2D buffer sizes)
"""
import argparse
import csv
import json
import os
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trt_runner import Cudart, TrtRunner  # noqa: E402

TEMP_LIMIT_MC = int(os.environ.get("E2E_TEMP_LIMIT_MC", "82000"))  # env override is for testing the abort path only
INTERPS = {"area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR}
BORDER = 0  # >0: engine built from a fold_norm.py --border graph; the input buffer carries a BORDER-pixel frame prefilled with mean*255
MEAN255 = np.array([0.485, 0.456, 0.406], np.float32) * 255.0


# ----------------------------------------------------------------------------- system probes
def read_int(path):
    with open(path) as f:
        s = f.read().strip()
    if not s.lstrip("-").isdigit():
        raise ValueError("non-integer content %r in %s" % (s, path))
    return int(s)


def thermal_zones():
    """{type: path} of the CPU and GPU thermal zones."""
    out = {}
    base = "/sys/class/thermal"
    for d in sorted(os.listdir(base)):
        if not d.startswith("thermal_zone"):
            continue
        with open(os.path.join(base, d, "type")) as f:
            t = f.read().strip()
        if t in ("CPU-therm", "GPU-therm"):
            out[t] = os.path.join(base, d, "temp")
    return out


def max_temp_mc(zones):
    return max(read_int(p) for p in zones.values())


def read_proc_stat():
    """(busy_jiffies, total_jiffies) of the aggregate cpu line."""
    with open("/proc/stat") as f:
        parts = f.readline().split()
    if parts[0] != "cpu":
        raise ValueError("unexpected /proc/stat first line: %r" % parts[:3])
    v = [int(x) for x in parts[1:9]]
    idle = v[3] + v[4]  # idle + iowait
    total = sum(v)
    return total - idle, total


def sys_info():
    info = {}
    with open("/sys/devices/system/cpu/online") as f:
        info["cpus_online"] = f.read().strip()
    info["nproc"] = os.cpu_count()
    freqs = []
    for i in range(info["nproc"]):
        p = "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq" % i
        if os.path.isfile(p):
            freqs.append(read_int(p) // 1000)
    info["cpu_mhz_now"] = freqs
    for g in ("/sys/devices/gpu.0/devfreq/17000000.gv11b/cur_freq", "/sys/devices/17000000.gv11b/devfreq/17000000.gv11b/cur_freq"):
        if os.path.isfile(g):
            info["gpu_mhz_now"] = read_int(g) // 1000000
            break
    try:
        with open("/etc/nv_tegra_release") as f:
            info["l4t"] = f.readline().strip()
    except IOError as e:
        info["l4t"] = "n/a (%s)" % e
    return info


# ----------------------------------------------------------------------------- post-processing on the host (P0)
def host_post(m, x, topk=10, sigma=4.0):
    """m [1,1,h,w], x [1,C,h,w] -> (score_max, score_top10, map [S,S] float32). Same math as graph_opt.host_post (float32)."""
    if m.ndim != 4 or x.ndim != 4 or m.shape[2:] != x.shape[2:]:
        raise ValueError("bad head shapes m=%s x=%s dtypes %s %s" % (m.shape, x.shape, m.dtype, x.dtype))
    xf = x[0].astype(np.float32)
    sq = np.einsum("chw,chw->hw", xf, xf)
    s = np.sqrt(np.maximum(sq - m[0, 0].astype(np.float32), 0.0))
    flat = s.ravel()
    smax = float(flat.max())
    top = float(np.partition(flat, flat.size - topk)[flat.size - topk:].mean())
    f = 8
    up = cv2.resize(s, (s.shape[1] * f, s.shape[0] * f), interpolation=cv2.INTER_NEAREST)
    mp = cv2.GaussianBlur(up, (33, 33), sigma, borderType=cv2.BORDER_REFLECT_101)  # radius int(4*sigma+.5)=16 like scipy
    return smax, top, mp


# ----------------------------------------------------------------------------- pipeline
class PostRunner(TrtRunner):
    """TrtRunner for the post-only engine, which has two inputs (m, xproj). Same buffers as TrtRunner; no single-input check."""

    def __init__(self, engine_path):
        import tensorrt as trt
        if not os.path.isfile(engine_path):
            raise IOError("engine file not found: %s" % engine_path)
        self.trt = trt
        self.cuda = Cudart()
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(engine_path, "rb") as f:
            data = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(data)
        if self.engine is None:
            raise RuntimeError("deserialize_cuda_engine failed for %s (%d bytes, GPU)" % (engine_path, len(data)))
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("create_execution_context failed for %s (GPU memory?)" % engine_path)
        self.stream = self.cuda.stream_create()
        self.names, self.is_input, self.dev, self.host, self.host_ptr, self.shapes = [], [], [], [], [], []
        self._freed = False
        for i in range(self.engine.num_bindings):
            shape = tuple(self.engine.get_binding_shape(i))
            name = self.engine.get_binding_name(i)
            if any(d < 0 for d in shape):
                raise ValueError("dynamic shape %s for binding %s is not supported" % (shape, name))
            dt = np.dtype({trt.float32: np.float32, trt.float16: np.float16, trt.int32: np.int32}[self.engine.get_binding_dtype(i)])
            nbytes = int(np.prod(shape)) * dt.itemsize
            import ctypes
            ptr = self.cuda.host_alloc(nbytes)
            self.host.append(np.frombuffer((ctypes.c_char * nbytes).from_address(ptr), dtype=dt).reshape(shape))
            if BORDER and self.engine.binding_is_input(i) and len(shape) == 4 and shape[3] == 3:
                self.host[-1][:] = MEAN255  # mean-filled frame; only the interior is rewritten per frame
            self.host_ptr.append(ptr)
            self.names.append(name)
            self.is_input.append(self.engine.binding_is_input(i))
            self.dev.append(self.cuda.malloc(nbytes))
            self.shapes.append(shape)
        self.bindings = list(self.dev)


class Pipe(object):
    """One worker: a runner plus the stage code. Not thread safe; use one Pipe per thread."""

    def __init__(self, kind, engine, dla_core, interp, nb_stream=True, post_engine=None, link="device"):
        self.kind = kind  # P0 | P1 | P2 | P4 | P5
        self.engine = engine
        self.dla_core = dla_core
        self.r = TrtRunner(engine, dla_core)
        r = self.r
        self.out_idx = dict((r.names[i], i) for i, f in enumerate(r.is_input) if not f)
        if kind in ("P0", "P4", "P5"):
            need = ("m", "xproj")
        else:
            need = ("score_max", "score_top10", "map")
        for n in need:
            if n not in self.out_idx:
                raise ValueError("%s: engine %s lacks output %s; bindings=%s" % (kind, engine, n, [
                    (r.names[i], r.shapes[i], str(r.host[i].dtype)) for i in range(len(r.names))]))
        s = r.in_shape
        if kind == "P2":
            if s[1] != 900 or s[2] != 900:
                raise ValueError("P2 needs a 900x900 input engine, got %s" % (s,))
        else:
            if s[1] != s[2]:
                raise ValueError("expected square network input, got %s" % (s,))
            self.small = np.empty((s[1] - 2 * BORDER, s[2] - 2 * BORDER, 3), np.uint8)
        self.interp = INTERPS[interp]
        if nb_stream:  # non-blocking stream: does not synchronize with the legacy default stream
            import ctypes
            st = ctypes.c_void_p()
            r.cuda.check(r.cuda.lib.cudaStreamCreateWithFlags(ctypes.byref(st), 1), "cudaStreamCreateWithFlags(nonblocking)")
            r.stream = st.value
        self.t = dict(pre=0.0, copyin=0.0, enq=0.0, gpu=0.0, post=0.0, n=0)
        self.r2 = None
        if kind == "P5":
            self._init_p5(post_engine, link)

    def _init_p5(self, post_engine, link):
        """Second runner (GPU post engine) on the backbone's stream; in 'device' link mode it reads the backbone's output buffers."""
        if link not in ("device", "host"):
            raise ValueError("p5 link must be device|host, got %r" % (link,))
        if not post_engine:
            raise ValueError("P5 needs --post-engine")
        r = self.r
        self.link = link
        self.t["mid"] = 0.0
        self.r2 = PostRunner(post_engine)
        r2 = self.r2
        self.r2_own_stream = r2.stream
        r2.stream = r.stream  # same stream: the post engine starts when the backbone has finished
        self.p_in = {}
        for n in ("m", "xproj"):
            if n not in r2.names:
                raise ValueError("post engine %s lacks input %s; bindings=%s" % (post_engine, n, r2.names))
            j = r2.names.index(n)
            i = self.out_idx[n]
            if not r2.is_input[j] or r2.shapes[j] != r.shapes[i] or r2.host[j].dtype != r.host[i].dtype:
                raise ValueError("post input %s %s %s (input=%s) does not match backbone output %s %s" % (
                    n, r2.shapes[j], r2.host[j].dtype, r2.is_input[j], r.shapes[i], r.host[i].dtype))
            self.p_in[n] = j
        self.p_out = {}
        for n in ("score_max", "score_top10", "map"):
            if n not in r2.names or r2.is_input[r2.names.index(n)]:
                raise ValueError("post engine %s lacks output %s; bindings=%s" % (post_engine, n, r2.names))
            self.p_out[n] = r2.names.index(n)
        self.r2.bindings = list(r2.dev)
        if link == "device":
            for n in ("m", "xproj"):
                self.r2.bindings[self.p_in[n]] = r.dev[self.out_idx[n]]

    def describe(self):
        r = self.r
        if self.r2 is not None:
            r2 = self.r2
            return {"engine": self.engine, "dla_core": self.dla_core, "in": [r.names[r.in_idx], list(r.in_shape), str(r.in_dtype)],
                    "out": [[r.names[i], list(r.shapes[i]), str(r.host[i].dtype)] for i in self.out_idx.values()], "p5_link": self.link,
                    "post_engine": [[r2.names[i], list(r2.shapes[i]), str(r2.host[i].dtype), "in" if r2.is_input[i] else "out"] for i in range(len(r2.names))]}
        return {"engine": self.engine, "dla_core": self.dla_core, "in": [r.names[r.in_idx], list(r.in_shape), str(r.in_dtype)],
                "out": [[r.names[i], list(r.shapes[i]), str(r.host[i].dtype)] for i in self.out_idx.values()]}

    def process_p5(self, frame):
        """P4's front end + DLA backbone, then the GPU post engine on the same stream. Same return value as process()."""
        r, r2, c = self.r, self.r2, self.r.cuda
        t0 = time.perf_counter()
        cv2.resize(frame, (self.small.shape[1], self.small.shape[0]), dst=self.small, interpolation=self.interp)
        t1 = time.perf_counter()
        if BORDER:
            r.host[r.in_idx][:, BORDER:-BORDER, BORDER:-BORDER] = self.small[None]
        else:
            np.copyto(r.host[r.in_idx], self.small[None])
        t2 = time.perf_counter()
        c.memcpy_h2d_async(r.dev[r.in_idx], r.host_ptr[r.in_idx], r.host[r.in_idx].nbytes, r.stream)
        te0 = time.perf_counter()
        if not r.context.execute_async_v2(r.bindings, r.stream):
            raise RuntimeError("backbone execute_async_v2 failed (input %s %s, engine %s, dla=%s)" % (r.in_shape, r.in_dtype, self.engine, self.dla_core))
        te1 = time.perf_counter()
        tm = None
        if self.link == "host":
            for n in ("m", "xproj"):
                i = self.out_idx[n]
                c.memcpy_d2h_async(r.host_ptr[i], r.dev[i], r.host[i].nbytes, r.stream)
            c.stream_sync(r.stream)
            tm = time.perf_counter()
            for n in ("m", "xproj"):
                j = self.p_in[n]
                np.copyto(r2.host[j], r.host[self.out_idx[n]])
                c.memcpy_h2d_async(r2.dev[j], r2.host_ptr[j], r2.host[j].nbytes, r.stream)
        te2 = time.perf_counter()
        if not r2.context.execute_async_v2(r2.bindings, r.stream):
            raise RuntimeError("post execute_async_v2 failed (link %s, engine shapes %s)" % (self.link, [tuple(x) for x in r2.shapes]))
        te3 = time.perf_counter()
        for n, j in self.p_out.items():
            c.memcpy_d2h_async(r2.host_ptr[j], r2.dev[j], r2.host[j].nbytes, r.stream)
        c.stream_sync(r.stream)
        t3 = time.perf_counter()
        res = (float(r2.host[self.p_out["score_max"]][0]), float(r2.host[self.p_out["score_top10"]][0]), r2.host[self.p_out["map"]][0, 0])
        t = self.t
        t["pre"] += t1 - t0
        t["copyin"] += t2 - t1
        t["gpu"] += t3 - t2  # backbone + (host link: copies) + post engine, all device-side waiting included
        t["enq"] += (te1 - te0) + (te3 - te2)
        t["mid"] += (tm - t2) if tm is not None else 0.0  # backbone incl. its D2H, host link only (the device link has no intermediate sync)
        t["n"] += 1
        return res

    def process(self, frame):
        """frame uint8 [900,900,3] RGB -> (score_max, score_top10, map [256,256] float32 view or copy)."""
        if self.kind == "P5":
            return self.process_p5(frame)
        r, c = self.r, self.r.cuda
        t0 = time.perf_counter()
        if self.kind == "P2":
            src = frame
        else:
            cv2.resize(frame, (self.small.shape[1], self.small.shape[0]), dst=self.small, interpolation=self.interp)
            src = self.small
        t1 = time.perf_counter()
        if BORDER and self.kind != "P2":
            r.host[r.in_idx][:, BORDER:-BORDER, BORDER:-BORDER] = src[None]  # border was prefilled with mean*255
        else:
            np.copyto(r.host[r.in_idx], src[None])  # uint8 -> float32 into the pinned buffer
        t2 = time.perf_counter()
        c.memcpy_h2d_async(r.dev[r.in_idx], r.host_ptr[r.in_idx], r.host[r.in_idx].nbytes, r.stream)
        te0 = time.perf_counter()
        ok = r.context.execute_async_v2(r.bindings, r.stream)
        te1 = time.perf_counter()
        if not ok:
            raise RuntimeError("execute_async_v2 failed (input %s %s, engine %s, dla=%s)" % (r.in_shape, r.in_dtype, self.engine, self.dla_core))
        for i in self.out_idx.values():
            c.memcpy_d2h_async(r.host_ptr[i], r.dev[i], r.host[i].nbytes, r.stream)
        c.stream_sync(r.stream)
        t3 = time.perf_counter()
        if self.kind in ("P0", "P4"):
            res = host_post(r.host[self.out_idx["m"]], r.host[self.out_idx["xproj"]])
        else:
            res = (float(r.host[self.out_idx["score_max"]][0]), float(r.host[self.out_idx["score_top10"]][0]),
                   r.host[self.out_idx["map"]][0, 0])
        t4 = time.perf_counter()
        t = self.t
        t["pre"] += t1 - t0
        t["copyin"] += t2 - t1
        t["gpu"] += t3 - t2
        t["enq"] += te1 - te0  # time inside execute_async_v2 (part of "gpu")
        t["post"] += t4 - t3
        t["n"] += 1
        return res

    def exec_staged(self, src_ptr):
        """Pipelined mode accelerator stage only: H2D from the pinned host buffer at src_ptr, execute, D2H of all outputs, sync.
        Resize and the CPU post are done elsewhere. Returns the seconds spent here. Not used by process()."""
        if self.kind not in ("P1", "P2", "P4"):
            raise ValueError("pipelined mode supports P1/P2/P4, got %s (engine %s)" % (self.kind, self.engine))
        r, c = self.r, self.r.cuda
        t0 = time.perf_counter()
        c.memcpy_h2d_async(r.dev[r.in_idx], src_ptr, r.host[r.in_idx].nbytes, r.stream)
        if not r.context.execute_async_v2(r.bindings, r.stream):
            raise RuntimeError("execute_async_v2 failed (input %s %s, engine %s, dla=%s)" % (r.in_shape, r.in_dtype, self.engine, self.dla_core))
        for i in self.out_idx.values():
            c.memcpy_d2h_async(r.host_ptr[i], r.dev[i], r.host[i].nbytes, r.stream)
        c.stream_sync(r.stream)
        return time.perf_counter() - t0

    def staged_outputs(self, copy):
        """After exec_staged: P1/P2 -> (score_max, score_top10, map); P4 -> (m, xproj) for host_post. copy=True detaches from the reused buffers."""
        r = self.r
        if self.kind == "P4":
            m, x = r.host[self.out_idx["m"]], r.host[self.out_idx["xproj"]]
            return (m.copy(), x.copy()) if copy else (m, x)
        mp = r.host[self.out_idx["map"]][0, 0]
        return (float(r.host[self.out_idx["score_max"]][0]), float(r.host[self.out_idx["score_top10"]][0]), mp.copy() if copy else mp)

    def close(self):
        if self.r2 is not None:
            self.r2.stream = self.r2_own_stream
            self.r2.close()
        self.r.close()


# ----------------------------------------------------------------------------- data
def load_frames(test_dir, labels_csv, limit=None):
    rows = list(csv.reader(open(labels_csv)))[1:]
    if limit:
        rows = rows[:limit]
    frames = []
    for name, _ in rows:
        p = os.path.join(test_dir, name + ".png")
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError("cannot read %s" % p)
        if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
            raise ValueError("frame %s has shape %s dtype %s, expected (H,W,3) uint8" % (p, bgr.shape, bgr.dtype))
        if frames and bgr.shape != frames[0].shape:
            raise ValueError("frame %s has shape %s, the first frame has %s (mixed sizes in one test dir)" % (p, bgr.shape, frames[0].shape))
        frames.append(np.ascontiguousarray(bgr[:, :, ::-1]))
    return [n for n, _ in rows], frames


def parse_workers(spec):
    """'gpu=a.engine,dla0=b.engine@P4' -> [(label, engine, dla_core or None, pipe override or None)]"""
    out = []
    for tok in spec.split(","):
        lab, _, eng = tok.partition("=")
        eng, _, pp = eng.partition("@")
        if pp and pp not in ("P0", "P1", "P2", "P4", "P5"):
            raise ValueError("bad pipe override %r in worker spec %r" % (pp, tok))
        if not eng:
            raise ValueError("bad worker spec %r (want gpu=ENGINE | dla0=ENGINE | dla1=ENGINE)" % tok)
        if lab.startswith("gpu"):
            core = None
        elif len(lab) >= 4 and lab.startswith("dla") and lab[3] in "01":
            # dla0, dla0b, ... would be several workers on one DLA core, BUT this board allows only 2 DLA execution contexts in total
            # (the 3rd fails with "Failed to deserialize DLA loadable", NvMap error 12), so use at most dla0 and dla1.
            core = int(lab[3])
        else:
            raise ValueError("worker label must be gpuN, dla0[x] or dla1[x], got %r" % lab)
        out.append((lab, eng, core, pp or None))
    return out


def placement(pipe, workers):
    """Which stage runs on which device, per worker."""
    out = []
    default_pipe = pipe
    for lab, _, core, pp in workers:
        pipe = pp or default_pipe
        dev = "DLA%d" % core if core is not None else "GPU"
        if pipe == "P4":
            st = {"resize+uint8->fp32": "CPU", "input scale + backbone + projection (m, xproj)": dev + (" (+GPU fallback of the input scale, if any)" if core is not None else ""),
                  "score/top10/upsample/blur": "CPU"}
        elif pipe == "P5":
            st = {"resize+uint8->fp32": "CPU", "input scale + backbone + projection (m, xproj)": dev,
                  "score/top10/upsample/blur (post engine, fp32)": "GPU",
                  "m/xproj hand-over": "device buffers shared (no copy)" if _P5_LINK[0] == "device" else "D2H + CPU copy + H2D"}
        elif pipe == "P0":
            st = {"resize+uint8->fp32": "CPU", "backbone+head (m, xproj)": dev, "score/top10/upsample/blur": "CPU"}
        elif pipe == "P2":
            st = {"uint8->fp32 copy": "CPU", "H2D 9.7MB": "copy engine", "resize 900->256 (2 MatMul)": "GPU", "backbone+head": dev,
                  "post (score/top10/upsample/blur)": "GPU"}
        else:
            st = {"resize+uint8->fp32": "CPU", "pre_transpose+scale operands": dev, "pre_sub/pre_div": "GPU" if core is None else "GPU (fallback)",
                  "backbone+projection": dev, "head ReduceMax, post (score/top10/upsample/blur)": "GPU" if core is None else "GPU (fallback)"}
        out.append({"worker": lab, "stages": st})
    return out


def pct(a, q):
    return float(np.percentile(np.asarray(a, np.float64), q))


_P5_LINK = ["device"]  # set from --p5-link, only used to label the placement


# ----------------------------------------------------------------------------- modes
def run_bench(a):
    names, frames = load_frames(a.test_dir, a.labels)
    zones = thermal_zones()
    if not zones:
        raise RuntimeError("no CPU-therm/GPU-therm thermal zones found")
    workers = parse_workers(a.workers)
    _P5_LINK[0] = a.p5_link
    pipes = [Pipe(pp or a.pipe, eng, core, a.interp, not a.blocking_stream, a.post_engine, a.p5_link) for _, eng, core, pp in workers]
    log = {"placement": placement(a.pipe, workers), "nb_stream": not a.blocking_stream, "config": a.name, "pipe": a.pipe, "interp": a.interp, "frames": len(frames), "duration_req_s": a.duration, "warmup_s": a.warmup,
           "workers": [dict(label=w[0], **p.describe()) for w, p in zip(workers, pipes)], "sysinfo_start": sys_info(),
           "temp_start_mc": {k: read_int(v) for k, v in zones.items()}, "status": "OK"}
    measuring = threading.Event()
    stop = threading.Event()
    lats = [[] for _ in pipes]
    errors = []

    def loop(k):
        p, i = pipes[k], k * 7
        try:
            while not stop.is_set():
                t0 = time.perf_counter()
                p.process(frames[i % len(frames)])
                t1 = time.perf_counter()
                i += 1
                if measuring.is_set():
                    lats[k].append(t1 - t0)
        except (RuntimeError, ValueError, IOError) as e:
            errors.append("worker %d (%s): %r" % (k, workers[k][0], e))
            stop.set()

    ths = [threading.Thread(target=loop, args=(k,)) for k in range(len(pipes))]
    for th in ths:
        th.start()
    peak = 0

    def watch(seconds):
        nonlocal peak
        end = time.time() + seconds
        while time.time() < end and not stop.is_set():
            tm = max_temp_mc(zones)
            peak = max(peak, tm)
            if tm >= TEMP_LIMIT_MC:
                log["status"] = "ABORTED over temperature %d mC (limit %d)" % (tm, TEMP_LIMIT_MC)
                stop.set()
                return
            time.sleep(0.5)

    watch(a.warmup)
    for p in pipes:  # discard warm-up stage timings
        p.t = dict((k, 0 if k == "n" else 0.0) for k in p.t)
    s0 = read_proc_stat()
    c0 = os.times()
    tw0 = time.perf_counter()
    measuring.set()
    watch(a.duration)
    measuring.clear()
    tw1 = time.perf_counter()
    s1 = read_proc_stat()
    c1 = os.times()
    snap = [dict(p.t) for p in pipes]
    stop.set()
    for th in ths:
        th.join(60)
    wall = tw1 - tw0
    allat = [x for l in lats for x in l]
    busy, tot = s1[0] - s0[0], s1[1] - s0[1]
    proc_cpu = (c1.user + c1.system) - (c0.user + c0.system)
    log.update({
        "errors": errors, "measured_wall_s": round(wall, 3), "frames_done": len(allat),
        "fps": round(len(allat) / wall, 2) if wall > 0 else 0.0,
        "per_worker_fps": [round(len(l) / wall, 2) for l in lats],
        "latency_ms": {"p50": round(1e3 * pct(allat, 50), 2), "p99": round(1e3 * pct(allat, 99), 2), "mean": round(1e3 * float(np.mean(allat)), 2)} if allat else None,
        "per_worker_latency_ms_p50": [round(1e3 * pct(l, 50), 2) if l else None for l in lats],
        "cpu_util_system_pct": round(100.0 * busy / tot, 1) if tot else None,
        "cpu_process_cores_busy": round(proc_cpu / wall, 2),
        "stage_ms_mean_per_frame": [dict((k, round(1e3 * s[k] / max(s["n"], 1), 3)) for k in sorted(s) if k != "n") for s in snap],
        "temp_peak_mc": peak, "sysinfo_end": sys_info(),
    })
    for p in pipes:
        p.close()
    return log


PIPELINED_NOTE = ("stage-decoupled: R feeder threads (resize or uint8->fp32 copy) fill pinned host buffers -> one shared bounded ready queue -> "
                  "accelerator threads (H2D + execute + D2H) -> [P4: optional post thread pool | inline CPU post]")


def run_pipelined(a, acc=False):
    """Stage-decoupled run. bench: timed (a.warmup + a.duration). acc: every labelled frame once, outputs stored by frame id.

    Latency = from the moment a feeder starts on a frame (a free buffer is held, frame taken from the in-RAM list) to the result being ready.
    It includes the ready-queue wait (ready_wait_ms_mean = from the end of the feeder's work, incl. a blocked put on a full queue, to dequeue). At saturation the latency is therefore about queue_depth x accelerator time.
    """
    import queue
    import itertools
    names, frames = load_frames(a.test_dir, a.labels)
    nf = len(frames)
    workers = parse_workers(a.workers)
    if acc:
        workers = workers[: a.acc_workers] if a.acc_workers else workers
    zones = thermal_zones()
    if not zones:
        raise RuntimeError("no CPU-therm/GPU-therm thermal zones found")
    if a.cv_threads is not None:
        cv2.setNumThreads(a.cv_threads)
    _P5_LINK[0] = a.p5_link
    pipes = [Pipe(pp or a.pipe, eng, core, a.interp, not a.blocking_stream, None, a.p5_link) for _, eng, core, pp in workers]
    kinds = set(p.kind for p in pipes)
    for p in pipes:
        if p.kind not in ("P1", "P2", "P4"):
            raise ValueError("pipelined mode supports P1/P2/P4, got %s for %s" % (p.kind, p.engine))
    if ("P2" in kinds) and len(kinds) > 1:
        raise ValueError("pipelined mode cannot mix P2 (900x900 input) with %s in one run (one shared buffer pool)" % sorted(kinds - {"P2"}))
    shape0, dt0 = pipes[0].r.in_shape, pipes[0].r.in_dtype
    for p in pipes:
        if p.r.in_shape != shape0 or p.r.in_dtype != dt0 or dt0 != np.float32:
            raise ValueError("input binding mismatch across workers: %s %s vs %s %s (engine %s)" % (p.r.in_shape, p.r.in_dtype, shape0, dt0, p.engine))
    full = "P2" in kinds
    nw, R, depth = len(pipes), max(1, a.resize_threads), max(1, a.queue_depth)
    cuda = pipes[0].r.cuda
    nbuf = depth + nw + R
    nbytes = int(np.prod(shape0)) * 4
    import ctypes
    ptrs = [cuda.host_alloc(nbytes) for _ in range(nbuf)]
    views = [np.frombuffer((ctypes.c_char * nbytes).from_address(p), dtype=np.float32).reshape(shape0) for p in ptrs]
    if BORDER:
        for v in views:
            v[:] = MEAN255  # border stays untouched: feeders only write the interior
    free_q, ready_q = queue.Queue(), queue.Queue(maxsize=depth)
    post_q = queue.Queue(maxsize=max(4, 2 * depth)) if a.post_threads > 0 else None
    for b in range(nbuf):
        free_q.put(b)
    cnt = itertools.count()
    stop, measuring = threading.Event(), threading.Event()
    errors, results = [], {}
    lats = [[] for _ in pipes]
    qw = [[] for _ in pipes]
    interp = INTERPS[a.interp]
    H, W = shape0[1], shape0[2]
    fst = [dict(resize=0.0, copy=0.0, free_wait=0.0, put_wait=0.0, n=0) for _ in range(R)]
    ast = [dict(wait=0.0, dev=0.0, post=0.0, post_put_wait=0.0, n=0) for _ in range(nw)]
    pst = [dict(wait=0.0, post=0.0, n=0) for _ in range(max(a.post_threads, 0))]
    log = {"placement": placement(a.pipe, workers), "pipelined": {"note": PIPELINED_NOTE, "resize_threads": R, "post_threads": a.post_threads, "queue_depth": depth,
                                                                    "buffers": nbuf, "buffer_bytes": nbytes, "cv_threads": cv2.getNumThreads(), "queue": "shared"},
           "nb_stream": not a.blocking_stream, "config": a.name, "pipe": a.pipe, "interp": a.interp, "frames": nf, "duration_req_s": a.duration, "warmup_s": a.warmup,
           "workers": [dict(label=w[0], **p.describe()) for w, p in zip(workers, pipes)], "sysinfo_start": sys_info(),
           "temp_start_mc": {k: read_int(v) for k, v in zones.items()}, "status": "OK"}

    def record(k, fid, t_av, t_enq, t_deq, t_done, res):
        if acc:
            results[fid] = res
        if measuring.is_set():
            lats[k].append(t_done - t_av)
            qw[k].append(t_deq - t_enq)

    def feeder(j):
        s = fst[j]
        small = None if full else np.empty((H - 2 * BORDER, W - 2 * BORDER, 3), np.uint8)
        try:
            while not stop.is_set():
                tw = time.perf_counter()
                try:
                    b = free_q.get(timeout=0.2)
                except queue.Empty:
                    s["free_wait"] += time.perf_counter() - tw
                    continue
                s["free_wait"] += time.perf_counter() - tw
                fid = next(cnt)
                if acc and fid >= nf:
                    free_q.put(b)
                    return
                t_av = time.perf_counter()
                fr = frames[fid % nf]
                if full:
                    np.copyto(views[b][0], fr)
                    t1 = time.perf_counter()
                    s["copy"] += t1 - t_av
                else:
                    cv2.resize(fr, (W - 2 * BORDER, H - 2 * BORDER), dst=small, interpolation=interp)
                    t1 = time.perf_counter()
                    if BORDER:
                        np.copyto(views[b][0][BORDER:-BORDER, BORDER:-BORDER], small)
                    else:
                        np.copyto(views[b][0], small)
                    s["resize"] += t1 - t_av
                    s["copy"] += time.perf_counter() - t1
                tp = time.perf_counter()
                while not stop.is_set():
                    try:
                        ready_q.put((fid, t_av, b, time.perf_counter()), timeout=0.2)
                        break
                    except queue.Full:
                        continue
                s["put_wait"] += time.perf_counter() - tp
                s["n"] += 1
        except (RuntimeError, ValueError, IOError, cv2.error) as e:
            errors.append("feeder %d: %r (frame shape %s dtype %s)" % (j, e, frames[0].shape, frames[0].dtype))
            stop.set()

    def accel(k):
        p, s = pipes[k], ast[k]
        try:
            while not stop.is_set():
                tw = time.perf_counter()
                try:
                    fid, t_av, b, t_enq = ready_q.get(timeout=0.2)
                except queue.Empty:
                    s["wait"] += time.perf_counter() - tw
                    continue
                t_deq = time.perf_counter()
                s["wait"] += t_deq - tw
                s["dev"] += p.exec_staged(ptrs[b])
                free_q.put(b)  # H2D has finished (stream synchronized)
                if p.kind == "P4":
                    if post_q is not None:
                        out = p.staged_outputs(True)
                        tp = time.perf_counter()
                        while not stop.is_set():
                            try:
                                post_q.put((k, fid, t_av, t_enq, t_deq, out), timeout=0.2)
                                break
                            except queue.Full:
                                continue
                        s["post_put_wait"] += time.perf_counter() - tp
                    else:
                        tp = time.perf_counter()
                        m, x = p.staged_outputs(False)
                        res = host_post(m, x)
                        td = time.perf_counter()
                        s["post"] += td - tp
                        record(k, fid, t_av, t_enq, t_deq, td, res)
                else:
                    res = p.staged_outputs(acc)
                    record(k, fid, t_av, t_enq, t_deq, time.perf_counter(), res)
                s["n"] += 1
        except (RuntimeError, ValueError, IOError) as e:
            errors.append("accel worker %d (%s): %r" % (k, workers[k][0], e))
            stop.set()

    def poster(j):
        s = pst[j]
        try:
            while not stop.is_set():
                tw = time.perf_counter()
                try:
                    k, fid, t_av, t_enq, t_deq, (m, x) = post_q.get(timeout=0.2)
                except queue.Empty:
                    s["wait"] += time.perf_counter() - tw
                    continue
                t0 = time.perf_counter()
                s["wait"] += t0 - tw
                res = host_post(m, x)
                td = time.perf_counter()
                s["post"] += td - t0
                s["n"] += 1
                record(k, fid, t_av, t_enq, t_deq, td, res)
        except (RuntimeError, ValueError, IOError, cv2.error) as e:
            errors.append("post thread %d: %r" % (j, e))
            stop.set()

    ths = [threading.Thread(target=feeder, args=(j,)) for j in range(R)]
    ths += [threading.Thread(target=accel, args=(k,)) for k in range(nw)]
    ths += [threading.Thread(target=poster, args=(j,)) for j in range(len(pst))]
    for th in ths:
        th.start()
    peak = 0
    occ = dict(ready=[], free=[], post=[])

    def snap():
        return ([dict(s) for s in fst], [dict(s) for s in ast], [dict(s) for s in pst])

    def watch(seconds, sample):
        nonlocal peak
        end = time.time() + seconds
        tick = 0
        while time.time() < end and not stop.is_set():
            if tick % 10 == 0:
                tm = max_temp_mc(zones)
                peak = max(peak, tm)
                if tm >= TEMP_LIMIT_MC:
                    log["status"] = "ABORTED over temperature %d mC (limit %d)" % (tm, TEMP_LIMIT_MC)
                    stop.set()
                    return
            if sample:
                occ["ready"].append(ready_q.qsize())
                occ["free"].append(free_q.qsize())
                if post_q is not None:
                    occ["post"].append(post_q.qsize())
            tick += 1
            time.sleep(0.05)

    if acc:
        deadline = time.time() + 120
        measuring.clear()
        while len(results) < nf and not stop.is_set() and time.time() < deadline:
            watch(0.5, False)
        if len(results) < nf and not errors:
            errors.append("acc timeout: %d of %d frames done" % (len(results), nf))
        stop.set()
        for th in ths:
            th.join(60)
        order = sorted(results)
        if order != list(range(nf)):
            errors.append("acc: frame ids %s... do not cover 0..%d" % (order[:5], nf - 1))
        else:
            sm = np.array([results[i][0] for i in order], np.float64)
            tp10 = np.array([results[i][1] for i in order], np.float64)
            mp = np.stack([np.array(results[i][2], np.float32) for i in order])
            np.savez(a.out, names=np.array(names), score_max=sm, score_top10=tp10, maps=mp)
            print("saved %s n=%d map shape=%s dtype=%s" % (a.out, nf, mp.shape, mp.dtype))
        for p in pipes:
            p.close()
        for q in ptrs:
            cuda.free_host(q)
        return {"config": a.name, "status": "OK" if not errors else "FAILED", "errors": errors, "pipelined": log["pipelined"], "pipe": a.pipe,
                "interp": a.interp, "n": nf, "workers": [w[0] for w in workers]}

    watch(a.warmup, False)
    s0 = read_proc_stat()
    c0 = os.times()
    sn0 = snap()
    tw0 = time.perf_counter()
    measuring.set()
    watch(a.duration, True)
    measuring.clear()
    tw1 = time.perf_counter()
    s1 = read_proc_stat()
    c1 = os.times()
    sn1 = snap()
    stop.set()
    for th in ths:
        th.join(60)
    for p in pipes:
        p.close()
    for q in ptrs:
        cuda.free_host(q)
    wall = tw1 - tw0
    allat = [x for l in lats for x in l]
    allq = [x for l in qw for x in l]
    busy, tot = s1[0] - s0[0], s1[1] - s0[1]
    proc_cpu = (c1.user + c1.system) - (c0.user + c0.system)

    def delta(i):
        return [dict((k, sn1[i][t][k] - sn0[i][t][k]) for k in sn1[i][t]) for t in range(len(sn1[i]))]
    df, da, dp = delta(0), delta(1), delta(2)
    nfeed = max(sum(d["n"] for d in df), 1)

    def frac(rows, key):
        return round(sum(d[key] for d in rows) / (wall * max(len(rows), 1)), 3) if rows else None
    r_occ = np.asarray(occ["ready"], np.float64) if occ["ready"] else np.zeros(1)
    f_occ = np.asarray(occ["free"], np.float64) if occ["free"] else np.zeros(1)
    qinfo = {
        "ready_mean": round(float(r_occ.mean()), 2), "ready_empty_pct": round(100.0 * float((r_occ == 0).mean()), 1),
        "ready_full_pct": round(100.0 * float((r_occ >= depth).mean()), 1), "free_buffers_mean": round(float(f_occ.mean()), 2),
        "post_mean": round(float(np.mean(occ["post"])), 2) if occ["post"] else None,
        "post_full_pct": round(100.0 * float((np.asarray(occ["post"]) >= post_q.maxsize).mean()), 1) if occ["post"] else None,
        "accel_wait_frac": frac(da, "wait"), "accel_busy_frac": frac(da, "dev"), "accel_inline_post_frac": frac(da, "post"),
        "accel_post_put_wait_frac": frac(da, "post_put_wait") if post_q is not None else None,
        "feeder_busy_frac": round(sum(d["resize"] + d["copy"] for d in df) / (wall * R), 3), "feeder_put_wait_frac": frac(df, "put_wait"),
        "post_thread_busy_frac": frac(dp, "post") if dp else None, "post_thread_wait_frac": frac(dp, "wait") if dp else None,
        "ready_wait_ms_mean": round(1e3 * float(np.mean(allq)), 2) if allq else None,
    }
    # heuristic verdict from the fractions above
    aw, fp = qinfo["accel_wait_frac"], qinfo["feeder_put_wait_frac"]
    if post_q is not None and (qinfo["accel_post_put_wait_frac"] or 0) > 0.1:
        verdict = "post stage (accelerators block on a full post queue)"
    elif aw > 0.15 and fp < 0.1:
        verdict = "feed stage (resize/copy): accelerators wait on an empty ready queue"
    elif fp > 0.15 and aw < 0.1:
        verdict = "accelerators: feeders wait on a full ready queue"
    else:
        verdict = "mixed (accelerator wait %.2f, feeder wait %.2f)" % (aw, fp)
    qinfo["bottleneck_heuristic"] = verdict
    log.update({
        "errors": errors, "measured_wall_s": round(wall, 3), "frames_done": len(allat),
        "fps": round(len(allat) / wall, 2) if wall > 0 else 0.0,
        "per_worker_fps": [round(len(l) / wall, 2) for l in lats],
        "latency_ms": {"p50": round(1e3 * pct(allat, 50), 2), "p99": round(1e3 * pct(allat, 99), 2), "mean": round(1e3 * float(np.mean(allat)), 2)} if allat else None,
        "per_worker_latency_ms_p50": [round(1e3 * pct(l, 50), 2) if l else None for l in lats],
        "cpu_util_system_pct": round(100.0 * busy / tot, 1) if tot else None,
        "cpu_process_cores_busy": round(proc_cpu / wall, 2),
        "queue": qinfo,
        "stage_ms_mean_per_frame": {
            "feeder_resize": round(1e3 * sum(d["resize"] for d in df) / nfeed, 3), "feeder_u8_to_f32_copy": round(1e3 * sum(d["copy"] for d in df) / nfeed, 3),
            "accel_h2d_exec_d2h": [round(1e3 * d["dev"] / max(d["n"], 1), 3) for d in da],
            "post": round(1e3 * (sum(d["post"] for d in dp) + sum(d["post"] for d in da)) / max(sum(d["n"] for d in da), 1), 3)},
        "temp_peak_mc": peak, "sysinfo_end": sys_info(),
    })
    return log


def run_acc(a):
    names, frames = load_frames(a.test_dir, a.labels)
    w = parse_workers(a.workers)[0]
    p = Pipe(w[3] or a.pipe, w[1], w[2], a.interp, True, a.post_engine, a.p5_link)
    for f in frames[:3]:
        p.process(f)
    smax, top, maps = [], [], []
    for f in frames:
        s, t, m = p.process(f)
        smax.append(s)
        top.append(t)
        maps.append(np.array(m, np.float32))
    np.savez(a.out, names=np.array(names), score_max=np.array(smax, np.float64), score_top10=np.array(top, np.float64), maps=np.stack(maps))
    print("saved %s n=%d map shape=%s dtype=%s" % (a.out, len(names), maps[0].shape, maps[0].dtype))
    p.close()
    return {"config": a.name, "status": "OK", "engine": w[1], "pipe": a.pipe, "interp": a.interp, "n": len(names)}


def run_check(a):
    """P5 post engine vs host_post() on the backbone engine's own outputs, all frames. Both links are checked."""
    names, frames = load_frames(a.test_dir, a.labels)
    w = parse_workers(a.workers)[0]
    res = {"config": a.name, "status": "OK", "n": len(names), "links": {}}
    for link in ("device", "host"):
        p = Pipe("P5", w[1], w[2], a.interp, True, a.post_engine, link)
        d = dict(score_max=0.0, score_top10=0.0, map_max=0.0, map_mean_sum=0.0)
        for f in frames:
            s, t, m = p.process(f)
            for n in ("m", "xproj"):  # the device link never copies the backbone outputs to the host; fetch them for the reference
                i = p.out_idx[n]
                p.r.cuda.memcpy_d2h_async(p.r.host_ptr[i], p.r.dev[i], p.r.host[i].nbytes, p.r.stream)
            p.r.cuda.stream_sync(p.r.stream)
            hs, ht, hm = host_post(p.r.host[p.out_idx["m"]], p.r.host[p.out_idx["xproj"]])
            d["score_max"] = max(d["score_max"], abs(s - hs))
            d["score_top10"] = max(d["score_top10"], abs(t - ht))
            d["map_max"] = max(d["map_max"], float(np.abs(m - hm).max()))
            d["map_mean_sum"] += float(np.abs(m - hm).mean())
        d["map_mean"] = d.pop("map_mean_sum") / len(frames)
        res["links"][link] = d
        p.close()
    return res


def run_micro(a):
    """Single-thread cost of the CPU/host-side stages and of the raw pinned H2D copy."""
    names, frames = load_frames(a.test_dir, a.labels, limit=10)
    cuda = Cudart()
    res = {"config": "micro", "status": "OK"}

    def timeit(fn, n=60):
        for _ in range(5):
            fn()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        return round(1e3 * (time.perf_counter() - t) / n, 3)

    small = np.empty((256, 256, 3), np.uint8)
    for k, it in INTERPS.items():
        res["resize900to256_%s_ms" % k] = timeit(lambda: cv2.resize(frames[0], (256, 256), dst=small, interpolation=it))
    f32_small = np.empty((1, 256, 256, 3), np.float32)
    f32_big = np.empty((1, 900, 900, 3), np.float32)
    res["u8_to_f32_256_ms"] = timeit(lambda: np.copyto(f32_small, small[None]))
    res["u8_to_f32_900_ms"] = timeit(lambda: np.copyto(f32_big, frames[0][None]))
    for tag, arr in (("900", f32_big), ("256", f32_small), ("900", f32_big)):  # 900 first and last: the first copy runs at idle clocks
        nb = arr.nbytes
        hp, dp = cuda.host_alloc(nb), cuda.malloc(nb)
        pin = np.frombuffer((ctypes_char(nb)).from_address(hp), dtype=np.float32).reshape(arr.shape)
        np.copyto(pin, arr)
        ms = timeit(lambda: cuda.memcpy_h2d(dp, hp, nb))
        res["h2d_pinned_%s_ms" % tag] = ms
        res["h2d_pinned_%s_MBps" % tag] = round(nb / 1e6 / (ms / 1e3), 0)
        res["bytes_%s" % tag] = nb
        cuda.free(dp)
        cuda.free_host(hp)
    # host post (P0) on random head outputs of the real shapes
    m = np.random.rand(1, 1, 32, 32).astype(np.float16)
    x = np.random.rand(1, 384, 32, 32).astype(np.float16)
    res["host_post_P0_ms"] = timeit(lambda: host_post(m, x))
    res["sysinfo"] = sys_info()
    return res


def ctypes_char(n):
    import ctypes
    return ctypes.c_char * n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["bench", "acc", "micro", "check"])
    ap.add_argument("--name", default="run")
    ap.add_argument("--pipe", default="P1", choices=["P0", "P1", "P2", "P4", "P5"])
    ap.add_argument("--post-engine", help="P5: GPU engine with inputs m, xproj and outputs score_max, score_top10, map")
    ap.add_argument("--p5-link", default="device", choices=["device", "host"], help="P5 hand-over of m/xproj between the engines")
    ap.add_argument("--workers", help="gpu=ENG[,gpu2=ENG,dla0=ENG,dla1=ENG]")
    ap.add_argument("--interp", default="area", choices=sorted(INTERPS))
    ap.add_argument("--test-dir", default=os.path.expanduser("~/e2e/test900"))
    ap.add_argument("--labels", default=os.path.expanduser("~/e2e/test_labels.csv"))
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--warmup", type=float, default=5.0)
    ap.add_argument("--blocking-stream", action="store_true", help="diagnostic: default blocking cudaStreamCreate stream (serializes mixed GPU+DLA threads)")
    ap.add_argument("--out", help="acc: npz path; bench/micro: json path")
    ap.add_argument("--pipelined", action="store_true", help="stage-decoupled mode (bench and acc; P1/P2/P4). Default off: sequential workers as before")
    ap.add_argument("--resize-threads", type=int, default=2, help="pipelined: feeder threads (cv2 resize + uint8->fp32 copy; for P2 only the copy)")
    ap.add_argument("--post-threads", type=int, default=0, help="pipelined P4: CPU post thread pool size; 0 = post inline in the accelerator thread")
    ap.add_argument("--border", type=int, default=0, help="engine from a fold_norm.py --border graph: pixels of the mean-filled frame around the image")
    ap.add_argument("--queue-depth", type=int, default=4, help="pipelined: depth of the shared ready queue")
    ap.add_argument("--cv-threads", type=int, default=None, help="pipelined: cv2.setNumThreads(N); default leaves OpenCV alone")
    ap.add_argument("--acc-workers", type=int, default=1, help="pipelined acc: use the first N workers of --workers (default 1)")
    a = ap.parse_args()
    global BORDER
    BORDER = a.border
    if a.mode == "micro":
        res = run_micro(a)
    elif not a.workers:
        ap.error("--workers is required")
    elif a.pipelined and a.mode in ("bench", "acc"):
        if a.mode == "acc" and not a.out:
            ap.error("--out is required for acc")
        res = run_pipelined(a, acc=(a.mode == "acc"))
    elif a.mode == "bench":
        res = run_bench(a)
    elif a.mode == "check":
        res = run_check(a)
    else:
        res = run_acc(a)
    txt = json.dumps(res, indent=1)
    print(txt)
    if a.mode != "acc" and a.out:
        with open(a.out, "w") as f:
            f.write(txt + "\n")
    return 0 if res.get("status") == "OK" and not res.get("errors") else 2


if __name__ == "__main__":
    sys.exit(main())
