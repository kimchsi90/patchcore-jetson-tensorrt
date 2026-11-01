#!/usr/bin/env python3
"""End-to-end PatchCore benchmark for Jetson (python3.8, numpy 1.17, cv2 4.5, TensorRT 8.5). Reuses trt_runner.py.

Pipelines (frames are uint8 RGB HWC 900x900 held in RAM, like a camera buffer; PNG decode is not timed):
  P0  CPU cv2.resize 900->256  | engine -> (m, xproj)                   | CPU numpy/cv2 post (score, top10, upsample, gaussian)
  P1  CPU cv2.resize 900->256  | engine -> (score_max, score_top10, map) post inside the engine
  P2  engine with 900x900 input: resize + post inside the engine (matmul resize)
  P3  = P1 with the backbone engine built for the DLA (post layers fall back to the GPU); selected by passing a DLA engine

Modes:
  bench : timed run, N worker threads (one engine/context/stream each), --workers gpu=ENG,dla0=ENG,dla1=ENG
  acc   : run the labelled test frames once through the first worker and save outputs (npz) for scoring on the PC
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
class Pipe(object):
    """One worker: a runner plus the stage code. Not thread safe; use one Pipe per thread."""

    def __init__(self, kind, engine, dla_core, interp, nb_stream=True):
        self.kind = kind  # P0 | P1 | P2
        self.engine = engine
        self.dla_core = dla_core
        self.r = TrtRunner(engine, dla_core)
        r = self.r
        self.out_idx = dict((r.names[i], i) for i, f in enumerate(r.is_input) if not f)
        if kind == "P0":
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
            self.small = np.empty((s[1], s[2], 3), np.uint8)
        self.interp = INTERPS[interp]
        if nb_stream:  # non-blocking stream: does not synchronize with the legacy default stream
            import ctypes
            st = ctypes.c_void_p()
            r.cuda.check(r.cuda.lib.cudaStreamCreateWithFlags(ctypes.byref(st), 1), "cudaStreamCreateWithFlags(nonblocking)")
            r.stream = st.value
        self.t = dict(pre=0.0, copyin=0.0, enq=0.0, gpu=0.0, post=0.0, n=0)

    def describe(self):
        r = self.r
        return {"engine": self.engine, "dla_core": self.dla_core, "in": [r.names[r.in_idx], list(r.in_shape), str(r.in_dtype)],
                "out": [[r.names[i], list(r.shapes[i]), str(r.host[i].dtype)] for i in self.out_idx.values()]}

    def process(self, frame):
        """frame uint8 [900,900,3] RGB -> (score_max, score_top10, map [256,256] float32 view or copy)."""
        r, c = self.r, self.r.cuda
        t0 = time.perf_counter()
        if self.kind == "P2":
            src = frame
        else:
            cv2.resize(frame, (self.small.shape[1], self.small.shape[0]), dst=self.small, interpolation=self.interp)
            src = self.small
        t1 = time.perf_counter()
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
        if self.kind == "P0":
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

    def close(self):
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
        if bgr.shape != (900, 900, 3) or bgr.dtype != np.uint8:
            raise ValueError("frame %s has shape %s dtype %s, expected (900,900,3) uint8" % (p, bgr.shape, bgr.dtype))
        frames.append(np.ascontiguousarray(bgr[:, :, ::-1]))
    return [n for n, _ in rows], frames


def parse_workers(spec):
    """'gpu=a.engine,dla0=b.engine' -> [(label, engine, dla_core or None)]"""
    out = []
    for tok in spec.split(","):
        lab, _, eng = tok.partition("=")
        if not eng:
            raise ValueError("bad worker spec %r (want gpu=ENGINE | dla0=ENGINE | dla1=ENGINE)" % tok)
        if lab.startswith("gpu"):
            core = None
        elif lab in ("dla0", "dla1"):
            core = int(lab[-1])
        else:
            raise ValueError("worker label must be gpuN, dla0 or dla1, got %r" % lab)
        out.append((lab, eng, core))
    return out


def placement(pipe, workers):
    """Which stage runs on which device, per worker."""
    out = []
    for lab, _, core in workers:
        dev = "DLA%d" % core if core is not None else "GPU"
        if pipe == "P0":
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


# ----------------------------------------------------------------------------- modes
def run_bench(a):
    names, frames = load_frames(a.test_dir, a.labels)
    zones = thermal_zones()
    if not zones:
        raise RuntimeError("no CPU-therm/GPU-therm thermal zones found")
    workers = parse_workers(a.workers)
    pipes = [Pipe(a.pipe, eng, core, a.interp, not a.blocking_stream) for _, eng, core in workers]
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
        p.t = dict(pre=0.0, copyin=0.0, enq=0.0, gpu=0.0, post=0.0, n=0)
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
        "stage_ms_mean_per_frame": [dict((k, round(1e3 * s[k] / max(s["n"], 1), 3)) for k in ("pre", "copyin", "enq", "gpu", "post")) for s in snap],
        "temp_peak_mc": peak, "sysinfo_end": sys_info(),
    })
    for p in pipes:
        p.close()
    return log


def run_acc(a):
    names, frames = load_frames(a.test_dir, a.labels)
    w = parse_workers(a.workers)[0]
    p = Pipe(a.pipe, w[1], w[2], a.interp)
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
    ap.add_argument("--mode", required=True, choices=["bench", "acc", "micro"])
    ap.add_argument("--name", default="run")
    ap.add_argument("--pipe", default="P1", choices=["P0", "P1", "P2"])
    ap.add_argument("--workers", help="gpu=ENG[,gpu2=ENG,dla0=ENG,dla1=ENG]")
    ap.add_argument("--interp", default="area", choices=sorted(INTERPS))
    ap.add_argument("--test-dir", default=os.path.expanduser("~/e2e/test900"))
    ap.add_argument("--labels", default=os.path.expanduser("~/e2e/test_labels.csv"))
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--warmup", type=float, default=5.0)
    ap.add_argument("--blocking-stream", action="store_true", help="diagnostic: default blocking cudaStreamCreate stream (serializes mixed GPU+DLA threads)")
    ap.add_argument("--out", help="acc: npz path; bench/micro: json path")
    a = ap.parse_args()
    if a.mode == "micro":
        res = run_micro(a)
    elif not a.workers:
        ap.error("--workers is required")
    elif a.mode == "bench":
        res = run_bench(a)
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
