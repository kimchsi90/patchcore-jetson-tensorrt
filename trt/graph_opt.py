"""Graph-level ONNX transformations for the PatchCore TensorRT port, verified with onnxruntime on the PC.

Usage (PYTHONUTF8=1, c:\\dev_venvs\\PatchcoreJetson\\Scripts\\python.exe):
  python graph_opt.py post        --cat bottle        # in-graph post-processing (score_max, score_top10, map), conv and matmul blur
  python graph_opt.py lowres_blur --cat bottle        # blur at 32x32 (sigma/8) before the nearest/bilinear upsample
  python graph_opt.py fullres     --cat bottle        # original-resolution front end (a/b/c), combined with the in-graph post
  python graph_opt.py split       --cat bottle        # cut points table + sub-graphs, chain check
  python graph_opt.py dla         --cat bottle        # static DLA notes (documentation based, nothing measured)
  python graph_opt.py all         --cat bottle        # everything, writes out\\graphopt_<cat>.json
Host reference math is copied from score_dump.py (float64 ||x||^2 - m, sqrt, np.kron upsample, scipy gaussian_filter mirror).
Opset of the source graph (12) is kept. Ops used: Mul, ReduceSum, Sub, Relu, Sqrt, Reshape, ReduceMax, TopK(largest=1), ReduceMean,
Resize, Pad(reflect), Conv, MatMul, AveragePool, Transpose, Floor.
"""
import argparse
import csv
import json
import os
import re
import sys
import time

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, checker, helper, numpy_helper, shape_inference
from onnx.utils import Extractor
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

sys.path.insert(0, r"D:\jetson_test\patchcore\rknn")
from eval_aupro import aupro  # noqa: E402

PC = r"D:\jetson_test\patchcore"
OUT = r"D:\jetson_test\trt\out"
BASE_NAME = "fused_random384_c0.01_rmax_pf_pre.onnx"
L2_NAME = "/ext/feature_extractor/layer2/layer2.3/act3/Relu_output_0"
L3_NAME = "/ext/feature_extractor/layer3/layer3.5/act3/Relu_output_0"
TOPK = 10
SIGMA = 4.0
F32 = TensorProto.FLOAT
I64 = TensorProto.INT64


# ----------------------------------------------------------------------------- paths / data
def work_dir(cat):
    return os.path.join(PC, "work", f"{cat}_256")


def base_path(cat):
    return os.path.join(work_dir(cat), BASE_NAME)


def opt_dir(cat):
    d = os.path.join(work_dir(cat), "graphopt")
    os.makedirs(d, exist_ok=True)
    return d


def read_rows(cat):
    rows = list(csv.reader(open(os.path.join(work_dir(cat), "test_labels.csv"))))[1:]
    return [(n, int(l)) for n, l in rows]


def load_masks(cat, rows, size=256):
    masks = []
    for n, l in rows:
        if l:
            mk = cv2.imread(os.path.join(work_dir(cat), "test_masks", f"{n}.png"), cv2.IMREAD_GRAYSCALE)
            if mk is None or mk.shape != (size, size):
                raise ValueError(f"bad mask {n}: {None if mk is None else mk.shape}, expected {(size, size)}")
            masks.append(mk > 127)
        else:
            masks.append(np.zeros((size, size), bool))
    return np.stack(masks)


def load_test_rgb(cat, name):
    return np.asarray(Image.open(os.path.join(work_dir(cat), "test", f"{name}.png")).convert("RGB"), np.float32)[None]


def original_path(cat, name):
    kind, stem = name.rsplit("_", 1)
    return os.path.join(PC, "datasets", "mvtec_ad", cat, "test", kind, f"{stem}.png")


def load_original_rgb(cat, name):
    """Original-resolution RGB float32 [1,H,W,3] 0..255 (grayscale images become 3 identical channels, as in prep_data.py)."""
    p = original_path(cat, name)
    img = Image.open(p)
    if img.size[0] != img.size[1]:
        raise ValueError(f"non-square original {p}: {img.size}")
    return np.asarray(img.convert("RGB"), np.float32)[None]


def session(model):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    data = model if isinstance(model, bytes) else model.SerializeToString()
    return ort.InferenceSession(data, so, providers=["CPUExecutionProvider"])


# ----------------------------------------------------------------------------- metrics
def host_post(m, x, topk=TOPK, sigma=SIGMA):
    """Exactly the math of score_dump.py. m [N,1,H,W], x [N,C,H,W] -> smax[N], stopk[N], maps[N,S,S] float32."""
    n, _, h, w = x.shape
    size = h * 8
    smax, stopk, up = [], [], []
    for i in range(n):
        sq = (x[i].astype(np.float64) ** 2).sum(0).reshape(-1)
        s = np.sqrt(np.clip(sq - m[i].astype(np.float64).reshape(-1), 0, None))
        smax.append(float(s.max()))
        stopk.append(float(np.sort(s)[::-1][:topk].mean()))
        up.append(np.kron(s.reshape(h, w), np.ones((size // h, size // w))).astype(np.float32))
    maps = np.stack([gaussian_filter(u, sigma=sigma, mode="mirror") for u in up]).astype(np.float32)
    return np.array(smax), np.array(stopk), maps


def metrics(rows, masks, smax, stopk, maps):
    labels = [l for _, l in rows]
    return {
        "image_auroc_max": round(float(roc_auc_score(labels, smax)), 4),
        "image_auroc_top10": round(float(roc_auc_score(labels, stopk)), 4),
        "pixel_auroc": round(float(roc_auc_score(masks.reshape(-1), maps.reshape(-1))), 4),
        "aupro": round(aupro(maps, masks), 4),
    }


def diffs(ref, got):
    smax, stopk, maps = ref
    g = got
    return {
        "max_abs_diff_score_max": float(np.abs(smax - g[0]).max()),
        "max_abs_diff_score_top10": float(np.abs(stopk - g[1]).max()),
        "max_abs_diff_map": float(np.abs(maps - g[2]).max()),
        "mean_abs_diff_map": float(np.abs(maps - g[2]).mean()),
    }


def host_baseline(cat):
    m = np.load(os.path.join(OUT, f"{cat}_256_ort_fp32_m.npy"))
    x = np.load(os.path.join(OUT, f"{cat}_256_ort_fp32_x.npy"))
    if m.ndim == 2:
        m = m.reshape(m.shape[0], 1, x.shape[2], x.shape[3])
    return host_post(m, x)


# ----------------------------------------------------------------------------- graph building helpers
class GB:
    def __init__(self):
        self.nodes, self.inits = [], []

    def const(self, name, arr):
        self.inits.append(numpy_helper.from_array(np.asarray(arr), name))
        return name

    def op(self, op_type, ins, out, **attrs):
        self.nodes.append(helper.make_node(op_type, ins, [out], name=out + "_n", **attrs))
        return out


def gauss1d(sigma, truncate=4.0):
    """Same taps as scipy.ndimage.gaussian_filter1d (radius int(truncate*sigma+0.5), normalized)."""
    r = int(truncate * sigma + 0.5)
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 / (sigma * sigma) * x * x)
    return (k / k.sum()), r


def reflect101(i, n):
    if i < 0:
        i = -i
    if i >= n:
        i = 2 * n - 2 - i
    return i


def blur_matrix(n, sigma):
    """[n,n] matrix B with B@v = gaussian filter of v, reflect-101 border (== scipy mode='mirror')."""
    k, r = gauss1d(sigma)
    if r >= n:
        raise ValueError(f"kernel radius {r} >= size {n}")
    b = np.zeros((n, n), np.float64)
    for i in range(n):
        for d in range(-r, r + 1):
            b[i, reflect101(i + d, n)] += k[d + r]
    return b


def upsample_matrix(n_out, n_in):
    f = n_out // n_in
    if f * n_in != n_out:
        raise ValueError(f"non-integer upsample {n_in}->{n_out}")
    u = np.zeros((n_out, n_in), np.float64)
    for i in range(n_out):
        u[i, i // f] = 1.0
    return u


def sep_blur(g, src, tag, sigma):
    """Reflect-101 Pad + two 1D depthwise (single channel) Convs. src [1,1,H,W] -> same shape."""
    k, r = gauss1d(sigma)
    pads = g.const(f"{tag}_pads", np.array([0, 0, r, r, 0, 0, r, r], np.int64))
    padded = g.op("Pad", [src, pads], f"{tag}_padded", mode="reflect")
    kh = g.const(f"{tag}_kh", k.astype(np.float32).reshape(1, 1, 1, -1))
    kv = g.const(f"{tag}_kv", k.astype(np.float32).reshape(1, 1, -1, 1))
    t = g.op("Conv", [padded, kh], f"{tag}_conv_h", kernel_shape=[1, 2 * r + 1])
    return g.op("Conv", [t, kv], f"{tag}_conv_v", kernel_shape=[2 * r + 1, 1])


def nearest_up(g, src, tag, f):
    scales = g.const(f"{tag}_scales", np.array([1, 1, f, f], np.float32))
    return g.op("Resize", [src, "", scales], f"{tag}_up", mode="nearest", coordinate_transformation_mode="asymmetric",
                nearest_mode="floor")


def add_post(model, mode="conv", sigma=SIGMA, topk=TOPK, keep_raw=False):
    """Append post-processing. Replaces outputs (m, xproj) by score_max [1], score_top10 [1], map [1,1,S,S].
    mode: conv (nearest up, Pad reflect, 2 Conv), mm (up+blur folded into two MatMuls), lowres (blur 32x32 with sigma/8, nearest up),
    lowres_bilinear (same, bilinear up), none (scores only)."""
    model = onnx.ModelProto.FromString(model.SerializeToString())
    gr = model.graph
    names = {o.name: o for o in gr.output}
    if "m" not in names or "xproj" not in names:
        raise ValueError(f"expected outputs m and xproj, got {list(names)}")
    gdim = [d.dim_value for d in names["xproj"].type.tensor_type.shape.dim]
    cdim = [d.dim_value for d in names["m"].type.tensor_type.shape.dim]
    if len(gdim) != 4 or cdim != [1, 1, gdim[2], gdim[3]]:
        raise ValueError(f"unexpected shapes xproj={gdim} m={cdim}")
    gh, gw = gdim[2], gdim[3]
    size = gh * 8
    g = GB()
    xsq = g.op("Mul", ["xproj", "xproj"], "post_xsq")
    sq = g.op("ReduceSum", [xsq], "post_sq", axes=[1], keepdims=1)
    d2 = g.op("Sub", [sq, "m"], "post_d2")
    d2c = g.op("Relu", [d2], "post_d2c")          # == Clip(d2, min=0); Relu is also a native DLA layer
    s = g.op("Sqrt", [d2c], "post_s")
    flat = g.op("Reshape", [s, g.const("post_flat_shape", np.array([1, gh * gw], np.int64))], "post_flat")
    g.op("ReduceMax", [flat], "score_max", axes=[1], keepdims=0)
    tv = f"post_topk_val"
    g.nodes.append(helper.make_node("TopK", [flat, g.const("post_k", np.array([topk], np.int64))], [tv, "post_topk_idx"],
                                    name="post_topk", axis=1, largest=1, sorted=1))
    g.op("ReduceMean", [tv], "score_top10", axes=[1], keepdims=0)
    outs = [helper.make_tensor_value_info("score_max", F32, [1]), helper.make_tensor_value_info("score_top10", F32, [1])]
    if mode != "none":
        if mode == "conv":
            up = nearest_up(g, s, "post_map", 8)
            res = sep_blur(g, up, "post_blur", sigma)
        elif mode == "mm":
            a = blur_matrix(size, sigma) @ upsample_matrix(size, gh)       # [S,gh]
            aw = blur_matrix(size, sigma) @ upsample_matrix(size, gw)
            t = g.op("MatMul", [g.const("post_A", a.astype(np.float32)), s], "post_mm1")                # [1,1,S,gw]
            res = g.op("MatMul", [t, g.const("post_Bt", aw.T.astype(np.float32).copy())], "post_mm2")   # [1,1,S,S]
        elif mode in ("lowres", "lowres_bilinear"):
            blurred = sep_blur(g, s, "post_blur_lr", sigma * gh / size)
            if mode == "lowres":
                res = nearest_up(g, blurred, "post_map", 8)
            else:
                sizes = g.const("post_up_sizes", np.array([1, 1, size, size], np.int64))
                res = g.op("Resize", [blurred, "", "", sizes], "post_map_up", mode="linear",
                           coordinate_transformation_mode="half_pixel")
        else:
            raise ValueError(f"unknown post mode {mode}")
        g.op("Identity", [res], "map")
        outs.append(helper.make_tensor_value_info("map", F32, [1, 1, size, size]))
    if keep_raw:
        outs += list(gr.output)
    del gr.output[:]
    gr.output.extend(outs)
    gr.node.extend(g.nodes)
    gr.initializer.extend(g.inits)
    del gr.value_info[:]
    checker.check_model(model)
    return model


# ----------------------------------------------------------------------------- full-resolution front end
def pil_bilinear_matrix(n_in, n_out):
    """Row-normalized triangle-filter weights, antialiased (support scales with n_in/n_out), as Pillow's BILINEAR resize."""
    scale = n_in / n_out
    fs = max(scale, 1.0)
    support = 1.0 * fs
    w = np.zeros((n_out, n_in), np.float64)
    for i in range(n_out):
        c = (i + 0.5) * scale
        lo = max(int(c - support + 0.5), 0)
        hi = min(int(c + support + 0.5), n_in)
        for j in range(lo, hi):
            x = (j + 0.5 - c) / fs
            w[i, j] = max(0.0, 1.0 - abs(x))
        w[i] /= w[i].sum()
    return w


def add_fullres(model, h_in, mode, expose_resized=False):
    """Replace the 256x256 NHWC input by a static [1,h_in,h_in,3] float input (RGB 0..255) and resize to the old size inside the graph.
    mode: bilinear (Resize linear half_pixel, no antialias), pool_bilinear (AveragePool k=floor(h/S) then bilinear if needed),
    matmul (separable antialiased triangle filter as two MatMuls, Pillow weights), matmul_round (same + Floor(x+0.5) like uint8 output)."""
    model = onnx.ModelProto.FromString(model.SerializeToString())
    gr = model.graph
    inp = [i for i in gr.input if i.name == "image_nhwc"]
    if len(inp) != 1:
        raise ValueError(f"input image_nhwc not found: {[i.name for i in gr.input]}")
    dims = [d.dim_value for d in inp[0].type.tensor_type.shape.dim]
    if dims[0] != 1 or dims[1] != dims[2] or dims[3] != 3:
        raise ValueError(f"unexpected input dims {dims}")
    s = dims[1]
    tr = [n for n in gr.node if n.name == "pre_transpose"]
    if len(tr) != 1:
        raise ValueError("pre_transpose node not found")
    pre_in = tr[0].output[0]                      # pre_nchw, consumed by pre_sub
    gr.node.remove(tr[0])
    g = GB()
    nin = "image_full_nhwc"
    if mode in ("bilinear", "pool_bilinear"):
        x = g.op("Transpose", [nin], "fr_nchw", perm=[0, 3, 1, 2])
        cur = h_in
        if mode == "pool_bilinear":
            k = h_in // s
            if k >= 2:
                x = g.op("AveragePool", [x], "fr_pool", kernel_shape=[k, k], strides=[k, k], pads=[0, 0, 0, 0], count_include_pad=0)
                cur = h_in // k
        if cur != s:
            sizes = g.const("fr_sizes", np.array([1, 3, s, s], np.int64))
            g.op("Resize", [x, "", "", sizes], pre_in, mode="linear", coordinate_transformation_mode="half_pixel")
        else:
            g.op("Identity", [x], pre_in)
    elif mode in ("matmul", "matmul_round"):
        w = pil_bilinear_matrix(h_in, s).astype(np.float32)
        a = g.op("Reshape", [nin, g.const("fr_s2d", np.array([h_in, h_in * 3], np.int64))], "fr_2d")
        t1 = g.op("MatMul", [g.const("fr_Wh", w), a], "fr_mm_h")                                     # [s, W*3]
        t2 = g.op("Reshape", [t1, g.const("fr_s3d", np.array([s, h_in, 3], np.int64))], "fr_3d")      # [s, W, 3]
        t3 = g.op("Transpose", [t2], "fr_cwh", perm=[2, 0, 1])                                      # [3, s, W]
        t4 = g.op("MatMul", [t3, g.const("fr_WwT", w.T.copy())], "fr_mm_w")                          # [3, s, s]
        if mode == "matmul_round":
            t4 = g.op("Floor", [g.op("Add", [t4, g.const("fr_half", np.array(0.5, np.float32))], "fr_add_half")], "fr_floor")
        g.op("Reshape", [t4, g.const("fr_s4d", np.array([1, 3, s, s], np.int64))], pre_in)
    else:
        raise ValueError(f"unknown fullres mode {mode}")
    existing = list(gr.node)
    del gr.node[:]
    gr.node.extend(g.nodes + existing)
    gr.initializer.extend(g.inits)
    del gr.input[:]
    gr.input.append(helper.make_tensor_value_info(nin, F32, [1, h_in, h_in, 3]))
    if expose_resized:
        gr.output.append(helper.make_tensor_value_info(pre_in, F32, [1, 3, s, s]))
    del gr.value_info[:]
    checker.check_model(model)
    return model


# ----------------------------------------------------------------------------- evaluation helpers
def run_post_model(model, cat, rows, loader=load_test_rgb, expose_resized=False):
    sess = session(model)
    in_name = sess.get_inputs()[0].name
    names = [o.name for o in sess.get_outputs()]
    smax, stopk, maps, resized = [], [], [], []
    for n, _ in rows:
        o = dict(zip(names, sess.run(None, {in_name: loader(cat, n)})))
        smax.append(float(o["score_max"][0]))
        stopk.append(float(o["score_top10"][0]))
        maps.append(o["map"][0, 0])
        if expose_resized:
            resized.append(o[names[-1]][0])
    res = (np.array(smax), np.array(stopk), np.stack(maps).astype(np.float32))
    return (res, np.stack(resized)) if expose_resized else res


def save(model, path):
    checker.check_model(model)
    onnx.save(model, path)
    opset = [(o.domain or "ai.onnx", o.version) for o in model.opset_import]
    return {"file": path, "opset": opset, "ir": model.ir_version, "bytes": os.path.getsize(path),
            "ops": sorted({n.op_type for n in model.graph.node})}


def clean_input_check(cat):
    """The base model is run once to confirm the stored fp32 reference is reproducible bit-for-bit here."""
    rows = read_rows(cat)
    sess = session(onnx.load(base_path(cat)))
    ref_m = np.load(os.path.join(OUT, f"{cat}_256_ort_fp32_m.npy"))
    ref_x = np.load(os.path.join(OUT, f"{cat}_256_ort_fp32_x.npy"))
    worst = 0.0
    for i in (0, len(rows) // 2, len(rows) - 1):
        m, x = sess.run(None, {"image_nhwc": load_test_rgb(cat, rows[i][0])})
        if x.shape[1] != ref_x.shape[1]:
            m, x = x, m
        worst = max(worst, float(np.abs(x[0] - ref_x[i]).max()), float(np.abs(m.reshape(-1) - ref_m[i].reshape(-1)).max()))
    return worst


# ----------------------------------------------------------------------------- subcommands
def cmd_post(cat, res):
    rows, masks = read_rows(cat), None
    masks = load_masks(cat, rows)
    base = onnx.load(base_path(cat))
    ref = host_baseline(cat)
    res["reproduce_stored_reference_max_abs"] = clean_input_check(cat)
    res["host_baseline"] = metrics(rows, masks, *ref)
    res["files"] = res.get("files", {})
    for mode, tag in (("conv", "post"), ("mm", "post_mm")):
        t0 = time.time()
        mdl = add_post(base, mode)
        res["files"][tag] = save(mdl, os.path.join(opt_dir(cat), f"{cat}_{tag}.onnx"))
        got = run_post_model(mdl, cat, rows)
        res[tag] = {**metrics(rows, masks, *got), **diffs(ref, got), "ort_seconds": round(time.time() - t0, 1)}
        print(cat, tag, res[tag], flush=True)
    return ref, masks, rows


def cmd_lowres(cat, res, ref=None, masks=None, rows=None):
    rows = rows or read_rows(cat)
    masks = masks if masks is not None else load_masks(cat, rows)
    ref = ref or host_baseline(cat)
    base = onnx.load(base_path(cat))
    res.setdefault("files", {})
    for mode, tag in (("lowres", "lowres_blur"), ("lowres_bilinear", "lowres_blur_bilinear_up")):
        mdl = add_post(base, mode)
        res["files"][tag] = save(mdl, os.path.join(opt_dir(cat), f"{cat}_{tag}.onnx"))
        got = run_post_model(mdl, cat, rows)
        res[tag] = {**metrics(rows, masks, *got), **diffs(ref, got)}
        print(cat, tag, res[tag], flush=True)


def cmd_fullres(cat, res, ref=None, masks=None, rows=None):
    rows = rows or read_rows(cat)
    masks = masks if masks is not None else load_masks(cat, rows)
    ref = ref or host_baseline(cat)
    h_in = load_original_rgb(cat, rows[0][0]).shape[1]
    res["fullres_input_size"] = h_in
    post = add_post(onnx.load(base_path(cat)), "conv")
    ref_pngs = None
    res.setdefault("files", {})
    for mode in ("bilinear", "pool_bilinear", "matmul", "matmul_round"):
        mdl = add_fullres(post, h_in, mode)
        res["files"][f"fullres_{mode}"] = save(mdl, os.path.join(opt_dir(cat), f"{cat}_post_fullres{h_in}_{mode}.onnx"))
        dbg = add_fullres(post, h_in, mode, expose_resized=True)
        got, resized = run_post_model(dbg, cat, rows, loader=load_original_rgb, expose_resized=True)
        if ref_pngs is None:
            ref_pngs = np.stack([load_test_rgb(cat, n)[0].transpose(2, 0, 1) for n, _ in rows])
        e = np.abs(resized - ref_pngs)
        res[f"fullres_{mode}"] = {**metrics(rows, masks, *got), **diffs(ref, got),
                                  "resized_vs_png_mean_abs": round(float(e.mean()), 4),
                                  "resized_vs_png_max_abs": round(float(e.max()), 2)}
        print(cat, f"fullres_{mode}", res[f"fullres_{mode}"], flush=True)


# --- split
def rename_tensors(model, mapping):
    for n in model.graph.node:
        for k, v in enumerate(n.input):
            if v in mapping:
                n.input[k] = mapping[v]
        for k, v in enumerate(n.output):
            if v in mapping:
                n.output[k] = mapping[v]
    return model


def cut_table(model):
    """For every node boundary, the tensors alive across it (produced earlier, consumed later) with fp16 bytes."""
    inferred = shape_inference.infer_shapes(model)
    shp = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        shp[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    nodes = list(model.graph.node)
    inits = {i.name for i in model.graph.initializer}
    last_use = {}
    for k, n in enumerate(nodes):
        for t in n.input:
            last_use[t] = k
    for o in model.graph.output:
        last_use[o.name] = len(nodes)
    produced = [i.name for i in model.graph.input if i.name not in inits]
    table = []
    for k, n in enumerate(nodes):
        produced += [o for o in n.output if o]
        live = [t for t in produced if last_use.get(t, -1) > k and t not in inits]
        table.append({"after_node": n.name or n.output[0], "out": n.output[0], "op": n.op_type,
                      "live": [{"tensor": t, "shape": shp.get(t), "fp16_bytes": int(np.prod(shp[t]) * 2) if t in shp else None} for t in live]})
    return table


def interesting(entry):
    out = entry["out"]
    return bool(re.search(r"maxpool/MaxPool_output|act3/Relu_output|^proj[23]$|^xproj$|^proj3_up$|pre_div|pool(_1)?/AveragePool_output|^m$|^score_max$|^map$", out)) \
        or entry["out"] in ("l2_feat", "l3_feat")


def cmd_split(cat, res):
    rows = read_rows(cat)
    post = add_post(onnx.load(base_path(cat)), "conv")
    post = rename_tensors(post, {L2_NAME: "l2_feat", L3_NAME: "l3_feat"})
    base = rename_tensors(onnx.load(base_path(cat)), {L2_NAME: "l2_feat", L3_NAME: "l3_feat"})
    table = cut_table(post)
    cuts = []
    for e in table:
        if not interesting(e):
            continue
        tot = sum(t["fp16_bytes"] or 0 for t in e["live"])
        cuts.append({"after": e["out"], "n_live": len(e["live"]), "fp16_bytes": tot,
                     "live": [(t["tensor"], t["shape"], t["fp16_bytes"]) for t in e["live"]]})
    res["cut_candidates"] = cuts
    img_bytes = 256 * 256 * 3
    res["input_bytes_256_uint8"] = img_bytes
    for c in cuts:
        print(f"  cut after {c['after'][-60:]:<60} live={c['n_live']} fp16={c['fp16_bytes']/1024:.1f} KiB {[x[1] for x in c['live']]}")

    def ext(model, ins, outs):
        mdl = Extractor(shape_inference.infer_shapes(model)).extract_model(ins, outs)
        return mdl

    img = "image_nhwc"
    parts = {
        "A_stem_to_l2": ext(post, [img], ["l2_feat"]),
        "B_l3": ext(post, ["l2_feat"], ["l3_feat"]),
        "C_head_m_xproj": ext(base, ["l2_feat", "l3_feat"], ["m", "xproj"]),
        "C_head_post": ext(post, ["l2_feat", "l3_feat"], ["score_max", "score_top10", "map"]),
        "AB_stem_to_l2_l3": ext(post, [img], ["l2_feat", "l3_feat"]),
        "BC_l3_head_post": ext(post, ["l2_feat"], ["score_max", "score_top10", "map"]),
    }
    res["files"] = res.get("files", {})
    for k, mdl in parts.items():
        res["files"][f"split_{k}"] = save(mdl, os.path.join(opt_dir(cat), f"{cat}_split_{k}.onnx"))
        res["files"][f"split_{k}"]["inputs"] = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in mdl.graph.input]
        res["files"][f"split_{k}"]["outputs"] = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in mdl.graph.output]
        res["files"][f"split_{k}"]["n_nodes"] = len(mdl.graph.node)
        print(cat, "split part", k, "nodes", len(mdl.graph.node), "in", res["files"][f"split_{k}"]["inputs"], flush=True)
    # Chain check against the unsplit graphs on every test image.
    sf, sA, sB, sCh, sCp, sAB, sBC = (session(post), session(parts["A_stem_to_l2"]), session(parts["B_l3"]), session(parts["C_head_m_xproj"]),
                                      session(parts["C_head_post"]), session(parts["AB_stem_to_l2_l3"]), session(parts["BC_l3_head_post"]))
    sbase = session(base)
    w = {"3way_post": 0.0, "3way_m": 0.0, "3way_xproj": 0.0, "2way_A|BC": 0.0, "2way_AB|C": 0.0}
    fn = [o.name for o in sf.get_outputs()]
    for n, _ in rows:
        x = load_test_rgb(cat, n)
        full = sf.run(None, {img: x})
        bm, bx = sbase.run(None, {img: x})
        if bx.shape[1] == 1:
            bm, bx = bx, bm
        l2 = sA.run(None, {img: x})[0]
        l3 = sB.run(None, {"l2_feat": l2})[0]
        cp = sCp.run(None, {"l2_feat": l2, "l3_feat": l3})
        ch = dict(zip([o.name for o in sCh.get_outputs()], sCh.run(None, {"l2_feat": l2, "l3_feat": l3})))
        bc = sBC.run(None, {"l2_feat": l2})
        ab = dict(zip([o.name for o in sAB.get_outputs()], sAB.run(None, {img: x})))
        c2 = sCp.run(None, {"l2_feat": ab["l2_feat"], "l3_feat": ab["l3_feat"]})
        w["3way_post"] = max(w["3way_post"], *[float(np.abs(a - b).max()) for a, b in zip(full, cp)])
        w["3way_m"] = max(w["3way_m"], float(np.abs(ch["m"] - bm).max()))
        w["3way_xproj"] = max(w["3way_xproj"], float(np.abs(ch["xproj"] - bx).max()))
        w["2way_A|BC"] = max(w["2way_A|BC"], *[float(np.abs(a - b).max()) for a, b in zip(full, bc)])
        w["2way_AB|C"] = max(w["2way_AB|C"], *[float(np.abs(a - b).max()) for a, b in zip(full, c2)])
    res["split_chain_max_abs_diff"] = w
    print(cat, "chain max abs diff", w, flush=True)


# --- DLA static notes (documentation, not measured)
DLA = {
    # op: (verdict, note)   verdicts: ok / limited / no
    "Conv": ("ok", "kernel [1,32], stride [1,8], in/out channels [1,8192], fp16/int8; fails the CBUF check for very large weights"),
    "Relu": ("ok", "Activation layer"),
    "MaxPool": ("ok", "window [1,8], pad [0,7], stride [1,16]"),
    "AveragePool": ("limited", "window [1,8] only; the fullres pool k=3/4 fits"),
    "Add": ("ok", "ElementWise Sum, same-shape or NCHW broadcast"),
    "Sub": ("limited", "ElementWise Sub is listed, but the m operand [1,1,G,G] broadcasts over channels only in the Sub of d2 (no channel dim use)"),
    "Mul": ("limited", "ElementWise Product; fine for same-shape x*x"),
    "Div": ("limited", "ElementWise Div needs INT8 or FP32-constant divisor (preproc Div by a per-channel constant -> Scale layer)"),
    "Transpose": ("limited", "Shuffle: 4D, batch dim untouched, non-batch dims <= 8192; the NHWC->NCHW input transpose of the base graph is unlikely to be a good DLA layer; use an NHWC/HWC4 input format or keep it on GPU"),
    "Reshape": ("limited", "Shuffle: 4D only, non-batch dims <= 8192; the [1,1024] flatten and the 2D MatMul reshapes are not DLA-able"),
    "Resize": ("limited", "4 scales, first two 1, last two integers: nearest [1,32], bilinear [1,4]. The head's 16->32 bilinear x2 fits; the post x8 nearest fits; Resize with `sizes` (fullres) or from 900 -> 256 does not"),
    "ReduceMax": ("limited", "Reduce MAX over CHW axes only on the latest doc; on Xavier DLA 1.0 / TRT 8.5 [Unknown], the channel max of the 2140-wide kNN output (2140 <= 8192) may fit"),
    "ReduceSum": ("no", "no Reduce SUM on DLA (only MAX); use GPU (or a 1x1 Conv with all-ones weights over channels to stay on DLA)"),
    "ReduceMean": ("no", "Reduce MEAN not supported"),
    "TopK": ("no", "not in the DLA layer list"),
    "Sqrt": ("no", "Unary: only ABS/SIN/COS/ATAN (INT8) listed on the latest doc"),
    "Pad": ("no", "no Padding layer on DLA (reflect mode is not available in any case)"),
    "MatMul": ("no", "MatrixMultiply not in the DLA list"),
    "Floor": ("no", "Unary FLOOR not on DLA"),
    "Identity": ("ok", "removed by the parser"),
}


def cmd_dla(cat, res):
    post = add_post(onnx.load(base_path(cat)), "conv")
    post = rename_tensors(post, {L2_NAME: "l2_feat", L3_NAME: "l3_feat"})
    base = rename_tensors(onnx.load(base_path(cat)), {L2_NAME: "l2_feat", L3_NAME: "l3_feat"})

    def summarize(model):
        cnt = {}
        for n in model.graph.node:
            cnt[n.op_type] = cnt.get(n.op_type, 0) + 1
        return {op: {"count": c, "dla": DLA.get(op, ("unknown", ""))[0], "note": DLA.get(op, ("unknown", ""))[1]} for op, c in sorted(cnt.items())}

    stages = {
        "backbone_stem_to_l2 (A)": Extractor(shape_inference.infer_shapes(post)).extract_model(["image_nhwc"], ["l2_feat"]),
        "layer3 (B)": Extractor(shape_inference.infer_shapes(post)).extract_model(["l2_feat"], ["l3_feat"]),
        "head (C, m/xproj)": Extractor(shape_inference.infer_shapes(base)).extract_model(["l2_feat", "l3_feat"], ["m", "xproj"]),
    }
    res["dla_static"] = {k: summarize(v) for k, v in stages.items()}
    post_nodes = [n for n in post.graph.node if n.name.startswith("post_") or n.output[0] in ("score_max", "score_top10", "map")]
    cnt = {}
    for n in post_nodes:
        cnt[n.op_type] = cnt.get(n.op_type, 0) + 1
    res["dla_static"]["post (conv blur)"] = {op: {"count": c, "dla": DLA.get(op, ("unknown", ""))[0], "note": DLA.get(op, ("unknown", ""))[1]}
                                              for op, c in sorted(cnt.items())}
    res["dla_static"]["_source"] = "documentation (TensorRT DLA layer-restriction page, latest version, plus memory); NOT measured, TRT 8.5 on Xavier DLA 1.0 may be stricter"
    for k, v in res["dla_static"].items():
        if k.startswith("_"):
            continue
        print(" ", k, {op: f"{d['count']}x {d['dla']}" for op, d in v.items()})


# --- fp16 simulation of the post math (numpy; not a TensorRT measurement)
def cmd_fp16sim(cat, res):
    rows = read_rows(cat)
    masks = load_masks(cat, rows)
    m = np.load(os.path.join(OUT, f"{cat}_256_ort_fp32_m.npy"))
    x = np.load(os.path.join(OUT, f"{cat}_256_ort_fp32_x.npy"))
    m = m.reshape(m.shape[0], 1, x.shape[2], x.shape[3])
    x16, m16 = x.astype(np.float16), m.astype(np.float16)
    sq16 = (x16.astype(np.float32) ** 2).sum(1, keepdims=True).astype(np.float16)   # fp32 accumulate, fp16 result
    d2 = np.maximum(sq16 - m16, np.float16(0))
    s = np.sqrt(d2.astype(np.float32)).astype(np.float16).astype(np.float32)
    n = s.shape[0]
    flat = s.reshape(n, -1)
    smax = flat.max(1)
    stopk = np.sort(flat, 1)[:, ::-1][:, :TOPK].mean(1)
    up = np.stack([np.kron(a[0], np.ones((8, 8))).astype(np.float32) for a in s])
    maps = np.stack([gaussian_filter(u, sigma=SIGMA, mode="mirror") for u in up]).astype(np.float32)
    ref = host_baseline(cat)
    res["fp16_post_sim"] = {**metrics(rows, masks, smax, stopk, maps), **diffs(ref, (smax, stopk, maps)),
                            "note": "numpy emulation: x,m rounded to fp16, ||x||^2 fp32-accumulated then rounded to fp16, d2 in fp16; not a TensorRT run"}
    print(cat, "fp16_post_sim", res["fp16_post_sim"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["post", "lowres_blur", "fullres", "split", "dla", "fp16sim", "all"])
    ap.add_argument("--cat", required=True)
    a = ap.parse_args()
    jpath = os.path.join(OUT, f"graphopt_{a.cat}.json")
    res = json.load(open(jpath)) if os.path.isfile(jpath) else {}
    if a.cmd in ("post", "all"):
        ref, masks, rows = cmd_post(a.cat, res)
    else:
        ref = masks = rows = None
    if a.cmd in ("lowres_blur", "all"):
        cmd_lowres(a.cat, res, ref, masks, rows)
    if a.cmd in ("fullres", "all"):
        cmd_fullres(a.cat, res, ref, masks, rows)
    if a.cmd in ("split", "all"):
        cmd_split(a.cat, res)
    if a.cmd in ("dla", "all"):
        cmd_dla(a.cat, res)
    if a.cmd in ("fp16sim", "all"):
        cmd_fp16sim(a.cat, res)
    json.dump(res, open(jpath, "w"), indent=1, default=str)
    print("wrote", jpath)
    return 0


if __name__ == "__main__":
    sys.exit(main())
