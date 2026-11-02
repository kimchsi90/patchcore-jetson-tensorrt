"""Fold the input normalization of a *_pre.onnx graph into the first convolution.

Original: image_nhwc -> Transpose -> Sub(mean255) -> Div(std255) -> Conv(W, b) ...
Folded:   image_nhwc_centered (= RGB - mean255, NHWC) -> Transpose -> Conv(W / std255, b) ...
The caller (feeder, calibrator, dump) subtracts mean255 on the CPU, so the convolution's zero padding stays identical to the
original (padding a centered image with 0 equals padding the normalized image with 0). The DLA then no longer needs the
Sub/Div fallback to the GPU. Border mode (--border): the caller instead feeds raw RGB inside a (S+2p)x(S+2p) buffer whose p-pixel border is prefilled once with
mean255, so no per-frame CPU subtraction is needed. The conv padding becomes 0 and its bias absorbs the mean:
b' = b - sum(W/std255 * mean255). The result equals the original graph exactly (border = mean -> centered 0 = zero padding).
Usage: python fold_norm.py <pre.onnx> <fold.onnx> [--border] [--verify-images DIR] [--n 5]
"""
import argparse
import glob
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import checker, numpy_helper

MEAN255 = np.array([0.485, 0.456, 0.406], np.float32) * 255.0
STD255 = np.array([0.229, 0.224, 0.225], np.float32) * 255.0


def fold(model, border=False, nchw=False):
    g = model.graph
    by_name = {n.name: n for n in g.node}
    for need in ("pre_transpose", "pre_sub", "pre_div"):
        if need not in by_name:
            raise ValueError(f"node {need} not found; input graph is not a *_pre.onnx (nodes: {[n.name for n in g.node][:6]})")
    div_out = by_name["pre_div"].output[0]
    consumers = [n for n in g.node if div_out in n.input]
    if len(consumers) != 1 or consumers[0].op_type != "Conv":
        raise ValueError(f"normalized tensor must feed exactly one Conv, got {[(n.name, n.op_type) for n in consumers]}")
    conv = consumers[0]
    if any(a.name == "group" and a.i != 1 for a in conv.attribute):
        raise ValueError(f"conv {conv.name} is grouped; folding per input channel is not valid")
    inits = {i.name: i for i in g.initializer}
    w = numpy_helper.to_array(inits[conv.input[1]]).astype(np.float32)
    if w.ndim != 4 or w.shape[1] != 3:
        raise ValueError(f"conv {conv.name} weight shape {w.shape}, expected [Cout,3,kh,kw]")
    w2 = w / STD255.reshape(1, 3, 1, 1)
    new_w = numpy_helper.from_array(w2, inits[conv.input[1]].name)
    inits[conv.input[1]].CopyFrom(new_w)
    if border:
        pads = [a for a in conv.attribute if a.name == "pads"]
        p = list(pads[0].ints) if pads else [0, 0, 0, 0]
        if len(set(p)) != 1 or p[0] <= 0:
            raise ValueError(f"border mode needs equal positive conv pads, got {p} on {conv.name}")
        pad = p[0]
        if len(conv.input) > 2:
            b = numpy_helper.to_array(inits[conv.input[2]]).astype(np.float32)
        else:
            raise ValueError(f"conv {conv.name} has no bias input to absorb the mean")
        b2 = b - (w2 * MEAN255.reshape(1, 3, 1, 1)).sum(axis=(1, 2, 3))
        inits[conv.input[2]].CopyFrom(numpy_helper.from_array(b2.astype(np.float32), inits[conv.input[2]].name))
        pads[0].ints[:] = [0, 0, 0, 0]
        dims = g.input[0].type.tensor_type.shape.dim  # NHWC
        dims[1].dim_value += 2 * pad
        dims[2].dim_value += 2 * pad
        print(f"border mode: pad {pad}, input now {[d.dim_value for d in dims]}")
    # rewire: Conv reads the transposed tensor directly
    conv.input[0] = by_name["pre_transpose"].output[0]
    for name in ("pre_sub", "pre_div"):
        g.node.remove(by_name[name])
    # drop the now unused mean/std constants
    used = {t for n in g.node for t in n.input}
    for init in [i for i in g.initializer if i.name in ("pre_mean255", "pre_std255") and i.name not in used]:
        g.initializer.remove(init)
    g.input[0].name = "image_nhwc"  # same name; the value now means RGB - mean255
    if nchw:  # latency experiment only: take NCHW input, no Transpose in the graph
        t = by_name["pre_transpose"]
        conv.input[0] = "image_nchw"
        g.node.remove(t)
        d = g.input[0].type.tensor_type.shape.dim
        h, w = d[1].dim_value, d[2].dim_value
        g.input[0].name = "image_nchw"
        d[1].dim_value, d[2].dim_value, d[3].dim_value = 3, h, w
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--border", action="store_true")
    ap.add_argument("--nchw", action="store_true", help="latency experiment: NCHW input without the Transpose node")
    ap.add_argument("--verify-images")
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    m = fold(onnx.load(a.src), a.border, a.nchw)
    checker.check_model(m)
    onnx.save(m, a.dst)
    print("saved", a.dst, [n.name for n in m.graph.node][:4])
    if a.verify_images:
        from PIL import Image
        files = sorted(glob.glob(os.path.join(a.verify_images, "*.png")))
        files = files[:: max(1, len(files) // a.n)][: a.n]
        so = ort.InferenceSession(a.src, providers=["CPUExecutionProvider"])
        sn = ort.InferenceSession(a.dst, providers=["CPUExecutionProvider"])
        worst = 0.0
        for f in files:
            x = np.asarray(Image.open(f).convert("RGB"), np.float32)[None]
            ro = so.run(None, {so.get_inputs()[0].name: x})
            if a.border:
                pad = (sn.get_inputs()[0].shape[1] - x.shape[1]) // 2
                xb = np.empty((1, x.shape[1] + 2 * pad, x.shape[2] + 2 * pad, 3), np.float32)
                xb[:] = MEAN255
                xb[:, pad:-pad, pad:-pad] = x
                rn = sn.run(None, {sn.get_inputs()[0].name: xb})
            else:
                rn = sn.run(None, {sn.get_inputs()[0].name: x - MEAN255.reshape(1, 1, 1, 3)})
            for o, n in zip(ro, rn):
                worst = max(worst, float(np.abs(o - n).max() / (np.abs(o).max() + 1e-9)))
        print(f"verified {len(files)} images, worst relative max error {worst:.2e}")
        return 0 if worst < 1e-4 else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
