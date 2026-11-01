"""Prepend an NHWC raw-RGB (0..255) preprocessing stage to a PatchCore ONNX graph.

Usage: python add_preproc.py <src.onnx> <dst.onnx> [--verify-images DIR] [--n 5]
New input: image_nhwc float32 [1,S,S,3], RGB 0..255.
Stage: Transpose(0,3,1,2) -> Sub(mean*255) -> Div(std*255), feeding the old `image` consumers.
Opset and IR version of the source graph are kept. Outputs are unchanged.
"""
import argparse
import glob
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, checker, helper, numpy_helper
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def add_preproc(model: onnx.ModelProto) -> onnx.ModelProto:
    g = model.graph
    if len(g.input) != 1:
        raise ValueError(f"expected 1 graph input, got {[i.name for i in g.input]}")
    old = g.input[0]
    dims = [d.dim_value for d in old.type.tensor_type.shape.dim]
    if len(dims) != 4 or dims[1] != 3 or old.type.tensor_type.elem_type != TensorProto.FLOAT:
        raise ValueError(f"unexpected input {old.name}: dims={dims} elem_type={old.type.tensor_type.elem_type}")
    n, _, h, w = dims
    old_name = old.name
    # Rename the old input tensor everywhere it is consumed, so the new stage can produce it.
    for node in g.node:
        for k, name in enumerate(node.input):
            if name == old_name:
                node.input[k] = old_name + "_pre_in"
    new_in = helper.make_tensor_value_info("image_nhwc", TensorProto.FLOAT, [n, h, w, 3])
    mean = numpy_helper.from_array((MEAN * 255.0).reshape(1, 3, 1, 1), "pre_mean255")
    std = numpy_helper.from_array((STD * 255.0).reshape(1, 3, 1, 1), "pre_std255")
    nodes = [
        helper.make_node("Transpose", ["image_nhwc"], ["pre_nchw"], perm=[0, 3, 1, 2], name="pre_transpose"),
        helper.make_node("Sub", ["pre_nchw", "pre_mean255"], ["pre_centered"], name="pre_sub"),
        helper.make_node("Div", ["pre_centered", "pre_std255"], [old_name + "_pre_in"], name="pre_div"),
    ]
    # Initializers/inputs must precede use; nodes are topologically sorted, so put the stage first.
    existing = list(g.node)
    del g.node[:]
    g.node.extend(nodes + existing)
    g.initializer.extend([mean, std])
    del g.input[:]
    g.input.extend([new_in])
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--verify-images", default=None)
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()

    model = onnx.load(a.src)
    opset = [(o.domain, o.version) for o in model.opset_import]
    model = add_preproc(model)
    checker.check_model(model)
    onnx.save(model, a.dst)
    print(f"saved {a.dst} opset={opset} ir={model.ir_version}")
    for t in list(model.graph.input) + list(model.graph.output):
        print(" ", t.name, helper.tensor_dtype_to_np_dtype(t.type.tensor_type.elem_type),
              [d.dim_value for d in t.type.tensor_type.shape.dim])

    if a.verify_images:
        files = sorted(glob.glob(os.path.join(a.verify_images, "*.png")))[:: max(1, 83 // a.n)][: a.n]
        if not files:
            print(f"no png files in {a.verify_images}")
            return 1
        so = ort.InferenceSession(a.src, providers=["CPUExecutionProvider"])
        sn = ort.InferenceSession(a.dst, providers=["CPUExecutionProvider"])
        names = [o.name for o in so.get_outputs()]
        worst = {k: 0.0 for k in names}
        for f in files:
            rgb = np.asarray(Image.open(f).convert("RGB"), np.float32)[None]   # [1,S,S,3] 0..255
            norm = ((rgb / 255.0 - MEAN) / STD).transpose(0, 3, 1, 2).astype(np.float32)
            ref = so.run(None, {so.get_inputs()[0].name: np.ascontiguousarray(norm)})
            new = sn.run(None, {"image_nhwc": rgb})
            for k, r, v in zip(names, ref, new):
                if r.shape != v.shape:
                    print(f"shape mismatch {k}: {r.shape} vs {v.shape} dtype={v.dtype} device=cpu")
                    return 1
                worst[k] = max(worst[k], float(np.abs(r - v).max()))
        print(f"verified on {len(files)} images, max abs diff: {worst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
