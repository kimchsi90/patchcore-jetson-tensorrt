"""Build the post-only ONNX (inputs m [1,1,32,32], xproj [1,384,32,32]; outputs score_max, score_top10, map) and verify it.

Reuses graph_opt.add_post (same nodes as the post part of bottle_post.onnx, opset 12). Checks the ONNX with onnxruntime against
graph_opt.host_post (float64 + scipy) on the stored fp32 dumps (83 frames). Run: PYTHONUTF8=1 python pc/build_post_only.py
"""
import json
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper

sys.path.insert(0, r"D:\jetson_test\trt")
import graph_opt as G  # noqa: E402

OUT = r"D:\jetson_test\trt\out"


def build(mode="conv", gh=32, c=384):
    """Dummy graph m_in/xproj_in -> Identity -> m/xproj so add_post can be reused, then the Identity nodes are removed."""
    f32 = TensorProto.FLOAT
    ins = [helper.make_tensor_value_info("m", f32, [1, 1, gh, gh]), helper.make_tensor_value_info("xproj", f32, [1, c, gh, gh])]
    outs = [helper.make_tensor_value_info("m", f32, [1, 1, gh, gh]), helper.make_tensor_value_info("xproj", f32, [1, c, gh, gh])]
    # Graph outputs must differ from inputs, so build with distinct input names and rename afterwards.
    ins[0].name, ins[1].name = "m_in", "xproj_in"
    nodes = [helper.make_node("Identity", ["m_in"], ["m"], name="id_m"), helper.make_node("Identity", ["xproj_in"], ["xproj"], name="id_x")]
    dummy = helper.make_model(helper.make_graph(nodes, "dummy", ins, outs), opset_imports=[helper.make_opsetid("", 12)])
    dummy.ir_version = 7
    mdl = G.add_post(dummy, mode)
    gr = mdl.graph
    for n in list(gr.node):
        if n.name in ("id_m", "id_x"):
            gr.node.remove(n)
    gr.input[0].name, gr.input[1].name = "m", "xproj"
    onnx.checker.check_model(mdl)
    return mdl


def main():
    mdl = build("conv")
    path = os.path.join(OUT, "bottle_post_only.onnx")
    onnx.save(mdl, path)
    print("saved", path, os.path.getsize(path), "bytes; ops", sorted({n.op_type for n in mdl.graph.node}),
          "opset", [(o.domain, o.version) for o in mdl.opset_import])
    m = np.load(os.path.join(OUT, "bottle_256_ort_fp32_m.npy"))
    x = np.load(os.path.join(OUT, "bottle_256_ort_fp32_x.npy"))
    if x.ndim != 4 or m.shape[0] != x.shape[0]:
        raise ValueError("unexpected dump shapes m=%s x=%s" % (m.shape, x.shape))
    m = m.reshape(m.shape[0], 1, x.shape[2], x.shape[3]).astype(np.float32)
    x = x.astype(np.float32)
    ref = G.host_post(m, x)
    sess = G.session(mdl)
    smax, stopk, maps = [], [], []
    for i in range(x.shape[0]):
        o = dict(zip([q.name for q in sess.get_outputs()], sess.run(None, {"m": m[i:i + 1], "xproj": x[i:i + 1]})))
        smax.append(float(o["score_max"][0]))
        stopk.append(float(o["score_top10"][0]))
        maps.append(o["map"][0, 0])
    got = (np.array(smax), np.array(stopk), np.stack(maps).astype(np.float32))
    d = G.diffs(ref, got)
    d["score_max_range"] = [float(ref[0].min()), float(ref[0].max())]
    d["map_range"] = [float(ref[2].min()), float(ref[2].max())]
    d["n_frames"] = int(x.shape[0])
    print(json.dumps(d, indent=1))
    with open(os.path.join(OUT, "e2e_split_post_only_check.json"), "w") as f:
        json.dump(d, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
