"""Dump m and xproj per test image from an ONNX (ort) or TensorRT engine (trt), same format as board/dump_full.py.

Usage: python dump_trt.py --backend ort|trt --model <onnx|engine> --images <dir> --labels <test_labels.csv> --out <prefix>
Input : <dir>/<name>.png, already resized to SxS by export/prep_data.py (not resized again here).
        Fed as float32 NHWC [1,S,S,3], RGB 0..255 (model must be the *_pre.onnx graph).
Output: <prefix>_m.npy [N, H*W] float32 (flattened rmax map, one row per image)
        <prefix>_x.npy [N, C, H, W] float32 (xproj, C=384, H=W=S/8)
"""
import argparse
import csv
import os
import sys

import numpy as np


class Backend:
    def infer(self, nhwc: np.ndarray):
        """nhwc float32 [1,S,S,3] -> (m, xproj) arrays."""
        raise NotImplementedError


class OrtBackend(Backend):
    def __init__(self, model: str):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_names = [o.name for o in self.sess.get_outputs()]

    def infer(self, nhwc):
        outs = self.sess.run(None, {self.in_name: nhwc})
        # xproj has channels and a grid; m has a single channel (same rule as dump_full.py)
        a, b = outs[0], outs[1]
        return (b, a) if a.shape[1] > 1 and a.shape[2] > 1 else (a, b)


class TrtBackend(Backend):
    """TensorRT engine through trt_runner.TrtRunner (ctypes CUDA, runs on the Jetson only)."""

    def __init__(self, model: str, dla_core=None):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from trt_runner import TrtRunner
        self.rn = TrtRunner(model, dla_core)

    def infer(self, nhwc):
        outs = self.rn.infer(nhwc)
        if "m" in outs and "xproj" in outs:
            return outs["m"], outs["xproj"]
        a, b = list(outs.values())[:2]
        return (b, a) if a.shape[1] > 1 and a.shape[2] > 1 else (a, b)


def load_rgb(path, use_cv2):
    """float32 NHWC [1,S,S,3], RGB 0..255. ort path uses PIL (unchanged); trt path uses cv2 (board has no PIL)."""
    if not use_cv2:
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGB"), np.float32)[None]
    import cv2
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    x = np.ascontiguousarray(bgr[:, :, ::-1]).astype(np.float32)[None]
    pad = int(os.environ.get("BORDER_INPUT", "0"))  # engines built from fold_norm.py --border graphs expect a mean-filled frame
    if pad:
        mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 1, 3) * 255.0
        xb = np.empty((1, x.shape[1] + 2 * pad, x.shape[2] + 2 * pad, 3), np.float32)
        xb[:] = mean
        xb[:, pad:-pad, pad:-pad] = x
        x = xb
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ort", "trt"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dla-core", type=int, default=None, help="trt backend: execute on this DLA core")
    a = ap.parse_args()

    names = [r[0] for r in list(csv.reader(open(a.labels)))[1:] if r]
    be = OrtBackend(a.model) if a.backend == "ort" else TrtBackend(a.model, a.dla_core)
    ms, xs = [], []
    for name in names:
        path = os.path.join(a.images, f"{name}.png")
        if not os.path.isfile(path):
            print(f"cannot read image: {path}")
            return 1
        rgb = load_rgb(path, a.backend == "trt")
        if rgb is None:
            print(f"cannot decode image: {path}")
            return 1
        m, xp = be.infer(np.ascontiguousarray(rgb))
        if xp.ndim != 4 or m.ndim != 4:
            print(f"bad outputs for {name}: m={m.shape} x={xp.shape} dtype={xp.dtype} input={rgb.shape}")
            return 1
        ms.append(m.reshape(-1).astype(np.float32))
        xs.append(xp[0].astype(np.float32))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.save(a.out + "_m.npy", np.stack(ms))
    np.save(a.out + "_x.npy", np.stack(xs))
    print(f"done: m={np.stack(ms).shape} x={np.stack(xs).shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
