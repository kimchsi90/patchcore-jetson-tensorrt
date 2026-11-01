"""Resize-method study: bank trained with resize M vs inference with resize M (aligned) and vs the Pillow-trained bank (mismatched).

Usage (PYTHONUTF8=1, c:\\dev_venvs\\PatchcoreJetson\\Scripts\\python.exe):
  python resize_study.py imgdiff  --cat screw                      # pixel difference between resize methods
  python resize_study.py run --cat screw --methods pil_aa,area,linear,tv_bilinear_noaa [--ratio 0.01] [--tag-suffix ""]
  python resize_study.py mismatch --cat screw --methods ...        # existing Pillow bank, test frames resized by M
Banks go to work\\<cat>_256\\resize_<tag>\\ (new subfolder). Results: out\\resize_study\\<cat>_<kind>_<tag>.json and results.csv.
Evaluation: original frames resized in-pipeline by M -> fused graph (NHWC 0..255 input) -> host_post() -> metrics().
"""
import argparse
import csv
import ctypes
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import onnx
import torch
from PIL import Image
from torchvision.transforms import v2

sys.path.insert(0, r"D:\jetson_test\trt")
import graph_opt as G  # noqa: E402

PY = r"c:\dev_venvs\PatchcoreJetson\Scripts\python.exe"
EXPORT = r"D:\jetson_test\patchcore\export"
HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = r"D:\jetson_test\trt\out\resize_study"
METHODS = ("pil_aa", "area", "linear", "tv_bilinear_noaa")
_aa = v2.Resize((256, 256), antialias=True)
_noaa = v2.Resize((256, 256), antialias=False)


def resize_img(arr, method):
    """arr uint8 [H,W,3] RGB -> uint8 [256,256,3] (same code path as make_knn_head_resize.load_resized)."""
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected uint8 HxWx3, got shape={arr.shape} dtype={arr.dtype}")
    if method in ("pil_aa", "tv_bilinear_noaa"):
        t = v2.functional.pil_to_tensor(Image.fromarray(arr))
        t = (_aa if method == "pil_aa" else _noaa)(t)
        return np.array(v2.functional.to_pil_image(t))
    if method in ("area", "linear"):
        return cv2.resize(arr, (256, 256), interpolation=cv2.INTER_AREA if method == "area" else cv2.INTER_LINEAR)
    raise ValueError(f"unknown method {method}")


def load_test(cat, rows, method):
    out = []
    for n, _ in rows:
        arr = np.asarray(Image.open(G.original_path(cat, n)).convert("RGB"))
        out.append(resize_img(arr, method).astype(np.float32)[None])
    return out


def evaluate(onnx_path, cat, rows, masks, frames):
    sess = G.session(onnx.load(onnx_path))
    names = [o.name for o in sess.get_outputs()]
    ms, xs = [], []
    for f in frames:
        if f.shape != (1, 256, 256, 3):
            raise ValueError(f"bad frame shape={f.shape} dtype={f.dtype}")
        o = dict(zip(names, sess.run(None, {"image_nhwc": f})))
        ms.append(o["m"][0])
        xs.append(o["xproj"][0])
    m, x = np.stack(ms), np.stack(xs)
    if m.shape[1] != 1:
        raise ValueError(f"unexpected m shape={m.shape} dtype={m.dtype}")
    smax, stopk, maps = G.host_post(m, x)
    return G.metrics(rows, masks, smax, stopk, maps), (smax, stopk, maps)


def run_step(args, env_extra, what):
    env = {**os.environ, "PYTHONUTF8": "1", **env_extra}
    r = subprocess.run([PY, "-W", "ignore"] + args, env=env, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(f"step failed ({what}) rc={r.returncode}\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
    return r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""


def build_bank(cat, method, ratio, tag, dim=384):
    """Same call order as p1_build_onnx.ps1 (steps after prep_data); output folder work\\<cat>_256\\resize_<tag>."""
    sub = rf"_256\resize_{tag}"
    wdir = rf"D:\jetson_test\patchcore\work\{cat}{sub}"
    os.makedirs(wdir, exist_ok=True)
    env = {"DATASET_ROOT": r"D:\jetson_test\patchcore\datasets\mvtec_ad", "NPU_SIZE": "256", "NPU_DIR_SUFFIX": sub,
           "NPU_BACKBONE_ONNX": r"D:\jetson_test\patchcore\work\backbones\wide_resnet50_2_256.onnx",
           "HEAD_SEED": "1", "RESIZE_METHOD": method}
    ttag = f"random{dim}_c{ratio}"
    final = os.path.join(wdir, f"fused_{ttag}_rmax_pf_pre.onnx")
    if os.path.isfile(final):
        raise FileExistsError(f"refusing to overwrite {final}")
    t0 = time.time()
    run_step([os.path.join(HERE, "make_knn_head_resize.py"), str(dim), str(ratio), "random", cat], env, "make_knn_head")
    run_step([os.path.join(EXPORT, "make_fused.py"), ttag, cat], env, "make_fused")
    run_step([os.path.join(EXPORT, "rewrite_knn_reduce.py"), os.path.join(wdir, f"fused_{ttag}.onnx"), os.path.join(wdir, f"fused_{ttag}_rmax.onnx")], env, "rmax")
    run_step([os.path.join(EXPORT, "rewrite_proj_first.py"), os.path.join(wdir, f"fused_{ttag}_rmax.onnx"), os.path.join(wdir, f"fused_{ttag}_rmax_pf.onnx")], env, "pf")
    run_step([r"D:\jetson_test\trt\add_preproc.py", os.path.join(wdir, f"fused_{ttag}_rmax_pf.onnx"), final], env, "preproc")
    return final, time.time() - t0


def record(cat, kind, method, tag, ratio, met, extra=None):
    row = {"cat": cat, "kind": kind, "test_resize": method, "bank_tag": tag, "ratio": ratio, **met, **(extra or {})}
    os.makedirs(OUTD, exist_ok=True)
    json.dump(row, open(os.path.join(OUTD, f"{cat}_{kind}_{tag}_test{method}_c{ratio}.json"), "w"), indent=1)
    path = os.path.join(OUTD, "results.csv")
    new = not os.path.isfile(path)
    keys = ["cat", "kind", "test_resize", "bank_tag", "ratio", "image_auroc_max", "image_auroc_top10", "pixel_auroc", "aupro", "build_s"]
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)
    print(row, flush=True)


def main():
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)   # BELOW_NORMAL
    torch.set_num_threads(4)
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["imgdiff", "run", "mismatch"])
    ap.add_argument("--cat", required=True)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--ratio", type=float, default=0.01)
    ap.add_argument("--tag-suffix", default="", help="bank folder tag = <method><suffix>; test resize stays <method> (use for determinism rebuild)")
    a = ap.parse_args()
    cat, methods = a.cat, a.methods.split(",")
    rows = G.read_rows(cat)
    if a.cmd == "imgdiff":
        res = {}
        files = [G.original_path(cat, n) for n, _ in rows[::max(1, len(rows) // 12)]]
        for f in files:
            arr = np.asarray(Image.open(f).convert("RGB"))
            r = {m: resize_img(arr, m).astype(np.int16) for m in METHODS}
            for p, q in (("linear", "tv_bilinear_noaa"), ("area", "pil_aa"), ("linear", "pil_aa"), ("area", "linear")):
                d = np.abs(r[p] - r[q])
                res.setdefault(f"{p}_vs_{q}", []).append((float(d.mean()), int(d.max())))
        out = {k: {"mean_abs": round(float(np.mean([v[0] for v in vs])), 4), "max_abs": max(v[1] for v in vs)} for k, vs in res.items()}
        out["n_images"] = len(files)
        print(cat, out)
        os.makedirs(OUTD, exist_ok=True)
        json.dump(out, open(os.path.join(OUTD, f"imgdiff_{cat}.json"), "w"), indent=1)
        return 0
    masks = G.load_masks(cat, rows)
    for m in methods:
        frames = load_test(cat, rows, m)
        if a.cmd == "mismatch":
            base = G.base_path(cat)
            met, _ = evaluate(base, cat, rows, masks, frames)
            record(cat, "mismatch", m, "pil_aa_existing", a.ratio, met)
        else:
            tag = m + a.tag_suffix
            c = a.ratio
            final, secs = build_bank(cat, m, c, tag)
            met, _ = evaluate(final, cat, rows, masks, frames)
            record(cat, "aligned", m, tag, c, met, {"build_s": round(secs, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
