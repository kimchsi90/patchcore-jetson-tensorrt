"""Score a dump: image AUROC (max and top-k mean), pixel AUROC, AUPRO, fp16 range peaks. Prints one JSON line.

Usage: python score_dump.py --dump <prefix> --labels <csv> --masks <dir> [--topk 10] [--sigma 4]
Semantics copied from rknn/eval_full_dump.py (score, nearest upsample x8, gaussian_filter) and
rknn/eval_aupro.py (aupro(), gaussian mode="mirror").
"""
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

sys.path.insert(0, r"D:\jetson_test\patchcore\rknn")
from eval_aupro import aupro  # noqa: E402  (module-level ROOT constant is unused here)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=4.0)
    a = ap.parse_args()

    m = np.load(a.dump + "_m.npy")
    x = np.load(a.dump + "_x.npy")
    rows = list(csv.reader(open(a.labels)))[1:]
    if m.shape[0] != len(rows) or x.shape[0] != len(rows):
        print(f"count mismatch: m={m.shape} x={x.shape} dtype={x.dtype} labels={len(rows)}")
        return 1
    h, w = x.shape[-2:]
    size = h * 8
    smaps, masks, labels, smax, stopk = [], [], [], [], []
    sqmax = 0.0
    for i, (name, label) in enumerate(rows):
        sq = (x[i].astype(np.float64) ** 2).sum(0).reshape(-1)
        sqmax = max(sqmax, float(sq.max()))
        s = np.sqrt(np.clip(sq - m[i].astype(np.float64), 0, None))
        smax.append(float(s.max()))
        stopk.append(float(np.sort(s)[::-1][: a.topk].mean()))
        smaps.append(s.reshape(h, w))
        labels.append(int(label))
        if int(label):
            mk = cv2.imread(os.path.join(a.masks, f"{name}.png"), cv2.IMREAD_GRAYSCALE)
            if mk is None or mk.shape != (size, size):
                print(f"bad mask {name}: {None if mk is None else mk.shape}, expected {(size, size)}")
                return 1
            masks.append(mk > 127)
        else:
            masks.append(np.zeros((size, size), bool))
    up = [np.kron(s, np.ones((size // h, size // w))).astype(np.float32) for s in smaps]
    maps = np.stack([gaussian_filter(u, sigma=a.sigma, mode="mirror") for u in up]).astype(np.float32)
    masks = np.stack(masks)
    res = {
        "image_auroc_max": round(float(roc_auc_score(labels, smax)), 4),
        f"image_auroc_top{a.topk}": round(float(roc_auc_score(labels, stopk)), 4),
        "pixel_auroc": round(float(roc_auc_score(masks.reshape(-1), maps.reshape(-1))), 4),
        "aupro": round(aupro(maps, masks), 4),
        "sigma": a.sigma,
        "peak_abs_m": float(np.abs(m).max()),
        "peak_xproj_sq_norm": sqmax,
        "fp16_limit": 65504,
        "n": len(rows),
    }
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
