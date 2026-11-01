"""Score e2e npz dumps (score_max, score_top10, maps) written by board/bench_e2e.py --mode acc.

Usage: python e2e_score.py NAME=path.npz [NAME=path.npz ...] --out out/e2e_accuracy.json
Uses graph_opt.metrics (image AUROC from score_max and top-10 mean, pixel AUROC, AUPRO) against the 256 masks, and the
difference to the fp32 host reference (stored ort dumps) for context. Light on RAM: one dump at a time.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, r"D:\jetson_test\trt")
import graph_opt as G  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cat = "bottle"
    rows = G.read_rows(cat)
    masks = G.load_masks(cat, rows)
    ref = G.host_baseline(cat)
    res = {"reference_fp32_host": G.metrics(rows, masks, *ref)}
    for it in a.items:
        name, _, path = it.partition("=")
        if not os.path.isfile(path):
            raise IOError("missing dump %s" % path)
        d = np.load(path)
        names = [str(n) for n in d["names"]]
        if names != [n for n, _ in rows]:
            raise ValueError("%s: frame order differs from test_labels.csv (%s...)" % (path, names[:2]))
        maps = d["maps"].astype(np.float32)
        if maps.shape != (len(rows), 256, 256):
            raise ValueError("%s: maps shape %s dtype %s, expected (%d,256,256)" % (path, maps.shape, maps.dtype, len(rows)))
        got = (d["score_max"], d["score_top10"], maps)
        res[name] = {**G.metrics(rows, masks, *got), **G.diffs(ref, got)}
        print(name, res[name], flush=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
