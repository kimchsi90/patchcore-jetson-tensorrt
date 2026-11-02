"""Pick a small, stratified set of full-resolution MVTec AD test frames per category for the end-to-end benchmark.

Output: <out>/<category>/<defect>_<idx>.png (original pixels, copied unchanged) and <out>/<category>/test_labels.csv (name,label).
Names follow the bottle test_labels.csv convention (<defect>_<NNN>, good_<NNN>). Usage: python prepare_e2e_frames.py <mvtec_root> <out> [n_good n_bad]
"""
import csv
import os
import shutil
import sys


def pick(items, n):
    if len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def main():
    root, out = sys.argv[1], sys.argv[2]
    n_good = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    n_bad = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    if not os.path.isdir(root):
        raise FileNotFoundError(f"MVTec root not found: {root}")
    for cat in sorted(os.listdir(root)):
        tdir = os.path.join(root, cat, "test")
        if not os.path.isdir(tdir):
            continue
        good, bad = [], []
        for defect in sorted(os.listdir(tdir)):
            for f in sorted(os.listdir(os.path.join(tdir, defect))):
                if not f.lower().endswith(".png"):
                    continue
                name = f"{defect}_{os.path.splitext(f)[0]}"
                (good if defect == "good" else bad).append((name, os.path.join(tdir, defect, f)))
        chosen = [(n, p, 0) for n, p in pick(good, n_good)] + [(n, p, 1) for n, p in pick(bad, n_bad)]
        dst = os.path.join(out, cat)
        os.makedirs(dst, exist_ok=True)
        with open(os.path.join(dst, "test_labels.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["name", "label"])
            for n, p, lab in chosen:
                shutil.copyfile(p, os.path.join(dst, n + ".png"))
                w.writerow([n, lab])
        print(cat, len(chosen), "frames")


if __name__ == "__main__":
    main()
