"""Summarize the GPU + DLA combination runs of board/combo_suite.sh.

Usage: python parse_combo.py <combo_dir>
Each group (one tegrastats log) runs several trtexec processes at the same time. The group FPS is the sum of the per-process throughput,
the power is the mean VDD_IN over the group log and the energy per image is power / FPS.
"""
import collections
import glob
import os
import re
import statistics
import sys

GROUPS = ("gpu_dla0dla1_int8mix", "gpu_dla0_int8mix", "dla0dla1_int8mix", "solo_gpu_mix", "solo_dla0_int8mix",
          "solo_dla1_int8mix", "solo_dla0_fp16", "gpu_dla0dla1_fp16")
RE_QPS = re.compile(r"Throughput:\s*([\d.]+)\s*qps")
RE_PWR = re.compile(r"VDD_IN (\d+)mW/")
RE_GPU_T = re.compile(r"GPU@([\d.]+)C")


def main():
    d = sys.argv[1]
    groups = collections.OrderedDict()
    for f in sorted(glob.glob(os.path.join(d, "*.log"))):
        base = os.path.basename(f)[:-4]
        for g in GROUPS:  # longest names first, so "gpu_dla0dla1_*" is not taken for "gpu_dla0_*"
            if base.startswith(g + "_"):
                groups.setdefault(g, []).append((base[len(g) + 1:], f))
                break
    print("%-28s %8s %8s %8s %6s  %s" % ("group", "FPS", "mW", "mJ/img", "Tmax", "parts"))
    for g, items in groups.items():
        total, parts = 0.0, []
        for label, f in items:
            with open(f, errors="replace") as fh:
                q = RE_QPS.findall(fh.read())
            v = float(q[-1]) if q else 0.0
            total += v
            parts.append("%s=%.1f" % (label, v))
        tegra = os.path.join(d, g + ".tegra")
        if not os.path.exists(tegra):
            print("%-28s no tegrastats log" % g)
            continue
        with open(tegra, errors="replace") as fh:
            text = fh.read()
        pw = [int(x) for x in RE_PWR.findall(text)]
        tmax = max([float(x) for x in RE_GPU_T.findall(text)] or [0.0])
        mw = statistics.mean(pw) if pw else 0.0
        print("%-28s %8.1f %8.0f %8.1f %6.1f  %s" % (g, total, mw, mw / total if total else 0.0, tmax, " ".join(parts)))


if __name__ == "__main__":
    main()
