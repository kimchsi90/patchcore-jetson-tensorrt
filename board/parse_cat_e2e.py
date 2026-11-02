#!/usr/bin/env python3
"""Per-category end-to-end summary: FPS, p50 latency, power (middle 60% of tegrastats), mJ/img for dla2 / gpu / mix."""
import json, os, re
R = os.path.expanduser("~/e2e/results/cat_e2e")
cats = "bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper".split()
print("category,frame_hw,config,fps,p50_ms,power_mW,mJ_per_img,cpu_pct")
for c in cats:
    for cfg in ("dla2", "gpu", "mix"):
        try:
            d = json.load(open(f"{R}/{c}_{cfg}.json"))
        except (OSError, ValueError):
            print(f"{c},,{cfg},,,,,"); continue
        p = []
        for l in open(f"{R}/{c}_{cfg}.tegra", errors="ignore"):
            m = re.search(r"VDD_IN (\d+)mW", l)
            if m: p.append(int(m[1]))
        n = len(p); p = p[n // 5: n * 4 // 5]
        pw = sum(p) / max(1, len(p))
        hw = d.get("frame_hw") or d.get("frame_shape") or ""
        print(f"{c},{hw},{cfg},{d['fps']:.1f},{d['latency_ms']['p50']:.1f},{pw:.0f},{pw / d['fps']:.1f},{d.get('cpu_util_pct', '')}")
