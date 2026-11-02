#!/usr/bin/env python3
"""Mean CPU utilisation (middle 60% of each tegrastats log) per category and config of the per-category e2e run."""
import os, re, statistics
R = os.path.expanduser("~/e2e/results/cat_e2e")
cats = "bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper".split()
print("category,config,cpu_busy_pct_mean,cores_online")
agg = {}
for c in cats:
    for cfg in ("dla2", "gpu", "mix"):
        vals = []
        try:
            for l in open(f"{R}/{c}_{cfg}.tegra", errors="ignore"):
                m = re.search(r"CPU \[([^\]]*)\]", l)
                if not m: continue
                cores = [x for x in m[1].split(",") if not x.startswith("off")]
                pct = [int(re.match(r"(\d+)%", x)[1]) for x in cores if re.match(r"(\d+)%", x)]
                if pct: vals.append(sum(pct) / len(pct))
        except OSError:
            continue
        n = len(vals); vals = vals[n // 5: n * 4 // 5]
        if vals:
            agg.setdefault(cfg, []).append(statistics.mean(vals))
            print(f"{c},{cfg},{statistics.mean(vals):.1f},6")
for k, v in agg.items():
    print(f"MEAN,{k},{statistics.mean(v):.1f},")
