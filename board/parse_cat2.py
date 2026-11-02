#!/usr/bin/env python3
"""Per-category e2e v2: 3 repetitions, FPS median, power (middle 60%), mJ/img and CPU busy % for dla2 / gpu / gpudla0 / mix."""
import json, os, re, statistics
R = os.path.expanduser("~/e2e/results/cat_e2e2")
cats = "bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper".split()
print("category,config,fps_reps,fps_median,fps_spread_pct,power_mW,mJ_per_img,cpu_busy_pct")
agg = {}
for c in cats:
    for cfg in ("dla2", "gpu", "gpudla0", "mix"):
        fps, pws, cpus = [], [], []
        for r in (1, 2, 3):
            try:
                fps.append(json.load(open(f"{R}/{c}_{cfg}_r{r}.json"))["fps"])
            except (OSError, ValueError, KeyError):
                continue
            p, u = [], []
            for l in open(f"{R}/{c}_{cfg}_r{r}.tegra", errors="ignore"):
                m = re.search(r"VDD_IN (\d+)mW", l)
                if m: p.append(int(m[1]))
                m = re.search(r"CPU \[([^\]]*)\]", l)
                if m:
                    pc = [int(re.match(r"(\d+)%", x)[1]) for x in m[1].split(",") if re.match(r"\d+%", x)]
                    if pc: u.append(sum(pc) / len(pc))
            n = len(p); p = p[n // 5: n * 4 // 5]; u = u[n // 5: n * 4 // 5]
            if p: pws.append(sum(p) / len(p))
            if u: cpus.append(sum(u) / len(u))
        if not fps: continue
        mf = statistics.median(fps); pw = statistics.mean(pws); cu = statistics.mean(cpus)
        spread = (max(fps) - min(fps)) / mf * 100
        agg.setdefault(cfg, []).append((mf, pw / mf, cu, spread, c))
        print(f"{c},{cfg},{'|'.join('%.1f' % x for x in fps)},{mf:.1f},{spread:.1f},{pw:.0f},{pw / mf:.1f},{cu:.1f}")
for k, v in agg.items():
    f = [x[0] for x in v]
    print(f"#MEAN,{k},,{statistics.mean(f):.1f},{statistics.mean(x[3] for x in v):.1f},,{statistics.mean(x[1] for x in v):.1f},{statistics.mean(x[2] for x in v):.1f}")
