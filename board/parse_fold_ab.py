#!/usr/bin/env python3
"""Summarize the fold A/B run (orig vs folded DLA backbone engine): FPS, p50 latency, power and GPU utilisation of the steady middle of each run."""
import json, os, re, statistics
R = os.path.expanduser("~/e2e/results/fold_ab")
print("variant,fps_r1,fps_r2,fps_r3,fps_median,p50_ms,power_mW,gr3d_pct,mJ_per_img")
for v in ("orig", "fold"):
    fps, lat, pw, gr = [], [], [], []
    for r in (1, 2, 3):
        d = json.load(open(f"{R}/ab_{v}_r{r}.json"))
        fps.append(d["fps"]); lat.append(d["latency_ms"]["p50"])
        p, g = [], []
        for l in open(f"{R}/{v}_r{r}.tegra", errors="ignore"):
            m = re.search(r"VDD_IN (\d+)mW", l)
            if m: p.append(int(m[1]))
            m = re.search(r"GR3D_FREQ (\d+)%", l)
            if m: g.append(int(m[1]))
        n = len(p); p = p[n // 5: n * 4 // 5]; g = g[n // 5: n * 4 // 5]
        pw.append(sum(p) / len(p)); gr.append(sum(g) / len(g))
    mf = statistics.median(fps)
    print(f"{v},{fps[0]:.1f},{fps[1]:.1f},{fps[2]:.1f},{mf:.1f},{statistics.median(lat):.1f},{statistics.mean(pw):.0f},{statistics.mean(gr):.1f},{statistics.mean(pw)/mf:.1f}")
