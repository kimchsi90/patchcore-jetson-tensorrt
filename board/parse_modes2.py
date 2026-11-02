#!/usr/bin/env python3
"""Power-mode sweep v2: per mode and config the FPS of 3 repetitions, mean power of the middle 60% of each run and mJ/img."""
import glob, json, os, re, statistics
R = os.path.expanduser("~/e2e/results/modes2")
names = {}
for l in open(f"{R}/modes.log"):
    m = re.match(r"== mode (\d+) NV Power Mode: (\S+) online=(\S+) gpu_max=(\d+)", l)
    if m: names[int(m[1])] = (m[2], m[3], int(m[4]) // 10**6)
print("mode,name,online,gpu_max_MHz,config,fps_reps,fps_median,power_mW,mJ_per_img,gpu_temp_max_C")
for mid in (8, 7, 6, 2, 1, 0, 5, 4, 3):
    for cfg in ("p1gpu", "dla2", "mix"):
        fps, pws, tmax = [], [], 0.0
        for r in (1, 2, 3):
            try:
                fps.append(json.load(open(f"{R}/m{mid}_{cfg}_r{r}.json"))["fps"])
            except (OSError, ValueError, KeyError):
                continue
            p = []
            for l in open(f"{R}/m{mid}_{cfg}_r{r}.tegra", errors="ignore"):
                m = re.search(r"VDD_IN (\d+)mW", l)
                if m: p.append(int(m[1]))
                m = re.search(r"GPU@([\d.]+)C", l)
                if m: tmax = max(tmax, float(m[1]))
            n = len(p); p = p[n // 5: n * 4 // 5]
            if p: pws.append(sum(p) / len(p))
        if not fps: continue
        nm = names.get(mid, ("?", "?", 0))
        mf = statistics.median(fps); pw = statistics.mean(pws) if pws else 0
        print(f"{mid},{nm[0]},{nm[1]},{nm[2]},{cfg},{'|'.join('%.1f' % x for x in fps)},{mf:.1f},{pw:.0f},{pw / mf:.1f},{tmax}")
