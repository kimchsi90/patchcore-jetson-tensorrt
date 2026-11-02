#!/usr/bin/env python3
"""Summarize the power-mode sweep: FPS (median of reps), mean VDD_IN while the config ran is approximated by the whole-mode log, and temperature."""
import glob, json, os, re, statistics, sys
R = os.path.expanduser("~/e2e/results/modes")
names = {}
for l in open(f"{R}/modes.log"):
    m = re.match(r"== mode (\d+) NV Power Mode: (\S+) online=(\S+) gpu_max=(\d+) cpu0_max=(\d+)", l)
    if m:
        names[int(m[1])] = (m[2], m[3], int(m[4]) // 10**6, int(m[5]) // 1000)
print("mode,name,online,gpu_max_MHz,cpu_max_MHz,config,fps_min,fps_max,reps,sweep_power_mean_mW,gpu_temp_max_C")
for mid in (8, 7, 6, 2, 1, 0, 5, 4, 3):
    pw, tmax = [], 0.0
    try:
        for l in open(f"{R}/tegra_{mid}.log", errors="ignore"):
            m = re.search(r"VDD_IN (\d+)mW", l)
            if m: pw.append(int(m[1]))
            m = re.search(r"GPU@([\d.]+)C", l)
            if m: tmax = max(tmax, float(m[1]))
    except FileNotFoundError:
        pass
    byc = {}
    for f in sorted(glob.glob(f"{R}/../modes_{mid}/*.json")):
        c = re.sub(r"_r\d+\.json$", "", os.path.basename(f))
        try:
            d = json.load(open(f))
        except ValueError:
            continue
        if d.get("fps") is not None:
            byc.setdefault(c, []).append(d["fps"])
    nm = names.get(mid, ("?", "?", 0, 0))
    for c, v in byc.items():
        print(f"{mid},{nm[0]},{nm[1]},{nm[2]},{nm[3]},{c},{min(v):.1f},{max(v):.1f},{len(v)},{sum(pw)//max(1,len(pw))},{tmax}")
