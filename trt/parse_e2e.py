"""Aggregate the end-to-end suite (board/e2e_final.sh + e2e/run_suite.sh) into one table.

Usage: python parse_e2e.py <results_dir> [out_csv]
Per config (repetitions *_r1..*_rN are merged): median FPS and the min/max over repetitions, p50 latency, system CPU %, peak temperature,
and from the single tegrastats log the mean VDD_IN during the measured window of each repetition (window = start + warm-up .. + measured wall time).
"""
import csv
import datetime
import glob
import json
import os
import re
import statistics
import sys

RE_START = re.compile(r"^== (\S+) .*? start=(\d\d:\d\d:\d\d)", re.M)
RE_PWR = re.compile(r"VDD_IN (\d+)mW/")
RE_TS = re.compile(r"^(\d\d)-(\d\d)-(\d{4}) (\d\d:\d\d:\d\d) ")


def secs(hms):
    h, m, s = (int(x) for x in hms.split(":"))
    return h * 3600 + m * 60 + s


def tegra_series(path):
    """[(seconds_of_day, VDD_IN mW)] from a tegrastats log with timestamps."""
    out = []
    with open(path, errors="replace") as f:
        for line in f:
            t = RE_TS.match(line)
            p = RE_PWR.search(line)
            if t and p:
                out.append((secs(t.group(4)), int(p.group(1))))
    return out


def main():
    d = sys.argv[1]
    out_csv = sys.argv[2] if len(sys.argv) > 2 else os.path.join(d, "e2e_summary.csv")
    suite = open(os.path.join(d, "suite.log"), errors="replace").read()
    starts = {m.group(1): secs(m.group(2)) for m in RE_START.finditer(suite)}
    series = tegra_series(os.path.join(d, "tegra.log")) if os.path.exists(os.path.join(d, "tegra.log")) else []

    runs = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        if os.path.basename(f) == "micro.json":
            continue
        j = json.load(open(f, encoding="utf-8"))
        if j.get("status") != "OK":
            continue
        name = j["config"]
        base = re.sub(r"_r\d+$", "", name)
        win = None
        if name in starts and series:
            t0 = starts[name] + int(j.get("warmup_s", 5))
            t1 = t0 + float(j["measured_wall_s"])
            vals = [mw for t, mw in series if t0 <= t <= t1]
            win = statistics.mean(vals) if vals else None
        runs.setdefault(base, []).append((j, win))

    rows = []
    for base, items in runs.items():
        fps = [j["fps"] for j, _ in items]
        p50 = [j["latency_ms"]["p50"] for j, _ in items]
        cpu = [j["cpu_util_system_pct"] for j, _ in items]
        tmax = [max(j["temp_peak_mc"].values()) if isinstance(j["temp_peak_mc"], dict) else j["temp_peak_mc"] for j, _ in items]
        pw = [w for _, w in items if w]
        f_med = statistics.median(fps)
        w_mean = statistics.mean(pw) if pw else None
        rows.append({
            "config": base, "reps": len(items), "fps_median": round(f_med, 1), "fps_min": round(min(fps), 1), "fps_max": round(max(fps), 1),
            "p50_ms": round(statistics.median(p50), 1), "cpu_pct": round(statistics.median(cpu), 1),
            "temp_peak_C": round(max(tmax) / 1000.0, 1), "power_mW": round(w_mean) if w_mean else None,
            "mJ_per_img": round(w_mean / f_med, 1) if w_mean and f_med else None,
        })
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("%-36s %4s %8s %15s %7s %6s %6s %8s %7s" % ("config", "reps", "FPS", "min..max", "p50ms", "CPU%", "Tmax", "mW", "mJ/img"))
    for r in rows:
        print("%-36s %4d %8.1f %7.1f..%-7.1f %7.1f %6.1f %6.1f %8s %7s" % (
            r["config"], r["reps"], r["fps_median"], r["fps_min"], r["fps_max"], r["p50_ms"], r["cpu_pct"], r["temp_peak_C"],
            r["power_mW"], r["mJ_per_img"]))


if __name__ == "__main__":
    main()
