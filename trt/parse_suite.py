"""Parse the Jetson speed suite output (trtexec logs + tegrastats logs) into one table.

Usage: python parse_suite.py <suite_dir> [out_csv]
Per run label: throughput (qps), GPU compute mean (ms), host latency mean (ms), mean power (VDD_IN, mW),
mean GPU load / clock, max GPU temperature. qps x batch 1 is images per second.
The soak run is summarised per minute to expose thermal throttling (GPU clock drop).
"""
import csv
import os
import re
import statistics
import sys

RE_QPS = re.compile(r"Throughput:\s*([\d.]+)\s*qps")
RE_GPU = re.compile(r"GPU Compute Time:.*?mean\s*=\s*([\d.]+)\s*ms")
RE_LAT = re.compile(r"\] Latency:.*?mean\s*=\s*([\d.]+)\s*ms")
# JetPack 4: "VDD_IN 5068/5068", "GR3D_FREQ 99%@1109"; JetPack 5: "VDD_IN 5561mW/5561mW", "GR3D_FREQ 0%@[114]" (bracket = per-TPC list)
RE_PWR = re.compile(r"VDD_IN (\d+)(?:mW)?/(\d+)(?:mW)?")
RE_GR3D = re.compile(r"GR3D_FREQ (\d+)%(?:@\[?(\d+))?")
RE_GTEMP = re.compile(r"GPU@([\d.]+)C")


def read(path):
    if not os.path.exists(path):
        return ""
    with open(path, "r", errors="replace") as f:
        return f.read()


def first(rx, text):
    m = rx.findall(text)
    return float(m[-1]) if m else None


def tegra_stats(text):
    pwr = [int(a) for a, _ in RE_PWR.findall(text)]
    gr = RE_GR3D.findall(text)
    util = [int(u) for u, _ in gr]
    freq = [int(f) for _, f in gr if f]
    temps = [float(t) for t in RE_GTEMP.findall(text)]
    return {
        "power_mw": round(statistics.mean(pwr)) if pwr else None,
        "gpu_util": round(statistics.mean(util)) if util else None,
        "gpu_mhz": round(statistics.mean(freq)) if freq else None,
        "gpu_temp_max": max(temps) if temps else None,
    }


def main():
    d = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(d, "suite_summary.csv")
    labels = sorted({f.rsplit(".", 1)[0] for f in os.listdir(d) if f.endswith(".log") and not f.startswith(("build_", "suite", "perf_"))})
    rows = []
    for lab in labels:
        log = read(os.path.join(d, lab + ".log"))
        tg = tegra_stats(read(os.path.join(d, lab + ".tegra")))
        qps = first(RE_QPS, log)
        rows.append({
            "label": lab,
            "qps": qps,
            "gpu_ms": first(RE_GPU, log),
            "lat_ms": first(RE_LAT, log),
            **tg,
            "mj_per_img": round(tg["power_mw"] / qps, 1) if qps and tg["power_mw"] else None,
            "passed": "PASSED" in log,
        })
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{'label':34} {'qps':>8} {'gpu_ms':>8} {'lat_ms':>8} {'mW':>6} {'mJ/img':>7} {'GPU%':>5} {'MHz':>6} {'Tmax':>5} ok")
    for r in rows:
        print(f"{r['label']:34} {str(r['qps']):>8} {str(r['gpu_ms']):>8} {str(r['lat_ms']):>8} {str(r['power_mw']):>6} "
              f"{str(r['mj_per_img']):>7} {str(r['gpu_util']):>5} {str(r['gpu_mhz']):>6} {str(r['gpu_temp_max']):>5} {r['passed']}")
    soak = read(os.path.join(d, "max_soak.tegra")).splitlines()
    if soak:
        print("\nsoak, per minute (GPU MHz mean / GPU temp max / power mW mean):")
        for i in range(0, len(soak), 60):
            chunk = "\n".join(soak[i:i + 60])
            s = tegra_stats(chunk)
            print(f"  min {i // 60:2d}: {s['gpu_mhz']} MHz, {s['gpu_temp_max']} C, {s['power_mw']} mW, util {s['gpu_util']}%")


if __name__ == "__main__":
    main()
