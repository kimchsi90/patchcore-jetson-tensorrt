import json, re, sys, os
R = os.path.expanduser("~/llm/results_conc")
for name in ("dla", "gpu"):
    t = {k: int(open(f"{R}/{name}.{k}").read()) for k in ("t_bench_start", "t_pipe_start", "t_pipe_end", "t_bench_end")}
    rows = [json.loads(l) for l in open(f"{R}/{name}.llm.jsonl") if l.strip().startswith("{")]
    r = rows[0]
    ts = r.get("samples_ts") or []
    ns = r.get("samples_ns") or []
    print(f"== {name}: llama avg_ts={r.get('avg_ts'):.2f} reps={len(ts)}")
    cum = 0.0; series = []
    for n, v in zip(ns, ts):
        cum += n / 1e9
        series.append((cum, v))
    print("   per-rep tg (cum s, tok/s):", " ".join(f"{c:.0f}:{v:.1f}" for c, v in series))
    print("   times rel to bench start: pipe", t["t_pipe_start"] - t["t_bench_start"], "-", t["t_pipe_end"] - t["t_bench_start"], "bench end", t["t_bench_end"] - t["t_bench_start"])
    for line in open(f"{R}/{name}.suite.out"):
        print("   suite:", line.strip())
    pw = []
    for l in open(f"{R}/{name}.tegra", errors="ignore"):
        m = re.search(r"VDD_IN (\d+)mW", l)
        if m: pw.append(int(m.group(1)))
    print("   VDD_IN mean mW", sum(pw) // max(1, len(pw)), "max", max(pw or [0]))
