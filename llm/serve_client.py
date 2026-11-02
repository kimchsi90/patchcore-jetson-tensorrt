"""Load client for llama-server (OpenAI-compatible, streaming). Reports TTFT, per-request tok/s and aggregate tok/s."""
import json, sys, time, threading, urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
PROMPT = "Explain in about 100 words how a GPU differs from a CPU."


def one(max_tokens, out):
    body = json.dumps({"model": "m", "stream": True, "max_tokens": max_tokens, "temperature": 0,
                       "messages": [{"role": "user", "content": PROMPT}]}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time(); first = None; n = 0
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])
            if d["choices"][0]["delta"].get("content"):
                if first is None:
                    first = time.time()
                n += 1
    t1 = time.time()
    out.append({"ttft": first - t0, "tokens": n, "decode_tps": (n - 1) / (t1 - first) if n > 1 else 0.0})


def run(conc, reps, max_tokens):
    res = []; lock = threading.Lock()
    def worker():
        for _ in range(reps):
            tmp = []
            one(max_tokens, tmp)
            with lock:
                res.extend(tmp)
    t0 = time.time()
    ths = [threading.Thread(target=worker) for _ in range(conc)]
    [t.start() for t in ths]; [t.join() for t in ths]
    wall = time.time() - t0
    tok = sum(r["tokens"] for r in res)
    ttfts = sorted(r["ttft"] for r in res)
    return {"concurrency": conc, "requests": len(res), "tokens": tok, "wall_s": round(wall, 2),
            "aggregate_tps": round(tok / wall, 2),
            "per_request_decode_tps": round(sum(r["decode_tps"] for r in res) / len(res), 2),
            "ttft_median_s": round(ttfts[len(ttfts) // 2], 3), "ttft_max_s": round(ttfts[-1], 3)}


if __name__ == "__main__":
    max_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    one(16, [])  # warmup
    for conc in (1, 2, 4):
        print(json.dumps(run(conc, 3, max_tokens)), flush=True)
