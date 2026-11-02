"""Single-model smoke test and timing of vLLM 0.2.2 on the Xavier NX (TinyLlama-1.1B-Chat, fp16)."""
import sys, time, json
import torch
import torch.distributed as _d
# This Jetson PyTorch wheel is built without distributed support. With world size 1 vLLM only needs trivial answers.
if not _d.is_available():
    _state = {"init": False}
    _d.is_initialized = lambda: _state["init"]
    _d.init_process_group = lambda *a, **k: _state.__setitem__("init", True)
    _d.get_world_size = lambda group=None: 1
    _d.get_rank = lambda group=None: 0
    _d.new_group = lambda ranks=None, *a, **k: object()
    _d.all_reduce = lambda t, *a, **k: t
    _d.barrier = lambda *a, **k: None
from vllm import LLM, SamplingParams

model = sys.argv[1]
n_out = int(sys.argv[2]) if len(sys.argv) > 2 else 128
llm = LLM(model=model, dtype="float16", gpu_memory_utilization=0.4, max_model_len=1024, swap_space=1)
sp = SamplingParams(temperature=0.0, max_tokens=n_out)
prompt = "<|user|>\nExplain in two sentences what a Jetson Xavier NX is.</s>\n<|assistant|>\n"
llm.generate([prompt], sp)  # warmup
res = {}
for bs in (1, 4, 8):
    t = time.time()
    outs = llm.generate([prompt] * bs, sp)
    dt = time.time() - t
    toks = sum(len(o.outputs[0].token_ids) for o in outs)
    res[f"bs{bs}"] = {"tokens": toks, "seconds": round(dt, 2), "tok_per_s": round(toks / dt, 2)}
print("TEXT:", outs[0].outputs[0].text[:200].replace("\n", " "))
print("RESULT", json.dumps(res))
