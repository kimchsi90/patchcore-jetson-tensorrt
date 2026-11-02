import torch, torch.nn.functional as F
w_shapes = ((2560, 2048, 2560), (2560, 2048, 2304), (2560, 2048, 5632), (1024, 2048, 2560))
for m, k, n in w_shapes:
    x = torch.randn(m, k, device="cuda", dtype=torch.float16); w = torch.randn(n, k, device="cuda", dtype=torch.float16)
    try:
        F.linear(x, w); torch.cuda.synchronize(); print((m, k, n), "ok")
    except RuntimeError as e:
        print((m, k, n), "FAIL", str(e)[:70])
