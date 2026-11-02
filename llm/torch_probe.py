import torch, torch.nn.functional as F
print("arch list", torch.cuda.get_arch_list(), "cap", torch.cuda.get_device_capability(0))
for dt in (torch.float32, torch.float16):
    for shp in ((4, 2048, 2048), (16, 2048, 6144), (1, 2048, 32000)):
        m, k, n = shp
        x = torch.randn(m, k, device="cuda", dtype=dt); w = torch.randn(n, k, device="cuda", dtype=dt)
        try:
            y = F.linear(x, w); torch.cuda.synchronize(); print(dt, shp, "ok")
        except RuntimeError as e:
            print(dt, shp, "FAIL", str(e)[:80])
