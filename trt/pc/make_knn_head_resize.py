"""Per-category kNN head as its own ONNX: embedding [1,1536,32,32] -> (m, xproj).

The shared INT8 backbone produces the embedding; this small head runs after it (also on the NPU, fp16).
  xproj = Conv1x1(emb)                       (random projection or PCA, centered)
  m     = max_j (2 x.b_j - ||b_j||^2)        via Conv chunks -> Reshape -> Transpose -> GlobalMaxPool -> Max
The CPU adds ||x||^2:  patch d^2 = ||xproj||^2 - m.
Usage: python make_knn_head.py <dim> <ratio> <random|pca> <cat1> [cat2 ...]
Writes work/<cat>_256/knn_head_<method><dim>_c<ratio>.onnx. dim=1536 means no projection (identity).
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper, numpy_helper
from PIL import Image
from torchvision.transforms import v2

DIM, RATIO, METHOD = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
CATS = sys.argv[4:]
import os
SEED = int(os.environ.get("HEAD_SEED", "1"))   # random-projection seed; a non-default seed is added to the file name
SUFFIX = "" if SEED == 1 else f"_s{SEED}"
ROOT = Path(os.environ.get("DATASET_ROOT", r"D:\jetson_test\patchcore\datasets\mvtec_ad"))   # <root>/<category>/train/good/*.png
WORKROOT = Path(r"D:\jetson_test\patchcore\work")
# variants (other input size / backbone) are selected with environment variables; defaults keep the 256 WRN50 behavior
SIZE = int(os.environ.get("NPU_SIZE", "256"))
DIRSUF = os.environ.get("NPU_DIR_SUFFIX", "_256")
ONNX = Path(os.environ.get("NPU_BACKBONE_ONNX", str(WORKROOT / "bottle_256" / "patchcore_npu_256.onnx")))
CHUNK = 2140
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
dev = torch.device("cuda")
resize = v2.Resize((SIZE, SIZE), antialias=True)
# RESIZE_METHOD switch (the only change vs make_knn_head.py): pil_aa | area | linear | tv_bilinear_noaa
RESIZE_METHOD = os.environ.get("RESIZE_METHOD", "pil_aa")
if RESIZE_METHOD not in ("pil_aa", "area", "linear", "tv_bilinear_noaa"):
    raise ValueError(f"unknown RESIZE_METHOD {RESIZE_METHOD}")
resize_noaa = v2.Resize((SIZE, SIZE), antialias=False)


def load_resized(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if RESIZE_METHOD == "pil_aa":
        t = resize(v2.functional.pil_to_tensor(img))
        return np.array(v2.functional.to_pil_image(t))
    if RESIZE_METHOD == "tv_bilinear_noaa":
        t = resize_noaa(v2.functional.pil_to_tensor(img))
        return np.array(v2.functional.to_pil_image(t))
    interp = cv2.INTER_AREA if RESIZE_METHOD == "area" else cv2.INTER_LINEAR
    return cv2.resize(np.asarray(img), (SIZE, SIZE), interpolation=interp)


def k_center_greedy(x: torch.Tensor, n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    z = x @ (torch.randn(x.shape[1], 128, generator=g).to(x.device) / np.sqrt(128))
    idx = [int(torch.randint(0, z.shape[0], (1,), generator=g))]
    mind = torch.full((z.shape[0],), float("inf"), device=z.device)
    for _ in range(n - 1):
        mind = torch.minimum(mind, ((z - z[idx[-1]]) ** 2).sum(1))
        idx.append(int(mind.argmax()))
    return torch.tensor(idx, device=x.device)


def build(cat: str, sess) -> Path:
    feats = []
    for p in sorted((ROOT / cat / "train" / "good").glob("*.png")):
        x = ((load_resized(p).astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
        e = sess.run(None, {"image": x.astype(np.float32)})[0][0]
        feats.append(torch.from_numpy(e).permute(1, 2, 0).reshape(-1, e.shape[0]))
    flat = torch.cat(feats).to(dev)
    n_patches, c = flat.shape
    n_bank = int(n_patches * RATIO)
    mean = flat.mean(0, keepdim=True)
    if METHOD == "base":
        if DIM != c:
            raise ValueError(f"method 'base' needs dim == embedding channels ({c}), got {DIM}")
        P = torch.eye(c, device=dev)
    elif DIM == c:
        P = torch.eye(c, device=dev)
    elif METHOD == "pca":
        sub = flat[torch.randperm(n_patches, device=dev)[:200000]] - mean
        _, vecs = torch.linalg.eigh(sub.T @ sub)
        P = vecs[:, -DIM:].flip(1).contiguous()
    else:
        g = torch.Generator(device="cpu").manual_seed(SEED)
        P = (torch.randn(c, DIM, generator=g) / np.sqrt(DIM)).to(dev)
    idx = k_center_greedy(flat, n_bank)
    bank = ((flat[idx] - mean) @ P).cpu().numpy().astype(np.float32)
    d = bank.shape[1]
    n = (SIZE // 8) ** 2

    inits = [numpy_helper.from_array(P.T.cpu().numpy()[:, :, None, None].astype(np.float32), "Wp"),
             numpy_helper.from_array((-(mean @ P)[0]).cpu().numpy().astype(np.float32), "Bp")]
    nodes = [helper.make_node("Conv", ["emb", "Wp", "Bp"], ["xproj"], kernel_shape=[1, 1])]
    outs = []
    for ci, s in enumerate(range(0, bank.shape[0], CHUNK)):
        part = bank[s:s + CHUNK]
        mc = part.shape[0]
        inits += [numpy_helper.from_array((2.0 * part)[:, :, None, None].astype(np.float32), f"Wk{ci}"),
                  numpy_helper.from_array((-(part * part).sum(1)).astype(np.float32), f"Bk{ci}"),
                  numpy_helper.from_array(np.array([1, mc, 1, n], dtype=np.int64), f"Sk{ci}")]
        nodes += [helper.make_node("Conv", ["xproj", f"Wk{ci}", f"Bk{ci}"], [f"dk{ci}"], kernel_shape=[1, 1]),
                  helper.make_node("Reshape", [f"dk{ci}", f"Sk{ci}"], [f"rk{ci}"]),
                  helper.make_node("Transpose", [f"rk{ci}"], [f"tk{ci}"], perm=[0, 3, 2, 1]),
                  helper.make_node("GlobalMaxPool", [f"tk{ci}"], [f"gk{ci}"])]
        outs.append(f"gk{ci}")
    nodes.append(helper.make_node("Max" if len(outs) > 1 else "Identity", outs, ["m"]))
    graph = helper.make_graph(
        nodes, "knn_head",
        [helper.make_tensor_value_info("emb", TensorProto.FLOAT, [1, c, SIZE // 8, SIZE // 8])],
        [helper.make_tensor_value_info("m", TensorProto.FLOAT, [1, n, 1, 1]),
         helper.make_tensor_value_info("xproj", TensorProto.FLOAT, [1, d, SIZE // 8, SIZE // 8])],
        initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    out = WORKROOT / f"{cat}{DIRSUF}" / f"knn_head_{METHOD}{DIM}_c{RATIO}{SUFFIX}.onnx"
    onnx.save(model, str(out))
    print(f"{cat}: wrote {out.name} bank={bank.shape} chunks={len(outs)} size_MB={out.stat().st_size / 1e6:.1f}", flush=True)
    return out


def main() -> int:
    if not ONNX.is_file():
        print(f"onnx not found: {ONNX}")
        return 1
    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    for cat in CATS:
        build(cat, sess)
    return 0


if __name__ == "__main__":
    sys.exit(main())
