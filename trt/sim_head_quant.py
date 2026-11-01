"""PC-side test of the head hypothesis: re-run the kNN head (Conv 1x1 + ReduceMax) in numpy on the ORT fp32 xproj with
simulated quantization of the head only. Writes dumps in the dump_trt.py format for score_dump.py / compare_dump.py.

Usage: python sim_head_quant.py --onnx <pre.onnx> --ref out\bottle_256_ort_fp32 --out out\bottle_256_sim
"""
import argparse
import sys

import numpy as np
import onnx
from onnx import numpy_helper


def q_sym(a, amax, bits=8):
    """Symmetric per-tensor fake quantization with a given clip range."""
    qmax = 2 ** (bits - 1) - 1
    s = amax / qmax
    return np.clip(np.round(a / s), -qmax, qmax) * s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    init = dict((i.name, numpy_helper.to_array(i)) for i in onnx.load(a.onnx).graph.initializer)
    W = init["Wk0"].reshape(init["Wk0"].shape[0], -1).astype(np.float32)  # [K, C]
    B = init["Bk0"].astype(np.float32)
    x = np.load(a.ref + "_x.npy")  # [N, C, H, W]
    m_ref = np.load(a.ref + "_m.npy")  # [N, H*W]
    if W.shape[1] != x.shape[1]:
        sys.exit("channel mismatch: W=%s x=%s dtype=%s" % (W.shape, x.shape, x.dtype))
    n = x.shape[0]
    xf = x.reshape(n, x.shape[1], -1).astype(np.float32)  # [N, C, P]

    def head(xin, w, quant_out=None, out_fp16=False, w_fp16=False):
        res = np.empty((n, xin.shape[2]), np.float32)
        stats = []
        for i in range(n):
            d = w @ xin[i] + B[:, None]  # [K, P], fp32 accumulate
            if out_fp16:
                d = d.astype(np.float16).astype(np.float32)
            stats.append(np.abs(d).max())
            if quant_out is not None:
                d = q_sym(d, quant_out)
            res[i] = d.max(0)
        return res, max(stats)

    m0, dk_absmax = head(xf, W)
    print("check vs ORT m: max_abs=%.4g (ref absmax %.4g); dk |max| over all channels = %.1f, max(m) = %.1f" %
          (np.abs(m0 - m_ref).max(), np.abs(m_ref).max(), dk_absmax, m0.max()))
    x_amax = float(np.abs(xf).max())
    w_amax = np.abs(W).max(1, keepdims=True)  # per output channel
    Wq = np.round(W / (w_amax / 127)).clip(-127, 127) * (w_amax / 127)
    xq = q_sym(xf, x_amax)
    variants = {
        "fp16out": lambda: head(xf.astype(np.float16).astype(np.float32), W.astype(np.float16).astype(np.float32), out_fp16=True),
        "int8dkout": lambda: head(xf, W, quant_out=dk_absmax),
        "int8dkout_mmax": lambda: head(xf, W, quant_out=float(m0.max())),
        "int8xw": lambda: head(xq, Wq),
        "int8all": lambda: head(xq, Wq, quant_out=dk_absmax),
    }
    for name, fn in variants.items():
        m, _ = fn()
        np.save("%s_%s_m.npy" % (a.out, name), m.astype(np.float32))
        np.save("%s_%s_x.npy" % (a.out, name), x)  # xproj kept as the fp32 reference
        print("wrote %s_%s (xproj fp32 reference; only the head is simulated)" % (a.out, name))
    print("x_amax=%.3f" % x_amax)


if __name__ == "__main__":
    main()
