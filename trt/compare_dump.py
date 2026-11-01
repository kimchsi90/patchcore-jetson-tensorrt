"""Compare two dumps: max abs / relative diff of m and xproj, correlation of image scores (max, top10)."""
import sys

import numpy as np

ref, tst = sys.argv[1], sys.argv[2]
out = {}
for k in ("m", "x"):
    a = np.load(ref + "_%s.npy" % k).astype(np.float64)
    b = np.load(tst + "_%s.npy" % k).astype(np.float64)
    if a.shape != b.shape:
        raise SystemExit("shape mismatch %s %s %s" % (k, a.shape, b.shape))
    d = np.abs(a - b)
    out[k] = (d.max(), d.max() / np.abs(a).max(), np.linalg.norm(a - b) / np.linalg.norm(a))
    print("%s: max_abs=%.4g max_abs/ref_absmax=%.4g rel_L2=%.4g (ref absmax %.4g)" % (k, out[k][0], out[k][1], out[k][2], np.abs(a).max()))


def scores(p):
    x = np.load(p + "_x.npy").astype(np.float64)
    m = np.load(p + "_m.npy").astype(np.float64)
    s = np.sqrt(np.clip((x ** 2).sum(1).reshape(len(x), -1) - m, 0, None))
    return s.max(1), np.sort(s, 1)[:, ::-1][:, :10].mean(1)


ra, rt = scores(ref)
ta, tt = scores(tst)
print("image score corr: max=%.5f top10=%.5f" % (np.corrcoef(ra, ta)[0, 1], np.corrcoef(rt, tt)[0, 1]))
