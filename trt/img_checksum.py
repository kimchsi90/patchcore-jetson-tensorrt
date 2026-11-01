"""Print sha1 of the float32 NHWC tensor for the first N images, using the trt (cv2) or ort (PIL) loader."""
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump_trt import load_rgb  # noqa: E402

loader, d = sys.argv[1], sys.argv[2]
for n in sorted(os.listdir(d))[:int(sys.argv[3])]:
    x = load_rgb(os.path.join(d, n), loader == "cv2")
    print(n, x.shape, x.dtype, hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest()[:16])
