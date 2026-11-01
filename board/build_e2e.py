#!/usr/bin/env python3
"""Engine builder for the e2e pipelines: trt_runner.build with a guarded pin_layers.

The shared trt_runner.pin_layers sets the precision of every selected layer. In the post-processing graphs that also hits
integer shape-tensor layers and the TopK index output (int32), which the TensorRT builder rejects. This wrapper skips every
layer whose outputs are not float32/float16 and only types the float outputs of the others. Same CLI as `trt_runner.py build`.
"""
import sys

import trt_runner


def guarded_pin_layers(trt, network, indices, dtype):
    floats = (trt.float32, trt.float16)
    for i in indices:
        layer = network.get_layer(i)
        if layer.type in (trt.LayerType.CONSTANT, trt.LayerType.IDENTITY):  # Identity: pure copy, rejected when pinned
            continue
        outs = [j for j in range(layer.num_outputs) if layer.get_output(j).dtype in floats]
        if not outs:
            continue
        layer.precision = dtype
        for j in outs:
            layer.set_output_type(j, dtype)


trt_runner.pin_layers = guarded_pin_layers

if __name__ == "__main__":
    sys.argv[0] = "trt_runner.py"
    sys.exit(trt_runner._cli())
