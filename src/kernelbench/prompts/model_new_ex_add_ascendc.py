import torch
import torch.nn as nn


# This example documents the candidate ABI. KernelBench generates the host
# wrapper/binding from the spec; the LLM edits the spec, tiling and device code.
ASCENDC_OP_SPEC = {
    "op_name": "add",
    "entrypoint": "kernel_entry",
    "inputs": [
        {"name": "a", "dtype": "float32", "layout": "contiguous", "shape": ["input[0].dim(0)"]},
        {"name": "b", "dtype": "float32", "layout": "contiguous", "shape": ["input[1].dim(0)"]},
    ],
    "outputs": [{"name": "out", "dtype": "float32", "layout": "contiguous", "shape": ["input[0].dim(0)"]}],
    "attributes": [],
    "tiling": {"block_size": {"c_type": "uint32_t", "default": 256}},
    "block_dim": 1,
    "workspace": 0,
}

ASCENDC_TILING = {"block_dim": 1, "block_size": 256}

ASCENDC_KERNEL_SOURCE = r'''
#include "kernel_operator.h"

// The framework-generated host wrapper supplies tensors and tiling data.
// Implement kernel_entry with GM/UB bounds checks and 32-byte alignment.
'''


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        # Replace with the generated private-OPP call. Do not add a fallback.
        raise RuntimeError("replace with generated Ascend C custom-op binding")
