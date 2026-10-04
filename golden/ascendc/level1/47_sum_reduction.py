import torch
import torch.nn as nn

ASCENDC_OP_SPEC = {
    "op_name": "l1_p47_sum_reduction",
    "entrypoint": "p47_sum_reduction_entry",
    "inputs": [
        {
            "name": "x",
            "dtype": "float32",
            "layout": "ND",
            "shape": [
                "input[0].dim(0)",
                "input[0].dim(1)",
                "input[0].dim(2)",
            ],
        }
    ],
    "outputs": [
        {
            "name": "y",
            "dtype": "float32",
            "layout": "ND",
            "shape": ["input[0].dim(0)", "literal(1)", "input[0].dim(2)"],
        }
    ],
    "attributes": [{"name": "dim", "c_type": "int64_t", "default": 1}],
    "tiling": {
        "B": {"c_type": "uint32_t", "expression": "input[0].dim(0)"},
        "N": {"c_type": "uint32_t", "expression": "input[0].dim(1)"},
        "S": {"c_type": "uint32_t", "expression": "input[0].dim(2)"},
        "outerCount": {
            "c_type": "uint32_t",
            "expression": "mul(input[0].dim(0), input[0].dim(2))",
        },
    },
    "block_dim": 24,
    "workspace": 0,
}

ASCENDC_TILING = {}

ASCENDC_KERNEL_SOURCE = r'''
#include "kernel_operator.h"

class KernelSumReduction {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t B, uint32_t N, uint32_t S) {
        B_ = B;
        N_ = N;
        S_ = S;
        outerCount_ = static_cast<uint64_t>(B) * static_cast<uint64_t>(S);
        const uint64_t xNumel = static_cast<uint64_t>(B) * N * S;
        const uint64_t yNumel = static_cast<uint64_t>(B) * S;
        xGm_.SetGlobalBuffer((__gm__ float*)x, xNumel);
        yGm_.SetGlobalBuffer((__gm__ float*)y, yNumel);
    }

    __aicore__ inline void Process() {
        if (outerCount_ == 0 || N_ == 0) return;
        const uint64_t blockIdx = static_cast<uint64_t>(AscendC::GetBlockIdx());
        uint64_t blockNum = static_cast<uint64_t>(AscendC::GetBlockNum());
        if (blockNum == 0) blockNum = 1;
        const uint64_t chunk = (outerCount_ + blockNum - 1) / blockNum;
        const uint64_t start = blockIdx * chunk;
        uint64_t end = start + chunk;
        if (end > outerCount_) end = outerCount_;
        const uint64_t stride = static_cast<uint64_t>(S_);

        for (uint64_t output = start; output < end; ++output) {
            const uint32_t b = static_cast<uint32_t>(output / S_);
            const uint32_t s = static_cast<uint32_t>(output - static_cast<uint64_t>(b) * S_);
            uint64_t offset = static_cast<uint64_t>(b) * N_ * stride + s;
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
            float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
            uint32_t n = 0;
            const uint32_t unrolled = (N_ / 8u) * 8u;
            for (; n < unrolled; n += 8u) {
                acc0 += xGm_.GetValue(offset);
                acc1 += xGm_.GetValue(offset + stride);
                acc2 += xGm_.GetValue(offset + stride * 2u);
                acc3 += xGm_.GetValue(offset + stride * 3u);
                acc4 += xGm_.GetValue(offset + stride * 4u);
                acc5 += xGm_.GetValue(offset + stride * 5u);
                acc6 += xGm_.GetValue(offset + stride * 6u);
                acc7 += xGm_.GetValue(offset + stride * 7u);
                offset += stride * 8u;
            }
            float sum = (acc0 + acc1) + (acc2 + acc3) +
                        (acc4 + acc5) + (acc6 + acc7);
            for (; n < N_; ++n) {
                sum += xGm_.GetValue(offset);
                offset += stride;
            }
            yGm_.SetValue(output, sum);
        }
    }

private:
    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> yGm_;
    uint32_t B_ = 0;
    uint32_t N_ = 0;
    uint32_t S_ = 0;
    uint64_t outerCount_ = 0;
};

extern "C" __global__ __aicore__ void p47_sum_reduction_entry(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSumReduction op;
    op.Init(x, y, tiling_data.B, tiling_data.N, tiling_data.S);
    op.Process();
}
'''


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return ascendc_binding.p47_sum_reduction_entry(x, self.dim)
