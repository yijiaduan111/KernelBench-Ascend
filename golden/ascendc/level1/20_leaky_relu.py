import torch
import torch.nn as nn

ASCENDC_OP_SPEC = {
    "op_name": "l1_p20_leaky_relu",
    "entrypoint": "p20_leaky_relu_entry",
    "inputs": [
        {
            "name": "x",
            "dtype": "float32",
            "layout": "ND",
            "shape": ["input[0].dim(0)", "input[0].dim(1)"],
        }
    ],
    "outputs": [
        {
            "name": "y",
            "dtype": "float32",
            "layout": "ND",
            "shape": ["input[0].dim(0)", "input[0].dim(1)"],
        }
    ],
    "attributes": [
        {"name": "negative_slope", "c_type": "float", "default": 0.01}
    ],
    "tiling": {
        "totalLength": {
            "c_type": "uint32_t",
            "expression": "input[0].numel()",
        },
        "tileLength": {"c_type": "uint32_t", "default": 8192},
        "negativeSlope": {
            "c_type": "float",
            "expression": "attr(\"negative_slope\")",
        },
    },
    "block_dim": 24,
    "workspace": 0,
}

ASCENDC_TILING = {"tileLength": 8192}

ASCENDC_KERNEL_SOURCE = r'''
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelLeakyRelu {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t totalLength, uint32_t tileLength,
                                float negativeSlope) {
        totalLen_ = totalLength;
        tileLen_ = tileLength;
        negCorr_ = negativeSlope - 1.0f;
        xGm_.SetGlobalBuffer((__gm__ float*)x, totalLen_);
        yGm_.SetGlobalBuffer((__gm__ float*)y, totalLen_);
        pipe_.InitBuffer(inQueue_, BUFFER_NUM, tileLen_ * sizeof(float));
        pipe_.InitBuffer(outQueue_, BUFFER_NUM, tileLen_ * sizeof(float));
        pipe_.InitBuffer(tmpQueue_, 1, tileLen_ * sizeof(float));
    }

    __aicore__ inline void Process() {
        if (totalLen_ == 0 || tileLen_ == 0) return;
        const uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
        const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t stride = blockNum * tileLen_;
        for (uint32_t base = blockIdx * tileLen_; base < totalLen_; base += stride) {
            uint32_t count = totalLen_ - base;
            if (count > tileLen_) count = tileLen_;
            CopyIn(base, count);
            Compute(count);
            CopyOut(base, count);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count) {
        auto x = inQueue_.AllocTensor<float>();
        AscendC::DataCopy(x, xGm_[offset], count);
        inQueue_.EnQue(x);
    }

    __aicore__ inline void Compute(uint32_t count) {
        auto x = inQueue_.DeQue<float>();
        auto y = outQueue_.AllocTensor<float>();
        auto tmp = tmpQueue_.AllocTensor<float>();
        AscendC::Mins(tmp, x, 0.0f, count);
        AscendC::Muls(tmp, tmp, negCorr_, count);
        AscendC::Add(y, x, tmp, count);
        tmpQueue_.FreeTensor(tmp);
        outQueue_.EnQue(y);
        inQueue_.FreeTensor(x);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count) {
        auto y = outQueue_.DeQue<float>();
        AscendC::DataCopy(yGm_[offset], y, count);
        outQueue_.FreeTensor(y);
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue_;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> tmpQueue_;
    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> yGm_;
    uint32_t totalLen_ = 0;
    uint32_t tileLen_ = 0;
    float negCorr_ = -0.99f;
};

extern "C" __global__ __aicore__ void p20_leaky_relu_entry(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelLeakyRelu op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileLength,
            tiling_data.negativeSlope);
    op.Process();
}
'''


class ModelNew(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return ascendc_binding.p20_leaky_relu_entry(x, self.negative_slope)
