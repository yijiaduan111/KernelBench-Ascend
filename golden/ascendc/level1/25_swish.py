import torch
import torch.nn as nn

ASCENDC_OP_SPEC = {
    "op_name": "l1_p25_swish",
    "entrypoint": "p25_swish_entry",
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
    "attributes": [],
    "tiling": {
        "totalLength": {"c_type": "uint32_t", "expression": "input[0].numel()"},
        "tileSize": {"c_type": "uint32_t", "default": 8192},
    },
    "block_dim": 32,
    "workspace": 0,
}

ASCENDC_TILING = {"tileSize": 8192}

ASCENDC_KERNEL_SOURCE = r'''
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}

__aicore__ inline uint32_t AlignUp32(uint32_t bytes) {
    return (bytes + 31u) & ~31u;
}

class KernelSwish {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t totalLength, uint32_t tileSize) {
        totalLength_ = totalLength;
        tileSize_ = tileSize == 0 ? 1 : tileSize;
        const uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
        const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t usedBlocks = blockNum == 0 ? 1 : blockNum;
        const uint32_t coreChunk = CeilDivU32(totalLength_, usedBlocks);
        const uint32_t coreStart = blockIdx * coreChunk;
        if (coreStart >= totalLength_) {
            coreLen_ = 0;
            return;
        }
        coreLen_ = totalLength_ - coreStart;
        if (coreLen_ > coreChunk) coreLen_ = coreChunk;
        xGm_.SetGlobalBuffer((__gm__ float*)x + coreStart, coreLen_);
        yGm_.SetGlobalBuffer((__gm__ float*)y + coreStart, coreLen_);
        if (tileSize_ > coreLen_) tileSize_ = coreLen_;
        const uint32_t bytes = AlignUp32(tileSize_ * sizeof(float));
        pipe_.InitBuffer(inQueue_, BUFFER_NUM, bytes);
        pipe_.InitBuffer(outQueue_, BUFFER_NUM, bytes);
        pipe_.InitBuffer(tmpBuffer_, AlignUp32(tileSize_ * sizeof(uint8_t)));
    }

    __aicore__ inline void Process() {
        if (coreLen_ == 0) return;
        const uint32_t tiles = CeilDivU32(coreLen_, tileSize_);
        CopyIn(0);
        if (tiles > 1) CopyIn(1);
        for (uint32_t i = 0; i < tiles + 1; ++i) {
            if (i < tiles) Compute(i);
            if (i + 2 < tiles) CopyIn(i + 2);
            if (i > 0) CopyOut(i - 1);
        }
    }

private:
    __aicore__ inline uint32_t TileLength(uint32_t tile) const {
        const uint32_t offset = tile * tileSize_;
        const uint32_t remaining = coreLen_ - offset;
        return remaining < tileSize_ ? remaining : tileSize_;
    }

    __aicore__ inline void CopyIn(uint32_t tile) {
        const uint32_t length = TileLength(tile);
        auto x = inQueue_.AllocTensor<float>();
        AscendC::DataCopy(x, xGm_[tile * tileSize_], length);
        inQueue_.EnQue(x);
    }

    __aicore__ inline void Compute(uint32_t tile) {
        const uint32_t length = TileLength(tile);
        auto x = inQueue_.DeQue<float>();
        auto y = outQueue_.AllocTensor<float>();
        auto tmp = tmpBuffer_.Get<uint8_t>();
        AscendC::Sigmoid(y, x, tmp, length);
        AscendC::Mul(y, y, x, length);
        outQueue_.EnQue(y);
        inQueue_.FreeTensor(x);
    }

    __aicore__ inline void CopyOut(uint32_t tile) {
        const uint32_t length = TileLength(tile);
        auto y = outQueue_.DeQue<float>();
        AscendC::DataCopy(yGm_[tile * tileSize_], y, length);
        outQueue_.FreeTensor(y);
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuffer_;
    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> yGm_;
    uint32_t totalLength_ = 0;
    uint32_t coreLen_ = 0;
    uint32_t tileSize_ = 1;
};

extern "C" __global__ __aicore__ void p25_swish_entry(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSwish op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileSize);
    op.Process();
}
'''


class ModelNew(nn.Module):
    def forward(self, x):
        return ascendc_binding.p25_swish_entry(x)
