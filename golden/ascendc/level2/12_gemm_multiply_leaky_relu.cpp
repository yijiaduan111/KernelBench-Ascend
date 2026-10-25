#include "kernel_operator.h"

class KernelGemmLeakyRelu {
public:
    __aicore__ inline void Init(GM_ADDR weight, GM_ADDR bias, GM_ADDR x, GM_ADDR y,
                                uint32_t outFeatures, uint32_t inFeatures, uint32_t batch,
                                float multiplier, float negativeSlope) {
        weight_.SetGlobalBuffer((__gm__ float*)weight, (uint64_t)outFeatures * inFeatures);
        bias_.SetGlobalBuffer((__gm__ float*)bias, outFeatures);
        x_.SetGlobalBuffer((__gm__ float*)x, (uint64_t)batch * inFeatures);
        y_.SetGlobalBuffer((__gm__ float*)y, (uint64_t)batch * outFeatures);
        outFeatures_ = outFeatures;
        inFeatures_ = inFeatures;
        batch_ = batch;
        multiplier_ = multiplier;
        negativeSlope_ = negativeSlope;
        pipe_.InitBuffer(xBuffer_, 8192 * sizeof(float));
        pipe_.InitBuffer(weightBuffer_, 8192 * sizeof(float));
        pipe_.InitBuffer(productBuffer_, 8192 * sizeof(float));
        pipe_.InitBuffer(scratchBuffer_, 8192 * sizeof(float));
        pipe_.InitBuffer(sumBuffer_, 64 * sizeof(float));
    }

    __aicore__ inline void Process() {
        uint64_t total = (uint64_t)batch_ * outFeatures_;
        uint64_t block = AscendC::GetBlockNum();
        if (block == 0) block = 1;
        uint64_t id = AscendC::GetBlockIdx();
        // Scalar GM writes use cache lines; split blocks on 128-byte boundaries
        // so two cores never update the same output cache line.
        uint64_t chunk = ((total + block - 1) / block + 31) / 32 * 32;
        uint64_t begin = id * chunk;
        uint64_t end = begin + chunk;
        if (end > total) end = total;
        auto xLocal = xBuffer_.Get<float>();
        auto weightLocal = weightBuffer_.Get<float>();
        auto product = productBuffer_.Get<float>();
        auto scratch = scratchBuffer_.Get<float>();
        auto result = sumBuffer_.Get<float>();
        uint32_t currentRow = 0xFFFFFFFF;
        for (uint64_t linear = begin; linear < end; ++linear) {
            uint32_t b = static_cast<uint32_t>(linear / outFeatures_);
            uint32_t j = static_cast<uint32_t>(linear % outFeatures_);
            float sum = 0.0f;
            uint64_t xbase = (uint64_t)b * inFeatures_;
            uint64_t wbase = (uint64_t)j * inFeatures_;
            if (inFeatures_ <= 8192 && inFeatures_ % 8 == 0) {
                if (currentRow != b) {
                    AscendC::DataCopy(xLocal, x_[xbase], inFeatures_);
                    currentRow = b;
                }
                AscendC::DataCopy(weightLocal, weight_[wbase], inFeatures_);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Mul(product, xLocal, weightLocal, inFeatures_);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::ReduceSum(result, product, scratch, inFeatures_);
                AscendC::PipeBarrier<PIPE_ALL>();
                sum = result.GetValue(0);
            } else {
                for (uint32_t k = 0; k < inFeatures_; ++k) {
                    sum += x_.GetValue(xbase + k) * weight_.GetValue(wbase + k);
                }
            }
            sum += bias_.GetValue(j);
            sum *= multiplier_;
            if (sum < 0.0f) sum *= negativeSlope_;
            y_.SetValue(linear, sum);
        }
    }

private:
    AscendC::GlobalTensor<float> weight_;
    AscendC::GlobalTensor<float> bias_;
    AscendC::GlobalTensor<float> x_;
    AscendC::GlobalTensor<float> y_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> weightBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> productBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scratchBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuffer_;
    uint32_t outFeatures_ = 0;
    uint32_t inFeatures_ = 0;
    uint32_t batch_ = 0;
    float multiplier_ = 1.0f;
    float negativeSlope_ = 0.1f;
};

extern "C" __global__ __aicore__ void p12_boundary_entry(
    GM_ADDR input_0, GM_ADDR input_1, GM_ADDR input_2, GM_ADDR output_0,
    GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelGemmLeakyRelu op;
    op.Init(input_0, input_1, input_2, output_0,
            tiling_data.input0_dim0, tiling_data.input0_dim1,
            tiling_data.input2_dim0, tiling_data.multiplier,
            tiling_data.init_arg_3);
    op.Process();
}
