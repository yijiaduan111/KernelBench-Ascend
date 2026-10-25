import torch
import torch.nn as nn

BOUNDARY_CONTRACT = {'version': 'boundary-contract-v1',
 'level': 2,
 'problem_id': 12,
 'source_hash': '0150537a4e53a29b1436d22667aa05b26c30301dd865d5ccf0f929b2c0cc29f6',
 'capture_mode': 'eval',
 'specialization': 'official_input',
 'inputs': ({'name': 'gemm_weight',
             'role': 'parameter',
             'dtype': 'float32',
             'rank': 2,
             'shape': (8192, 8192),
             'source_path': 'gemm.weight',
             'layout': 'ND'},
            {'name': 'gemm_bias',
             'role': 'parameter',
             'dtype': 'float32',
             'rank': 1,
             'shape': (8192,),
             'source_path': 'gemm.bias',
             'layout': 'ND'},
            {'name': 'x',
             'role': 'user_input',
             'dtype': 'float32',
             'rank': 2,
             'shape': (1024, 8192),
             'source_path': None,
             'layout': 'ND'}),
 'attributes': ({'name': 'multiplier',
                 'c_type': 'float',
                 'source': 'module_attribute',
                 'default': 2.0},
                {'name': 'init_arg_0', 'c_type': 'int64_t', 'source': 'init_arg', 'default': 8192},
                {'name': 'init_arg_1', 'c_type': 'int64_t', 'source': 'init_arg', 'default': 8192},
                {'name': 'init_arg_3', 'c_type': 'float', 'source': 'init_arg', 'default': 0.1}),
 'outputs': ({'index': 0, 'dtype': 'float32', 'rank': 2, 'shape': (1024, 8192), 'path': ()},),
 'output_tree': {'kind': 'tensor'},
 'input_guards': ({'input_index': 2, 'shape': [1024, 8192], 'dtype': 'float32', 'layout': 'ND'},),
 'state_mutations': (),
 'status': 'supported',
 'failure_type': None,
 'failure_message': None}
ASCENDC_OP_SPEC = {'op_name': 'l2_p12_boundary',
 'entrypoint': 'p12_boundary_entry',
 'inputs': [{'name': 'gemm_weight',
             'dtype': 'float32',
             'layout': 'ND',
             'shape': ['literal(8192)', 'literal(8192)']},
            {'name': 'gemm_bias', 'dtype': 'float32', 'layout': 'ND', 'shape': ['literal(8192)']},
            {'name': 'x',
             'dtype': 'float32',
             'layout': 'ND',
             'shape': ['literal(1024)', 'literal(8192)']}],
 'outputs': [{'name': 'output_0',
              'dtype': 'float32',
              'layout': 'ND',
              'shape': ['literal(1024)', 'literal(8192)']}],
 'attributes': [{'name': 'multiplier', 'c_type': 'float', 'default': 2.0},
                {'name': 'init_arg_0', 'c_type': 'int64_t', 'default': 8192},
                {'name': 'init_arg_1', 'c_type': 'int64_t', 'default': 8192},
                {'name': 'init_arg_3', 'c_type': 'float', 'default': 0.1}],
 'tiling': {'input0_numel': {'c_type': 'uint64_t', 'expression': 'input[0].numel()'},
            'input0_dim0': {'c_type': 'uint64_t', 'expression': 'input[0].dim(0)'},
            'input0_dim1': {'c_type': 'uint64_t', 'expression': 'input[0].dim(1)'},
            'input1_numel': {'c_type': 'uint64_t', 'expression': 'input[1].numel()'},
            'input1_dim0': {'c_type': 'uint64_t', 'expression': 'input[1].dim(0)'},
            'input2_numel': {'c_type': 'uint64_t', 'expression': 'input[2].numel()'},
            'input2_dim0': {'c_type': 'uint64_t', 'expression': 'input[2].dim(0)'},
            'input2_dim1': {'c_type': 'uint64_t', 'expression': 'input[2].dim(1)'},
            'multiplier': {'c_type': 'float', 'expression': 'attr("multiplier")'},
            'init_arg_0': {'c_type': 'int64_t', 'expression': 'attr("init_arg_0")'},
            'init_arg_1': {'c_type': 'int64_t', 'expression': 'attr("init_arg_1")'},
            'init_arg_3': {'c_type': 'float', 'expression': 'attr("init_arg_3")'}},
 'block_dim': 24,
 'workspace': 0}
ASCENDC_TILING = {'fields': [{'name': 'input0_numel', 'c_type': 'uint64_t', 'expression': 'input[0].numel()'}, {'name': 'input0_dim0', 'c_type': 'uint64_t', 'expression': 'input[0].dim(0)'}, {'name': 'input0_dim1', 'c_type': 'uint64_t', 'expression': 'input[0].dim(1)'}, {'name': 'input1_numel', 'c_type': 'uint64_t', 'expression': 'input[1].numel()'}, {'name': 'input1_dim0', 'c_type': 'uint64_t', 'expression': 'input[1].dim(0)'}, {'name': 'input2_numel', 'c_type': 'uint64_t', 'expression': 'input[2].numel()'}, {'name': 'input2_dim0', 'c_type': 'uint64_t', 'expression': 'input[2].dim(0)'}, {'name': 'input2_dim1', 'c_type': 'uint64_t', 'expression': 'input[2].dim(1)'}, {'name': 'multiplier', 'c_type': 'float', 'expression': 'attr("multiplier")'}, {'name': 'init_arg_0', 'c_type': 'int64_t', 'expression': 'attr("init_arg_0")'}, {'name': 'init_arg_1', 'c_type': 'int64_t', 'expression': 'attr("init_arg_1")'}, {'name': 'init_arg_3', 'c_type': 'float', 'expression': 'attr("init_arg_3")'}], 'block_dim': 24, 'workspace': 0}

# <<<IMPROVE:ascendc_kernel>>>
ASCENDC_KERNEL_SOURCE = '#include "kernel_operator.h"\n\nclass KernelGemmLeakyRelu {\npublic:\n    __aicore__ inline void Init(GM_ADDR weight, GM_ADDR bias, GM_ADDR x, GM_ADDR y,\n                                uint32_t outFeatures, uint32_t inFeatures, uint32_t batch,\n                                float multiplier, float negativeSlope) {\n        weight_.SetGlobalBuffer((__gm__ float*)weight, (uint64_t)outFeatures * inFeatures);\n        bias_.SetGlobalBuffer((__gm__ float*)bias, outFeatures);\n        x_.SetGlobalBuffer((__gm__ float*)x, (uint64_t)batch * inFeatures);\n        y_.SetGlobalBuffer((__gm__ float*)y, (uint64_t)batch * outFeatures);\n        outFeatures_ = outFeatures;\n        inFeatures_ = inFeatures;\n        batch_ = batch;\n        multiplier_ = multiplier;\n        negativeSlope_ = negativeSlope;\n        pipe_.InitBuffer(xBuffer_, 8192 * sizeof(float));\n        pipe_.InitBuffer(weightBuffer_, 8192 * sizeof(float));\n        pipe_.InitBuffer(productBuffer_, 8192 * sizeof(float));\n        pipe_.InitBuffer(scratchBuffer_, 8192 * sizeof(float));\n        pipe_.InitBuffer(sumBuffer_, 64 * sizeof(float));\n    }\n\n    __aicore__ inline void Process() {\n        uint64_t total = (uint64_t)batch_ * outFeatures_;\n        uint64_t block = AscendC::GetBlockNum();\n        if (block == 0) block = 1;\n        uint64_t id = AscendC::GetBlockIdx();\n        // Scalar GM writes use cache lines; split blocks on 128-byte boundaries\n        // so two cores never update the same output cache line.\n        uint64_t chunk = ((total + block - 1) / block + 31) / 32 * 32;\n        uint64_t begin = id * chunk;\n        uint64_t end = begin + chunk;\n        if (end > total) end = total;\n        auto xLocal = xBuffer_.Get<float>();\n        auto weightLocal = weightBuffer_.Get<float>();\n        auto product = productBuffer_.Get<float>();\n        auto scratch = scratchBuffer_.Get<float>();\n        auto result = sumBuffer_.Get<float>();\n        uint32_t currentRow = 0xFFFFFFFF;\n        for (uint64_t linear = begin; linear < end; ++linear) {\n            uint32_t b = static_cast<uint32_t>(linear / outFeatures_);\n            uint32_t j = static_cast<uint32_t>(linear % outFeatures_);\n            float sum = 0.0f;\n            uint64_t xbase = (uint64_t)b * inFeatures_;\n            uint64_t wbase = (uint64_t)j * inFeatures_;\n            if (inFeatures_ <= 8192 && inFeatures_ % 8 == 0) {\n                if (currentRow != b) {\n                    AscendC::DataCopy(xLocal, x_[xbase], inFeatures_);\n                    currentRow = b;\n                }\n                AscendC::DataCopy(weightLocal, weight_[wbase], inFeatures_);\n                AscendC::PipeBarrier<PIPE_ALL>();\n                AscendC::Mul(product, xLocal, weightLocal, inFeatures_);\n                AscendC::PipeBarrier<PIPE_V>();\n                AscendC::ReduceSum(result, product, scratch, inFeatures_);\n                AscendC::PipeBarrier<PIPE_ALL>();\n                sum = result.GetValue(0);\n            } else {\n                for (uint32_t k = 0; k < inFeatures_; ++k) {\n                    sum += x_.GetValue(xbase + k) * weight_.GetValue(wbase + k);\n                }\n            }\n            sum += bias_.GetValue(j);\n            sum *= multiplier_;\n            if (sum < 0.0f) sum *= negativeSlope_;\n            y_.SetValue(linear, sum);\n        }\n    }\n\nprivate:\n    AscendC::GlobalTensor<float> weight_;\n    AscendC::GlobalTensor<float> bias_;\n    AscendC::GlobalTensor<float> x_;\n    AscendC::GlobalTensor<float> y_;\n    AscendC::TPipe pipe_;\n    AscendC::TBuf<AscendC::TPosition::VECCALC> xBuffer_;\n    AscendC::TBuf<AscendC::TPosition::VECCALC> weightBuffer_;\n    AscendC::TBuf<AscendC::TPosition::VECCALC> productBuffer_;\n    AscendC::TBuf<AscendC::TPosition::VECCALC> scratchBuffer_;\n    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuffer_;\n    uint32_t outFeatures_ = 0;\n    uint32_t inFeatures_ = 0;\n    uint32_t batch_ = 0;\n    float multiplier_ = 1.0f;\n    float negativeSlope_ = 0.1f;\n};\n\nextern "C" __global__ __aicore__ void p12_boundary_entry(\n    GM_ADDR input_0, GM_ADDR input_1, GM_ADDR input_2, GM_ADDR output_0,\n    GM_ADDR workspace, GM_ADDR tiling) {\n    GET_TILING_DATA(tiling_data, tiling);\n    KernelGemmLeakyRelu op;\n    op.Init(input_0, input_1, input_2, output_0,\n            tiling_data.input0_dim0, tiling_data.input0_dim1,\n            tiling_data.input2_dim0, tiling_data.multiplier,\n            tiling_data.init_arg_3);\n    op.Process();\n}\n'
# <<<END_IMPROVE>>>

class ModelNew(nn.Module):
    """
        Simple model that performs a Gemm, multiplies the result, and applies LeakyReLU.
        """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = multiplier
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        if tuple(self.gemm.weight.shape) != (8192, 8192) or self.gemm.weight.dtype != torch.float32 or not self.gemm.weight.is_contiguous():
            raise ValueError('boundary input guard failed: gemm_weight')
        if tuple(self.gemm.bias.shape) != (8192,) or self.gemm.bias.dtype != torch.float32 or not self.gemm.bias.is_contiguous():
            raise ValueError('boundary input guard failed: gemm_bias')
        if tuple(x.shape) != (1024, 8192) or x.dtype != torch.float32 or not x.is_contiguous():
            raise ValueError('boundary input guard failed: x')
        if self.multiplier != 2.0:
            raise ValueError('boundary attribute requires recapture: multiplier')
        if 8192 != 8192:
            raise ValueError('boundary attribute requires recapture: init_arg_0')
        if 8192 != 8192:
            raise ValueError('boundary attribute requires recapture: init_arg_1')
        if 0.1 != 0.1:
            raise ValueError('boundary attribute requires recapture: init_arg_3')
        return ascendc_binding.p12_boundary_entry(self.gemm.weight, self.gemm.bias, x, self.multiplier, 8192, 8192, 0.1)
