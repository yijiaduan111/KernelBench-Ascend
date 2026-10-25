# P20/P25 可信 Ascend C spec：第一阶段结果

本阶段在 Ascend 910B3 的单卡 `npu:0` 上验证了由题目 contract 生成的完整算子 spec。PyramidKernel 和 KernelBench 基线提交分别为 `ce35cedcafda9804d54783cd3d358e0ba9bdd5e4` 与 `e44cef35d8412f3698a8e7f9944adf5399f56ba4`；本阶段的工作树改动尚未提交。环境为 CANN 9.0.1、PyTorch 2.7.1+cpu、torch-npu 2.7.1.post4。250 题参考实现审计清单有 244 题 supported、6 题 excluded；P45 的 contract 状态已与审计清单同步为 supported。

`AscendTaskContract` 提供 P20/P25 的 FP32、二维输入输出名称、ND layout、同形输出规则，以及 P20 的 `negative_slope=0.01` 默认值。`build_full_operator_spec()` 从可信 contract 生成语义字段和固定的算子入口名称，从独立的 `AscendCImplementationConfig` 获取 tiling、`block_dim` 和 workspace。验证过的 P20/P25 实现配置保存在 `spec_factory.py`，与原 golden 数值一致。候选模板中的原 spec 可以是占位值；渲染器会整体替换它，保留 device kernel、`ASCENDC_TILING` 和 `ModelNew`。

在 Python 3.10 环境中，服务器原先缺少编译 binding 所需的开发头文件。本次将对应 Ubuntu 头文件包解压在个人结果目录，通过 `KERNELBENCH_PYTHON_INCLUDE_DIR` 指向该目录；构建器的生成模板支持这个私有路径。未修改系统 Python、CANN 或系统 OPP。

官方输入 `4096 × 393216`，默认评测参数，NPU Event 稳态 p50：

| 题目 | 生成候选正确性 | 生成候选 p50 | golden p50 | torch-npu reference p50 | 生成候选加速比 |
| --- | --- | ---: | ---: | ---: | ---: |
| P20 LeakyReLU | 通过 | 10.861 ms | 10.938 ms | 10.318 ms | 0.950× |
| P25 Swish | 通过 | 10.446 ms | 10.430 ms | 27.407 ms | 2.624× |

两题的静态检查、Ascend C 编译、私有 OPP 安装、binding 编译/加载、冷启动及预热后正确性均通过，且记录到真实 custom op 调用。生成候选与 golden 的 p50 差异分别为 −0.70% 和 +0.15%，可视为本次独立运行的测量波动。P20 官方输入仅有非负数，因此另外用带负值的小输入测试了默认斜率和 `negative_slope=0.2`；P20/P25 均通过 `8 × 64` 与 `16 × 128` 两种尺寸。

相关单元回归为 95 passed，覆盖另外 13 道题的原有 contract 范围、Ascend C 静态检查及 CUDA 代码的静态检查。当前服务器没有 NVIDIA GPU，未运行 CUDA 设备评测。完整单测收集时，仓库既有的 `test_eval_timing.py` 在模块级调用 `pytest.skip` 导致收集错误；本阶段用相关测试文件完成回归。

完整归档位于服务器 `/home/ma-user/work/dyj/results/ascendc_spec_factory_phase1_20261022/`：`metadata.json` 记录提交、环境、审计与源码 hash；`summary.json` 汇总两题对照；`p20/`、`p25/` 分别保存生成的 spec、候选、golden、reference、官方评测和小输入评测 JSON；`cache/` 保存私有 OPP、binding、构建日志及 worker 日志。当前结果只证明 P20/P25 的接口可自动生成并完成评测；Pointwise 模板、LLM 生成和 tiling 搜索属于后续阶段。
