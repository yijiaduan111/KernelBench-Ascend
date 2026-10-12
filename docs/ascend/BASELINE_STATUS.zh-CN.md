# Ascend NPU 基线状态

更新时间：2026-10-12

## 结论

KernelBench Level 1-3 共 250 道官方题目。当前环境中，244 道题可以在
Ascend 910B3 上运行保持原题语义的 Torch-NPU eager reference；其余 6 道
受 CANN 9.0.1 的算子或 dtype contract 限制，暂时作为 diagnostic suite。

因此，本仓库可以作为后续 NPU 算子优化的正式 benchmark，准确口径是：

```text
244-task Ascend active benchmark with 6 documented exclusions
```

后续主测试集的分母是 244。6 道诊断题不能被静默删除，也不能计为候选生成失败。

## 验证环境

```text
Hardware:   Ascend 910B3，每张卡 64 GiB HBM
CANN:       9.0.1
PyTorch:    2.7.1+cpu
torch_npu:  2.7.1.post4
Python:     3.10
Audit NPUs: 物理卡 4、5、6、7
```

审计记录中的 `PyTorch 2.7.1+cpu` 是 Python 包版本标签；实际 NPU 执行由
配套的 torch_npu 提供，并不代表性能基线跑在 CPU 上。

## 覆盖情况

| Level | 官方题目数 | Active | Diagnostic |
| --- | ---: | ---: | ---: |
| Level 1 | 100 | 98 | 2 |
| Level 2 | 100 | 100 | 0 |
| Level 3 | 50 | 46 | 4 |
| 合计 | **250** | **244** | **6** |

机器可读清单位于：

- [`manifest.json`](compatibility/ascend910b3-cann9.0.1/manifest.json)：脱敏后的
  250 题审计结果，包含源码 hash 与计时摘要
- [`supported.json`](compatibility/ascend910b3-cann9.0.1/supported.json)
- [`excluded.json`](compatibility/ascend910b3-cann9.0.1/excluded.json)

## active baseline 的定义

对每一道 active task，审计器会：

1. 使用官方题目中的 `Model`、`get_init_inputs` 和 `get_inputs`。
2. 递归地把模型、参数和输入移动到 `npu:0`。
3. 通过 Torch-NPU eager 执行 reference。
4. 检查输出结构、shape、dtype、device 和有限性。
5. 检查 CPU fallback。
6. 保存冷启动耗时与 warm-up 后的 NPU Event 稳态耗时。

Ascend C candidate 在同一张 NPU、等价输入和同一题目语义下与该 reference
比较正确性和 speedup。baseline 既不是 CUDA 结果，也不是 CPU fallback。

## 隔离审计流程

每道题在独立 worker 进程中执行。四张卡可以并行审计，但单题 OOM、超时或
NPU runtime error 只会销毁当前 worker，不会中断剩余任务。初始超时为 120 秒；
失败题使用全新进程复核一次，复核超时为 300 秒。

完整记录包含源码 hash、软硬件版本、worker 返回码、冷启动延迟、NPU Event
样本、输出摘要、NaN/Inf 和 fallback 检查。原始日志保留在实验归档中；脱敏后的
250 题 manifest 以及稳定的 active/excluded 清单随仓库版本管理。

## 6 道 diagnostic task

| 题目 | 分类 | 原因 |
| --- | --- | --- |
| L1/P41 MaxPool1D | `unsupported_operator` | CANN 拒绝官方 `dilation=3` |
| L1/P43 MaxPool3D | `unsupported_operator` | CANN 拒绝官方 3D `dilation=3` |
| L3/P39 GRU | `unsupported_dtype` | DynamicGRUV2 拒绝官方 FP32 权重 contract |
| L3/P40 GRUHidden | `unsupported_dtype` | 同上 |
| L3/P41 GRUBidirectional | `unsupported_dtype` | 同上 |
| L3/P42 GRUBidirectionalHidden | `unsupported_dtype` | 同上 |

把 MaxPool dilation 改成 1 或把 GRU 权重改成 FP16 都会改变题目语义，不能
作为迁移方案。只有经过原始 reference 等价性验证的 NPU semantic adapter，
或者能直接支持原 contract 的新版 CANN/torch_npu，才能让这些题进入 active suite。

## 已修复的误报

L1/P45 的官方输入约为 17.2 GiB。旧审计器为重复稳定性检查额外 clone 输入，
引发了审计器自身造成的 OOM。修复后，超过 8 GiB 的输入只跳过这一次额外 clone，
仍保留冷启动、稳态计时、device 和有限性检查。P45 随后通过，reference Event
latency 约为 4.56 秒。

L3/P48 和 L3/P49 最初缺少 `einops`。补齐依赖后两题均已通过。Ascend runtime
依赖文件还记录了复核 CANN AOE/TBE 时使用的辅助包。

## 当前成熟度边界

已经证明：

- 244 道官方 reference 可以在 Ascend 910B3 上执行。
- 题目发现、进程隔离、输出检查和 NPU Event 计时能够批量运行。
- 通用 Ascend C 流程可以做静态校验、生成 host/device 工程、构建私有 OPP、
  加载 torch_npu binding、检查正确性并报告 speedup。
- 仓库内三道 golden candidate 已覆盖 elementwise 和 reduction 构建路径。

尚未证明：

- 244 道题都能由 LLM 生成可编译且正确的 Ascend C candidate。
- 244 道 active task 都能超过 Torch-NPU eager。
- 严格可信的 Ascend C ABI contract 已覆盖 244 道题；目前覆盖选定的 Level 1
  十五题。
- 6 道 diagnostic task 在当前环境中已有保持原语义的 NPU reference。
- Level 4 Hugging Face 题目达到同等审计覆盖度。

因此，“NPU 参考基线迁移”已经达到可用阶段。下一阶段的重点应转向扩展可信
task contract，以及提高 Ascend C 生成、联合修复、正确率和性能。

## 后续评测规则

1. 默认从 `supported.json` 读取 244 道 active task。
2. 分别报告 compile、correctness、reference runtime、candidate runtime 和 speedup。
3. 只有正确、输出完整、无 NaN/Inf、不修改输入且确实调用 custom op 的 candidate
   才能进入性能排名。
4. 6 道 diagnostic task 不进入 active-suite success rate 的分母。
5. 每次正式实验记录硬件、CANN、PyTorch、torch_npu、仓库 commit 和题目 source hash。
