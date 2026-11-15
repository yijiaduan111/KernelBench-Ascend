# Ascend 开发快照：2026-11-15

本次将 worker-1 可读取的现有开发版本经 Windows Git bundle 中转发布到 `yijiaduan111/KernelBench-Ascend`，不改变算子实现或评测策略。

- 源版本：`b9db5383263b21ec354067fb9fed9ffef2966e74`，分支 `codex/boundary-baseline-20261031`。
- 保留已有运行时、Ascend C 工程生成、私有 OPP、binding、隔离 worker、BoundaryContract 和 tiling 相关实现。
- 纳入服务器上尚未提交的 `src/kernelbench/backends/worker.py`、`src/kernelbench/eval.py` 改动：FP32 默认容差为 `1e-3`，可通过 `KERNELBENCH_TOLERANCE_FP32` 配置；隔离 worker 复用同一容差入口。
- 不提交运行环境、编译缓存、凭据和整批原始实验制品。
- Windows 发布副本与 worker-1 当前选定的 407 个源码及文档文件已逐文件进行 SHA-256 比对；发布说明为本次新增文件。

相关框架位于 `yijiaduan111/PyramidKernel-Ascend`，由其子模块固定本仓库的发布提交。

环境沿用 Ascend 910B3、CANN 9.0.1、PyTorch 2.7.1、torch_npu 2.7.1.post4。Ascend 环境安装应沿用专用约束，不直接使用默认 GPU 依赖覆盖已验证环境。
