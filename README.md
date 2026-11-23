# KernelBench-Ascend

KernelBench-Ascend is an Ascend NPU benchmark based on the official [KernelBench](https://github.com/ScalingIntelligence/KernelBench) task suite. It provides Torch-NPU reference baselines and a generic build-and-evaluate interface for complete Ascend C candidates. Candidate generation and search remain the responsibility of the optimizer using the benchmark.

The original task definitions and upstream MIT license are retained. This fork adds the Ascend runtime, compatibility audit, public candidate protocol, generic Ascend C build pipeline, and evaluation methodology.

## Validated Ascend scope

The published compatibility audit targets **Ascend 910B3**, **CANN 9.0.1**, **Python 3.10**, **PyTorch 2.7.1**, and **torch_npu 2.7.1.post4**.

| Level | Official tasks | Torch-NPU baseline available | Diagnostic exclusions |
| --- | ---: | ---: | ---: |
| 1 | 100 | 98 | 2 |
| 2 | 100 | 100 | 0 |
| 3 | 50 | 46 | 4 |
| **Total** | **250** | **244** | **6** |

The active suite contains 244 tasks. Six tasks remain visible as diagnostics because the validated CANN stack cannot execute their original operator or dtype contract. They are not counted as optimizer failures. Level 4 has not been audited for this release.

The generic Ascend C pipeline has been exercised with four checked-in candidates: Level 1/P20, P25, and P47, and Level 2/P12. This demonstrates the candidate build and evaluation path; it does not claim custom-kernel coverage for all 244 reference tasks.

See the [baseline methodology](docs/ascend/BASELINE_STATUS.md), the [中文基线说明](docs/ascend/BASELINE_STATUS.zh-CN.md), the [architecture and ownership boundary](docs/ascend/ARCHITECTURE.md), and the [machine-readable audit reports](docs/ascend/compatibility/ascend910b3-cann9.0.1/).

## Benchmark and optimizer responsibilities

```text
Official Model + inputs ──> Torch-NPU reference ─────────────┐
                                                             │
Optimizer ──> ModelNew + ASCENDC_OP_SPEC + tiling + kernel   │
                    │                                        │
             static validation                               │
                    │                                        │
       generated host / OPP / binding                        │
                    │                                        │
        Ascend C build + private OPP                         │
                    │                                        │
        isolated correctness + NPU Events ───────────────────┘
```

KernelBench owns the task definitions, reference execution, candidate format, generic validation/build steps, and result measurement. It does not generate candidate code, define a task-specific `BoundaryContract`, select tiling, or run an optimizer's search. The candidate interface is documented in [ASCENDC_SUBMISSION_PROTOCOL.md](docs/ascend/ASCENDC_SUBMISSION_PROTOCOL.md).

## Installation

### Ascend

Use a Python 3.10 environment where CANN and its vendor-matched PyTorch/torch_npu packages are already installed. Do not install the CUDA extras or use a generic PyTorch wheel to provision an Ascend environment.

```bash
python -m pip install -r requirements-ascend-runtime.txt
```

The Ascend requirements file does not install `torch` or `torch_npu`; provision their vendor-matched wheels separately. The `ascend` extra cannot replace them, and the validated versions are listed above.

### CUDA and inherited workflows

Install the optional GPU, generation, Modal, and Hugging Face dependencies in an environment configured for the target GPU:

```bash
python -m pip install -r requirements.txt
```

For Ascend development tools, install `.[ascend,dev]` in the validated runtime. For CUDA development, use `.[gpu,dev]`. PyTorch and device compiler packages must match the target platform and driver/toolkit.

## Audit Torch-NPU references

Each official reference runs in its own worker process. The audit records compatibility, output properties, device/fallback checks, and NPU Event timing. Device indices refer to the devices visible to the process.

```bash
python scripts/audit_ascend_level.py \
  --kernelbench-root . \
  --levels 1 2 3 \
  --devices 0 1 2 3 \
  --output results/ascend910b3/manifest.json \
  --journal results/ascend910b3/journal.jsonl \
  --logs-dir results/ascend910b3/logs
```

Use `--problem-ids` to audit a smaller set. Resume is enabled by default; the JSONL journal lets an interrupted audit continue.

## Evaluate an Ascend C candidate

A candidate is one Python file containing `ModelNew`, a literal operator spec, declarative tiling values, and the Ascend C device source. The generic builder generates the host project, operator registration, and Torch-NPU binding. Candidate artifacts use a hash-isolated private OPP; the system OPP is not modified.

```bash
python scripts/evaluate_ascendc_candidate.py \
  --level 1 \
  --problem-id 25 \
  --candidate golden/ascendc/level1/25_swish.py \
  --cache-root cache/ascendc \
  --output results/p25.json
```

The result records build stages, correctness, reference and candidate latency, and speedup. Only candidates that pass correctness and runtime-integrity checks are eligible for performance comparison. See [EVAL.md](EVAL.md) for the measurement policy.

## Public API and layout

External optimizer integrations should use the versioned [`kernelbench.ascendc`](src/kernelbench/ascendc.py) API. `API_VERSION` identifies this integration surface. Modules under `kernelbench.backends` are implementation details and may change without preserving import compatibility.

```text
KernelBench/                    Official Level 1-3 reference tasks
src/kernelbench/runtimes/       CUDA, ROCm, and Ascend runtime adapters
src/kernelbench/backends/       Candidate validation, build, OPP, and worker
src/kernelbench/ascendc.py      Public Ascend C integration API
scripts/                        Reference audit and candidate evaluation CLIs
golden/ascendc/                 Representative validated candidates
docs/ascend/                    Scope, protocol, and compatibility reports
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and style checks. Hardware-backed Ascend build and timing tests require the validated CANN, PyTorch, and torch_npu environment. CANN binaries, private OPP artifacts, credentials, and experiment caches are not included in the source repository.

## Attribution and license

This project is derived from [ScalingIntelligence/KernelBench](https://github.com/ScalingIntelligence/KernelBench) and retains its MIT license and task definitions. See [LICENSE](LICENSE) for the license text and the upstream repository for the original paper and benchmark history.
