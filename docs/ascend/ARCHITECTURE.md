# Ascend benchmark architecture

KernelBench-Ascend separates benchmark-owned behavior from optimizer-owned
behavior. The benchmark defines the tasks and score evidence. An optimizer
produces a candidate in the documented submission format and may choose any
prompting, contract, tiling, or search strategy.

## Evaluation flow

```text
Official task source
      │
      ├── Model / init inputs / test inputs
      │          │
      │          └── Torch-NPU reference ───────────────────┐
      │                                                     │
      └── Candidate file                                    │
             │                                              │
       Literal protocol parsing                            │
             │                                              │
       Static/security validation                           │
             │                                              │
       Generated host + operator registration               │
       + CMake + Torch-NPU binding                          │
             │                                              │
       Ascend C compile → candidate-private OPP             │
             │                                              │
       Isolated worker: correctness + integrity + NPU Event ┘
             │
       Structured result JSON
```

The reference path never depends on the candidate generator. Both outputs are
checked against the original official task semantics. A candidate is eligible
for timing comparison only after correctness and runtime-integrity checks pass.

## Ownership boundary

| Benchmark owns | Optimizer owns |
| --- | --- |
| Official task definitions and input construction | Task analysis and prompts |
| Torch-NPU reference execution and compatibility reports | BoundaryContract or other task metadata |
| Public candidate schema and generic static checks | Candidate `ModelNew`, spec, tiling, and device algorithm |
| Host/registration/binding project generation | Tiling selection and launch strategy |
| Private OPP build and isolated evaluation | Search, retries, and candidate selection |
| Correctness and timing evidence | Model/provider configuration and credentials |

The benchmark treats optional optimizer metadata as opaque. No BoundaryContract,
scaffold convention, model route, or search policy is required by the public
candidate protocol.

## Code map

| Path | Responsibility | Stability |
| --- | --- | --- |
| `KernelBench/` | Official Level 1-3 reference task source | Benchmark data |
| `src/kernelbench/runtimes/` | CUDA, ROCm, and Ascend device adapters | Internal implementation |
| `src/kernelbench/ascendc.py` | Ascend C candidate/build/evaluation exports | Public integration API (`API_VERSION`) |
| `src/kernelbench/backends/spec.py` | Literal candidate schema and expression validation | Internal implementation |
| `src/kernelbench/backends/security.py` | Candidate source restrictions | Internal implementation |
| `src/kernelbench/backends/generator.py` | Host/OPP/binding project generation | Internal implementation |
| `src/kernelbench/backends/builder.py` | Compile, cache, and private OPP installation | Internal implementation |
| `src/kernelbench/backends/worker.py` | Isolated correctness and event timing | Internal implementation |
| `scripts/audit_ascend_level.py` | Official reference compatibility audit | CLI |
| `scripts/evaluate_ascendc_candidate.py` | Single candidate evaluation CLI | CLI |
| `docs/ascend/ASCENDC_SUBMISSION_PROTOCOL.md` | Candidate submission contract | Public protocol |

External consumers should import the stable façade instead of implementation
modules:

```python
from kernelbench.ascendc import (
    API_VERSION,
    build_ascendc_artifact,
    evaluate_ascendc_artifact,
    validate_ascendc_candidate,
)
```

## Failure and artifact boundaries

Static rejection, host/device compilation, OPP installation, binding loading,
runtime execution, and correctness are reported as separate stages. Build logs
and artifacts are stored beneath the configured cache root; only bounded error
summaries belong in result metadata. Build and worker subprocesses receive a
sanitized environment. Each candidate's OPP and binding live under a
candidate-specific artifact directory, and the system OPP is not modified.

## Compatibility boundary

The published Ascend audit establishes Torch-NPU reference availability for
244 of 250 Level 1-3 tasks on the documented 910B3/CANN stack. It does not
establish that custom candidates exist or compile for every supported
reference. See [BASELINE_STATUS.md](BASELINE_STATUS.md) and the checked-in
compatibility manifest for the precise active and diagnostic task lists.
