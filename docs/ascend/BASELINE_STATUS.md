# Ascend NPU baseline status

Updated: 2026-10-12

## Decision

KernelBench Levels 1-3 contain 250 official tasks. In the validated Ascend
environment, 244 tasks run an unmodified-semantics Torch-NPU eager reference.
The remaining six tasks are explicit diagnostics because CANN 9.0.1 cannot
execute their original operator or dtype contracts.

This repository can therefore be used as an Ascend optimization benchmark
with the following precise scope:

```text
244-task Ascend active benchmark with 6 documented exclusions
```

The main-suite denominator is 244. The exclusions must not be silently removed
or counted as LLM candidate failures.

## Validated environment

```text
Hardware:   Ascend 910B3, 64 GiB HBM per device
CANN:       9.0.1
PyTorch:    2.7.1+cpu
torch_npu:  2.7.1.post4
Python:     3.10
Audit NPUs: physical devices 4, 5, 6, 7
```

The `+cpu` PyTorch package label is the value recorded by the audit. NPU
execution is provided by the matching torch_npu package; it does not mean the
performance baseline ran on CPU.

## Coverage

| Level | Official | Active | Diagnostic |
| --- | ---: | ---: | ---: |
| Level 1 | 100 | 98 | 2 |
| Level 2 | 100 | 100 | 0 |
| Level 3 | 50 | 46 | 4 |
| **Total** | **250** | **244** | **6** |

The exact task lists are versioned alongside this document:

- [`manifest.json`](compatibility/ascend910b3-cann9.0.1/manifest.json), a
  sanitized 250-task audit manifest with source hashes and timing summaries
- [`supported.json`](compatibility/ascend910b3-cann9.0.1/supported.json)
- [`excluded.json`](compatibility/ascend910b3-cann9.0.1/excluded.json)

## What an active baseline means

For every active task, the auditor:

1. Uses the official `Model`, `get_init_inputs`, and `get_inputs` definitions.
2. Recursively moves the model, parameters, and inputs to `npu:0`.
3. Executes the reference with Torch-NPU eager.
4. Checks output structure, shape, dtype, device, and finite values.
5. Detects CPU fallback.
6. Records cold-start latency and steady-state NPU Event samples.

An Ascend C candidate is compared against this reference on the same NPU and
with equivalent inputs. The baseline is not a CUDA result or a CPU fallback.

## Isolated audit protocol

Each task executes in a fresh worker process. Four device workers can run in
parallel, but an OOM, timeout, or NPU runtime failure only destroys the current
task worker. The initial timeout is 120 seconds; failed tasks are retried once
in a clean process with a 300-second timeout.

Full records include source hashes, hardware and software versions, worker
return codes, cold-start latency, NPU Event samples, output summaries,
NaN/Inf checks, and fallback checks. Raw logs remain archived with the
experiment. A sanitized 250-record manifest and the stable active/excluded
lists are versioned here.

## Diagnostic tasks

| Task | Classification | Reason |
| --- | --- | --- |
| L1/P41 MaxPool1D | `unsupported_operator` | CANN rejects the official `dilation=3` |
| L1/P43 MaxPool3D | `unsupported_operator` | CANN rejects the official 3D `dilation=3` |
| L3/P39 GRU | `unsupported_dtype` | DynamicGRUV2 rejects the official FP32 weight contract |
| L3/P40 GRUHidden | `unsupported_dtype` | Same contract limitation |
| L3/P41 GRUBidirectional | `unsupported_dtype` | Same contract limitation |
| L3/P42 GRUBidirectionalHidden | `unsupported_dtype` | Same contract limitation |

Changing MaxPool dilation to 1 or converting GRU weights to FP16 would alter
the benchmark semantics, so those shortcuts are not accepted. A task may move
to the active suite only after an NPU semantic adapter is validated against the
original reference, or a newer CANN/torch_npu combination supports the
original contract.

## Corrected false failures

L1/P45 uses roughly 17.2 GiB of input. The first auditor cloned that input for
a repeat-stability check and caused an audit-induced OOM. The corrected worker
skips only this extra clone above an explicit 8 GiB threshold while retaining
cold execution, steady-state timing, device, and finite-value checks. P45 then
passed with reference Event latency of approximately 4.56 seconds.

L3/P48 and L3/P49 initially failed because `einops` was absent. Both passed
after the runtime dependency was installed. The Ascend runtime dependency file
also records the CANN AOE/TBE helper dependencies used during remediation.

## Maturity boundary

The following has been demonstrated:

- 244 official references execute on Ascend 910B3.
- Reference discovery, process isolation, validation, and NPU Event timing run
  in batch.
- The generic Ascend C pipeline performs static validation, generates host and
  device projects, builds a private OPP, loads the torch_npu binding, checks
  correctness, and reports speedup.
- Three checked-in golden candidates exercise elementwise and reduction paths.

The following has not yet been demonstrated:

- LLM-generated Ascend C candidates compile and pass for all 244 tasks.
- All active tasks can outperform Torch-NPU eager.
- Strict trusted Ascend C task contracts cover all 244 tasks. They currently
  cover the selected Level 1 15-task suite.
- The six diagnostic tasks preserve original semantics on this environment.
- Level 4 Hugging Face tasks have equivalent Ascend audit coverage.

Reference migration is ready for use. The next engineering focus is expanding
trusted task contracts and improving Ascend C generation, repair, correctness,
and performance.

## Evaluation policy

1. Use `supported.json` as the default active task set.
2. Report compile success, correctness, reference runtime, candidate runtime,
   and speedup separately.
3. Rank only candidates that are correct, complete, finite, non-mutating, and
   verified to call the custom operator.
4. Do not include diagnostic tasks in active-suite success-rate denominators.
5. Record hardware, CANN, PyTorch, torch_npu, repository commit, and task source
   hash for each formal experiment.
