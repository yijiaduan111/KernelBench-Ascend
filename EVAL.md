# Evaluation methodology

This document describes the evaluation contract for KernelBench-Ascend. The
benchmark measures candidates against the task's original framework reference;
it does not generate candidates or define the optimizer's search procedure.

## Reference and candidate

For Ascend, the reference is the official `Model`, `get_init_inputs`, and
`get_inputs` executed with Torch-NPU on the selected NPU. Do not alter the
reference semantics to make an unsupported operator appear compatible. The
published audit identifies which tasks can run unchanged on the validated
Ascend stack.

The candidate implements the same task through the public Ascend C submission
protocol. The builder validates the declared interface, generates host and
binding code, compiles the operator, installs it into a candidate-private OPP,
and loads the binding in a fresh worker. Candidate generation and optimization
strategy are external to the benchmark.

## Correctness gate

A candidate must pass all required correctness trials and the runtime integrity
checks before its latency is eligible for comparison. The evaluator checks the
output structure and tensor metadata, numerical agreement with the reference,
finite outputs, input immutability, random-state stability, and evidence that
the custom operator was actually called. A failed candidate has no speedup
score.

The tolerance is precision-dependent and is recorded with the evaluation
result. Formal comparisons should record the exact candidate source hash,
reference source hash, benchmark commit, hardware, CANN, PyTorch, torch_npu,
device, and evaluator settings.

## Performance measurement

Ascend candidate and reference are measured on the same NPU with the NPU Event
timer. Compilation and first-call cost are reported separately from warmed
steady-state latency. Report at least:

- compilation and binding-load outcome;
- correctness outcome and failure stage, if any;
- reference and candidate steady-state latency;
- speedup, computed as reference latency divided by candidate latency;
- warm-up, trial count, device, and software versions.

Use the evaluator's structured result rather than timing only the host-side
Python call. Report per-task outcomes as well as aggregate scores. For a
multi-attempt optimizer, declare the attempt budget and whether task-level
success means at least one compiling or correct candidate. Do not compare
aggregates with different denominators or correctness tolerances.

## Compatibility and denominators

The validated Level 1-3 reference suite has 250 official tasks: 244 active
Torch-NPU baselines and six diagnostic exclusions. Use the checked-in
`supported.json` and `excluded.json` lists as denominators. Excluded tasks must
remain visible in reports and must not be counted as candidate failures.

Reference compatibility is different from custom-kernel coverage. A runnable
Torch-NPU reference makes a task measurable; it does not imply that a custom
Ascend C candidate has been built, validated, or accelerated for that task.

For inherited CUDA analysis, use the upstream KernelBench documentation and
the corresponding evaluation scripts. Do not mix CUDA timings with Ascend
NPU Event timings in one hardware aggregate.

## Reproducibility and result integrity

Keep raw logs and build artifacts in a run-specific directory outside version
control. Commit sanitized audit manifests and small, reproducible examples;
do not commit API credentials, CANN installations, private OPP trees, or
unbounded experiment caches. Never report performance for candidates that
failed correctness or runtime-integrity checks.

See [the Ascend baseline report](docs/ascend/BASELINE_STATUS.md),
[the submission protocol](docs/ascend/ASCENDC_SUBMISSION_PROTOCOL.md), and
[the architecture overview](docs/ascend/ARCHITECTURE.md) for implementation
and task-scope details.
