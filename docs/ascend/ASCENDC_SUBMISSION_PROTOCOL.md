# Ascend C candidate submission protocol

KernelBench-Ascend evaluates complete Ascend C candidates through one public,
backend-specific format. Candidate generation is left to the caller's optimizer;
the benchmark reads the official reference and inputs itself for correctness and
performance evaluation.

## Required candidate surface

A candidate is one Python source file containing:

- `ModelNew`, with a constructor compatible with the official task and a
  `forward` method that invokes the generated binding.
- A literal `ASCENDC_OP_SPEC` mapping. It declares `op_name`, `entrypoint`,
  tensor `inputs`, tensor `outputs`, optional `attributes`, declarative `tiling`,
  `block_dim`, and `workspace`.
- A literal `ASCENDC_TILING` mapping that supplies the values for the declared
  tiling fields, block dimension, and workspace.
- A literal `ASCENDC_KERNEL_SOURCE` string containing the Ascend C device
  implementation and the declared entrypoint.

Tensor descriptors include a name, dtype, shape expressions, and optional
layout. The expression language is intentionally small and checked before code
generation. Expressions can read input dimensions and element counts, refer to
declared scalar attributes, use numeric literals, and compose supported
arithmetic functions. Tiling expressions use the same checked language. Use
`validate_ascendc_candidate()` before building to validate a candidate against
the published protocol. The [P20 candidate](../../golden/ascendc/level1/20_leaky_relu.py)
is a complete example.

Optimizer integrations should use the stable `kernelbench.ascendc` API and
check `API_VERSION` when integrating. Modules under `kernelbench.backends` are
implementation details and are not a compatibility promise.

Python-side candidate imports are restricted to PyTorch. Device code may
include `kernel_operator.h`, and `ModelNew.forward` must call the declared
binding entrypoint.

The candidate may include additional literal metadata, such as a
`BOUNDARY_CONTRACT`, but the benchmark does not require or interpret
optimizer-specific contract or scaffold data.

## What the benchmark generates

The Builder turns the declared operator interface and kernel into an executable
artifact. It generates host tiling code, output shape and dtype inference,
operator registration, CMake project files, and the Torch-NPU C++ binding. It
then compiles the device and host code, packages the operator, installs it into
a candidate-isolated private OPP directory, and compiles the binding. The
system OPP is not modified.

This build service implements the public submission protocol. It does not
choose a task-specific algorithm, tile size, block mapping, or scaffold on the
optimizer's behalf. Those decisions are part of the candidate.

## Evaluation

The evaluator loads `ModelNew` and the private binding in an isolated worker,
compares candidate outputs with the official Torch-NPU reference, checks output
structure and finite values, input mutation, and actual custom-op invocation,
then measures steady-state latency with NPU events. Speedup is calculated
against the same task's Torch-NPU reference on the same device.

The task suite and reference baseline can therefore be shared by optimizers
that use different prompt strategies, BoundaryContract implementations, or
search algorithms. To reuse the existing Ascend C evaluator, they submit this
candidate format; a different code-generation backend can provide its own
candidate adapter while reusing the task definitions and baseline methodology.
