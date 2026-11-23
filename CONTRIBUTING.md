# Contributing

KernelBench-Ascend keeps benchmark behavior independent from candidate
generation and optimizer policy. Contributions should preserve official task
semantics and the public candidate protocol.

## Development environment

Use Python 3.10. For Ascend development, first activate a CANN environment with
the matching PyTorch and torch_npu packages, then install the project and its
development dependencies:

```bash
python -m pip install -e ".[ascend,dev]"
```

The `ascend` extra does not install PyTorch or torch_npu. Use the vendor
installation procedure for those packages. For the inherited CUDA workflow,
install `.[gpu,dev]` in an environment configured for the target CUDA stack.

## Checks

Run the unit suite and style checks before opening a change:

```bash
python -m pytest
python -m ruff check
python -m ruff format --check
```

The pytest configuration adds `src/` to the import path. Hardware-backed tests
must run in a matching CUDA or Ascend environment. The focused Ruff gate covers
the maintained Ascend runtime, candidate build/evaluation, public API, and
related tests; inherited upstream modules are excluded to avoid unrelated
formatting churn.

## Change guidelines

- Keep official reference sources and task semantics unchanged.
- Keep optimizer-specific contracts, prompts, model providers, and search
  policies outside the benchmark.
- Make candidate-format changes through the public `kernelbench.ascendc` API;
  update `API_VERSION` for integration-breaking changes.
- Keep CANN binaries, credentials, OPP installations, and generated run data
  out of commits.
- For compatibility changes, update the machine-readable manifests and both
  baseline reports together.
- Include the environment, task IDs, checks performed, and remaining hardware
  validation in the change description.
