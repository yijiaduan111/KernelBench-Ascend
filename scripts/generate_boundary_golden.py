#!/usr/bin/env python3
"""Adapt a checked-in Ascend C golden candidate to BoundaryContract v1."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from kernelbench.backends.boundary import capture_boundary_contract, lower_boundary_contract, render_boundary_scaffold
from kernelbench.backends.spec import extract_literal_assignment, extract_literal_spec
from kernelbench.backends.spec_factory import AscendCImplementationConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernelbench-root", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--problem-id", type=int, required=True)
    args = parser.parse_args()

    golden = args.golden.read_text(encoding="utf-8")
    reference_path = next((args.kernelbench_root / f"level{args.level}").glob(f"{args.problem_id}_*.py"))
    contract = capture_boundary_contract(reference_path.read_text(encoding="utf-8"), args.level, args.problem_id, source_path=str(reference_path))
    if contract.status != "supported":
        raise SystemExit(f"unsupported BoundaryContract: {contract.failure_type}: {contract.failure_message}")
    golden_spec = extract_literal_spec(golden)
    implementation = AscendCImplementationConfig.from_spec(golden_spec)
    spec = lower_boundary_contract(contract, implementation)
    source = render_boundary_scaffold(_problem_info_from_reference(reference_path), contract, spec)
    kernel = str(extract_literal_assignment(golden, "ASCENDC_KERNEL_SOURCE"))
    kernel = re.sub(
        rf'(extern\s+"C"\s+__global__\s+__aicore__\s+void\s+){re.escape(golden_spec.entrypoint)}(\s*\()',
        rf"\g<1>{spec.entrypoint}\g<2>",
        kernel,
        count=1,
    )
    begin = source.index("# <<<IMPROVE:ascendc_kernel>>>")
    end = source.index("# <<<END_IMPROVE>>>", begin)
    body_start = source.index("\n", begin) + 1
    replacement = "ASCENDC_KERNEL_SOURCE = " + repr(kernel) + "\n"
    source = source[:body_start] + replacement + source[end:]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(source, encoding="utf-8")
    print(args.output)
    return 0


def _problem_info_from_reference(path: Path):
    # The loader is the single source for class/forward extraction; importing
    # it here avoids duplicating the scaffold parser in this adapter tool.
    from pyramidkernel.core.loader import KernelBenchLoader

    return KernelBenchLoader()._inspect_problem(path)


if __name__ == "__main__":
    raise SystemExit(main())
