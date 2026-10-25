#!/usr/bin/env python3
"""Regenerate a golden candidate's semantic spec from a trusted task contract."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from kernelbench.backends.ascendc_tasks import (
    get_ascend_task_contract,
    validate_spec_for_task,
)
from kernelbench.backends.security import validate_candidate_source
from kernelbench.backends.spec_factory import render_candidate_with_generated_spec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--problem-id", type=int, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.input.read_text(encoding="utf-8")
    candidate = render_candidate_with_generated_spec(
        source, args.level, args.problem_id
    )
    spec = validate_candidate_source(candidate)
    validate_spec_for_task(
        spec, get_ascend_task_contract(args.level, args.problem_id)
    )
    if args.output.exists():
        if args.output.read_text(encoding="utf-8") != candidate:
            parser.error(f"output exists with different content: {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(candidate, encoding="utf-8")
    digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
    print(f"candidate={args.output.resolve()} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
