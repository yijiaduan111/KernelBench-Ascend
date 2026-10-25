#!/usr/bin/env python3
"""Capture and audit canonical BoundaryContracts for KernelBench tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kernelbench.backends.boundary import capture_boundary_contract, lower_boundary_contract


DEFAULT_PROBLEMS = (1, 10, 20, 25, 33, 40, 42, 45, 47, 50, 61, 82, 89, 95, 97)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernelbench-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--problem-id", type=int, action="append", dest="problem_ids")
    args = parser.parse_args()
    problem_ids = tuple(args.problem_ids or DEFAULT_PROBLEMS)
    level_dir = args.kernelbench_root / f"level{args.level}"
    results = []
    for problem_id in problem_ids:
        matches = sorted(level_dir.glob(f"{problem_id}_*.py"))
        if not matches:
            results.append({"level": args.level, "problem_id": problem_id, "status": "unsupported", "failure_type": "problem_not_found"})
            continue
        path = matches[0]
        source = path.read_text(encoding="utf-8")
        contract = capture_boundary_contract(source, args.level, problem_id, source_path=str(path))
        item = {
            "level": args.level,
            "problem_id": problem_id,
            "problem": path.name,
            "source_hash": contract.source_hash,
            "contract_hash": contract.contract_hash(),
            "status": contract.status,
            "failure_type": contract.failure_type,
            "failure_message": contract.failure_message,
            "inputs": [entry.__dict__ for entry in contract.inputs],
            "attributes": [entry.__dict__ for entry in contract.attributes],
            "outputs": [entry.__dict__ for entry in contract.outputs],
        }
        if contract.status == "supported":
            try:
                spec = lower_boundary_contract(contract)
                item["spec_ok"] = True
                item["spec"] = spec.to_dict()
            except Exception as exc:
                item["spec_ok"] = False
                item["spec_error"] = f"{exc.__class__.__name__}: {exc}"
        results.append(item)
    payload = {"version": "boundary-audit-v1", "level": args.level, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=list) + "\n", encoding="utf-8")
    supported = sum(item["status"] == "supported" for item in results)
    spec_ok = sum(item.get("spec_ok", False) for item in results)
    print(json.dumps({"output": str(args.output), "supported": supported, "spec_ok": spec_ok, "total": len(results)}))
    return 0 if supported == len(results) and spec_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
