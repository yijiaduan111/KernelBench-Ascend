#!/usr/bin/env python3
"""Validate the fixed 15-problem Ascend reference report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED = (1, 10, 20, 25, 33, 40, 42, 45, 47, 50, 61, 82, 89, 95, 97)
KNOWN_FAILURES = {"timeout", "out_of_memory", "unsupported", "failed", "audit_error"}


def validate(report: dict) -> tuple[bool, dict]:
    rows = report.get("results")
    if not isinstance(rows, list):
        return False, {"failure": "results is not a list"}
    by_id = {int(row.get("problem_id")): row for row in rows if row.get("problem_id") is not None}
    missing = [problem_id for problem_id in EXPECTED if problem_id not in by_id]
    extra = sorted(problem_id for problem_id in by_id if problem_id not in EXPECTED)
    invalid_status = [
        {"problem_id": problem_id, "status": row.get("status")}
        for problem_id, row in by_id.items()
        if row.get("status") not in {"supported", "failed"}
    ]
    no_hash = [problem_id for problem_id, row in by_id.items() if not row.get("source_hash")]
    supported = sorted(problem_id for problem_id, row in by_id.items() if row.get("status") == "supported")
    failed = sorted(problem_id for problem_id, row in by_id.items() if row.get("status") == "failed")
    malformed_failure = [
        {"problem_id": problem_id, "failure_type": row.get("failure_type")}
        for problem_id, row in by_id.items()
        if row.get("status") == "failed" and row.get("failure_type") not in KNOWN_FAILURES and not row.get("failure_type", "").endswith("Error")
    ]
    summary = {
        "expected_count": len(EXPECTED),
        "observed_count": len(by_id),
        "missing": missing,
        "extra": extra,
        "supported": supported,
        "failed": failed,
        "invalid_status": invalid_status,
        "missing_source_hash": no_hash,
        "malformed_failure": malformed_failure,
    }
    return not any((missing, extra, invalid_status, no_hash, malformed_failure)), summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    ok, summary = validate(json.loads(args.report.read_text(encoding="utf-8")))
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.summary:
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
