#!/usr/bin/env python3
"""Build and evaluate one Ascend C candidate through the generic backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

from kernelbench.backends.ascendc import (
    build_ascendc_artifact,
    evaluate_ascendc_artifact,
    validate_ascendc_candidate,
)


def _reference_source(level: int, problem_id: int) -> str:
    from kernelbench.eval import fetch_ref_arch_from_level_problem_id

    return fetch_ref_arch_from_level_problem_id(level, problem_id)


def _write_result(result: dict, path: Path | None) -> None:
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    print(payload, end="")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile, load, check and time a KernelBench Ascend C candidate."
    )
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--problem-id", type=int, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("cache/ascendc"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--correctness-trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--performance-trials", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--build-timeout", type=int, default=900)
    parser.add_argument("--target-soc", default="Ascend910B3")
    parser.add_argument(
        "--validate-task-contract",
        action="store_true",
        help="Require the candidate custom op to implement the suite's full-operator ABI.",
    )
    args = parser.parse_args()

    for name in ("correctness_trials", "performance_trials", "timeout", "build_timeout"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")

    result = {
        "level": args.level,
        "problem_id": args.problem_id,
        "candidate_path": str(args.candidate.resolve()),
        "target_soc": args.target_soc,
        "device": args.device,
        "precision": args.precision,
        "task_contract_validated": bool(args.validate_task_contract),
    }
    started = time.perf_counter()
    try:
        candidate_source = args.candidate.read_text(encoding="utf-8")
        result["source_sha256"] = hashlib.sha256(candidate_source.encode("utf-8")).hexdigest()
        result["stages"] = {
            "static_validation": "running",
            "ascendc_compile": "pending",
            "opp_install": "pending",
            "binding_compile": "pending",
            "worker": "pending",
        }
        validate_ascendc_candidate(
            candidate_source,
            level=args.level if args.validate_task_contract else None,
            problem_id=args.problem_id if args.validate_task_contract else None,
        )
        result["stages"]["static_validation"] = "passed"
        reference_source = _reference_source(args.level, args.problem_id)
        result["reference_sha256"] = hashlib.sha256(reference_source.encode("utf-8")).hexdigest()
        artifact = build_ascendc_artifact(
            candidate_source,
            build_dir=args.cache_root,
            target_soc=args.target_soc,
            level=args.level if args.validate_task_contract else None,
            problem_id=args.problem_id if args.validate_task_contract else None,
            build_timeout=args.build_timeout,
        )
        result["artifact"] = {
            "candidate_hash": artifact.candidate_hash,
            "root": str(artifact.root),
            "cache_hit": bool(artifact.metadata.get("cache_hit", False)),
            "private_opp": True,
            "system_opp_modified": bool(
                artifact.metadata.get("system_opp_modified", False)
            ),
            "build_timings": artifact.metadata.get("timings", {}),
        }
        result["stages"].update(
            {
                "ascendc_compile": "passed",
                "opp_install": "passed",
                "binding_compile": "passed",
                "worker": "running",
            }
        )
        result.update(
            evaluate_ascendc_artifact(
                artifact,
                reference_source,
                device_index=args.device,
                precision=args.precision,
                num_correct_trials=args.correctness_trials,
                num_warmup=args.warmup,
                num_perf_trials=args.performance_trials,
                timeout=args.timeout,
            )
        )
        result["stages"]["worker"] = "passed" if result.get("correctness") else "failed"
    except Exception as exc:
        result.update(
            {
                "compiled": False,
                "correctness": False,
                "eligible_for_performance": False,
                "failure_type": getattr(exc, "failure_type", "evaluation_error"),
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "speedup": None,
            }
        )
        if "stages" in result:
            for stage, status in list(result["stages"].items()):
                if status in {"running", "pending"}:
                    result["stages"][stage] = "failed"
    result["total_wall_seconds"] = time.perf_counter() - started
    _write_result(result, args.output)
    return 0 if result.get("correctness") else 1


if __name__ == "__main__":
    raise SystemExit(main())
