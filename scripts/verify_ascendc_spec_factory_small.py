#!/usr/bin/env python3
"""Exercise a generated P20/P25 candidate on small official-semantics inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from kernelbench.backends.ascendc import build_ascendc_artifact, evaluate_ascendc_artifact
from kernelbench.eval import fetch_ref_arch_from_level_problem_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-id", type=int, choices=(20, 25), required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    source = args.candidate.read_text(encoding="utf-8")
    artifact = build_ascendc_artifact(
        source,
        build_dir=args.cache_root,
        level=1,
        problem_id=args.problem_id,
    )
    official_reference = fetch_ref_arch_from_level_problem_id(1, args.problem_id)
    cases = [(8, 64, None), (16, 128, None)]
    if args.problem_id == 20:
        cases.append((8, 64, 0.2))

    results = []
    for rows, cols, slope in cases:
        # Keep the official Model and override only its input factory.  The
        # non-default slope probes forwarding of the Model constructor arg.
        reference = official_reference + (
            f"\ndef get_inputs():\n    return [torch.randn({rows}, {cols})]\n"
        )
        if slope is not None:
            reference += f"\ndef get_init_inputs():\n    return [{slope!r}]\n"
        result = evaluate_ascendc_artifact(
            artifact,
            reference,
            device_index=args.device,
            num_correct_trials=2,
            num_warmup=1,
            num_perf_trials=3,
        )
        results.append(
            {"shape": [rows, cols], "negative_slope": slope, "result": result}
        )

    report = {
        "problem_id": args.problem_id,
        "candidate_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "reference_sha256": hashlib.sha256(official_reference.encode("utf-8")).hexdigest(),
        "artifact_root": str(artifact.root),
        "cases": results,
        "all_correct": all(case["result"].get("correctness") for case in results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "all_correct": report["all_correct"]}))
    return 0 if report["all_correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
