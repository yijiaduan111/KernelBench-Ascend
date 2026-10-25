#!/usr/bin/env python3
"""Archive the P20/P25 trusted-spec experiment inputs and environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from kernelbench.backends.artifact import environment_fingerprint
from kernelbench.backends.spec import extract_literal_spec


ROOT = Path(__file__).resolve().parents[1]
TASKS = {20: ("20_LeakyReLU.py", "20_leaky_relu.py"),
         25: ("25_Swish.py", "25_swish.py")}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)

    manifest = ROOT / "docs/ascend/compatibility/ascend910b3-cann9.0.1/manifest.json"
    audit = json.loads(manifest.read_text(encoding="utf-8"))
    shutil.copy2(manifest, output / "compatibility_manifest.json")
    status_counts: dict[str, int] = {}
    for item in audit["results"]:
        status = item["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    tasks = {}
    for problem_id, (reference_name, golden_name) in TASKS.items():
        task_root = output / f"p{problem_id}"
        task_root.mkdir(exist_ok=True)
        reference = ROOT / "KernelBench/level1" / reference_name
        golden = ROOT / "golden/ascendc/level1" / golden_name
        generated = task_root / "generated_candidate.py"
        shutil.copy2(reference, task_root / "reference_source.py")
        shutil.copy2(golden, task_root / "golden_candidate.py")
        spec = extract_literal_spec(generated.read_text(encoding="utf-8"))
        (task_root / "generated_spec.json").write_text(
            json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tasks[str(problem_id)] = {
            "reference_sha256": _sha256(reference),
            "golden_sha256": _sha256(golden),
            "generated_sha256": _sha256(generated),
            "spec_sha256": _sha256(task_root / "generated_spec.json"),
            "audit_status": next(
                item["status"] for item in audit["results"]
                if item["level"] == 1 and item["problem_id"] == problem_id
            ),
        }

    metadata = {
        "pyramidkernel_commit": _commit(ROOT.parent),
        "kernelbench_commit": _commit(ROOT),
        "environment": environment_fingerprint("Ascend910B3"),
        "audit_manifest_sha256": _sha256(manifest),
        "audit_total": len(audit["results"]),
        "audit_status_counts": status_counts,
        "tasks": tasks,
    }
    path = output / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    comparison = {}
    for problem_id in TASKS:
        task_root = output / f"p{problem_id}"
        generated = json.loads((task_root / "generated_official.json").read_text())
        golden = json.loads((task_root / "golden_official.json").read_text())
        small = json.loads((task_root / "generated_small.json").read_text())
        for label, result in (("generated", generated), ("golden", golden)):
            if not (
                result.get("correctness")
                and result.get("task_contract_validated")
                and result.get("artifact", {}).get("private_opp")
                and not result.get("artifact", {}).get("system_opp_modified")
                and result.get("custom_op_call_count", 0) > 0
                and all(trial.get("ok") for trial in result.get("correctness_trials", []))
                and all(stage == "passed" for stage in result.get("stages", {}).values())
            ):
                raise RuntimeError(f"P{problem_id} {label} official evaluation is incomplete")
        if not (
            small.get("all_correct")
            and all(case["result"].get("correctness") for case in small["cases"])
        ):
            raise RuntimeError(f"P{problem_id} small-input evaluation is incomplete")
        generated_ms = generated["candidate"]["p50_ms"]
        golden_ms = golden["candidate"]["p50_ms"]
        comparison[str(problem_id)] = {
            "generated_candidate_p50_ms": generated_ms,
            "golden_candidate_p50_ms": golden_ms,
            "generated_reference_p50_ms": generated["reference"]["p50_ms"],
            "golden_reference_p50_ms": golden["reference"]["p50_ms"],
            "generated_speedup": generated["speedup"],
            "golden_speedup": golden["speedup"],
            "candidate_latency_change_pct": 100.0 * (generated_ms / golden_ms - 1.0),
            "small_cases": [
                {"shape": case["shape"], "negative_slope": case["negative_slope"]}
                for case in small["cases"]
            ],
        }
    summary = {"phase1_passed": True, "comparison": comparison}
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"metadata": str(path), "summary": str(summary_path), "comparison": comparison}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
