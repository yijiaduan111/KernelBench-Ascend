"""Public Ascend C candidate protocol."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path

from .binding import private_opp_environment
from .builder import AscendCArtifact, AscendCBuilder, scrubbed_environment
from .errors import CandidateTimeoutError, NpuRuntimeError
from .security import validate_candidate_source
from .spec import AscendCSpec
from .ascendc_tasks import get_ascend_task_contract, validate_spec_for_task


def validate_ascendc_candidate(
    source: str,
    *,
    level: int | None = None,
    problem_id: int | None = None,
) -> AscendCSpec:
    spec = validate_candidate_source(source)
    if (level is None) != (problem_id is None):
        raise ValueError("level and problem_id must be provided together")
    if level is not None and problem_id is not None:
        validate_spec_for_task(spec, get_ascend_task_contract(level, problem_id))
    return spec


def build_ascendc_artifact(
    candidate_source: str,
    build_dir: str | Path = "cache/ascendc",
    target_soc: str = "Ascend910B3",
    *,
    level: int | None = None,
    problem_id: int | None = None,
    build_timeout: int = 900,
) -> AscendCArtifact:
    validate_ascendc_candidate(
        candidate_source,
        level=level,
        problem_id=problem_id,
    )
    return AscendCBuilder(
        build_dir,
        target_soc,
        build_timeout=build_timeout,
    ).build(
        candidate_source,
        level=level,
        problem_id=problem_id,
    )


def evaluate_ascendc_artifact(
    artifact: AscendCArtifact,
    reference_source: str,
    *,
    device_index: int = 0,
    precision: str = "fp32",
    seed: int = 42,
    num_correct_trials: int = 3,
    num_warmup: int = 3,
    num_perf_trials: int = 10,
    timeout: int = 300,
) -> dict:
    """Evaluate in a fresh process so CANN sees the private OPP at startup."""
    if artifact.binding_path is None:
        raise NpuRuntimeError("artifact has no compiled binding")
    evaluation_root = artifact.root / "evaluation" / uuid.uuid4().hex
    evaluation_root.mkdir(parents=True, exist_ok=True)
    reference_path = evaluation_root / "reference.py"
    reference_path.write_text(reference_source, encoding="utf-8")
    result_path = evaluation_root / "result.json"
    job = {
        "reference_path": str(reference_path),
        "candidate_path": str(artifact.root / "source" / "candidate.py"),
        "binding_path": str(artifact.binding_path),
        "binding_module": artifact.names.binding_module,
        # ASCEND_RT_VISIBLE_DEVICES below remaps the selected physical device
        # to logical npu:0 inside the isolated worker.
        "device_index": 0,
        "physical_device_index": device_index,
        "precision": precision,
        "seed": seed,
        "num_correct_trials": num_correct_trials,
        "num_warmup": num_warmup,
        "num_perf_trials": num_perf_trials,
    }
    job_path = evaluation_root / "job.json"
    job_path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    env = scrubbed_environment()
    env.update(private_opp_environment(artifact.opp_root, artifact.binding_path.parent))
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(device_index)
    package_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(
        [package_root, env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    command = [
        sys.executable,
        "-m",
        "kernelbench.backends.worker",
        str(job_path),
        str(result_path),
    ]
    process = subprocess.Popen(
            command,
            env=env,
            cwd=artifact.root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
        raise CandidateTimeoutError(f"Ascend C worker timed out after {timeout}s") from exc
    (evaluation_root / "worker.log").write_text(output, encoding="utf-8")
    if process.returncode != 0 or not result_path.is_file():
        tail = "\n".join(output.splitlines()[-40:])
        raise NpuRuntimeError(
            f"Ascend C worker failed with return code {process.returncode}\n{tail}"
        )
    report = json.loads(result_path.read_text(encoding="utf-8"))
    report["evaluation_root"] = str(evaluation_root)
    return report


__all__ = [
    "AscendCSpec",
    "AscendCBuilder",
    "AscendCArtifact",
    "validate_ascendc_candidate",
    "build_ascendc_artifact",
    "evaluate_ascendc_artifact",
]
