#!/usr/bin/env python3
"""Audit official KernelBench reference programs on an Ascend NPU.

Each problem is executed in a fresh worker process.  The source is never
modified, and a problem timeout kills only that worker so the matrix can still
finish for the remaining problems.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _version(module_name: str) -> str:
    try:
        package_name = {"torch_npu": "torch-npu", "torch": "torch"}.get(module_name, module_name)
        return importlib.metadata.version(package_name)
    except Exception:
        return "unavailable"


def _walk(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.to("npu:0")
    if isinstance(value, tuple):
        return tuple(_walk(item) for item in value)
    if isinstance(value, list):
        return [_walk(item) for item in value]
    if isinstance(value, dict):
        return {key: _walk(item) for key, item in value.items()}
    return value


def _clone(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    return value


def _same_structure(left, right) -> bool:
    import torch
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.shape != right.shape or left.dtype != right.dtype or left.device != right.device:
            return False
        if left.is_floating_point():
            return bool(torch.allclose(left, right, atol=1e-5, rtol=1e-5, equal_nan=True))
        return bool(torch.equal(left, right))
    if isinstance(left, (tuple, list)) and type(left) is type(right) and len(left) == len(right):
        return all(_same_structure(a, b) for a, b in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict) and list(left) == list(right):
        return all(_same_structure(left[key], right[key]) for key in left)
    return left == right


def _all_finite(value) -> bool:
    import torch
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all().item())
    if isinstance(value, (tuple, list)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    return True


def _output_facts(value, path="output") -> dict[str, Any]:
    import torch
    if isinstance(value, torch.Tensor):
        return {
            "path": path,
            "type": "Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "finite": bool(torch.isfinite(value).all().item()) if value.is_floating_point() else True,
        }
    if isinstance(value, (tuple, list)):
        return {"path": path, "type": type(value).__name__, "items": [_output_facts(item, f"{path}[{index}]") for index, item in enumerate(value)]}
    if isinstance(value, dict):
        return {"path": path, "type": "dict", "items": {str(key): _output_facts(item, f"{path}[{key!r}]") for key, item in value.items()}}
    return {"path": path, "type": type(value).__name__}


def _find_problems(root: Path, level: int) -> list[Path]:
    directory = root / f"level{level}"
    if not directory.exists():
        directory = root / "KernelBench" / f"level{level}"
    return sorted(directory.glob("*.py"), key=lambda path: int(path.name.split("_", 1)[0]))


def _load(path: Path):
    name = f"kernelbench_audit_{path.stem}_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker(path: Path, warmup: int, repeats: int) -> dict[str, Any]:
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_device(0)
    module = _load(path)
    model = module.Model(*_walk(module.get_init_inputs()))
    model = model.to("npu:0")
    cold_inputs = _walk(module.get_inputs())
    torch.npu.synchronize()
    started = time.perf_counter()
    output = model(*cold_inputs)
    torch.npu.synchronize()
    cold_ms = (time.perf_counter() - started) * 1000.0
    repeat_output = model(*_clone(cold_inputs))
    torch.npu.synchronize()
    repeat_stable = _same_structure(output, repeat_output)
    finite = _all_finite(output) and _all_finite(repeat_output)
    for _ in range(warmup):
        model(*_walk(module.get_inputs()))
        torch.npu.synchronize()
    timings = []
    for _ in range(repeats):
        inputs = _walk(module.get_inputs())
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        model(*inputs)
        end.record()
        torch.npu.synchronize()
        timings.append(float(start.elapsed_time(end)))
    status = "supported" if repeat_stable and finite else "failed"
    return {
        "status": status,
        "failure_type": None if status == "supported" else ("non_finite_output" if not finite else "repeat_instability"),
        "cold_ms": cold_ms,
        "warmup_ms": timings,
        "output": _output_facts(output),
        "nan_inf": not finite,
        "repeat_stable": repeat_stable,
    }


def _run_worker(path: Path, warmup: int, repeats: int) -> None:
    try:
        print(json.dumps(worker(path, warmup, repeats), sort_keys=True), flush=True)
    except Exception as exc:
        print(json.dumps({"status": "failed", "failure_type": type(exc).__name__, "error": str(exc)}, sort_keys=True), flush=True)
        raise


def _base_record(path: Path, level: int) -> dict[str, Any]:
    return {
        "hardware": os.environ.get("ASCEND_SOC_VERSION", "Ascend910B3"),
        "cann": os.environ.get("ASCEND_TOOLKIT_VERSION", "9.0.1"),
        "torch": _version("torch"),
        "torch_npu": _version("torch_npu"),
        "level": level,
        "problem_id": int(path.name.split("_", 1)[0]),
        "problem": path.name,
        "source_hash": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.kernelbench_root).resolve()
    problems = _find_problems(root, args.level)
    if args.problem_ids:
        selected = set(args.problem_ids)
        problems = [path for path in problems if int(path.name.split("_", 1)[0]) in selected]
    results = []
    for path in problems:
        record = _base_record(path, args.level)
        env = os.environ.copy()
        for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY", "DASHSCOPE_API_KEY", "PYRAMIDKERNEL_ENV_FILE"):
            env.pop(key, None)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(path), "--warmup", str(args.warmup), "--repeats", str(args.repeats)]
        started = time.monotonic()
        try:
            process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
            stdout, stderr = process.communicate(timeout=args.timeout)
            lines = [line.strip() for line in stdout.splitlines() if line.strip()]
            payload = None
            for line in reversed(lines):
                try:
                    candidate = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict) and "status" in candidate:
                    payload = candidate
                    break
            if payload is None:
                record.update(
                    status="failed",
                    failure_type=("timeout" if process.returncode == -signal.SIGKILL else "worker_no_output"),
                    error=stderr[-4000:] or stdout[-4000:],
                )
            else:
                record.update(payload)
            if process.returncode != 0 and record.get("status") == "supported":
                record.update(status="failed", failure_type="worker_error", error=stderr[-4000:])
        except subprocess.TimeoutExpired:
            # Some CANN components can leave descendants holding the worker's
            # stdout/stderr descriptors.  Kill the whole process group and do
            # not block trying to drain a pipe owned by a dead descendant.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.kill()
            except ProcessLookupError:
                pass
            try:
                process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                if process.stdout:
                    process.stdout.close()
                if process.stderr:
                    process.stderr.close()
            record.update(status="failed", failure_type="timeout")
        except Exception as exc:
            record.update(status="failed", failure_type="audit_error", error=str(exc))
        record["elapsed_wall_s"] = round(time.monotonic() - started, 3)
        results.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    return {"metadata": {"hardware": "Ascend910B3", "cann": "9.0.1", "torch": _version("torch"), "torch_npu": _version("torch_npu"), "level": args.level}, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernelbench-root", default=".")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--output", default="ascend_level_audit.json")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--problem-ids", type=int, nargs="*")
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        _run_worker(args.worker, args.warmup, args.repeats)
        return 0
    report = audit(args)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
