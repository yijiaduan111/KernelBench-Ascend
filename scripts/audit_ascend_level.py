#!/usr/bin/env python3
"""Audit official KernelBench reference programs on an Ascend NPU.

Each problem is executed in a fresh worker process.  The source is never
modified, and a problem timeout kills only that worker so the matrix can still
finish for the remaining problems.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import importlib.metadata
import json
import os
import queue
import signal
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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


def _all_on_npu(value) -> bool:
    import torch
    if isinstance(value, torch.Tensor):
        return value.device.type == "npu"
    if isinstance(value, (tuple, list)):
        return all(_all_on_npu(item) for item in value)
    if isinstance(value, dict):
        return all(_all_on_npu(item) for item in value.values())
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
    torch.manual_seed(42)
    torch.npu.manual_seed(42)
    module = _load(path)
    model = module.Model(*_walk(module.get_init_inputs()))
    model = model.to("npu:0")
    cold_inputs = _walk(module.get_inputs())
    torch.npu.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(*cold_inputs)
        torch.npu.synchronize()
        cold_ms = (time.perf_counter() - started) * 1000.0
        repeat_output = model(*_clone(cold_inputs))
        torch.npu.synchronize()
        repeat_stable = _same_structure(output, repeat_output)
        finite = _all_finite(output) and _all_finite(repeat_output)
        outputs_on_npu = _all_on_npu(output) and _all_on_npu(repeat_output)
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
    if not outputs_on_npu:
        failure_type = "output_device_mismatch"
    elif not finite:
        failure_type = "numerical_error"
    else:
        failure_type = None
    status = "supported" if failure_type is None else "failed"
    ordered = sorted(timings)
    p90_index = max(0, min(len(ordered) - 1, int(0.9 * len(ordered) + 0.999999) - 1))
    return {
        "status": status,
        "failure_type": failure_type,
        "cold_ms": cold_ms,
        "event_ms": timings,
        "reference_runtime_ms": {
            "p50": statistics.median(timings),
            "mean": statistics.fmean(timings),
            "p90": ordered[p90_index],
        },
        "output": _output_facts(output),
        "nan_inf": not finite,
        "outputs_on_npu": outputs_on_npu,
        "repeat_stable": repeat_stable,
        "stochastic_output": not repeat_stable,
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


_SECRET_ENV_KEYS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "DASHSCOPE_API_KEY",
    "PYRAMIDKERNEL_ENV_FILE",
)

_CPU_FALLBACK_MARKERS = (
    "fall back to run on the cpu",
    "falling back to cpu",
    "fallback to cpu",
    "cpu fallback",
)


def _record_key(record: dict[str, Any]) -> tuple[int, int, str]:
    return (record["level"], record["problem_id"], record["source_hash"])


def _parse_worker_payload(stdout: str) -> dict[str, Any] | None:
    for line in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "status" in candidate:
            return candidate
    return None


def _classify_failure(payload: dict[str, Any], diagnostic: str) -> str:
    if payload.get("failure_type") in {
        "timeout",
        "cpu_fallback",
        "numerical_error",
        "output_device_mismatch",
    }:
        return str(payload["failure_type"])
    message = diagnostic.lower()
    if "out of memory" in message or "memory allocation" in message:
        return "out_of_memory"
    if "dtype" in message and ("not support" in message or "unsupported" in message):
        return "unsupported_dtype"
    if "layout" in message and ("not support" in message or "unsupported" in message):
        return "unsupported_layout"
    if "notimplementederror" in message or "not supported on the npu" in message:
        return "unsupported_operator"
    if "no module named" in message or "importerror" in message:
        return "environment_error"
    return "runtime_error"


def _terminate_process_group(process: subprocess.Popen) -> None:
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


def _run_attempt(
    path: Path,
    level: int,
    device: int,
    attempt: int,
    timeout: float,
    warmup: int,
    repeats: int,
    logs_dir: Path,
) -> dict[str, Any]:
    record = _base_record(path, level)
    record.update(physical_device=device, logical_device=0, attempt=attempt)
    env = os.environ.copy()
    for key in _SECRET_ENV_KEYS:
        env.pop(key, None)
    # A worker sees exactly one physical NPU and consistently addresses it as npu:0.
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(device)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        str(path),
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
    ]
    started = time.monotonic()
    stdout = ""
    stderr = ""
    process = None
    try:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        stdout, stderr = process.communicate(timeout=timeout)
        payload = _parse_worker_payload(stdout)
        if payload is None:
            payload = {
                "status": "failed",
                "failure_type": "worker_no_output",
                "error": stderr[-4000:] or stdout[-4000:],
            }
        record.update(payload)
        diagnostic = "\n".join((stdout, stderr, str(record.get("error", ""))))
        fallback_detected = any(marker in diagnostic.lower() for marker in _CPU_FALLBACK_MARKERS)
        record["fallback_detected"] = fallback_detected
        if fallback_detected:
            record.update(status="failed", failure_type="cpu_fallback")
        elif process.returncode != 0 and record.get("status") == "supported":
            record.update(status="failed", failure_type="worker_error", error=stderr[-4000:])
        if record.get("status") != "supported":
            original_failure_type = record.get("failure_type")
            if original_failure_type:
                record["exception_type"] = original_failure_type
            record["failure_type"] = _classify_failure(record, diagnostic)
    except subprocess.TimeoutExpired:
        if process is not None:
            _terminate_process_group(process)
        record.update(status="failed", failure_type="timeout", fallback_detected=False)
    except Exception as exc:
        if process is not None and process.poll() is None:
            _terminate_process_group(process)
        record.update(
            status="failed",
            failure_type="audit_error",
            error=str(exc),
            fallback_detected=False,
        )
    record["worker_returncode"] = process.returncode if process is not None else None
    record["elapsed_wall_s"] = round(time.monotonic() - started, 3)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / (
        f"level{level}_p{record['problem_id']:03d}_attempt{attempt}_device{device}.log"
    )
    log_path.write_text(
        f"command: {' '.join(command)}\nphysical_device: {device}\ntimeout_s: {timeout}\n"
        f"\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
        encoding="utf-8",
    )
    record["log"] = str(log_path)
    return record


def _run_problem(
    path: Path,
    level: int,
    device: int,
    args: argparse.Namespace,
    logs_dir: Path,
) -> dict[str, Any]:
    attempts = []
    final = None
    for attempt in range(1, args.retries + 2):
        timeout = args.timeout if attempt == 1 else args.retry_timeout
        final = _run_attempt(
            path,
            level,
            device,
            attempt,
            timeout,
            args.warmup,
            args.repeats,
            logs_dir,
        )
        attempts.append(
            {
                key: final.get(key)
                for key in (
                    "attempt",
                    "status",
                    "failure_type",
                    "elapsed_wall_s",
                    "worker_returncode",
                    "log",
                )
            }
        )
        if final.get("status") == "supported":
            break
        if attempt <= args.retries:
            time.sleep(args.failure_cooldown)
    assert final is not None
    final["status"] = "supported" if final.get("status") == "supported" else "excluded"
    final["attempt_count"] = len(attempts)
    final["attempts"] = attempts
    return final


def _load_journal(path: Path) -> dict[tuple[int, int, str], dict[str, Any]]:
    records = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            records[_record_key(record)] = record
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return records


def _write_report(
    output: Path,
    metadata: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    records = sorted(records, key=lambda item: (item["level"], item["problem_id"]))
    report = {"metadata": metadata, "results": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)

    csv_path = output.with_suffix(".csv")
    csv_temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    columns = [
        "level",
        "problem_id",
        "problem",
        "status",
        "failure_type",
        "physical_device",
        "attempt_count",
        "cold_ms",
        "reference_p50_ms",
        "reference_mean_ms",
        "reference_p90_ms",
        "fallback_detected",
        "repeat_stable",
        "elapsed_wall_s",
        "source_hash",
        "log",
    ]
    with csv_temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            runtime = record.get("reference_runtime_ms") or {}
            writer.writerow(
                {
                    **{key: record.get(key) for key in columns},
                    "reference_p50_ms": runtime.get("p50"),
                    "reference_mean_ms": runtime.get("mean"),
                    "reference_p90_ms": runtime.get("p90"),
                }
            )
    csv_temporary.replace(csv_path)

    supported = [
        {"level": item["level"], "problem_id": item["problem_id"], "problem": item["problem"]}
        for item in records
        if item["status"] == "supported"
    ]
    excluded = [
        {
            "level": item["level"],
            "problem_id": item["problem_id"],
            "problem": item["problem"],
            "failure_type": item.get("failure_type"),
        }
        for item in records
        if item["status"] != "supported"
    ]
    output.with_name("supported.json").write_text(
        json.dumps(supported, indent=2, sort_keys=True), encoding="utf-8"
    )
    output.with_name("excluded.json").write_text(
        json.dumps(excluded, indent=2, sort_keys=True), encoding="utf-8"
    )


def audit(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.kernelbench_root).resolve()
    levels = args.levels or ([args.level] if args.level is not None else [1])
    tasks = []
    for level in levels:
        for path in _find_problems(root, level):
            problem_id = int(path.name.split("_", 1)[0])
            if not args.problem_ids or problem_id in set(args.problem_ids):
                tasks.append((level, path))

    output = Path(args.output).resolve()
    journal = Path(args.journal).resolve() if args.journal else output.with_suffix(".jsonl")
    logs_dir = Path(args.logs_dir).resolve() if args.logs_dir else output.parent / "logs"
    existing = _load_journal(journal) if args.resume else {}
    pending = []
    for level, path in tasks:
        base = _base_record(path, level)
        if _record_key(base) not in existing:
            pending.append((level, path))

    metadata = {
        "hardware": "Ascend910B3",
        "cann": "9.0.1",
        "torch": _version("torch"),
        "torch_npu": _version("torch_npu"),
        "levels": levels,
        "devices": args.devices,
        "timeout_s": args.timeout,
        "retry_timeout_s": args.retry_timeout,
        "retries": args.retries,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "total_discovered": len(tasks),
        "resumed_records": len(existing),
    }
    work_queue: queue.Queue[tuple[int, Path]] = queue.Queue()
    for task in pending:
        work_queue.put(task)
    records = dict(existing)
    persist_lock = threading.Lock()
    completed = len(existing)

    def persist(record: dict[str, Any]) -> None:
        nonlocal completed
        with persist_lock:
            journal.parent.mkdir(parents=True, exist_ok=True)
            with journal.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            records[_record_key(record)] = record
            completed += 1
            _write_report(output, metadata, list(records.values()))
            print(
                f"[{completed}/{len(tasks)}] device={record['physical_device']} "
                f"level={record['level']} problem={record['problem_id']} "
                f"status={record['status']} failure={record.get('failure_type')}",
                flush=True,
            )

    def device_loop(device: int) -> None:
        while True:
            try:
                level, path = work_queue.get_nowait()
            except queue.Empty:
                return
            try:
                persist(_run_problem(path, level, device, args, logs_dir))
            except Exception as exc:
                record = _base_record(path, level)
                record.update(
                    status="excluded",
                    failure_type="scheduler_error",
                    error=str(exc),
                    physical_device=device,
                    logical_device=0,
                    attempt_count=0,
                    attempts=[],
                    elapsed_wall_s=0.0,
                )
                persist(record)
            finally:
                work_queue.task_done()

    print(
        f"Starting Ascend audit: {len(pending)} pending / {len(tasks)} total, "
        f"devices={args.devices}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=len(args.devices)) as executor:
        futures = [executor.submit(device_loop, device) for device in args.devices]
        for future in futures:
            future.result()
    final_records = list(records.values())
    _write_report(output, metadata, final_records)
    return {"metadata": metadata, "results": final_records}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernelbench-root", default=".")
    parser.add_argument("--level", type=int)
    parser.add_argument("--levels", type=int, nargs="+")
    parser.add_argument("--output", default="ascend_level_audit.json")
    parser.add_argument("--journal")
    parser.add_argument("--logs-dir")
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--retry-timeout", type=float, default=300)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--failure-cooldown", type=float, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from the append-only JSONL journal (default: enabled).",
    )
    parser.add_argument("--problem-ids", type=int, nargs="*")
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        _run_worker(args.worker, args.warmup, args.repeats)
        return 0
    audit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
