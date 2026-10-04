"""Isolated NPU worker for an already-built Ascend C artifact."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import traceback
from pathlib import Path
from typing import Any


def _load_reference(source: str):
    namespace: dict[str, Any] = {}
    exec(compile(source, "<kernelbench-reference>", "exec"), namespace)
    return namespace["Model"], namespace["get_init_inputs"], namespace["get_inputs"]


def _move(value: Any, device, precision):
    import torch

    if isinstance(value, torch.Tensor):
        dtype = precision if value.is_floating_point() else value.dtype
        return value.to(device=device, dtype=dtype)
    if isinstance(value, tuple):
        return tuple(_move(item, device, precision) for item in value)
    if isinstance(value, list):
        return [_move(item, device, precision) for item in value]
    if isinstance(value, dict):
        return {key: _move(item, device, precision) for key, item in value.items()}
    return value


def _clone(value: Any):
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


def _compare(reference: Any, candidate: Any, path: str = "output") -> dict[str, Any]:
    import torch

    if isinstance(reference, torch.Tensor) and isinstance(candidate, torch.Tensor):
        result = {
            "ok": False,
            "path": path,
            "max_abs_error": None,
            "mean_abs_error": None,
            "nan_count": 0,
            "inf_count": 0,
        }
        if reference.shape != candidate.shape:
            result["reason"] = f"shape mismatch: {tuple(reference.shape)} != {tuple(candidate.shape)}"
            return result
        if reference.dtype != candidate.dtype:
            result["reason"] = f"dtype mismatch: {reference.dtype} != {candidate.dtype}"
            return result
        if reference.device != candidate.device:
            result["reason"] = f"device mismatch: {reference.device} != {candidate.device}"
            return result
        if not reference.is_floating_point():
            result["ok"] = bool(torch.equal(reference, candidate))
            result["reason"] = None if result["ok"] else "integer/boolean values differ"
            return result
        tolerance = 1e-4 if reference.dtype == torch.float32 else 1e-2
        reference_flat = reference.reshape(-1)
        candidate_flat = candidate.reshape(-1)
        # Full-sized temporary tensors can add several GiB for official L1
        # shapes. Chunking keeps correctness memory bounded independently of
        # the problem size.
        chunk_elements = 16 * 1024 * 1024
        max_error = 0.0
        error_sum = 0.0
        nan_count = 0
        inf_count = 0
        close = True
        for start in range(0, reference_flat.numel(), chunk_elements):
            end = min(start + chunk_elements, reference_flat.numel())
            reference_chunk = reference_flat[start:end]
            candidate_chunk = candidate_flat[start:end]
            chunk_nan = int(torch.isnan(candidate_chunk).sum().item())
            chunk_inf = int(torch.isinf(candidate_chunk).sum().item())
            nan_count += chunk_nan
            inf_count += chunk_inf
            if chunk_nan or chunk_inf:
                close = False
                continue
            difference = (reference_chunk.float() - candidate_chunk.float()).abs()
            if difference.numel():
                max_error = max(max_error, float(difference.max().item()))
                error_sum += float(difference.sum().item())
            if close and not torch.allclose(
                reference_chunk,
                candidate_chunk,
                atol=tolerance,
                rtol=tolerance,
            ):
                close = False
        result["nan_count"] = nan_count
        result["inf_count"] = inf_count
        result["max_abs_error"] = max_error
        result["mean_abs_error"] = (
            error_sum / reference_flat.numel() if reference_flat.numel() else 0.0
        )
        result["ok"] = bool(close and not nan_count and not inf_count)
        result["reason"] = (
            None
            if result["ok"]
            else "candidate contains NaN/Inf"
            if nan_count or inf_count
            else "floating values differ"
        )
        return result
    if isinstance(reference, (list, tuple)) and isinstance(candidate, type(reference)):
        if len(reference) != len(candidate):
            return {"ok": False, "path": path, "reason": "sequence length mismatch"}
        children = [
            _compare(left, right, f"{path}[{index}]")
            for index, (left, right) in enumerate(zip(reference, candidate))
        ]
        return {"ok": all(child["ok"] for child in children), "path": path, "children": children}
    if isinstance(reference, dict) and isinstance(candidate, dict):
        if reference.keys() != candidate.keys():
            return {"ok": False, "path": path, "reason": "mapping keys differ"}
        children = [
            _compare(reference[key], candidate[key], f"{path}.{key}") for key in reference
        ]
        return {"ok": all(child["ok"] for child in children), "path": path, "children": children}
    if isinstance(reference, (str, int, float, bool, type(None))) and type(reference) is type(candidate):
        return {
            "ok": reference == candidate,
            "path": path,
            "reason": None if reference == candidate else "scalar values differ",
        }
    return {
        "ok": False,
        "path": path,
        "reason": f"output structure mismatch: {type(reference).__name__} != {type(candidate).__name__}",
    }


def _capture_rng_state():
    import torch

    cpu_state = torch.random.get_rng_state().clone()
    try:
        npu_state = torch.npu.get_rng_state().clone()
    except Exception:
        npu_state = None
    return cpu_state, npu_state


def _exact_equal(reference: Any, candidate: Any, path: str) -> dict[str, Any]:
    """Compare inputs without allocating a full-size floating-point difference."""
    import torch

    if isinstance(reference, torch.Tensor) and isinstance(candidate, torch.Tensor):
        ok = bool(
            reference.shape == candidate.shape
            and reference.dtype == candidate.dtype
            and reference.device == candidate.device
            and torch.equal(reference, candidate)
        )
        return {"ok": ok, "path": path, "reason": None if ok else "tensor was modified"}
    if isinstance(reference, (list, tuple)) and isinstance(candidate, type(reference)):
        if len(reference) != len(candidate):
            return {"ok": False, "path": path, "reason": "sequence length mismatch"}
        children = [
            _exact_equal(left, right, f"{path}[{index}]")
            for index, (left, right) in enumerate(zip(reference, candidate))
        ]
        return {"ok": all(child["ok"] for child in children), "path": path, "children": children}
    if isinstance(reference, dict) and isinstance(candidate, dict):
        if reference.keys() != candidate.keys():
            return {"ok": False, "path": path, "reason": "mapping keys differ"}
        children = [
            _exact_equal(reference[key], candidate[key], f"{path}.{key}")
            for key in reference
        ]
        return {"ok": all(child["ok"] for child in children), "path": path, "children": children}
    ok = type(reference) is type(candidate) and reference == candidate
    return {"ok": bool(ok), "path": path, "reason": None if ok else "value was modified"}


def _rng_state_unchanged(before) -> bool:
    import torch

    cpu_before, npu_before = before
    if not torch.equal(cpu_before, torch.random.get_rng_state()):
        return False
    if npu_before is None:
        return True
    try:
        return bool(torch.equal(npu_before, torch.npu.get_rng_state()))
    except Exception:
        return False


def _event_call(model, inputs, device) -> tuple[Any, float, float]:
    import torch

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    torch.npu.synchronize(device)
    host_start = time.perf_counter()
    start.record()
    output = model(*inputs)
    end.record()
    torch.npu.synchronize(device)
    host_ms = (time.perf_counter() - host_start) * 1000.0
    return output, float(start.elapsed_time(end)), host_ms


def _stats(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p90_index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * 0.9) - 1))
    return {
        "mean_ms": float(statistics.fmean(samples)),
        "p50_ms": float(statistics.median(samples)),
        "p90_ms": float(ordered[p90_index]),
        "min_ms": float(ordered[0]),
        "max_ms": float(ordered[-1]),
    }


def evaluate(job: dict[str, Any]) -> dict[str, Any]:
    import torch
    import torch_npu  # noqa: F401 - registers the NPU backend

    from .loader import load_binding, load_candidate_source

    device = torch.device(f"npu:{int(job.get('device_index', 0))}")
    torch.npu.set_device(device)
    precision = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[job.get("precision", "fp32")]
    reference_source = Path(job["reference_path"]).read_text(encoding="utf-8")
    candidate_source = Path(job["candidate_path"]).read_text(encoding="utf-8")
    Model, get_init_inputs, get_inputs = _load_reference(reference_source)
    binding = load_binding(job["binding_path"], job["binding_module"])
    candidate_module = load_candidate_source(candidate_source, binding)

    seed = int(job.get("seed", 42))
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    init_inputs = _move(get_init_inputs(), device, precision)
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    reference_model = Model(*init_inputs).to(device=device, dtype=precision)
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    candidate_model = candidate_module.ModelNew(*init_inputs).to(device=device, dtype=precision)
    reference_model.eval()
    candidate_model.eval()

    trials: list[dict[str, Any]] = []
    binding.reset_call_count()
    first_call_ms = None
    first_host_ms = None
    with torch.no_grad():
        for trial in range(max(1, int(job.get("num_correct_trials", 3)))):
            trial_seed = seed + trial * 1009
            torch.manual_seed(trial_seed)
            torch.npu.manual_seed(trial_seed)
            inputs = _move(get_inputs(), device, precision)
            reference_inputs = _clone(inputs)
            candidate_inputs = _clone(inputs)
            reference_output = reference_model(*reference_inputs)
            torch.npu.synchronize(device)
            rng_before = _capture_rng_state()
            if trial == 0:
                candidate_output, first_call_ms, first_host_ms = _event_call(
                    candidate_model, candidate_inputs, device
                )
            else:
                candidate_output = candidate_model(*candidate_inputs)
                torch.npu.synchronize(device)
            comparison = _compare(reference_output, candidate_output)
            input_check = _exact_equal(inputs, candidate_inputs, "inputs")
            comparison["input_unchanged"] = bool(input_check["ok"])
            comparison["input_check"] = input_check
            comparison["rng_unchanged"] = _rng_state_unchanged(rng_before)
            comparison["ok"] = bool(
                comparison["ok"]
                and comparison["input_unchanged"]
                and comparison["rng_unchanged"]
            )
            comparison["trial"] = trial
            comparison["phase"] = "cold"
            trials.append(comparison)
            del inputs, reference_inputs, candidate_inputs, reference_output, candidate_output

        custom_calls_after_correctness = int(binding.get_call_count())
        cold_correct = all(item["ok"] for item in trials) and custom_calls_after_correctness > 0

        warmup_ms: list[float] = []
        timing_inputs = _move(get_inputs(), device, precision)
        timing_inputs_before = _clone(timing_inputs)
        if cold_correct:
            for _ in range(max(0, int(job.get("num_warmup", 3)))):
                warmup_output, elapsed, _ = _event_call(candidate_model, timing_inputs, device)
                warmup_ms.append(elapsed)
                del warmup_output

            warmup_mutation = _exact_equal(timing_inputs_before, timing_inputs, "warmup_inputs")
            candidate_inputs = _clone(timing_inputs_before)
            reference_output = reference_model(*timing_inputs_before)
            torch.npu.synchronize(device)
            rng_before = _capture_rng_state()
            candidate_output = candidate_model(*candidate_inputs)
            torch.npu.synchronize(device)
            post_warmup = _compare(reference_output, candidate_output)
            post_input_check = _exact_equal(timing_inputs_before, candidate_inputs, "inputs")
            post_warmup.update(
                {
                    "trial": len(trials),
                    "phase": "post_warmup",
                    "input_unchanged": bool(
                        warmup_mutation["ok"] and post_input_check["ok"]
                    ),
                    "input_check": post_input_check,
                    "warmup_input_check": warmup_mutation,
                    "rng_unchanged": _rng_state_unchanged(rng_before),
                }
            )
            post_warmup["ok"] = bool(
                post_warmup["ok"]
                and post_warmup["input_unchanged"]
                and post_warmup["rng_unchanged"]
            )
            trials.append(post_warmup)
            del candidate_inputs, reference_output, candidate_output, timing_inputs_before
        else:
            post_warmup = {"ok": False, "phase": "post_warmup", "skipped": True}

        correct = bool(cold_correct and post_warmup["ok"])
        cold_warm_divergence = bool(cold_correct and not post_warmup["ok"])
        if not correct:
            return {
                "compiled": True,
                "correctness": False,
                "eligible_for_performance": False,
                "failure_type": "correctness_error",
                "custom_op_call_count": int(binding.get_call_count()),
                "correctness_trials": trials,
                "cold_warm_divergence": cold_warm_divergence,
                "first_call_ms": first_call_ms,
                "first_call_host_ms": first_host_ms,
                "warmup_ms": warmup_ms,
                "candidate": {},
                "reference": {},
                "speedup": None,
            }

        candidate_samples: list[float] = []
        candidate_host: list[float] = []
        reference_samples: list[float] = []
        for _ in range(max(1, int(job.get("num_perf_trials", 10)))):
            timing_output, elapsed, host = _event_call(candidate_model, timing_inputs, device)
            candidate_samples.append(elapsed)
            candidate_host.append(host)
            del timing_output
        for _ in range(max(1, int(job.get("num_perf_trials", 10)))):
            timing_output, elapsed, _ = _event_call(reference_model, timing_inputs, device)
            reference_samples.append(elapsed)
            del timing_output

    candidate_stats = _stats(candidate_samples)
    reference_stats = _stats(reference_samples)
    candidate_stats["host_mean_ms"] = float(statistics.fmean(candidate_host))
    speedup = reference_stats["p50_ms"] / candidate_stats["p50_ms"]
    return {
        "compiled": True,
        "correctness": correct,
        "eligible_for_performance": correct,
        "failure_type": None if correct else "correctness_error",
        "custom_op_call_count": int(binding.get_call_count()),
        "correctness_trials": trials,
        "first_call_ms": first_call_ms,
        "first_call_host_ms": first_host_ms,
        "warmup_ms": warmup_ms,
        "cold_warm_divergence": cold_warm_divergence,
        "candidate": candidate_stats,
        "reference": reference_stats,
        "speedup": speedup,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("job")
    parser.add_argument("result")
    args = parser.parse_args()
    result_path = Path(args.result)
    try:
        job = json.loads(Path(args.job).read_text(encoding="utf-8"))
        report = evaluate(job)
    except Exception as exc:
        failure_type = (
            "out_of_memory"
            if "out of memory" in str(exc).lower()
            else "npu_runtime_error"
        )
        report = {
            "compiled": True,
            "correctness": False,
            "eligible_for_performance": False,
            "failure_type": failure_type,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    result_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
