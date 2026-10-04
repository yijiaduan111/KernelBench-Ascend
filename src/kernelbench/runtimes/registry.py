"""Runtime registry kept separate from code-generation backends."""

from __future__ import annotations

from .ascend import AscendRuntime
from .cuda import CudaRuntime


_RUNTIMES = {
    "cuda": CudaRuntime(),
    "rocm": CudaRuntime(),
    "ascend": AscendRuntime(),
    "npu": AscendRuntime(),
}


def get_runtime(name: str):
    normalized = str(name).strip().lower()
    try:
        return _RUNTIMES[normalized]
    except KeyError as exc:
        raise ValueError(f"Unknown runtime '{name}'.") from exc


def get_runtime_for_backend(backend: str):
    normalized = str(backend or "").strip().lower()
    if normalized == "ascendc":
        return _RUNTIMES["ascend"]
    return _RUNTIMES["cuda"]


def list_runtimes() -> tuple[str, ...]:
    return ("ascend", "cuda")
