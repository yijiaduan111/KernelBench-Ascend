"""Validated loading boundary for Ascend C artifacts."""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

from .errors import BindingLoadError


def load_python_candidate(path: str | Path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location("kernelbench_ascendc_candidate", path)
    if spec is None or spec.loader is None:
        raise BindingLoadError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise BindingLoadError(f"candidate import failed: {exc}") from exc
    if not hasattr(module, "ModelNew"):
        raise BindingLoadError("loaded candidate has no ModelNew")
    return module


def load_binding(path: str | Path, module_name: str):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise BindingLoadError(f"cannot create binding import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise BindingLoadError(f"binding import failed: {exc}") from exc
    for name in ("get_call_count", "reset_call_count"):
        if not hasattr(module, name):
            raise BindingLoadError(f"binding is missing required method: {name}")
    return module


def load_candidate_source(source: str, binding):
    """Execute validated candidate Python with only the generated binding injected."""
    module = types.ModuleType("kernelbench_ascendc_candidate")
    module.__dict__["ascendc_binding"] = binding
    try:
        code = compile(source, "<ascendc-candidate>", "exec")
        exec(code, module.__dict__)
    except Exception as exc:
        raise BindingLoadError(f"candidate import failed: {exc}") from exc
    if not hasattr(module, "ModelNew"):
        raise BindingLoadError("loaded candidate has no ModelNew")
    return module
