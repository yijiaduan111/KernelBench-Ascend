"""Immutable metadata and isolated paths for an Ascend C candidate."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any


def source_hash(source: str, spec_json: str, cache_context: dict[str, Any] | None = None) -> str:
    payload = {
        "source": source,
        "spec": json.loads(spec_json),
        "context": cache_context or {},
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def environment_fingerprint(target_soc: str = "Ascend910B3") -> dict[str, str]:
    def command(*args: str) -> str:
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"
    return {
        "soc": target_soc,
        "cann_version": os.environ.get("ASCEND_TOOLKIT_VERSION", "9.0.1"),
        "torch": _package_version("torch"),
        "torch_npu": _package_version("torch-npu"),
        "python": platform.python_version(),
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "unknown"


def artifact_metadata(
    source: str,
    spec_json: str,
    target_soc: str,
    root: Path,
    *,
    template_version: str = "unknown",
    compiler_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    environment = environment_fingerprint(target_soc)
    cache_context = {
        "environment": environment,
        "template_version": template_version,
        "compiler_options": compiler_options or {},
    }
    digest = source_hash(source, spec_json, cache_context)
    return {
        "candidate_hash": digest,
        "source_hash": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "spec_hash": hashlib.sha256(spec_json.encode("utf-8")).hexdigest(),
        "environment": environment,
        "template_version": template_version,
        "compiler_options": compiler_options or {},
        "cache_root": str(root),
        "private_opp": True,
        "system_opp_modified": False,
    }


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
