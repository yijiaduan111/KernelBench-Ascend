"""Static checks for untrusted Ascend C candidate Python."""

from __future__ import annotations

import ast
import re

from .errors import StaticRejectionError
from .spec import AscendCSpec, extract_literal_spec, extract_literal_tiling

_ALLOWED_IMPORT_ROOTS = {"torch"}
_FORBIDDEN_CALLS = {
    "__import__",
    "compile",
    "dlopen",
    "eval",
    "exec",
    "execve",
    "fork",
    "getattr",
    "globals",
    "load",
    "loads",
    "locals",
    "manual_seed",
    "open",
    "popen",
    "read_bytes",
    "read_text",
    "save",
    "set_rng_state",
    "setattr",
    "spawn",
    "system",
    "urlopen",
    "write_bytes",
    "write_text",
}
_SECRET_ENV = re.compile(r"(?:OPENAI|ANTHROPIC|DASHSCOPE|PYRAMIDKERNEL).*KEY|API_KEY|API_BASE", re.I)


def validate_candidate_source(source: str) -> AscendCSpec:
    try:
        tree = ast.parse(source, filename="candidate.py")
    except SyntaxError as exc:
        raise StaticRejectionError(f"candidate syntax error: {exc}") from exc
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    if "ModelNew" not in names:
        raise StaticRejectionError("ModelNew class is required")
    spec = extract_literal_spec(source)
    assignments = {
        node.targets[0].id
        for node in tree.body
        if isinstance(node, ast.Assign)
        and node.targets
        and isinstance(node.targets[0], ast.Name)
    }
    if "ASCENDC_KERNEL_SOURCE" not in assignments:
        raise StaticRejectionError("ASCENDC_KERNEL_SOURCE assignment is required")
    if "ASCENDC_TILING" not in assignments:
        raise StaticRejectionError("ASCENDC_TILING assignment is required")
    if "kernel_operator.h" not in source:
        raise StaticRejectionError("Ascend C source must include kernel_operator.h")
    kernel_assign = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "ASCENDC_KERNEL_SOURCE" for target in node.targets)
    )
    try:
        kernel_source = str(ast.literal_eval(kernel_assign))
    except Exception as exc:
        raise StaticRejectionError("ASCENDC_KERNEL_SOURCE must be a literal string") from exc
    includes = [line.strip() for line in kernel_source.splitlines() if line.strip().startswith("#include")]
    if includes != ['#include "kernel_operator.h"']:
        raise StaticRejectionError("only #include \\\"kernel_operator.h\\\" is allowed in Ascend C source")
    if re.search(r"\b(?:system|popen|fopen|socket|connect|fork|execv|execve|dlopen|getenv|syscall)\s*\(", kernel_source):
        raise StaticRejectionError("forbidden host/process operation in Ascend C source")
    entrypoint_pattern = re.compile(
        rf'extern\s+"C"\s+__global__\s+__aicore__\s+void\s+{re.escape(spec.entrypoint)}\s*\('
    )
    if entrypoint_pattern.search(kernel_source) is None:
        raise StaticRejectionError(
            f"Ascend C source does not define entrypoint '{spec.entrypoint}'"
        )
    if "GET_TILING_DATA(" not in kernel_source:
        raise StaticRejectionError("Ascend C entrypoint must load generated tiling data")
    # Validate the complete structured tiling contract before any files are
    # generated or any candidate Python is executed.
    extract_literal_tiling(source, spec)
    # The device source may define the entrypoint, but the Python candidate
    # must also invoke the generated binding.  Remove the raw C string before
    # checking so a C declaration cannot satisfy this requirement.
    python_surface = re.sub(
        r"ASCENDC_KERNEL_SOURCE\s*=\s*(?:r|u|f|b|fr|rf)?('''.*?'''|\"\"\".*?\"\"\"|'[^']*'|\"[^\"]*\")",
        "",
        source,
        flags=re.S,
    )
    binding_call = rf"\bascendc_binding\s*\.\s*{re.escape(spec.entrypoint)}\s*\("
    if not re.search(binding_call, python_surface):
        raise StaticRejectionError(
            f"candidate must call ascendc_binding.{spec.entrypoint}(...)"
        )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad = {
                alias.name for alias in node.names
                if alias.name.split(".")[0] not in _ALLOWED_IMPORT_ROOTS
            }
            if bad:
                raise StaticRejectionError(f"forbidden import(s): {', '.join(sorted(bad))}")
        elif (
            isinstance(node, ast.ImportFrom)
            and (
                not node.module
                or node.module.split(".")[0] not in _ALLOWED_IMPORT_ROOTS
            )
        ):
            raise StaticRejectionError(f"forbidden import: {node.module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _FORBIDDEN_CALLS:
            raise StaticRejectionError(f"forbidden call: {node.func.attr}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
            raise StaticRejectionError(f"forbidden call: {node.func.id}")
        elif isinstance(node, ast.Try):
            raise StaticRejectionError("try/except fallback is forbidden for Ascend C candidates")
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "environ":
            key = node.slice.value if isinstance(node.slice, ast.Constant) else ""
            if isinstance(key, str) and _SECRET_ENV.search(key):
                raise StaticRejectionError("candidate cannot read API credentials from environment")
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                root = target
                while isinstance(root, (ast.Attribute, ast.Subscript)):
                    root = root.value
                if isinstance(root, ast.Name) and root.id in {"torch", "ascendc_binding"}:
                    raise StaticRejectionError(
                        f"candidate cannot modify global module {root.id}"
                    )
    if re.search(r"(?:torch\.npu\.(?:Event|synchronize)|time\.perf_counter)\s*=", source):
        raise StaticRejectionError("candidate cannot monkey-patch timing")
    return spec
