"""Reference-derived BoundaryContract for whole-Model Ascend C candidates."""

from __future__ import annotations

import hashlib
import json
import pprint
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

CONTRACT_VERSION = "boundary-contract-v1"
_IDENTIFIER = re.compile(r"[^A-Za-z0-9_]+")


@dataclass(frozen=True)
class BoundaryTensor:
    name: str
    role: Literal["user_input", "parameter", "buffer", "constant_tensor"]
    dtype: str
    rank: int
    shape: tuple[int, ...]
    source_path: str | None = None
    layout: str = "ND"


@dataclass(frozen=True)
class BoundaryAttribute:
    name: str
    c_type: str
    source: Literal["forward_arg", "init_arg", "module_attribute", "constant"]
    default: int | float | bool | None = None


@dataclass(frozen=True)
class BoundaryOutput:
    index: int
    dtype: str
    rank: int
    shape: tuple[int, ...]
    path: tuple[str | int, ...] = ()


@dataclass(frozen=True)
class BoundaryContract:
    version: str
    level: int
    problem_id: int
    source_hash: str
    capture_mode: Literal["eval"]
    specialization: Literal["official_input"]
    inputs: tuple[BoundaryTensor, ...] = ()
    attributes: tuple[BoundaryAttribute, ...] = ()
    outputs: tuple[BoundaryOutput, ...] = ()
    output_tree: dict[str, Any] = field(default_factory=dict)
    input_guards: tuple[dict[str, Any], ...] = ()
    state_mutations: tuple[dict[str, Any], ...] = ()
    status: Literal["supported", "unsupported"] = "supported"
    failure_type: str | None = None
    failure_message: str | None = None

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_mapping(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def contract_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "BoundaryContract":
        return cls(
            version=str(value.get("version", CONTRACT_VERSION)),
            level=int(value["level"]),
            problem_id=int(value["problem_id"]),
            source_hash=str(value["source_hash"]),
            capture_mode=str(value.get("capture_mode", "eval")),
            specialization=str(value.get("specialization", "official_input")),
            inputs=tuple(
                BoundaryTensor(
                    name=str(item["name"]), role=str(item["role"]), dtype=str(item["dtype"]),
                    rank=int(item["rank"]), shape=tuple(int(dim) for dim in item.get("shape", [])),
                    source_path=item.get("source_path"), layout=str(item.get("layout", "ND")),
                ) for item in value.get("inputs", [])
            ),
            attributes=tuple(
                BoundaryAttribute(
                    name=str(item["name"]), c_type=str(item["c_type"]),
                    source=str(item["source"]), default=item.get("default"),
                ) for item in value.get("attributes", [])
            ),
            outputs=tuple(
                BoundaryOutput(
                    index=int(item["index"]), dtype=str(item["dtype"]), rank=int(item["rank"]),
                    shape=tuple(int(dim) for dim in item.get("shape", [])), path=tuple(item.get("path", [])),
                ) for item in value.get("outputs", [])
            ),
            output_tree=dict(value.get("output_tree", {})),
            input_guards=tuple(dict(item) for item in value.get("input_guards", [])),
            state_mutations=tuple(dict(item) for item in value.get("state_mutations", [])),
            status=str(value.get("status", "supported")),
            failure_type=value.get("failure_type"), failure_message=value.get("failure_message"),
        )


def _dtype_name(tensor: Any) -> str:
    return str(getattr(tensor, "dtype", "")).removeprefix("torch.") or "unknown"


def _c_type(value: Any) -> str:
    if isinstance(value, bool): return "bool"
    if isinstance(value, int): return "int64_t"
    if isinstance(value, float): return "float"
    raise ValueError(f"unsupported scalar attribute type: {type(value).__name__}")


def _sanitize(value: str, fallback: str) -> str:
    name = _IDENTIFIER.sub("_", str(value)).strip("_")
    return f"{fallback}_{name}" if not name or name[0].isdigit() else name


def _unique_name(value: str, used: set[str], fallback: str) -> str:
    base = _sanitize(value, fallback)
    name, index = base, 1
    while name in used:
        index += 1
        name = f"{base}_{index}"
    used.add(name)
    return name


def _python_path_expr(root: str, path: str) -> str:
    expression = root
    for piece in str(path).split("."):
        expression += f"[{piece}]" if piece.isdigit() else f".{piece}"
    return expression


def _lookup_state(model: Any, path: str) -> Any:
    current = model
    for piece in str(path).split("."):
        if hasattr(current, piece): current = getattr(current, piece)
        elif isinstance(current, (list, tuple)) and piece.isdigit(): current = current[int(piece)]
        else: raise KeyError(path)
    return current


def _flatten(value: Any, path: tuple[str | int, ...] = ()) -> list[tuple[tuple[str | int, ...], Any]]:
    if isinstance(value, (tuple, list)):
        return [item for index, child in enumerate(value) for item in _flatten(child, path + (index,))]
    if isinstance(value, dict):
        return [item for key, child in value.items() for item in _flatten(child, path + (str(key),))]
    return [(path, value)]


def _tree_mapping(value: Any) -> dict[str, Any]:
    if hasattr(value, "shape") and hasattr(value, "dtype"): return {"kind": "tensor"}
    if isinstance(value, (tuple, list)):
        return {"kind": "tuple" if isinstance(value, tuple) else "list", "children": [_tree_mapping(child) for child in value]}
    if isinstance(value, dict):
        return {"kind": "dict", "keys": [str(key) for key in value], "children": [_tree_mapping(child) for child in value.values()]}
    return {"kind": "scalar"}


def _meta_tensor(value: Any, torch: Any) -> Any:
    if isinstance(value, torch.Tensor): return torch.empty(tuple(value.shape), dtype=value.dtype, device="meta")
    return value


def _unsupported(level: int, problem_id: int, source_hash: str, failure_type: str, message: str) -> BoundaryContract:
    return BoundaryContract(
        version=CONTRACT_VERSION, level=level, problem_id=problem_id, source_hash=source_hash,
        capture_mode="eval", specialization="official_input", status="unsupported",
        failure_type=failure_type, failure_message=message[:2000],
    )


def capture_boundary_contract(reference_source: str, level: int, problem_id: int, *, source_path: str | None = None) -> BoundaryContract:
    """Capture one official eval-mode boundary without allocating real workload tensors."""
    source_hash = hashlib.sha256(reference_source.encode("utf-8")).hexdigest()
    try:
        import torch
        from torch.utils._pytree import tree_flatten
        namespace: dict[str, Any] = {}
        with torch.device("meta"):
            exec(compile(reference_source, source_path or "reference.py", "exec"), namespace)
            model = namespace["Model"](*namespace["get_init_inputs"]())
            model.eval()
            raw_inputs = namespace["get_inputs"]()
        inputs = [_meta_tensor(value, torch) for value in raw_inputs]
        try: model = model.to(device="meta")
        except Exception: pass
        exported = torch.export.export(model, tuple(inputs), strict=False)
        input_specs = list(exported.graph_signature.input_specs)
        state_dict = dict(getattr(exported, "state_dict", {}))
        used_names: set[str] = set(); contract_inputs: list[BoundaryTensor] = []; attributes: list[BoundaryAttribute] = []
        user_index = 0
        for spec in input_specs:
            kind = getattr(getattr(spec, "kind", None), "name", str(getattr(spec, "kind", "")))
            arg_name = str(getattr(getattr(spec, "arg", None), "name", "")); target_name = str(getattr(spec, "target", "") or "")
            if kind == "USER_INPUT":
                value = raw_inputs[user_index]; user_index += 1
                if isinstance(value, torch.Tensor):
                    name = _unique_name(arg_name or f"input_{user_index - 1}", used_names, f"input_{user_index - 1}")
                    contract_inputs.append(BoundaryTensor(name, "user_input", _dtype_name(value), value.dim(), tuple(int(x) for x in value.shape)))
                else:
                    attributes.append(BoundaryAttribute(_unique_name(arg_name or f"arg_{user_index - 1}", used_names, f"arg_{user_index - 1}"), _c_type(value), "forward_arg", value))
                continue
            if kind not in {"PARAMETER", "BUFFER", "CONSTANT_TENSOR"}: continue
            value = state_dict.get(target_name)
            if value is None:
                try: value = _lookup_state(model, target_name)
                except KeyError: value = None
            if value is None or not isinstance(value, torch.Tensor):
                return _unsupported(level, problem_id, source_hash, "boundary_tensor_unresolved", f"cannot resolve {kind} {target_name!r}")
            role = {"PARAMETER": "parameter", "BUFFER": "buffer", "CONSTANT_TENSOR": "constant_tensor"}[kind]
            name = _unique_name(target_name, used_names, f"input_{len(contract_inputs)}")
            contract_inputs.append(BoundaryTensor(name, role, _dtype_name(value), value.dim(), tuple(int(x) for x in value.shape), target_name))
        init_args = namespace["get_init_inputs"]()
        module_scalars = [
            (name, value)
            for name, value in vars(model).items()
            if isinstance(value, (bool, int, float)) and not name.startswith("_") and name != "training"
        ]
        existing_attrs = {item.name for item in attributes}
        for name, value in module_scalars:
            if name not in existing_attrs:
                attributes.append(BoundaryAttribute(name, _c_type(value), "module_attribute", value))
                existing_attrs.add(name)
        for index, value in enumerate(item for item in init_args if isinstance(item, (bool, int, float))):
            match = next(((name, current) for name, current in module_scalars if current == value), None)
            attr_name = match[0] if match else f"init_arg_{index}"
            if attr_name not in existing_attrs:
                attributes.append(BoundaryAttribute(attr_name, _c_type(value), "module_attribute" if match else "init_arg", value)); existing_attrs.add(attr_name)
        output = exported.module()(*inputs)
        leaves, _ = tree_flatten(output)
        outputs: list[BoundaryOutput] = []
        for index, (path, leaf) in enumerate(_flatten(output)):
            if not isinstance(leaf, torch.Tensor): return _unsupported(level, problem_id, source_hash, "boundary_output_structure_unsupported", "non-Tensor output leaf")
            outputs.append(BoundaryOutput(index, _dtype_name(leaf), leaf.dim(), tuple(int(x) for x in leaf.shape), path))
        guards = tuple({"input_index": index, "shape": list(item.shape), "dtype": item.dtype, "layout": item.layout} for index, item in enumerate(contract_inputs) if item.role == "user_input")
        return BoundaryContract(version=CONTRACT_VERSION, level=level, problem_id=problem_id, source_hash=source_hash, capture_mode="eval", specialization="official_input", inputs=tuple(contract_inputs), attributes=tuple(attributes), outputs=tuple(outputs), output_tree=_tree_mapping(output), input_guards=guards)
    except Exception as exc:
        return _unsupported(level, problem_id, source_hash, "boundary_export_error", f"{exc.__class__.__name__}: {exc}")


def load_boundary_contract(source: str | dict[str, Any]) -> BoundaryContract:
    return BoundaryContract.from_mapping(json.loads(source) if isinstance(source, str) else source)


def lower_boundary_contract(contract: BoundaryContract, implementation_config: Any = None) -> Any:
    """Lower semantic boundary facts to the existing canonical AscendCSpec."""
    from .spec import AscendCSpec

    if contract.status != "supported":
        raise ValueError(f"cannot lower unsupported boundary: {contract.failure_type}: {contract.failure_message}")
    inputs = [
        {"name": item.name, "dtype": item.dtype, "layout": item.layout, "shape": [f"literal({dim})" for dim in item.shape]}
        for item in contract.inputs
    ]
    outputs = [
        {"name": f"output_{item.index}", "dtype": item.dtype, "layout": "ND", "shape": [f"literal({dim})" for dim in item.shape]}
        for item in contract.outputs
    ]
    attributes = [
        {"name": item.name, "c_type": item.c_type, "default": item.default}
        for item in contract.attributes
    ]
    tiling: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(contract.inputs):
        tiling[f"input{index}_numel"] = {"c_type": "uint64_t", "expression": f"input[{index}].numel()"}
        for dim in range(item.rank):
            tiling[f"input{index}_dim{dim}"] = {"c_type": "uint64_t", "expression": f"input[{index}].dim({dim})"}
    for item in contract.attributes:
        tiling[item.name] = {"c_type": item.c_type, "expression": f'attr("{item.name}")'}
    return AscendCSpec.from_mapping({
        "op_name": f"l{contract.level}_p{contract.problem_id}_boundary",
        "entrypoint": f"p{contract.problem_id}_boundary_entry",
        "inputs": inputs,
        "outputs": outputs,
        "attributes": attributes,
        "tiling": tiling,
        "block_dim": "literal(1)",
        "workspace": "literal(0)",
    })


def validate_boundary_spec(spec: Any, contract: BoundaryContract) -> None:
    if contract.status != "supported":
        raise ValueError(f"unsupported boundary contract: {contract.failure_type}: {contract.failure_message}")
    expected_inputs = [item.name for item in contract.inputs]
    actual_inputs = [str(item.get("name")) for item in spec.inputs]
    if actual_inputs != expected_inputs:
        raise ValueError(f"boundary input ABI mismatch: expected {expected_inputs}, got {actual_inputs}")
    if len(spec.outputs) != len(contract.outputs):
        raise ValueError(f"boundary output count mismatch: expected {len(contract.outputs)}, got {len(spec.outputs)}")
    for index, (actual, expected) in enumerate(zip(spec.outputs, contract.outputs)):
        if str(actual.get("dtype")) != expected.dtype or len(actual.get("shape", [])) != expected.rank:
            raise ValueError(f"boundary output {index} dtype/rank mismatch")
    actual_attrs = [(str(item.get("name")), str(item.get("c_type", item.get("type", "")))) for item in spec.attributes]
    expected_attrs = [(item.name, item.c_type) for item in contract.attributes]
    if actual_attrs != expected_attrs:
        raise ValueError(f"boundary attribute ABI mismatch: expected {expected_attrs}, got {actual_attrs}")


def render_boundary_scaffold(info: Any, contract: BoundaryContract, spec: Any) -> str:
    """Render a frozen canonical Python surface with only device kernel editable."""
    imports = str(info.imports_block).strip()
    lines: list[str] = [imports] if imports else []
    existing = imports.splitlines()
    if "import torch" not in existing: lines.append("import torch")
    if not any(line.strip().startswith("import torch.nn") or line.strip().startswith("from torch import nn") for line in existing): lines.append("import torch.nn as nn")
    lines.extend([
        "", "BOUNDARY_CONTRACT = " + pprint.pformat(contract.to_mapping(), sort_dicts=False, width=100),
        "ASCENDC_OP_SPEC = " + pprint.pformat(spec.to_dict(), sort_dicts=False, width=100),
        "ASCENDC_TILING = {" + repr("fields") + ": [" + ", ".join(
            repr({"name": name, "c_type": definition.get("c_type", "uint64_t"), "expression": definition.get("expression", definition.get("default", "literal(0)"))})
            for name, definition in spec.tiling.items()
        ) + "], 'block_dim': 'literal(1)', 'workspace': 'literal(0)'}",
        "",
        "# <<<IMPROVE:ascendc_kernel>>>",
        "ASCENDC_KERNEL_SOURCE = r'''",
        '#include "kernel_operator.h"',
    ])
    tensor_args = [f"GM_ADDR input_{index}" for index, _ in enumerate(contract.inputs)]
    tensor_args += [f"GM_ADDR output_{index}" for index, _ in enumerate(contract.outputs)]
    tensor_args += ["GM_ADDR workspace", "GM_ADDR tiling"]
    lines += [
        "// Implement the device kernel. Host tiling, OPP registration and binding are framework-owned.",
        "extern \"C\" __global__ __aicore__ void " + spec.entrypoint + "(",
        "    " + ", ".join(tensor_args),
        ") {",
        "    GET_TILING_DATA(tiling_data, tiling);",
        "    (void)tiling_data;",
        "}",
        "'''",
        "# <<<END_IMPROVE>>>",
        "",
        "class ModelNew(nn.Module):",
    ]
    if info.class_preamble:
        lines.extend([f"    {line}" if line else "" for line in info.class_preamble.splitlines()])
    lines += [f"    def {info.init_signature}:", "        super().__init__()"]
    if info.init_body:
        lines.extend([f"        {line}" if line else "" for line in info.init_body.splitlines()])
    lines += ["", f"    def {info.forward_signature}:"]
    arg_names = []
    try:
        import ast
        node = ast.parse(f"def forward{info.forward_signature}: pass").body[0]
        arg_names = [arg.arg for arg in node.args.args if arg.arg != "self"]
    except Exception:
        arg_names = []
    call_args: list[str] = []
    user_pos = 0
    for item in contract.inputs:
        if item.role == "user_input":
            call_args.append(arg_names[user_pos] if user_pos < len(arg_names) else item.name)
            user_pos += 1
        elif item.source_path:
            call_args.append(_python_path_expr("self", item.source_path))
        else:
            call_args.append(item.name)
    for attr in contract.attributes:
        if attr.source == "module_attribute": call_args.append(_python_path_expr("self", attr.name))
        elif attr.source == "forward_arg" and attr.name in arg_names: call_args.append(attr.name)
        else: call_args.append(repr(attr.default))
    lines += ["        return ascendc_binding." + spec.entrypoint + "(" + ", ".join(call_args) + ")", ""]
    return "\n".join(lines).rstrip() + "\n"


__all__ = ["CONTRACT_VERSION", "BoundaryTensor", "BoundaryAttribute", "BoundaryOutput", "BoundaryContract", "capture_boundary_contract", "load_boundary_contract", "lower_boundary_contract", "validate_boundary_spec", "render_boundary_scaffold"]
