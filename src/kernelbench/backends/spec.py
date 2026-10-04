"""Declarative ABI and safe expression DSL for Ascend C candidates."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .errors import SpecValidationError


_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_C_TYPE = re.compile(r"^(?:u?int(?:8|16|32|64)_t|float|double|half|bool|size_t)$")
_DTYPES = {
    "float32": ("ge::DT_FLOAT", "at::kFloat", "ACL_FLOAT"),
    "float16": ("ge::DT_FLOAT16", "at::kHalf", "ACL_FLOAT16"),
    "bfloat16": ("ge::DT_BF16", "at::kBFloat16", "ACL_BF16"),
    "int64": ("ge::DT_INT64", "at::kLong", "ACL_INT64"),
    "int32": ("ge::DT_INT32", "at::kInt", "ACL_INT32"),
    "bool": ("ge::DT_BOOL", "at::kBool", "ACL_BOOL"),
}
_ATTR_TYPES = {
    "float": ("float", "double"),
    "double": ("double", "double"),
    "int32_t": ("int32_t", "int64_t"),
    "int64_t": ("int64_t", "int64_t"),
    "bool": ("bool", "bool"),
}


def _literal_expr(value: Any) -> str:
    if isinstance(value, bool):
        return f"literal({'true' if value else 'false'})"
    if isinstance(value, (int, float)):
        return f"literal({value!r})"
    if isinstance(value, str):
        return value
    raise SpecValidationError(f"expression value is not scalar: {value!r}")


def _subscript_index(node: ast.AST) -> int:
    value = node.value if isinstance(node, ast.Constant) else None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SpecValidationError("input index must be a non-negative integer")
    return value


def _constant(node: ast.AST) -> int | float | bool | str:
    if not isinstance(node, ast.Constant) or not isinstance(
        node.value, (int, float, bool, str)
    ):
        raise SpecValidationError("expression literal must be a scalar constant")
    return node.value


def _parse_expression(expression: str, input_count: int, attributes: set[str]) -> ast.AST:
    if not isinstance(expression, str) or not expression.strip():
        raise SpecValidationError("expression must be a non-empty string")
    normalized = expression.replace("literal(true)", "literal(True)").replace(
        "literal(false)", "literal(False)"
    )
    try:
        root = ast.parse(normalized, mode="eval").body
    except SyntaxError as exc:
        raise SpecValidationError(f"invalid expression {expression!r}: {exc}") from exc

    def check(node: ast.AST) -> None:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name == "literal":
                if len(node.args) != 1 or isinstance(_constant(node.args[0]), str):
                    raise SpecValidationError("literal() requires one numeric or boolean value")
                return
            if name == "attr":
                if len(node.args) != 1:
                    raise SpecValidationError("attr() requires one attribute name")
                attr_name = _constant(node.args[0])
                if not isinstance(attr_name, str) or attr_name not in attributes:
                    raise SpecValidationError(f"unknown attribute in expression: {attr_name!r}")
                return
            if name not in {"add", "sub", "mul", "floor_div", "ceil_div", "min", "max"}:
                raise SpecValidationError(f"expression function is not allowed: {name}")
            if len(node.args) != 2:
                raise SpecValidationError(f"{name}() requires two arguments")
            for argument in node.args:
                check(argument)
            return
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func
            if attr.attr not in {"dim", "size", "numel"}:
                raise SpecValidationError(f"input method is not allowed: {attr.attr}")
            if not isinstance(attr.value, ast.Subscript) or not isinstance(
                attr.value.value, ast.Name
            ) or attr.value.value.id != "input":
                raise SpecValidationError("only input[index] methods are allowed")
            index = _subscript_index(attr.value.slice)
            if index >= input_count:
                raise SpecValidationError(f"input index {index} is out of range")
            if attr.attr == "numel":
                if node.args:
                    raise SpecValidationError("numel() takes no arguments")
            else:
                if len(node.args) != 1:
                    raise SpecValidationError(f"{attr.attr}() requires one dimension")
                dim = _constant(node.args[0])
                if not isinstance(dim, int) or isinstance(dim, bool) or dim < 0:
                    raise SpecValidationError("dimension must be a non-negative integer")
            return
        raise SpecValidationError(
            "unsafe expression; use input[i].dim(j), input[i].numel(), attr(name), "
            "literal(value), or a supported arithmetic function"
        )

    check(root)
    return root


ExpressionTarget = Literal["tiling", "infer", "binding"]


def compile_expression(
    expression: str,
    *,
    target: ExpressionTarget,
    input_count: int,
    attribute_names: list[str],
    attribute_types: dict[str, str] | None = None,
) -> str:
    """Compile the restricted expression DSL to a C++ expression."""

    root = _parse_expression(expression, input_count, set(attribute_names))
    attr_indexes = {name: index for index, name in enumerate(attribute_names)}
    attribute_types = attribute_types or {}

    def emit(node: ast.AST) -> str:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name == "literal":
                value = _constant(node.args[0])
                if isinstance(value, bool):
                    return "true" if value else "false"
                return repr(value)
            if name == "attr":
                attr_name = str(_constant(node.args[0]))
                if target == "binding":
                    return attr_name
                index = attr_indexes[attr_name]
                c_type = attribute_types.get(attr_name, "float")
                return f"(*context->GetAttrs()->GetAttrPointer<{c_type}>({index}))"
            left, right = (emit(argument) for argument in node.args)
            if name == "add":
                return f"(({left}) + ({right}))"
            if name == "sub":
                return f"(({left}) - ({right}))"
            if name == "mul":
                return f"(({left}) * ({right}))"
            if name == "floor_div":
                return f"(({left}) / ({right}))"
            if name == "ceil_div":
                return f"((({left}) + ({right}) - 1) / ({right}))"
            return (
                f"std::{name}<int64_t>(static_cast<int64_t>({left}), "
                f"static_cast<int64_t>({right}))"
            )

        call = node
        assert isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
        method = call.func.attr
        subscript = call.func.value
        assert isinstance(subscript, ast.Subscript)
        index = _subscript_index(subscript.slice)
        if target == "binding":
            base = f"input_{index}"
            return f"{base}.numel()" if method == "numel" else f"{base}.size({_constant(call.args[0])})"
        if target == "tiling":
            base = f"context->GetInputShape({index})->GetOriginShape()"
        else:
            base = f"(*context->GetInputShape({index}))"
        return f"{base}.GetShapeSize()" if method == "numel" else f"{base}.GetDim({_constant(call.args[0])})"

    return emit(root)


@dataclass(frozen=True)
class AscendCTilingField:
    name: str
    c_type: str
    expression: str


@dataclass(frozen=True)
class AscendCTilingSpec:
    fields: tuple[AscendCTilingField, ...]
    block_dim: str
    workspace: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fields": [asdict(item) for item in self.fields],
            "block_dim": self.block_dim,
            "workspace": self.workspace,
        }


@dataclass(frozen=True)
class AscendCSpec:
    op_name: str
    entrypoint: str
    inputs: list[dict[str, Any]]
    outputs: list[dict[str, Any]]
    attributes: list[dict[str, Any]] = field(default_factory=list)
    tiling: dict[str, Any] = field(default_factory=dict)
    block_dim: int | str | None = None
    workspace: int | str = 0

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "AscendCSpec":
        if not isinstance(value, dict):
            raise SpecValidationError("ASCENDC_OP_SPEC must be a dictionary")
        required = ("op_name", "entrypoint", "inputs", "outputs")
        missing = [key for key in required if key not in value]
        if missing:
            raise SpecValidationError(f"ASCENDC_OP_SPEC missing fields: {', '.join(missing)}")
        spec = cls(
            op_name=value["op_name"],
            entrypoint=value["entrypoint"],
            inputs=value["inputs"],
            outputs=value["outputs"],
            attributes=value.get("attributes", []),
            tiling=value.get("tiling", {}),
            block_dim=value.get("block_dim"),
            workspace=value.get("workspace", 0),
        )
        spec.validate()
        return spec

    @property
    def attribute_names(self) -> list[str]:
        return [str(item["name"]) for item in self.attributes]

    @property
    def attribute_host_types(self) -> dict[str, str]:
        return {
            item["name"]: _ATTR_TYPES[str(item.get("c_type", item.get("type", "float")))][0]
            for item in self.attributes
        }

    def validate(self) -> None:
        if not isinstance(self.op_name, str) or not _IDENTIFIER.fullmatch(self.op_name):
            raise SpecValidationError("op_name must be a C/Python identifier")
        if not isinstance(self.entrypoint, str) or not _IDENTIFIER.fullmatch(self.entrypoint):
            raise SpecValidationError("entrypoint must be a C/Python identifier")
        if not isinstance(self.inputs, list) or not isinstance(self.outputs, list):
            raise SpecValidationError("inputs and outputs must be lists")
        if not self.inputs:
            raise SpecValidationError("at least one input is required")
        if not self.outputs:
            raise SpecValidationError("at least one output is required")
        if not isinstance(self.attributes, list):
            raise SpecValidationError("attributes must be a list")

        attr_names: set[str] = set()
        for index, item in enumerate(self.attributes):
            if not isinstance(item, dict):
                raise SpecValidationError(f"attributes[{index}] must be a mapping")
            name = item.get("name")
            if not isinstance(name, str) or not _IDENTIFIER.fullmatch(name):
                raise SpecValidationError(f"attributes[{index}].name must be an identifier")
            if name in attr_names:
                raise SpecValidationError(f"duplicate attribute name: {name}")
            attr_names.add(name)
            c_type = str(item.get("c_type", item.get("type", "float")))
            if c_type not in _ATTR_TYPES:
                raise SpecValidationError(f"unsupported attribute type: {c_type}")

        tensor_names: set[str] = set()
        for group_name, group in (("inputs", self.inputs), ("outputs", self.outputs)):
            for index, item in enumerate(group):
                if not isinstance(item, dict):
                    raise SpecValidationError(f"{group_name}[{index}] must be a mapping")
                name = item.get("name")
                if not isinstance(name, str) or not _IDENTIFIER.fullmatch(name):
                    raise SpecValidationError(f"{group_name}[{index}].name must be an identifier")
                if name in tensor_names:
                    raise SpecValidationError(f"duplicate tensor name: {name}")
                tensor_names.add(name)
                dtype = item.get("dtype", "float32")
                if dtype not in _DTYPES:
                    raise SpecValidationError(f"unsupported dtype: {dtype}")
                if item.get("layout", "ND") not in {"ND", "contiguous"}:
                    raise SpecValidationError("only ND/contiguous layout is supported")
                shape = item.get("shape", [])
                if not isinstance(shape, list):
                    raise SpecValidationError(f"{group_name}[{index}].shape must be a list")
                for expression in shape:
                    _parse_expression(expression, len(self.inputs), attr_names)

        if not isinstance(self.tiling, dict):
            raise SpecValidationError("tiling must be a mapping")
        for name, definition in self.tiling.items():
            if not isinstance(name, str) or not _IDENTIFIER.fullmatch(name):
                raise SpecValidationError(f"invalid tiling field name: {name!r}")
            if not isinstance(definition, dict):
                raise SpecValidationError(f"tiling.{name} must be a mapping")
            c_type = str(definition.get("c_type", "uint32_t"))
            if not _C_TYPE.fullmatch(c_type):
                raise SpecValidationError(f"tiling.{name}.c_type is not allowed")

        for expression in (self.block_dim, self.workspace):
            if expression is not None:
                _parse_expression(_literal_expr(expression), len(self.inputs), attr_names)

    def compile_expression(self, expression: str | int | float, target: ExpressionTarget) -> str:
        return compile_expression(
            _literal_expr(expression),
            target=target,
            input_count=len(self.inputs),
            attribute_names=self.attribute_names,
            attribute_types=self.attribute_host_types,
        )

    def dtype_info(self, item: dict[str, Any]) -> tuple[str, str, str]:
        return _DTYPES[str(item.get("dtype", "float32"))]

    def attribute_types(self, item: dict[str, Any]) -> tuple[str, str]:
        return _ATTR_TYPES[str(item.get("c_type", item.get("type", "float")))]

    def parse_tiling(self, value: dict[str, Any]) -> AscendCTilingSpec:
        if not isinstance(value, dict):
            raise SpecValidationError("ASCENDC_TILING must be a literal mapping")
        explicit = value.get("fields")
        if explicit is not None:
            if not isinstance(explicit, list):
                raise SpecValidationError("ASCENDC_TILING.fields must be a list")
            entries = explicit
        else:
            entries = []
            for name, definition in self.tiling.items():
                expression = value.get(name, definition.get("expression", definition.get("default")))
                if expression is None:
                    raise SpecValidationError(f"tiling field has no expression: {name}")
                entries.append(
                    {
                        "name": name,
                        "c_type": definition.get("c_type", "uint32_t"),
                        "expression": expression,
                    }
                )

        fields: list[AscendCTilingField] = []
        names: set[str] = set()
        for index, item in enumerate(entries):
            if not isinstance(item, dict):
                raise SpecValidationError(f"tiling fields[{index}] must be a mapping")
            name = item.get("name")
            c_type = str(item.get("c_type", "uint32_t"))
            if not isinstance(name, str) or not _IDENTIFIER.fullmatch(name):
                raise SpecValidationError(f"invalid tiling field name: {name!r}")
            if name in names:
                raise SpecValidationError(f"duplicate tiling field: {name}")
            if not _C_TYPE.fullmatch(c_type):
                raise SpecValidationError(f"tiling field type is not allowed: {c_type}")
            names.add(name)
            expression = _literal_expr(item.get("expression", item.get("value")))
            _parse_expression(expression, len(self.inputs), set(self.attribute_names))
            fields.append(AscendCTilingField(name, c_type, expression))

        block_dim = _literal_expr(value.get("block_dim", self.block_dim or 1))
        workspace = _literal_expr(value.get("workspace", self.workspace))
        _parse_expression(block_dim, len(self.inputs), set(self.attribute_names))
        _parse_expression(workspace, len(self.inputs), set(self.attribute_names))
        return AscendCTilingSpec(tuple(fields), block_dim, workspace)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)


def extract_literal_assignment(source: str, name: str) -> Any:
    try:
        tree = ast.parse(source, filename="candidate.py")
    except SyntaxError as exc:
        raise SpecValidationError(f"candidate syntax error: {exc}") from exc
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            try:
                return ast.literal_eval(node.value)
            except Exception as exc:
                raise SpecValidationError(f"{name} must be a literal value") from exc
    raise SpecValidationError(f"{name} assignment was not found")


def extract_literal_spec(source: str) -> AscendCSpec:
    return AscendCSpec.from_mapping(extract_literal_assignment(source, "ASCENDC_OP_SPEC"))


def extract_literal_tiling(source: str, spec: AscendCSpec) -> AscendCTilingSpec:
    return spec.parse_tiling(extract_literal_assignment(source, "ASCENDC_TILING"))
