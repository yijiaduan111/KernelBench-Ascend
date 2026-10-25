"""Generate full-operator Ascend C specs from trusted task semantics."""

from __future__ import annotations

import ast
import copy
import pprint
from dataclasses import dataclass
from typing import Any

from .ascendc_tasks import get_ascend_task_contract, validate_spec_for_task
from .errors import SpecValidationError
from .spec import AscendCSpec


@dataclass(frozen=True)
class AscendCImplementationConfig:
    """Implementation choices supplied by a template, not by the task contract."""

    tiling: dict[str, dict[str, Any]]
    block_dim: int | str
    workspace: int | str = 0

    @classmethod
    def from_spec(cls, spec: AscendCSpec) -> "AscendCImplementationConfig":
        return cls(
            tiling=copy.deepcopy(spec.tiling),
            block_dim=spec.block_dim if spec.block_dim is not None else 1,
            workspace=spec.workspace,
        )


_VERIFIED_IMPLEMENTATIONS: dict[tuple[int, int], AscendCImplementationConfig] = {
    (1, 20): AscendCImplementationConfig(
        tiling={
            "totalLength": {"c_type": "uint32_t", "expression": "input[0].numel()"},
            "tileLength": {"c_type": "uint32_t", "default": 8192},
            "negativeSlope": {
                "c_type": "float",
                "expression": 'attr("negative_slope")',
            },
        },
        block_dim=24,
    ),
    (1, 25): AscendCImplementationConfig(
        tiling={
            "totalLength": {"c_type": "uint32_t", "expression": "input[0].numel()"},
            "tileSize": {"c_type": "uint32_t", "default": 8192},
        },
        block_dim=32,
    ),
}


def verified_implementation_config(level: int, problem_id: int) -> AscendCImplementationConfig:
    """Return the first-stage implementation choices, independent of candidate spec."""

    try:
        return copy.deepcopy(_VERIFIED_IMPLEMENTATIONS[(level, problem_id)])
    except KeyError as exc:
        raise SpecValidationError(
            f"no verified implementation config for level={level}, problem_id={problem_id}"
        ) from exc


def build_full_operator_spec(
    level: int,
    problem_id: int,
    implementation_config: AscendCImplementationConfig,
) -> AscendCSpec:
    """Build the semantic part of a spec without copying it from a candidate."""

    contract = get_ascend_task_contract(level, problem_id)
    if not contract.strict_schema:
        raise SpecValidationError(
            f"task P{problem_id} has no complete trusted spec-generation contract"
        )

    def tensors(items: tuple) -> list[dict[str, Any]]:
        result = []
        for item in items:
            if item.name is None or item.shape is None or item.layout is None:
                raise SpecValidationError("spec-generation tensor contract is incomplete")
            result.append(
                {
                    "name": item.name,
                    "dtype": item.dtype,
                    "layout": item.layout,
                    "shape": list(item.shape),
                }
            )
        return result

    defaults = dict(contract.attribute_defaults)
    if set(defaults) != {name for name, _ in contract.required_attributes}:
        raise SpecValidationError("spec-generation attribute defaults are incomplete")
    attributes = [
        {"name": name, "c_type": c_type, "default": defaults[name]}
        for name, c_type in contract.required_attributes
    ]
    if not contract.operator_slug:
        raise SpecValidationError("spec-generation operator name is incomplete")
    slug = contract.operator_slug
    spec = AscendCSpec.from_mapping(
        {
            "op_name": f"l{level}_p{problem_id}_{slug}",
            "entrypoint": f"p{problem_id}_{slug}_entry",
            "inputs": tensors(contract.inputs),
            "outputs": tensors(contract.outputs),
            "attributes": attributes,
            "tiling": copy.deepcopy(implementation_config.tiling),
            "block_dim": implementation_config.block_dim,
            "workspace": implementation_config.workspace,
        }
    )
    validate_spec_for_task(spec, contract)
    return spec


def render_candidate_with_generated_spec(
    source: str,
    level: int,
    problem_id: int,
    implementation_config: AscendCImplementationConfig | None = None,
) -> str:
    """Keep a trusted kernel, tiling values and ModelNew while regenerating its spec."""

    if implementation_config is None:
        implementation_config = verified_implementation_config(level, problem_id)
    generated = build_full_operator_spec(level, problem_id, implementation_config)
    tree = ast.parse(source, filename="candidate.py")
    assignment = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "ASCENDC_OP_SPEC"
                for target in node.targets
            )
        ),
        None,
    )
    if assignment is None:
        raise SpecValidationError("candidate template has no ASCENDC_OP_SPEC assignment")
    lines = source.splitlines(keepends=True)
    rendered_spec = pprint.pformat(generated.to_dict(), sort_dicts=False, width=88)
    lines[assignment.lineno - 1 : assignment.end_lineno] = [
        f"ASCENDC_OP_SPEC = {rendered_spec}\n"
    ]
    return "".join(lines)


__all__ = [
    "AscendCImplementationConfig",
    "build_full_operator_spec",
    "render_candidate_with_generated_spec",
    "verified_implementation_config",
]
