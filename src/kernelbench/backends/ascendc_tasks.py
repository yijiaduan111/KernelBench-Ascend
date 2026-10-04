"""Trusted task contracts for the first Ascend Level-1 migration suite."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import SpecValidationError
from .spec import AscendCSpec


@dataclass(frozen=True)
class TensorContract:
    dtype: str
    rank: int


@dataclass(frozen=True)
class AscendTaskContract:
    level: int
    problem_id: int
    name: str
    inputs: tuple[TensorContract, ...]
    outputs: tuple[TensorContract, ...]
    init_input_count: int
    required_attributes: tuple[tuple[str, str], ...] = ()
    reference_status: str = "supported"


def _tensor(dtype: str, rank: int) -> TensorContract:
    return TensorContract(dtype=dtype, rank=rank)


# These contracts describe full-operator candidates. Hybrid candidates are
# allowed by the evaluator, but should skip strict ABI validation when their
# custom op intentionally implements only part of ModelNew.forward.
ASCEND_L1_15: dict[int, AscendTaskContract] = {
    1: AscendTaskContract(1, 1, "Square matrix multiplication", (_tensor("float32", 2), _tensor("float32", 2)), (_tensor("float32", 2),), 0),
    10: AscendTaskContract(1, 10, "3D tensor matrix multiplication", (_tensor("float32", 3), _tensor("float32", 2)), (_tensor("float32", 3),), 0),
    20: AscendTaskContract(1, 20, "LeakyReLU", (_tensor("float32", 2),), (_tensor("float32", 2),), 0, (("negative_slope", "float"),)),
    25: AscendTaskContract(1, 25, "Swish", (_tensor("float32", 2),), (_tensor("float32", 2),), 0),
    33: AscendTaskContract(1, 33, "BatchNorm", (_tensor("float32", 4),), (_tensor("float32", 4),), 1),
    40: AscendTaskContract(1, 40, "LayerNorm", (_tensor("float32", 4),), (_tensor("float32", 4),), 1),
    42: AscendTaskContract(1, 42, "MaxPool2d", (_tensor("float32", 4),), (_tensor("float32", 4),), 4),
    45: AscendTaskContract(1, 45, "AvgPool2d", (_tensor("float32", 4),), (_tensor("float32", 4),), 1, reference_status="resource_timeout"),
    47: AscendTaskContract(1, 47, "Sum reduction dim 1", (_tensor("float32", 3),), (_tensor("float32", 3),), 1, (("dim", "int64_t"),)),
    50: AscendTaskContract(1, 50, "Conv2d", (_tensor("float32", 4),), (_tensor("float32", 4),), 1),
    61: AscendTaskContract(1, 61, "ConvTranspose3d", (_tensor("float32", 5),), (_tensor("float32", 5),), 3),
    82: AscendTaskContract(1, 82, "Depthwise Conv2d", (_tensor("float32", 4),), (_tensor("float32", 4),), 4),
    89: AscendTaskContract(1, 89, "Cumulative sum", (_tensor("float32", 2),), (_tensor("float32", 2),), 1),
    95: AscendTaskContract(1, 95, "Cross entropy loss", (_tensor("float32", 2), _tensor("int64", 1)), (_tensor("float32", 0),), 0),
    97: AscendTaskContract(1, 97, "Scaled dot-product attention", (_tensor("float32", 4), _tensor("float32", 4), _tensor("float32", 4)), (_tensor("float32", 4),), 0),
}


def get_ascend_task_contract(level: int, problem_id: int) -> AscendTaskContract:
    if level != 1 or problem_id not in ASCEND_L1_15:
        raise SpecValidationError(
            f"no trusted Ascend task contract for level={level}, problem_id={problem_id}"
        )
    return ASCEND_L1_15[problem_id]


def validate_spec_for_task(spec: AscendCSpec, contract: AscendTaskContract) -> None:
    """Validate a full-operator candidate against a trusted task ABI."""

    if len(spec.inputs) != len(contract.inputs):
        raise SpecValidationError(
            f"task P{contract.problem_id} requires {len(contract.inputs)} custom-op inputs, "
            f"candidate declares {len(spec.inputs)}"
        )
    if len(spec.outputs) != len(contract.outputs):
        raise SpecValidationError(
            f"task P{contract.problem_id} requires {len(contract.outputs)} custom-op outputs, "
            f"candidate declares {len(spec.outputs)}"
        )

    for group_name, actual, expected in (
        ("input", spec.inputs, contract.inputs),
        ("output", spec.outputs, contract.outputs),
    ):
        for index, (item, tensor) in enumerate(zip(actual, expected)):
            dtype = str(item.get("dtype", "float32"))
            rank = len(item.get("shape", []))
            if dtype != tensor.dtype:
                raise SpecValidationError(
                    f"task P{contract.problem_id} {group_name}[{index}] dtype must be "
                    f"{tensor.dtype}, got {dtype}"
                )
            if rank != tensor.rank:
                raise SpecValidationError(
                    f"task P{contract.problem_id} {group_name}[{index}] rank must be "
                    f"{tensor.rank}, got {rank}"
                )

    attributes = {
        str(item["name"]): str(item.get("c_type", item.get("type", "float")))
        for item in spec.attributes
    }
    for name, c_type in contract.required_attributes:
        if attributes.get(name) != c_type:
            raise SpecValidationError(
                f"task P{contract.problem_id} requires attribute {name!r} with type {c_type}"
            )


__all__ = [
    "ASCEND_L1_15",
    "AscendTaskContract",
    "TensorContract",
    "get_ascend_task_contract",
    "validate_spec_for_task",
]
