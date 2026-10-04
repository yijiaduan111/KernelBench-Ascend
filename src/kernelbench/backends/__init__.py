"""KernelBench code-generation backends."""

from .ascendc import (
    AscendCSpec,
    build_ascendc_artifact,
    evaluate_ascendc_artifact,
    validate_ascendc_candidate,
)
from .ascendc_tasks import ASCEND_L1_15, get_ascend_task_contract

__all__ = [
    "ASCEND_L1_15",
    "AscendCSpec",
    "build_ascendc_artifact",
    "evaluate_ascendc_artifact",
    "get_ascend_task_contract",
    "validate_ascendc_candidate",
]
