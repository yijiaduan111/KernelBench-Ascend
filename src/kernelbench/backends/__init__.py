"""KernelBench code-generation backends."""

from .ascendc import (
    AscendCSpec,
    build_ascendc_artifact,
    evaluate_ascendc_artifact,
    validate_ascendc_candidate,
)
__all__ = [
    "AscendCSpec",
    "build_ascendc_artifact",
    "evaluate_ascendc_artifact",
    "validate_ascendc_candidate",
]
