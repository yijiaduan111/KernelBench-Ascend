"""Public KernelBench Ascend C candidate and evaluation API.

Optimizer integrations should import candidate types and evaluator operations
from this module. Internal modules under kernelbench.backends may change
without changing this integration surface.
"""

from __future__ import annotations

from .backends.artifact import environment_fingerprint
from .backends.ascendc import (
    build_ascendc_artifact,
    evaluate_ascendc_artifact,
    validate_ascendc_candidate,
)
from .backends.builder import AscendCArtifact, AscendCBuilder
from .backends.errors import (
    AscendCCompileError,
    AscendCError,
    BindingCompileError,
    BindingLoadError,
    CandidateTimeoutError,
    NpuRuntimeError,
    OPPInstallError,
    SpecValidationError,
    StaticRejectionError,
)
from .backends.security import validate_candidate_source
from .backends.spec import (
    AscendCSpec,
    AscendCTilingField,
    AscendCTilingSpec,
    extract_literal_assignment,
    extract_literal_spec,
    extract_literal_tiling,
)

API_VERSION = 1

__all__ = [
    "API_VERSION",
    "AscendCArtifact",
    "AscendCBuilder",
    "AscendCCompileError",
    "AscendCError",
    "AscendCSpec",
    "AscendCTilingField",
    "AscendCTilingSpec",
    "BindingCompileError",
    "BindingLoadError",
    "CandidateTimeoutError",
    "NpuRuntimeError",
    "OPPInstallError",
    "SpecValidationError",
    "StaticRejectionError",
    "build_ascendc_artifact",
    "evaluate_ascendc_artifact",
    "environment_fingerprint",
    "extract_literal_assignment",
    "extract_literal_spec",
    "extract_literal_tiling",
    "validate_ascendc_candidate",
    "validate_candidate_source",
]
