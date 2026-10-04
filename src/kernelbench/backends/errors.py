"""Structured errors for generated Ascend C candidates."""


class AscendCError(RuntimeError):
    failure_type = "ascendc_error"


class StaticRejectionError(AscendCError):
    failure_type = "static_rejection"


class SpecValidationError(AscendCError):
    failure_type = "spec_validation_error"


class AscendCCompileError(AscendCError):
    failure_type = "ascendc_compile_error"


class HostCompileError(AscendCError):
    failure_type = "host_compile_error"


class OPPInstallError(AscendCError):
    failure_type = "opp_install_error"


class BindingCompileError(AscendCError):
    failure_type = "binding_compile_error"


class BindingLoadError(AscendCError):
    failure_type = "binding_load_error"


class NpuRuntimeError(AscendCError):
    failure_type = "npu_runtime_error"


class CandidateTimeoutError(AscendCError):
    failure_type = "timeout"


class BackendUnavailableError(AscendCError):
    failure_type = "backend_unavailable"
