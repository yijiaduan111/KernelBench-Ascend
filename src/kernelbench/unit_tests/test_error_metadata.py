from kernelbench.backends.errors import SpecValidationError
from kernelbench.eval import (
    _bounded_error_text,
    _record_backend_exception,
    _record_backend_failure,
)


def test_ascendc_exception_records_structured_failure_and_legacy_alias():
    metadata = {}

    _record_backend_exception(metadata, SpecValidationError("tiling.tileSize must be a mapping"))

    assert metadata["failure_type"] == "spec_validation_error"
    assert metadata["failure_stage"] == "compile"
    assert metadata["failure_exception"] == "SpecValidationError"
    assert metadata["failure_message"] == "tiling.tileSize must be a mapping"
    assert metadata["ascendc_error"] == metadata["failure_message"]


def test_backend_failure_message_is_bounded():
    metadata = {}
    message = "a" * 7000

    _record_backend_failure(
        metadata,
        failure_type="binding_compile_error",
        failure_stage="compile",
        message=message,
        exception_name="RuntimeError",
    )

    assert len(metadata["failure_message"]) <= 6000
    assert "[error truncated]" in metadata["failure_message"]
    assert metadata["failure_message"].startswith("a")
    assert metadata["failure_message"].endswith("a")


def test_bounded_error_text_preserves_short_messages():
    assert _bounded_error_text("shape expression is unsafe") == "shape expression is unsafe"
