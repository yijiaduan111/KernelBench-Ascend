from pathlib import Path

import pytest

from kernelbench.backends.ascendc_tasks import (
    get_ascend_task_contract,
    validate_spec_for_task,
)
from kernelbench.backends.errors import SpecValidationError
from kernelbench.backends.security import validate_candidate_source
from kernelbench.backends.spec import (
    AscendCSpec,
    extract_literal_assignment,
    extract_literal_spec,
    extract_literal_tiling,
)
from kernelbench.backends.spec_factory import (
    AscendCImplementationConfig,
    build_full_operator_spec,
    render_candidate_with_generated_spec,
    verified_implementation_config,
)


ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("problem_id", "filename"),
    [(20, "20_leaky_relu.py"), (25, "25_swish.py")],
)
def test_generated_spec_preserves_golden_implementation(problem_id, filename):
    source = (ROOT / "golden" / "ascendc" / "level1" / filename).read_text()
    original = extract_literal_spec(source)
    config = AscendCImplementationConfig.from_spec(original)
    assert verified_implementation_config(1, problem_id) == config
    spec = build_full_operator_spec(1, problem_id, config)
    candidate = render_candidate_with_generated_spec(source, 1, problem_id)

    assert spec.to_dict() == original.to_dict()
    assert validate_candidate_source(candidate).to_dict() == spec.to_dict()
    assert extract_literal_tiling(candidate, spec) == extract_literal_tiling(source, original)
    assert extract_literal_assignment(candidate, "ASCENDC_KERNEL_SOURCE") == extract_literal_assignment(
        source, "ASCENDC_KERNEL_SOURCE"
    )
    assert candidate.split("class ModelNew", 1)[1] == source.split("class ModelNew", 1)[1]
    validate_spec_for_task(spec, get_ascend_task_contract(1, problem_id))


def test_generated_semantics_do_not_come_from_old_candidate():
    source = (ROOT / "golden" / "ascendc" / "level1" / "25_swish.py").read_text()
    original = extract_literal_spec(source)
    wrong = original.to_dict()
    wrong["outputs"][0]["shape"][0] = "literal(1)"
    wrong["inputs"][0]["name"] = "incorrect"
    with pytest.raises(SpecValidationError, match="name must be"):
        validate_spec_for_task(AscendCSpec.from_mapping(wrong), get_ascend_task_contract(1, 25))

    generated = build_full_operator_spec(
        1, 25, AscendCImplementationConfig.from_spec(original)
    )
    assert generated.inputs[0]["name"] == "x"
    assert generated.outputs[0]["shape"] == [
        "input[0].dim(0)", "input[0].dim(1)"
    ]
    stale_source = source.replace('"name": "x"', '"name": "incorrect"', 1)
    repaired = render_candidate_with_generated_spec(stale_source, 1, 25)
    assert extract_literal_spec(repaired).to_dict() == original.to_dict()
    placeholder = source[: source.index("ASCENDC_OP_SPEC")] + (
        "ASCENDC_OP_SPEC = None\n\n" + source[source.index("ASCENDC_TILING") :]
    )
    assert extract_literal_spec(
        render_candidate_with_generated_spec(placeholder, 1, 25)
    ).to_dict() == original.to_dict()


def test_same_rank_wrong_output_shape_is_rejected():
    source = (ROOT / "golden" / "ascendc" / "level1" / "25_swish.py").read_text()
    mapping = extract_literal_spec(source).to_dict()
    mapping["outputs"][0]["shape"] = ["literal(4096)", "input[0].dim(1)"]
    with pytest.raises(SpecValidationError, match="shape must be"):
        validate_spec_for_task(
            AscendCSpec.from_mapping(mapping), get_ascend_task_contract(1, 25)
        )
    mapping = extract_literal_spec(source).to_dict()
    del mapping["outputs"][0]["layout"]
    with pytest.raises(SpecValidationError, match="layout must be"):
        validate_spec_for_task(
            AscendCSpec.from_mapping(mapping), get_ascend_task_contract(1, 25)
        )


def test_p20_attribute_default_and_p25_dtype_are_trusted():
    p20 = (ROOT / "golden" / "ascendc" / "level1" / "20_leaky_relu.py").read_text()
    mapping = extract_literal_spec(p20).to_dict()
    mapping["attributes"][0]["default"] = 0.2
    with pytest.raises(SpecValidationError, match="default must be"):
        validate_spec_for_task(
            AscendCSpec.from_mapping(mapping), get_ascend_task_contract(1, 20)
        )
    mapping = extract_literal_spec(p20).to_dict()
    mapping["attributes"] = []
    with pytest.raises(SpecValidationError, match="requires attribute"):
        validate_spec_for_task(
            AscendCSpec.from_mapping(mapping), get_ascend_task_contract(1, 20)
        )

    p25 = (ROOT / "golden" / "ascendc" / "level1" / "25_swish.py").read_text()
    mapping = extract_literal_spec(p25).to_dict()
    mapping["outputs"][0]["dtype"] = "float16"
    with pytest.raises(SpecValidationError, match="dtype must be"):
        validate_spec_for_task(
            AscendCSpec.from_mapping(mapping), get_ascend_task_contract(1, 25)
        )


def test_p20_operator_and_entrypoint_are_trusted():
    source = (ROOT / "golden" / "ascendc" / "level1" / "20_leaky_relu.py").read_text()
    mapping = extract_literal_spec(source).to_dict()
    mapping["entrypoint"] = "other_entry"
    with pytest.raises(SpecValidationError, match="op name and entrypoint"):
        validate_spec_for_task(
            AscendCSpec.from_mapping(mapping), get_ascend_task_contract(1, 20)
        )


def test_factory_waits_for_complete_contract_and_p45_matches_audit():
    assert get_ascend_task_contract(1, 45).reference_status == "supported"
    with pytest.raises(SpecValidationError, match="no complete trusted"):
        build_full_operator_spec(
            1, 47, AscendCImplementationConfig(tiling={}, block_dim=1)
        )


@pytest.mark.parametrize(
    "problem_id", (1, 10, 33, 40, 42, 45, 47, 50, 61, 82, 89, 95, 97)
)
def test_other_suite_contracts_keep_their_existing_validation_scope(problem_id):
    contract = get_ascend_task_contract(1, problem_id)
    assert contract.reference_status == "supported"
    assert not contract.strict_schema
    assert contract.operator_slug is None
    assert all(tensor.shape is None for tensor in (*contract.inputs, *contract.outputs))
