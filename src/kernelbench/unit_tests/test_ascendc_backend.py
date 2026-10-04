import json
from pathlib import Path

import pytest

from kernelbench.backends.ascendc_tasks import (
    ASCEND_L1_15,
    get_ascend_task_contract,
    validate_spec_for_task,
)
from kernelbench.backends.builder import AscendCBuilder, scrubbed_environment
from kernelbench.backends.generator import generated_names, render_project
from kernelbench.backends.security import validate_candidate_source
from kernelbench.backends.spec import (
    AscendCSpec,
    extract_literal_assignment,
    extract_literal_tiling,
)


VALID = r"""import torch
import torch.nn as nn
ASCENDC_OP_SPEC = {
    "op_name": "add",
    "entrypoint": "kernel_entry",
    "inputs": [
        {"name": "a", "dtype": "float32", "shape": ["input[0].dim(0)"]},
        {"name": "b", "dtype": "float32", "shape": ["input[1].dim(0)"]},
    ],
    "outputs": [
        {"name": "out", "dtype": "float32", "shape": ["input[0].dim(0)"]},
    ],
    "attributes": [],
    "tiling": {
        "total": {"c_type": "uint64_t", "expression": "input[0].numel()"},
        "tile_size": {"c_type": "uint32_t", "default": 256},
    },
    "block_dim": "min(literal(32), ceil_div(input[0].numel(), literal(256)))",
    "workspace": 0,
}
ASCENDC_TILING = {"tile_size": 256}
ASCENDC_KERNEL_SOURCE = r'''
#include "kernel_operator.h"
extern "C" __global__ __aicore__ void kernel_entry(
    GM_ADDR a, GM_ADDR b, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
}
'''
class ModelNew(nn.Module):
    def forward(self, a, b):
        return ascendc_binding.kernel_entry(a, b)
"""


def test_ascendc_candidate_contract():
    spec = validate_candidate_source(VALID)
    tiling = extract_literal_tiling(VALID, spec)
    assert spec.op_name == "add"
    assert [item.name for item in tiling.fields] == ["total", "tile_size"]
    assert "GetShapeSize" in spec.compile_expression(tiling.fields[0].expression, "tiling")


def test_ascendc_rejects_process_import():
    with pytest.raises(Exception, match="forbidden import"):
        validate_candidate_source(VALID.replace("import torch", "import os\nimport torch"))


def test_ascendc_rejects_non_torch_import_and_file_write():
    with pytest.raises(Exception, match="forbidden import"):
        validate_candidate_source(VALID.replace("import torch", "import urllib.request\nimport torch"))
    with pytest.raises(Exception, match="forbidden call: save"):
        validate_candidate_source(
            VALID.replace(
                "return ascendc_binding.kernel_entry(a, b)",
                'torch.save(a, "leak.pt")\n        return ascendc_binding.kernel_entry(a, b)',
            )
        )


def test_ascendc_rejects_unsafe_shape():
    source = VALID.replace(
        '"shape": ["input[0].dim(0)"]',
        '"shape": ["__import__(\\"os\\")"]',
        1,
    )
    with pytest.raises(Exception, match="expression function is not allowed"):
        validate_candidate_source(source)


def test_ascendc_requires_binding_call():
    with pytest.raises(Exception, match="must call ascendc_binding"):
        validate_candidate_source(
            VALID.replace(
                "return ascendc_binding.kernel_entry(a, b)",
                "return torch.add(a, b)",
            )
        )


def test_generator_emits_unique_private_project(tmp_path: Path):
    spec = validate_candidate_source(VALID)
    tiling = extract_literal_tiling(VALID, spec)
    names = generated_names(spec, "a" * 64)
    kernel = extract_literal_assignment(VALID, "ASCENDC_KERNEL_SOURCE")
    render_project(tmp_path, spec, tiling, kernel, names, Path("/opt/cann"))

    host = (tmp_path / "project/op_host" / f"{names.op_file}.cpp").read_text()
    device = (tmp_path / "project/op_kernel" / f"{names.op_file}.cpp").read_text()
    binding = (tmp_path / "binding/binding.cpp").read_text()
    preset = json.loads((tmp_path / "project/CMakePresets.json").read_text())

    assert names.op_type in host
    assert names.kernel_symbol in device
    assert "kernel_entry" not in device
    assert f"aclnn{names.op_type}GetWorkspaceSize" in binding
    assert preset["configurePresets"][0]["cacheVariables"]["ASCEND_CANN_PACKAGE_PATH"] == str(Path("/opt/cann"))


def test_l1_15_manifest_and_p47_contract():
    assert tuple(ASCEND_L1_15) == (1, 10, 20, 25, 33, 40, 42, 45, 47, 50, 61, 82, 89, 95, 97)
    contract = get_ascend_task_contract(1, 47)
    spec = AscendCSpec.from_mapping(
        {
            "op_name": "sum_dim_one",
            "entrypoint": "sum_entry",
            "inputs": [
                {
                    "name": "x",
                    "dtype": "float32",
                    "shape": [
                        "input[0].dim(0)",
                        "input[0].dim(1)",
                        "input[0].dim(2)",
                    ],
                }
            ],
            "outputs": [
                {
                    "name": "y",
                    "dtype": "float32",
                    "shape": [
                        "input[0].dim(0)",
                        "literal(1)",
                        "input[0].dim(2)",
                    ],
                }
            ],
            "attributes": [{"name": "dim", "c_type": "int64_t", "default": 1}],
        }
    )
    validate_spec_for_task(spec, contract)
    with pytest.raises(Exception, match="requires attribute"):
        validate_spec_for_task(
            AscendCSpec.from_mapping({**spec.to_dict(), "attributes": []}),
            contract,
        )


def test_builder_cache_hit_preserves_timings(tmp_path: Path):
    builder = AscendCBuilder(tmp_path)
    artifact = builder.prepare(VALID)
    binding = artifact.root / "binding" / f"{artifact.names.binding_module}.so"
    binding.touch()
    op_api = artifact.custom_vendor_root / "op_api" / "lib"
    op_api.mkdir(parents=True)
    (op_api / "libcust_opapi.so").touch()
    (artifact.root / "success.json").write_text(
        json.dumps(
            {
                "binding_path": str(binding),
                "timings": {
                    "compile_seconds": 1.0,
                    "opp_install_seconds": 2.0,
                    "binding_compile_seconds": 3.0,
                },
            }
        ),
        encoding="utf-8",
    )
    cached = builder.build(VALID)
    assert cached.metadata["cache_hit"] is True
    assert cached.metadata["timings"]["compile_seconds"] == 1.0


def test_build_environment_removes_api_credentials():
    env = scrubbed_environment(
        {
            "PATH": "/bin",
            "OPENAI_API_KEY": "secret",
            "OPENAI_BASE_URL": "https://example.invalid",
            "ANTHROPIC_API_KEY": "secret",
            "DASHSCOPE_API_KEY": "secret",
            "PYRAMIDKERNEL_ENV_FILE": "/tmp/secrets.env",
        }
    )
    assert env == {"PATH": "/bin"}
