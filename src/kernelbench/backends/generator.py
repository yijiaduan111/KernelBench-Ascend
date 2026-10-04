"""CANN 9.0.1 project generator for validated Ascend C candidates."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .errors import SpecValidationError
from .spec import AscendCSpec, AscendCTilingSpec


TEMPLATE_VERSION = "cann-9.0.1-v4"


def _snake(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    return value.strip("_").lower()


def _pascal(value: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in _snake(value).split("_") if part)


@dataclass(frozen=True)
class GeneratedNames:
    op_type: str
    op_file: str
    kernel_symbol: str
    tiling_class: str
    binding_module: str


def generated_names(spec: AscendCSpec, candidate_hash: str) -> GeneratedNames:
    suffix = candidate_hash[:12]
    op_type = f"Kb{_pascal(spec.op_name)}{suffix}"
    op_file = _snake(op_type)
    return GeneratedNames(
        op_type=op_type,
        op_file=op_file,
        # CANN's generated compile script resolves the device entrypoint from
        # the snake_case op type. The symbol therefore must equal op_file.
        kernel_symbol=op_file,
        tiling_class=f"{op_type}TilingData",
        binding_module=f"kb_ascendc_{candidate_hash[:16]}",
    )


def rewrite_kernel_entrypoint(source: str, old: str, new: str) -> str:
    pattern = re.compile(
        rf'(extern\s+"C"\s+__global__\s+__aicore__\s+void\s+){re.escape(old)}(\s*\()'
    )
    rewritten, count = pattern.subn(rf"\g<1>{new}\g<2>", source)
    if count != 1:
        raise SpecValidationError(
            f"expected exactly one Ascend C entrypoint definition for {old}, found {count}"
        )
    return rewritten


def _render_tiling_header(names: GeneratedNames, tiling: AscendCTilingSpec) -> str:
    fields = "\n".join(
        f"    TILING_DATA_FIELD_DEF({item.c_type}, {item.name});" for item in tiling.fields
    )
    return f'''#pragma once
#include "register/tilingdata_base.h"

namespace optiling {{
BEGIN_TILING_DATA_DEF({names.tiling_class})
{fields}
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS({names.op_type}, {names.tiling_class})
}}  // namespace optiling
'''


def _render_infer_shape(spec: AscendCSpec) -> str:
    lines = ["static ge::graphStatus InferShape(gert::InferShapeContext* context) {"]
    for output_index, output in enumerate(spec.outputs):
        shape = output.get("shape", [])
        lines.extend(
            [
                f"    auto* output_{output_index} = context->GetOutputShape({output_index});",
                f"    if (output_{output_index} == nullptr) return ge::GRAPH_FAILED;",
                f"    output_{output_index}->SetDimNum({len(shape)});",
            ]
        )
        for dim_index, expression in enumerate(shape):
            cpp = spec.compile_expression(expression, "infer")
            lines.append(
                f"    output_{output_index}->SetDim({dim_index}, static_cast<int64_t>({cpp}));"
            )
    lines.extend(["    return ge::GRAPH_SUCCESS;", "}"])
    return "\n".join(lines)


def _render_infer_dtype(spec: AscendCSpec) -> str:
    lines = ["static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {"]
    for index, output in enumerate(spec.outputs):
        ge_dtype, _, _ = spec.dtype_info(output)
        lines.append(f"    context->SetOutputDataType({index}, {ge_dtype});")
    lines.extend(["    return ge::GRAPH_SUCCESS;", "}"])
    return "\n".join(lines)


def _render_registration(spec: AscendCSpec, names: GeneratedNames) -> str:
    lines: list[str] = []
    for item in spec.inputs:
        ge_dtype, _, _ = spec.dtype_info(item)
        lines.extend(
            [
                f'        this->Input("{item["name"]}").ParamType(REQUIRED)',
                f"            .DataType({{{ge_dtype}}}).Format({{ge::FORMAT_ND}})",
                "            .UnknownShapeFormat({ge::FORMAT_ND});",
            ]
        )
    for item in spec.outputs:
        ge_dtype, _, _ = spec.dtype_info(item)
        lines.extend(
            [
                f'        this->Output("{item["name"]}").ParamType(REQUIRED)',
                f"            .DataType({{{ge_dtype}}}).Format({{ge::FORMAT_ND}})",
                "            .UnknownShapeFormat({ge::FORMAT_ND});",
            ]
        )
    method = {
        "float": "Float",
        "double": "Float",
        "int32_t": "Int",
        "int64_t": "Int",
        "bool": "Bool",
    }
    for item in spec.attributes:
        c_type = str(item.get("c_type", item.get("type", "float")))
        default = item.get("default", item.get("default_value"))
        if default is None:
            raise SpecValidationError(
                f"attribute {item['name']} requires a default in the first Ascend C backend version"
            )
        rendered = str(default).lower() if isinstance(default, bool) else repr(default)
        lines.append(
            f'        this->Attr("{item["name"]}").AttrType(OPTIONAL).{method[c_type]}({rendered});'
        )
    return "\n".join(lines)


def _render_host(
    spec: AscendCSpec, tiling: AscendCTilingSpec, names: GeneratedNames
) -> str:
    field_setters = "\n".join(
        f"    tiling.set_{item.name}(static_cast<{item.c_type}>({spec.compile_expression(item.expression, 'tiling')}));"
        for item in tiling.fields
    )
    registration = _render_registration(spec, names)
    return f'''#include "{names.op_file}_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <cstdint>

namespace optiling {{
static ge::graphStatus TilingFunc(gert::TilingContext* context) {{
    if (context == nullptr) return ge::GRAPH_FAILED;
    {names.tiling_class} tiling;
    const uint32_t block_dim = static_cast<uint32_t>({spec.compile_expression(tiling.block_dim, 'tiling')});
    if (block_dim == 0) return ge::GRAPH_FAILED;
    context->SetBlockDim(block_dim);
{field_setters}
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->GetWorkspaceSizes(1)[0] = static_cast<size_t>({spec.compile_expression(tiling.workspace, 'tiling')});
    return ge::GRAPH_SUCCESS;
}}
}}  // namespace optiling

namespace ge {{
{_render_infer_shape(spec)}

{_render_infer_dtype(spec)}
}}  // namespace ge

namespace ops {{
class {names.op_type} : public OpDef {{
public:
    explicit {names.op_type}(const char* name) : OpDef(name) {{
{registration}
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }}
}};
OP_ADD({names.op_type});
}}  // namespace ops
'''


def _render_common_header() -> str:
    return r'''#pragma once
#include <torch/extension.h>
#include <cstdint>
#include <vector>
#include "aclnn/acl_meta.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

inline void check_status(aclnnStatus status, const char* operation) {
    TORCH_CHECK(status == 0, operation, " failed with ACLNN status ", status);
}

inline aclDataType acl_dtype(const at::Tensor& tensor) {
    switch (tensor.scalar_type()) {
        case at::kFloat: return ACL_FLOAT;
        case at::kHalf: return ACL_FLOAT16;
        case at::kBFloat16: return ACL_BF16;
        case at::kLong: return ACL_INT64;
        case at::kInt: return ACL_INT32;
        case at::kBool: return ACL_BOOL;
        default: TORCH_CHECK(false, "unsupported tensor dtype for Ascend C binding");
    }
}

struct AclTensorGuard {
    explicit AclTensorGuard(const at::Tensor& tensor) {
        TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, "tensor must be on NPU");
        TORCH_CHECK(tensor.is_contiguous(), "tensor must be contiguous");
        std::vector<int64_t> shape(tensor.sizes().begin(), tensor.sizes().end());
        std::vector<int64_t> strides(tensor.strides().begin(), tensor.strides().end());
        value = aclCreateTensor(shape.data(), shape.size(), acl_dtype(tensor), strides.data(), 0,
                                ACL_FORMAT_ND, shape.data(), shape.size(),
                                const_cast<void*>(tensor.const_data_ptr()));
        TORCH_CHECK(value != nullptr, "aclCreateTensor returned null");
    }
    ~AclTensorGuard() { if (value != nullptr) aclDestroyTensor(value); }
    aclTensor* value = nullptr;
};

inline at::Tensor make_workspace(uint64_t bytes, const at::Tensor& reference, void** ptr) {
    if (bytes == 0) { *ptr = nullptr; return at::Tensor(); }
    auto workspace = at::empty({static_cast<int64_t>(bytes)}, reference.options().dtype(at::kByte));
    *ptr = workspace.mutable_data_ptr();
    return workspace;
}
'''


def _render_binding(spec: AscendCSpec, names: GeneratedNames) -> str:
    input_parameters = [
        f"const at::Tensor& input_{index}" for index, _ in enumerate(spec.inputs)
    ]
    attr_parameters: list[str] = []
    for item in spec.attributes:
        _, binding_type = spec.attribute_types(item)
        attr_parameters.append(f"{binding_type} {item['name']}")
    parameters = ", ".join(input_parameters + attr_parameters)

    output_lines: list[str] = []
    for index, output in enumerate(spec.outputs):
        shape = ", ".join(
            f"static_cast<int64_t>({spec.compile_expression(expr, 'binding')})"
            for expr in output.get("shape", [])
        )
        _, torch_dtype, _ = spec.dtype_info(output)
        output_lines.append(
            f"    auto output_{index} = at::empty({{{shape}}}, input_0.options().dtype({torch_dtype}));"
        )

    guards = [
        f"    AclTensorGuard input_{index}_acl(input_{index});"
        for index, _ in enumerate(spec.inputs)
    ] + [
        f"    AclTensorGuard output_{index}_acl(output_{index});"
        for index, _ in enumerate(spec.outputs)
    ]
    call_args = (
        [f"input_{index}_acl.value" for index, _ in enumerate(spec.inputs)]
        + [str(item["name"]) for item in spec.attributes]
        + [f"output_{index}_acl.value" for index, _ in enumerate(spec.outputs)]
    )
    workspace_args = ", ".join(call_args + ["&workspace_size", "&executor"])
    launch_args = "workspace_ptr, workspace_size, executor, stream"
    if len(spec.outputs) == 1:
        result = "output_0"
    else:
        result = "std::make_tuple(" + ", ".join(
            f"output_{index}" for index, _ in enumerate(spec.outputs)
        ) + ")"
    return f'''#include "common.h"
#include "aclnn_{names.op_file}.h"
#include <atomic>

static std::atomic<uint64_t> call_count{{0}};

auto {spec.entrypoint}({parameters}) {{
{chr(10).join(output_lines)}
{chr(10).join(guards)}
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    check_status(aclnn{names.op_type}GetWorkspaceSize({workspace_args}),
                 "aclnn{names.op_type}GetWorkspaceSize");
    void* workspace_ptr = nullptr;
    auto workspace = make_workspace(workspace_size, input_0, &workspace_ptr);
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    check_status(aclnn{names.op_type}({launch_args}), "aclnn{names.op_type}");
    ++call_count;
    return {result};
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {{
    module.def("{spec.entrypoint}", &{spec.entrypoint}, "KernelBench Ascend C custom op");
    module.def("get_call_count", []() {{ return call_count.load(); }});
    module.def("reset_call_count", []() {{ call_count.store(0); }});
}}
'''


def _render_setup(names: GeneratedNames, cann_path: Path) -> str:
    return f'''from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch_npu.utils.cpp_extension import NpuExtension
import torch_npu
import sys
import sysconfig

root = Path(__file__).resolve().parent
artifact = root.parent
opp = artifact / "opp" / "vendors" / "customize"
cann = Path({str(cann_path)!r})
torch_npu_root = Path(torch_npu.__file__).resolve().parent
python_version = f"{{sys.version_info.major}}.{{sys.version_info.minor}}"
python_header_candidates = [
    Path(sysconfig.get_path("include")),
    Path(f"/usr/include/python{{python_version}}"),
]
python_header_candidates.extend(
    sorted(Path.home().glob(f"miniconda*/pkgs/python-{{python_version}}*/include/python{{python_version}}"), reverse=True)
)
python_header_candidates.extend(
    sorted(Path("/home").glob(f"*/miniconda*/pkgs/python-{{python_version}}*/include/python{{python_version}}"), reverse=True)
)
python_headers = next(
    (path for path in python_header_candidates if (path / "Python.h").is_file()),
    None,
)
if python_headers is None:
    raise RuntimeError(f"Python {{python_version}} development headers were not found")

setup(
    name={names.binding_module!r},
    ext_modules=[NpuExtension(
        name={names.binding_module!r},
        sources=[str(root / "binding.cpp")],
        include_dirs=[str(root), str(python_headers), str(opp / "op_api" / "include"), str(cann / "include")],
        library_dirs=[str(opp / "op_api" / "lib"), str(cann / "lib64")],
        libraries=["cust_opapi", "nnopbase"],
        extra_compile_args=["-O3"],
        extra_link_args=[
            f"-Wl,-rpath,{{torch_npu_root / 'lib'}}",
            f"-Wl,-rpath,{{opp / 'op_api' / 'lib'}}",
            f"-Wl,-rpath,{{cann / 'lib64'}}",
        ],
    )],
    cmdclass={{"build_ext": BuildExtension.with_options(use_ninja=True)}},
)
'''


def render_project(
    root: Path,
    spec: AscendCSpec,
    tiling: AscendCTilingSpec,
    kernel_source: str,
    names: GeneratedNames,
    cann_path: Path,
) -> None:
    project = root / "project"
    host = project / "op_host"
    kernel = project / "op_kernel"
    binding = root / "binding"
    for directory in (host, kernel, binding):
        directory.mkdir(parents=True, exist_ok=True)

    (project / "CMakeLists.txt").write_text(
        '''cmake_minimum_required(VERSION 3.16.0)
project(opp)
find_package(ASC REQUIRED)
set(package_name customize)
npu_op_package(${package_name} TYPE RUN CONFIG INSTALL_PATH ${CMAKE_BINARY_DIR}/)
add_subdirectory(op_host)
add_subdirectory(op_kernel)
''',
        encoding="utf-8",
    )
    (project / "CMakePresets.json").write_text(
        f'''{{
  "version": 1,
  "cmakeMinimumRequired": {{"major": 3, "minor": 16, "patch": 0}},
  "configurePresets": [{{
    "name": "default",
    "generator": "Unix Makefiles",
    "binaryDir": "${{sourceDir}}/build_out",
    "cacheVariables": {{
      "CMAKE_BUILD_TYPE": "Release",
      "ENABLE_SOURCE_PACKAGE": "False",
      "ENABLE_BINARY_PACKAGE": "True",
      "ASCEND_COMPUTE_UNIT": "ascend910b",
      "ENABLE_TEST": "False",
      "vendor_name": "customize",
      "ASCEND_CANN_PACKAGE_PATH": {json.dumps(str(cann_path))},
      "ASCEND_PYTHON_EXECUTABLE": "python3",
      "CMAKE_INSTALL_PREFIX": "${{sourceDir}}/build_out",
      "ENABLE_CROSS_COMPILE": "False",
      "ASCEND_PACK_SHARED_LIBRARY": "False"
    }}
  }}]
}}
''',
        encoding="utf-8",
    )
    (project / "build.sh").write_text(
        f'''#!/bin/bash
set -euo pipefail
export ASCEND_HOME_PATH="${{ASCEND_HOME_PATH:-{cann_path}}}"
cmake -S . -B build_out --preset=default
cmake --build build_out --target binary package -j"${{KERNELBENCH_BUILD_JOBS:-$(nproc)}}"
''',
        encoding="utf-8",
    )
    (host / "CMakeLists.txt").write_text(
        '''aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} ops_srcs)
npu_op_code_gen(SRC ${ops_srcs} PACKAGE ${package_name} OUT_DIR ${ASCEND_AUTOGEN_PATH} JOIN_OP_DEF True)
file(GLOB autogen_aclnn_src ${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp)
set_source_files_properties(${autogen_aclnn_src} PROPERTIES GENERATED TRUE)
npu_op_library(cust_opapi ACLNN ${autogen_aclnn_src})
file(GLOB group_proto_src ${ASCEND_AUTOGEN_PATH}/group_op_proto/*.cc)
file(GLOB proto_src ${ASCEND_AUTOGEN_PATH}/op_proto.cc)
npu_op_library(cust_op_proto GRAPH ${ops_srcs} ${group_proto_src} ${proto_src})
file(GLOB fallback_src ${ASCEND_AUTOGEN_PATH}/fallback_*.cpp)
npu_op_library(cust_optiling TILING ${ops_srcs} ${fallback_src})
npu_op_package_add(${package_name} LIBRARY cust_optiling cust_opapi cust_op_proto)
''',
        encoding="utf-8",
    )
    (kernel / "CMakeLists.txt").write_text(
        '''npu_op_kernel_sources(ascendc_kernels KERNEL_DIR ./)
npu_op_kernel_library(ascendc_kernels SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}/ TILING_LIBRARY cust_optiling)
npu_op_package_add(${package_name} LIBRARY ascendc_kernels)
''',
        encoding="utf-8",
    )
    tiling_header = _render_tiling_header(names, tiling)
    (host / f"{names.op_file}_tiling.h").write_text(tiling_header, encoding="utf-8")
    (kernel / f"{names.op_file}_tiling.h").write_text(tiling_header, encoding="utf-8")
    (host / f"{names.op_file}.cpp").write_text(
        _render_host(spec, tiling, names), encoding="utf-8"
    )
    (kernel / f"{names.op_file}.cpp").write_text(
        rewrite_kernel_entrypoint(kernel_source, spec.entrypoint, names.kernel_symbol),
        encoding="utf-8",
    )
    (binding / "common.h").write_text(_render_common_header(), encoding="utf-8")
    (binding / "binding.cpp").write_text(_render_binding(spec, names), encoding="utf-8")
    (binding / "setup.py").write_text(_render_setup(names, cann_path), encoding="utf-8")
