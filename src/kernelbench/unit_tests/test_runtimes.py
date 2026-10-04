"""Runtime abstraction tests that do not require accelerator hardware."""

from kernelbench.runtimes import get_runtime, get_runtime_for_backend, list_runtimes


def test_runtime_registry_maps_code_backends():
    assert get_runtime_for_backend("cuda").name == "cuda"
    assert get_runtime_for_backend("triton").name == "cuda"
    assert get_runtime_for_backend("ascendc").name == "ascend"
    assert get_runtime("npu").device_type == "npu"
    assert set(list_runtimes()) == {"cuda", "ascend"}


def test_ascend_runtime_is_lazy():
    runtime = get_runtime("ascend")
    # On a CUDA-only developer machine this is simply False; importantly the
    # import of kernelbench itself does not require torch_npu.
    assert isinstance(runtime.is_available(), bool)


def test_runtime_event_interface_exists():
    assert callable(get_runtime("cuda").event)
    assert callable(get_runtime("ascend").event)
