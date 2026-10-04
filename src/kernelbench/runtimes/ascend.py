"""Lazy Ascend NPU runtime adapter.

Importing KernelBench remains safe on CUDA-only hosts.  ``torch_npu`` is only
imported when an Ascend runtime operation is requested.
"""

from __future__ import annotations

import importlib

import torch

from .base import DeviceRuntime, RuntimeUnavailableError


class AscendRuntime(DeviceRuntime):
    name = "ascend"
    device_type = "npu"

    def _npu(self):
        try:
            importlib.import_module("torch_npu")
        except Exception as exc:  # pragma: no cover - depends on host SDK
            raise RuntimeUnavailableError(
                "torch_npu could not be imported. Install a wheel matching "
                "the installed PyTorch and CANN versions."
            ) from exc
        npu = getattr(torch, "npu", None)
        if npu is None:
            raise RuntimeUnavailableError(
                "torch_npu is installed but torch.npu is unavailable."
            )
        return npu

    def is_available(self) -> bool:
        try:
            return bool(self._npu().is_available())
        except RuntimeUnavailableError:
            return False

    def require_available(self) -> None:
        npu = self._npu()
        if not npu.is_available():
            raise RuntimeUnavailableError("Ascend NPU runtime is not available.")

    def device(self, index: int = 0):
        if isinstance(index, torch.device):
            return index
        return torch.device(f"npu:{index}")

    def device_count(self) -> int:
        return int(self._npu().device_count())

    def current_device(self) -> int:
        return int(self._npu().current_device())

    def set_device(self, device) -> None:
        self._npu().set_device(device)

    def synchronize(self, device=None) -> None:
        npu = self._npu()
        if device is None:
            npu.synchronize()
        else:
            npu.synchronize(device=device)

    def empty_cache(self) -> None:
        empty_cache = getattr(self._npu(), "empty_cache", None)
        if empty_cache is not None:
            empty_cache()

    def reset_peak_memory_stats(self, device=None) -> None:
        fn = getattr(self._npu(), "reset_peak_memory_stats", None)
        if fn is not None:
            fn(device=device)

    def manual_seed(self, seed: int) -> None:
        fn = getattr(self._npu(), "manual_seed", None)
        if fn is not None:
            fn(seed)

    def device_name(self, device=None) -> str:
        return str(self._npu().get_device_name(device))

    def event(self, enable_timing: bool = True):
        return self._npu().Event(enable_timing=enable_timing)

    def visible_devices_env(self, index: int) -> tuple[str, str]:
        return "ASCEND_RT_VISIBLE_DEVICES", str(index)
