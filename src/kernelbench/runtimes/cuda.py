"""CUDA/ROCm-compatible runtime adapter."""

from __future__ import annotations

import torch

from .base import DeviceRuntime


class CudaRuntime(DeviceRuntime):
    name = "cuda"
    device_type = "cuda"

    def is_available(self) -> bool:
        return bool(torch.cuda.is_available())

    def device(self, index: int = 0):
        if isinstance(index, torch.device):
            return index
        return torch.device(f"cuda:{index}")

    def device_count(self) -> int:
        return int(torch.cuda.device_count())

    def current_device(self) -> int:
        return int(torch.cuda.current_device())

    def set_device(self, device) -> None:
        torch.cuda.set_device(device)

    def synchronize(self, device=None) -> None:
        if device is None:
            torch.cuda.synchronize()
        else:
            torch.cuda.synchronize(device=device)

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def reset_peak_memory_stats(self, device=None) -> None:
        torch.cuda.reset_peak_memory_stats(device=device)

    def manual_seed(self, seed: int) -> None:
        torch.cuda.manual_seed(seed)

    def device_name(self, device=None) -> str:
        return str(torch.cuda.get_device_name(device=device))

    def event(self, enable_timing: bool = True):
        return torch.cuda.Event(enable_timing=enable_timing)

    def visible_devices_env(self, index: int) -> tuple[str, str]:
        return "CUDA_VISIBLE_DEVICES", str(index)
