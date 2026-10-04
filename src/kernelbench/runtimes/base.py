"""Small device-runtime protocol shared by CUDA and Ascend."""

from __future__ import annotations

from typing import Any


class RuntimeUnavailableError(RuntimeError):
    """Raised when a requested accelerator runtime is unavailable."""


class DeviceRuntime:
    """Backend-neutral accelerator operations used by the evaluator."""

    name = "unknown"
    device_type = "unknown"

    def is_available(self) -> bool:
        raise NotImplementedError

    def require_available(self) -> None:
        if not self.is_available():
            raise RuntimeUnavailableError(
                f"Runtime '{self.name}' is not available on this host."
            )

    def device(self, index: int = 0):
        raise NotImplementedError

    def device_count(self) -> int:
        raise NotImplementedError

    def current_device(self) -> int:
        raise NotImplementedError

    def set_device(self, device) -> None:
        raise NotImplementedError

    def synchronize(self, device=None) -> None:
        raise NotImplementedError

    def empty_cache(self) -> None:
        return None

    def reset_peak_memory_stats(self, device=None) -> None:
        return None

    def manual_seed(self, seed: int) -> None:
        return None

    def device_name(self, device=None) -> str:
        return self.name

    def event(self, enable_timing: bool = True):
        raise NotImplementedError

    def move_tensor(self, tensor: Any, device, dtype=None):
        kwargs = {"device": device}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return tensor.to(**kwargs)

    def visible_devices_env(self, index: int) -> tuple[str, str]:
        """Return (environment variable, value) for a single-device worker."""
        raise NotImplementedError
