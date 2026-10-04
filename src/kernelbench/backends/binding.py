"""Binding/OPP loading hooks kept separate from candidate source."""

from __future__ import annotations

import os
from pathlib import Path


def private_opp_environment(
    opp_root: str | Path,
    binding_root: str | Path,
    system_opp: str | Path = "/usr/local/Ascend/cann-9.0.1/opp",
) -> dict[str, str]:
    """Return runtime environment additions without hiding built-in operators."""
    opp = Path(opp_root).resolve()
    vendor = opp / "vendors" / "customize"
    binding = str(Path(binding_root).resolve())
    env = {
        "ASCEND_OPP_PATH": str(Path(system_opp).resolve()),
        "ASCEND_CUSTOM_OPP_PATH": str(vendor),
        "LD_LIBRARY_PATH": os.pathsep.join(
            [binding, str(vendor / "op_api" / "lib"), os.environ.get("LD_LIBRARY_PATH", "")]
        ).rstrip(os.pathsep),
    }
    return env
