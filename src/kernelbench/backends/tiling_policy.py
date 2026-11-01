"""Automatic tiling policy inference from BoundaryContract.

Infers block_dim, workspace, and tiling fields based on tensor shapes,
operator semantics, and available hardware resources.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .boundary import BoundaryContract


@dataclass
class TilingDecision:
    """Result of tiling policy inference."""
    block_dim: str  # e.g. "literal(20)"
    workspace: str  # e.g. "literal(0)" or "expression(...)"
    tiling: dict[str, dict[str, Any]]  # Additional fields for kernel


class TilingPolicy:
    """Infers execution parameters from boundary contract."""
    
    def __init__(self, max_cores: int = 20):
        """
        Args:
            max_cores: Maximum AI cores available (Ascend 910B3 has 20)
        """
        self.max_cores = max_cores
    
    def infer(self, contract: BoundaryContract) -> TilingDecision:
        """Infer tiling parameters from boundary contract.
        
        Strategy:
        1. Compute total work size from input shapes
        2. Choose block_dim based on work size and max_cores
        3. Infer workspace requirements (0 for element-wise ops)
        4. Generate tiling struct fields (numel, dims, attributes)
        """
        if contract.status != "supported":
            return self._fallback_decision(contract)
        
        # Compute total work size (use largest input tensor)
        max_numel = 0
        for item in contract.inputs:
            numel = 1
            for dim in item.shape:
                numel *= dim
            max_numel = max(max_numel, numel)
        
        block_dim = self._choose_block_dim(max_numel)
        workspace = "literal(0)"
        
        # Generate tiling fields
        tiling: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(contract.inputs):
            tiling[f"input{index}_numel"] = {
                "c_type": "uint64_t",
                "expression": f"input[{index}].numel()"
            }
            for dim in range(item.rank):
                tiling[f"input{index}_dim{dim}"] = {
                    "c_type": "uint64_t",
                    "expression": f"input[{index}].dim({dim})"
                }
        
        for item in contract.attributes:
            tiling[item.name] = {
                "c_type": item.c_type,
                "expression": f"attr(\"{item.name}\")"
            }
        
        return TilingDecision(
            block_dim=block_dim,
            workspace=workspace,
            tiling=tiling
        )
    
    def _choose_block_dim(self, numel: int) -> str:
        """Choose number of cores based on work size.
        
        Heuristic:
        - numel < 100K: single core (overhead not worth it)
        - numel < 1M: use min(8, max_cores)
        - numel < 10M: use min(16, max_cores)
        - numel >= 10M: use all max_cores
        """
        if numel < 100_000:
            cores = 1
        elif numel < 1_000_000:
            cores = min(8, self.max_cores)
        elif numel < 10_000_000:
            cores = min(16, self.max_cores)
        else:
            cores = self.max_cores
        
        return f"literal({cores})"
    
    def _fallback_decision(self, contract: BoundaryContract) -> TilingDecision:
        """Fallback for unsupported contracts (single-core)."""
        tiling: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(contract.inputs):
            tiling[f"input{index}_numel"] = {
                "c_type": "uint64_t",
                "expression": f"input[{index}].numel()"
            }
            for dim in range(item.rank):
                tiling[f"input{index}_dim{dim}"] = {
                    "c_type": "uint64_t",
                    "expression": f"input[{index}].dim({dim})"
                }
        
        return TilingDecision(
            block_dim="literal(1)",
            workspace="literal(0)",
            tiling=tiling
        )
