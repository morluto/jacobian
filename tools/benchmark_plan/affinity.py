"""Re-export product affinity packing for Harbor timing helpers."""

from __future__ import annotations

from tools.test_plan.affinity import (
    AffinityNode,
    affinity_index_from_inventory,
    balance_affinity_shards,
)

__all__ = [
    "AffinityNode",
    "affinity_index_from_inventory",
    "balance_affinity_shards",
]
