"""Unit tests for affinity-aware shard balancing."""

from __future__ import annotations

from tools.test_plan.affinity import (
    AffinityNode,
    affinity_index_from_inventory,
    balance_affinity_shards,
)


def test_affinity_groups_prefer_shared_setup_colocation() -> None:
    nodes = (
        AffinityNode("a::t1", "sqlite", 10.0),
        AffinityNode("a::t2", "sqlite", 9.0),
        AffinityNode("b::t1", "complete-runtime", 8.0),
        AffinityNode("b::t2", "complete-runtime", 7.0),
        AffinityNode("c::t1", "default", 1.0),
    )
    shards = balance_affinity_shards(nodes, shard_count=2)
    flat = {nodeid: index for index, shard in enumerate(shards) for nodeid in shard}
    assert flat["a::t1"] == flat["a::t2"]
    assert flat["b::t1"] == flat["b::t2"]


def test_affinity_index_from_inventory_defaults_missing_affinity() -> None:
    nodes = affinity_index_from_inventory(
        [{"nodeid": "x::test", "setup_affinity": []}],
        {"x::test": 2.5},
    )
    assert nodes == (AffinityNode("x::test", "default", 2.5),)
