"""Affinity-aware shard balancing for timing-sharded pytest lanes.

Durations alone cannot keep shared setup (complete-runtime templates, SQLite
services, providers) co-located. This module owns product-test affinity packing.
Small affinity groups stay on one shard; oversized groups are duration-balanced
across shards so a single shared profile (for example composition) still splits.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AffinityNode:
    nodeid: str
    affinity: str
    duration: float


def balance_affinity_shards(
    nodes: tuple[AffinityNode, ...] | list[AffinityNode],
    *,
    shard_count: int,
) -> tuple[tuple[str, ...], ...]:
    """Return ``shard_count`` shards preferring affinity co-location.

    One- and two-node affinity groups stay together. Larger groups whose total
    duration is at most ``total / shard_count`` also stay on one shard. Only
    larger, dominant groups are packed node-by-node onto the lightest shards so
    a shared profile such as composition can still parallelize.
    """

    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if not nodes:
        return tuple(() for _ in range(shard_count))

    by_affinity: dict[str, list[AffinityNode]] = defaultdict(list)
    for node in nodes:
        key = node.affinity.strip() or "default"
        by_affinity[key].append(node)

    shards: list[list[str]] = [[] for _ in range(shard_count)]
    weights = [0.0] * shard_count
    total_weight = sum(max(node.duration, 0.0) for node in nodes)
    colocate_limit = total_weight / shard_count if shard_count else total_weight

    ordered_groups = sorted(
        by_affinity.items(),
        key=lambda item: (-sum(max(node.duration, 0.0) for node in item[1]), item[0]),
    )
    for _affinity, group in ordered_groups:
        group_weight = sum(max(node.duration, 0.0) for node in group)
        if len(group) <= 2 or group_weight <= colocate_limit:
            target = min(range(shard_count), key=lambda index: (weights[index], index))
            for node in sorted(group, key=lambda item: item.nodeid):
                shards[target].append(node.nodeid)
            weights[target] += group_weight
            continue
        for node in sorted(
            group, key=lambda item: (-max(item.duration, 0.0), item.nodeid)
        ):
            target = min(range(shard_count), key=lambda index: (weights[index], index))
            shards[target].append(node.nodeid)
            weights[target] += max(node.duration, 0.0)

    return tuple(tuple(shard) for shard in shards)


def affinity_for_nodeid(nodeid: str, *, suite: str) -> str:
    """Infer setup affinity from a historical duration node id and suite."""

    path = nodeid.split("::", 1)[0]
    if suite == "composition" or "/composition/" in path:
        return "complete-runtime"
    if "/boundary/storage/" in path or path.startswith("tests/boundary/storage"):
        return "sqlite"
    if "/boundary/mcp/" in path:
        return "mcp"
    if "/boundary/providers/lean/" in path:
        return "lean"
    if "/boundary/providers/" in path:
        return "provider"
    if "/domain/" in path and any(
        token in path
        for token in (
            "/finite/",
            "/matrix/",
            "/polynomial/",
            "/universal_algebra/",
            "/graph_symmetry/",
        )
    ):
        return "sqlite"
    return "default"


def inventory_rows_from_durations(
    durations: dict[str, float],
    *,
    suite: str,
) -> list[dict[str, object]]:
    """Build inventory rows for affinity packing from a durations map."""

    return [
        {
            "nodeid": nodeid,
            "setup_affinity": [affinity_for_nodeid(nodeid, suite=suite)],
        }
        for nodeid in sorted(durations)
    ]


def affinity_index_from_inventory(
    inventory_rows: list[dict[str, object]],
    durations: dict[str, float],
) -> tuple[AffinityNode, ...]:
    """Build affinity nodes from resource-inventory rows and duration maps."""

    nodes: list[AffinityNode] = []
    for row in inventory_rows:
        nodeid = str(row.get("nodeid") or "")
        if not nodeid:
            continue
        affinities = row.get("setup_affinity") or ()
        if isinstance(affinities, str):
            affinity = affinities
        elif isinstance(affinities, (list, tuple)) and affinities:
            affinity = str(affinities[0])
        else:
            affinity = "default"
        nodes.append(
            AffinityNode(
                nodeid=nodeid,
                affinity=affinity,
                duration=float(durations.get(nodeid, 1.0)),
            )
        )
    return tuple(nodes)


def assign_collected_to_shards(
    collected_nodeids: tuple[str, ...] | list[str],
    *,
    suite: str,
    shard_count: int,
    durations: dict[str, float] | None = None,
) -> tuple[tuple[str, ...], ...]:
    """Partition currently collectable node IDs with affinity-aware balancing.

    Historical durations bias weights; unknown nodes default to ``1.0`` seconds
    and still receive an inferred affinity so new tests are not dropped.
    """

    duration_map = dict(durations or {})
    nodes = tuple(
        AffinityNode(
            nodeid=nodeid,
            affinity=affinity_for_nodeid(nodeid, suite=suite),
            duration=float(duration_map.get(nodeid, 1.0)),
        )
        for nodeid in collected_nodeids
    )
    return balance_affinity_shards(nodes, shard_count=shard_count)


__all__ = [
    "AffinityNode",
    "affinity_for_nodeid",
    "affinity_index_from_inventory",
    "assign_collected_to_shards",
    "balance_affinity_shards",
    "inventory_rows_from_durations",
]
