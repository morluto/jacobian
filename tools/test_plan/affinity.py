"""Affinity-aware shard balancing for timing-sharded pytest lanes.

Durations alone cannot keep shared setup (complete-runtime templates, SQLite
services, providers) co-located. This module owns product-test affinity packing;
Harbor timing helpers may re-export it. Partition node IDs by setup affinity
first, then greedily pack each affinity group onto shards.
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
    """Return ``shard_count`` shards of node IDs preferring affinity co-location.

    Within each affinity group, nodes are packed onto the current lightest shard
    (least total duration). Empty affinities are treated as ``"default"``.
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

    # Larger affinity groups first so shared setup dominates placement.
    # Place each affinity group onto one shard so shared setup stays co-located.
    ordered_groups = sorted(
        by_affinity.items(),
        key=lambda item: (-sum(node.duration for node in item[1]), item[0]),
    )
    for _affinity, group in ordered_groups:
        group_weight = sum(max(node.duration, 0.0) for node in group)
        target = min(range(shard_count), key=lambda index: (weights[index], index))
        for node in sorted(group, key=lambda item: item.nodeid):
            shards[target].append(node.nodeid)
        weights[target] += group_weight

    return tuple(tuple(shard) for shard in shards)


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


__all__ = [
    "AffinityNode",
    "affinity_index_from_inventory",
    "balance_affinity_shards",
]
