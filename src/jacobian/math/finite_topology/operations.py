"""Exact native kernels for finite topological spaces."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

from jacobian.math.finite_topology.values import (
    BeatPointWitness,
    FiniteTopology,
    PointMap,
)


@dataclass(frozen=True, slots=True)
class ContinuityAnalysis:
    is_continuous: bool
    violating_open_set: tuple[int, ...] | None
    violating_preimage: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class BeatPointAnalysis:
    down_beat_points: tuple[BeatPointWitness, ...]
    up_beat_points: tuple[BeatPointWitness, ...]


def _open_sets(topology: FiniteTopology) -> set[frozenset[int]]:
    return {frozenset(open_set) for open_set in topology.open_sets}


def _subset(topology: FiniteTopology, values: Iterable[int]) -> frozenset[int]:
    result = frozenset(values)
    if any(not 0 <= point < topology.point_count for point in result):
        raise ValueError("subset point is outside the topology carrier")
    return result


def specialization_preorder(
    topology: FiniteTopology,
) -> tuple[tuple[bool, ...], ...]:
    """Return ``relation[x][y]`` iff ``x`` lies in the closure of ``{y}``."""
    containing = [
        tuple(
            frozenset(open_set) for open_set in topology.open_sets if point in open_set
        )
        for point in range(topology.point_count)
    ]
    return tuple(
        tuple(
            all(upper in open_set for open_set in containing[lower])
            for upper in range(topology.point_count)
        )
        for lower in range(topology.point_count)
    )


def minimal_open_neighborhoods(
    topology: FiniteTopology,
) -> tuple[frozenset[int], ...]:
    return tuple(
        frozenset.intersection(
            *(
                frozenset(open_set)
                for open_set in topology.open_sets
                if point in open_set
            )
        )
        for point in range(topology.point_count)
    )


def closure(topology: FiniteTopology, subset: Iterable[int]) -> frozenset[int]:
    selected = _subset(topology, subset)
    relation = specialization_preorder(topology)
    return frozenset(
        point
        for point in range(topology.point_count)
        if any(relation[point][source] for source in selected)
    )


def interior(topology: FiniteTopology, subset: Iterable[int]) -> frozenset[int]:
    selected = _subset(topology, subset)
    neighborhoods = minimal_open_neighborhoods(topology)
    return frozenset(
        point
        for point, neighborhood in enumerate(neighborhoods)
        if neighborhood <= selected
    )


def connected_components(topology: FiniteTopology) -> tuple[tuple[int, ...], ...]:
    relation = specialization_preorder(topology)
    visited: set[int] = set()
    components: list[tuple[int, ...]] = []
    for start in range(topology.point_count):
        if start in visited:
            continue
        queue = deque([start])
        visited.add(start)
        component: list[int] = []
        while queue:
            point = queue.popleft()
            component.append(point)
            for neighbor in range(topology.point_count):
                if neighbor not in visited and (
                    relation[point][neighbor] or relation[neighbor][point]
                ):
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(components)


def continuity(
    domain: FiniteTopology, codomain: FiniteTopology, point_map: PointMap
) -> ContinuityAnalysis:
    if point_map.domain_point_count != domain.point_count:
        raise ValueError("map domain size does not match the domain topology")
    if point_map.codomain_point_count != codomain.point_count:
        raise ValueError("map codomain size does not match the codomain topology")
    domain_opens = _open_sets(domain)
    for open_set in codomain.open_sets:
        target_open = frozenset(open_set)
        preimage = tuple(
            point
            for point, target in enumerate(point_map.values)
            if target in target_open
        )
        if frozenset(preimage) not in domain_opens:
            return ContinuityAnalysis(False, open_set, preimage)
    return ContinuityAnalysis(True, None, None)


def is_continuous(
    domain: FiniteTopology, codomain: FiniteTopology, point_map: PointMap
) -> bool:
    return continuity(domain, codomain, point_map).is_continuous


def is_t0(topology: FiniteTopology) -> bool:
    relation = specialization_preorder(topology)
    return all(
        not (relation[left][right] and relation[right][left])
        for left in range(topology.point_count)
        for right in range(left + 1, topology.point_count)
    )


def _unique_extremum(
    points: tuple[int, ...],
    relation: tuple[tuple[bool, ...], ...],
    *,
    minimum: bool,
) -> int | None:
    candidates = tuple(
        candidate
        for candidate in points
        if all(
            relation[candidate][other] if minimum else relation[other][candidate]
            for other in points
        )
    )
    return candidates[0] if len(candidates) == 1 else None


def beat_points(topology: FiniteTopology) -> BeatPointAnalysis:
    if not is_t0(topology):
        raise ValueError("beat points require a T0 finite topology")
    relation = specialization_preorder(topology)
    down: list[BeatPointWitness] = []
    up: list[BeatPointWitness] = []
    for point in range(topology.point_count):
        below = tuple(
            other
            for other in range(topology.point_count)
            if other != point and relation[other][point]
        )
        above = tuple(
            other
            for other in range(topology.point_count)
            if other != point and relation[point][other]
        )
        down_witness = _unique_extremum(below, relation, minimum=False)
        if down_witness is not None:
            down.append(BeatPointWitness(point=point, witness=down_witness))
        up_witness = _unique_extremum(above, relation, minimum=True)
        if up_witness is not None:
            up.append(BeatPointWitness(point=point, witness=up_witness))
    return BeatPointAnalysis(tuple(down), tuple(up))


__all__ = [
    "BeatPointAnalysis",
    "ContinuityAnalysis",
    "beat_points",
    "closure",
    "connected_components",
    "continuity",
    "interior",
    "is_continuous",
    "is_t0",
    "minimal_open_neighborhoods",
    "specialization_preorder",
]
