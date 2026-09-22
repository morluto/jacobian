"""Deterministic exact path-and-cycle decomposition of sparse flow tensors."""

from __future__ import annotations

from collections import deque
from fractions import Fraction
from itertools import pairwise

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
)
from jacobian.canonical import format_canonical_integer
from jacobian.math.graphs.flows.multicommodity._models import (
    FlowCycleTerm,
    FlowPathTerm,
    MulticommodityFlow,
)


def _rational_sides(value: Fraction) -> tuple[int, int]:
    return (
        len(format_canonical_integer(abs(value.numerator))),
        len(format_canonical_integer(value.denominator)),
    )


_MAX_INTERMEDIATE_DIGITS = 2 * MAX_CANONICAL_RATIONAL_DIGITS + 8


def _add(left: Fraction, right: Fraction) -> Fraction:
    """Add divergence terms only after bounding cross-denominator growth."""

    left_num, left_den = _rational_sides(left)
    right_num, right_den = _rational_sides(right)
    if (
        max(left_num + right_den, right_num + left_den) + 1 > _MAX_INTERMEDIATE_DIGITS
        or left_den + right_den > _MAX_INTERMEDIATE_DIGITS
    ):
        raise OverflowError("decomposition rational addition exceeds its envelope")
    value = left + right
    num_digits, den_digits = _rational_sides(value)
    if max(num_digits, den_digits) > _MAX_INTERMEDIATE_DIGITS:
        raise OverflowError("decomposition divergence exceeds its envelope")
    return value


def _subtract(
    left: Fraction,
    right: Fraction,
    *,
    max_result_digits: int = MAX_CANONICAL_RATIONAL_DIGITS,
) -> Fraction:
    """Subtract after bounding the unreduced and reduced intermediate."""

    left_num, left_den = _rational_sides(left)
    right_num, right_den = _rational_sides(right)
    if (
        max(left_num + right_den, right_num + left_den) + 1 > _MAX_INTERMEDIATE_DIGITS
        or left_den + right_den > _MAX_INTERMEDIATE_DIGITS
    ):
        raise OverflowError("decomposition rational subtraction exceeds its envelope")
    value = left - right
    num_digits, den_digits = _rational_sides(value)
    if max(num_digits, den_digits) > max_result_digits:
        raise OverflowError("decomposition residual exceeds its envelope")
    return value


def _path_from_predecessors(
    predecessors: dict[int, int | None], end: int
) -> tuple[int, ...]:
    trail: list[int] = []
    vertex: int | None = end
    while vertex is not None:
        trail.append(vertex)
        vertex = predecessors[vertex]
    trail.reverse()
    return tuple(trail)


def _find_path_to_negative(
    start: int,
    residual: dict[tuple[int, int], Fraction],
    divergence: dict[int, Fraction],
    adjacency: dict[int, tuple[int, ...]],
) -> tuple[int, ...] | None:
    """Return a deterministic shortest residual path to a deficit vertex.

    A global visited set and predecessor map make this traversal O(V + E).
    The previous recursive simple-path search could enumerate exponentially
    many trails before discovering a reachable deficit.
    """

    queue: deque[int] = deque((start,))
    predecessors: dict[int, int | None] = {start: None}
    while queue:
        vertex = queue.popleft()
        if vertex != start and divergence[vertex] < 0:
            return _path_from_predecessors(predecessors, vertex)
        for target in adjacency.get(vertex, ()):
            if target in predecessors or residual[(vertex, target)] <= 0:
                continue
            predecessors[target] = vertex
            queue.append(target)
    return None


def _find_cycle(
    first_edge: tuple[int, int],
    residual: dict[tuple[int, int], Fraction],
    adjacency: dict[int, tuple[int, ...]],
) -> tuple[int, ...] | None:
    """Return a deterministic shortest residual cycle containing ``first_edge``."""

    start, first_target = first_edge
    if first_target == start:
        return (start, start)

    queue: deque[int] = deque((first_target,))
    predecessors: dict[int, int | None] = {start: None, first_target: start}
    while queue:
        vertex = queue.popleft()
        for target in adjacency.get(vertex, ()):
            if residual[(vertex, target)] <= 0:
                continue
            if target == start:
                return (*_path_from_predecessors(predecessors, vertex), start)
            if target in predecessors:
                continue
            predecessors[target] = vertex
            queue.append(target)
    return None


def _decompose_one(
    flow: MulticommodityFlow,
    commodity_id: str,
) -> tuple[list[FlowPathTerm], list[FlowCycleTerm]]:
    edge_keys = tuple((edge.source, edge.target) for edge in flow.network.edges)
    adjacency: dict[int, tuple[int, ...]] = {}
    for source, target in edge_keys:
        adjacency[source] = (*adjacency.get(source, ()), target)
    adjacency = {
        vertex: tuple(sorted(targets)) for vertex, targets in adjacency.items()
    }
    residual = {edge: Fraction(0) for edge in edge_keys}
    for entry in flow.entries:
        if entry.commodity_id == commodity_id:
            residual[(entry.source, entry.target)] = entry.amount.as_fraction()

    divergence = {vertex: Fraction(0) for vertex in range(flow.network.vertex_count)}
    for (source, target), amount in residual.items():
        divergence[source] = _add(divergence[source], amount)
        divergence[target] = _subtract(
            divergence[target], amount, max_result_digits=_MAX_INTERMEDIATE_DIGITS
        )

    paths: list[FlowPathTerm] = []
    while True:
        starts = tuple(
            vertex for vertex in sorted(divergence) if divergence[vertex] > 0
        )
        if not starts:
            break
        start = starts[0]
        path = _find_path_to_negative(start, residual, divergence, adjacency)
        if path is None:
            raise ValueError("the nonnegative tensor has no path/cycle decomposition")
        end = path[-1]
        amount = min(
            divergence[start],
            -divergence[end],
            *(residual[edge] for edge in pairwise(path)),
        )
        for edge in pairwise(path):
            residual[edge] = _subtract(residual[edge], amount)
        divergence[start] = _subtract(
            divergence[start], amount, max_result_digits=_MAX_INTERMEDIATE_DIGITS
        )
        divergence[end] = _subtract(
            divergence[end], -amount, max_result_digits=_MAX_INTERMEDIATE_DIGITS
        )
        paths.append(
            FlowPathTerm(
                commodity_id=commodity_id,
                vertices=path,
                amount=CanonicalRational.from_fraction(amount),
            )
        )

    if any(value != 0 for value in divergence.values()):
        raise ValueError("the tensor divergence does not balance")

    cycles: list[FlowCycleTerm] = []
    while True:
        remaining = tuple(edge for edge in edge_keys if residual[edge] > 0)
        if not remaining:
            break
        first_edge = min(remaining)
        cycle = _find_cycle(first_edge, residual, adjacency)
        if cycle is None:
            raise ValueError("the balanced residual has no directed cycle")
        amount = min(residual[edge] for edge in pairwise(cycle))
        for edge in pairwise(cycle):
            residual[edge] = _subtract(residual[edge], amount)
        cycles.append(
            FlowCycleTerm(
                commodity_id=commodity_id,
                vertices=cycle,
                amount=CanonicalRational.from_fraction(amount),
            )
        )
    return paths, cycles


def decompose_flow(
    flow: MulticommodityFlow,
) -> tuple[tuple[FlowPathTerm, ...], tuple[FlowCycleTerm, ...]]:
    """Decompose each commodity in canonical commodity and edge order."""

    paths: list[FlowPathTerm] = []
    cycles: list[FlowCycleTerm] = []
    for commodity in flow.commodities:
        commodity_paths, commodity_cycles = _decompose_one(flow, commodity.commodity_id)
        paths.extend(commodity_paths)
        cycles.extend(commodity_cycles)
    return tuple(paths), tuple(cycles)


__all__ = ["decompose_flow"]
