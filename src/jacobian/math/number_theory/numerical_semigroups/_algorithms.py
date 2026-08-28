"""Exact bounded algorithms shared by numerical-semigroup contracts and tools.

Betti candidates and minimal-presentation witnesses use the generator-graph
method of Rosales (International Journal of Algebra and Computation 6 (1996),
441--455). The global delta sweep follows the maintained GAP NumericalSgps
``DeltaSetPeriodicityBoundForNumericalSemigroup`` construction.
"""

from __future__ import annotations

from fractions import Fraction
from heapq import heappop, heappush
from itertools import pairwise
from math import gcd


def minimal_generating_system(generators: tuple[int, ...]) -> tuple[int, ...]:
    """Return the increasing minimal generating system for ``generators``."""
    values = tuple(sorted(set(generators)))
    # A presentation containing 1 is the free numerical semigroup N.  This
    # avoids allocating up to a redundant raw generator when the canonical
    # axis is already known to be the singleton (1,).
    if 1 in values:
        return (1,)
    minimal: list[int] = []
    for generator in values:
        reachable = [False] * (generator + 1)
        reachable[0] = True
        for value in range(1, generator + 1):
            reachable[value] = any(
                value >= other and reachable[value - other]
                for other in values
                if other != generator
            )
        if not reachable[generator]:
            minimal.append(generator)
    return tuple(minimal)


def apery_set(generators: tuple[int, ...]) -> tuple[int, ...]:
    """Compute ``Ap(S, m)`` by shortest paths on residues modulo ``m``."""
    multiplicity = generators[0]
    distances: list[int | None] = [None] * multiplicity
    distances[0] = 0
    queue = [(0, 0)]
    while queue:
        distance, residue = heappop(queue)
        if distances[residue] != distance:
            continue
        for generator in generators:
            candidate = distance + generator
            candidate_residue = candidate % multiplicity
            previous = distances[candidate_residue]
            if previous is None or candidate < previous:
                distances[candidate_residue] = candidate
                heappush(queue, (candidate, candidate_residue))
    if any(value is None for value in distances):
        raise ValueError("generators do not define a numerical semigroup")
    return tuple(value for value in distances if value is not None)


def belongs(value: int, apery: tuple[int, ...]) -> bool:
    """Decide membership from an Apéry set."""
    return value >= 0 and value >= apery[value % len(apery)]


def factorization_count(generators: tuple[int, ...], target: int) -> int:
    """Count all nonnegative factorizations without materializing them."""
    if target < 0:
        return 0
    if generators == (1,):
        return 1
    counts = [0] * (target + 1)
    counts[0] = 1
    for generator in generators:
        for value in range(generator, target + 1):
            counts[value] += counts[value - generator]
    return counts[target]


def factorizations(
    generators: tuple[int, ...], target: int
) -> tuple[tuple[int, ...], ...]:
    """Exhaustively enumerate all factorizations in lexicographic order."""
    if target < 0:
        return ()
    if generators == (1,):
        return ((target,),)
    result: list[tuple[int, ...]] = []

    def visit(index: int, remainder: int, prefix: tuple[int, ...]) -> None:
        generator = generators[index]
        if index == len(generators) - 1:
            if remainder % generator == 0:
                result.append((*prefix, remainder // generator))
            return
        for coefficient in range(remainder // generator + 1):
            visit(
                index + 1,
                remainder - coefficient * generator,
                (*prefix, coefficient),
            )

    visit(0, target, ())
    return tuple(result)


def factorization_lengths(generators: tuple[int, ...], target: int) -> tuple[int, ...]:
    """Compute the complete factorization-length set by dynamic programming."""
    if target < 0:
        return ()
    if generators == (1,):
        return (target,)
    lengths: list[set[int]] = [set() for _ in range(target + 1)]
    lengths[0].add(0)
    for value in range(1, target + 1):
        for generator in generators:
            if value >= generator:
                lengths[value].update(
                    length + 1 for length in lengths[value - generator]
                )
    return tuple(sorted(lengths[target]))


def factorization_length_extrema(
    generators: tuple[int, ...], target: int
) -> tuple[int, int]:
    """Compute the minimum and maximum factorization lengths without sets."""
    if target < 0:
        raise ValueError("target must be non-negative")
    if generators == (1,):
        return target, target
    minimum: list[int | None] = [None] * (target + 1)
    maximum: list[int | None] = [None] * (target + 1)
    minimum[0] = maximum[0] = 0
    for value in range(1, target + 1):
        min_candidates: list[int] = []
        max_candidates: list[int] = []
        for generator in generators:
            predecessor = value - generator
            if predecessor >= 0:
                min_length = minimum[predecessor]
                max_length = maximum[predecessor]
                if min_length is not None and max_length is not None:
                    min_candidates.append(min_length)
                    max_candidates.append(max_length)
        if min_candidates:
            minimum[value] = min(min_candidates) + 1
            maximum[value] = max(max_candidates) + 1
    min_result = minimum[target]
    max_result = maximum[target]
    if min_result is None or max_result is None:
        raise ValueError("target is not in the numerical semigroup")
    return min_result, max_result


def catenary_degree_from_factorizations(
    family: tuple[tuple[int, ...], ...],
) -> int:
    """Compute the least distance threshold connecting a factorization family."""
    if len(family) <= 1:
        return 0
    parents = list(range(len(family)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    edges: list[tuple[int, int, int]] = []
    for left in range(len(family)):
        for right in range(left + 1, len(family)):
            common_length = sum(
                min(a, b) for a, b in zip(family[left], family[right], strict=True)
            )
            distance = max(
                sum(family[left]) - common_length,
                sum(family[right]) - common_length,
            )
            edges.append((distance, left, right))
    maximum = 0
    joined = 0
    for distance, left, right in sorted(edges):
        left_root = root(left)
        right_root = root(right)
        if left_root == right_root:
            continue
        parents[right_root] = left_root
        maximum = max(maximum, distance)
        joined += 1
        if joined == len(family) - 1:
            return maximum
    raise RuntimeError("complete factorization distance graph did not connect")


def factorization_predecessors(
    generators: tuple[int, ...], limit: int
) -> tuple[tuple[int, int] | None, ...]:
    """Build one deterministic predecessor chain for each reachable value."""
    predecessors: list[tuple[int, int] | None] = [None] * (limit + 1)
    reachable = [False] * (limit + 1)
    reachable[0] = True
    for value in range(1, limit + 1):
        for index, generator in enumerate(generators):
            if value >= generator and reachable[value - generator]:
                reachable[value] = True
                predecessors[value] = (value - generator, index)
                break
    return tuple(predecessors)


def reconstruct_factorization(
    generators: tuple[int, ...],
    predecessors: tuple[tuple[int, int] | None, ...],
    target: int,
) -> tuple[int, ...] | None:
    """Reconstruct one factorization from a shared predecessor table."""
    if target < 0 or target >= len(predecessors):
        return None
    if target and predecessors[target] is None:
        return None
    result = [0] * len(generators)
    value = target
    while value:
        predecessor = predecessors[value]
        if predecessor is None:
            raise RuntimeError("factorization predecessor chain is incomplete")
        value, index = predecessor
        result[index] += 1
    return tuple(result)


def generator_graph_components(
    generators: tuple[int, ...], apery: tuple[int, ...], value: int
) -> tuple[tuple[int, ...], ...]:
    """Return components of Rosales' graph associated to ``value``."""
    vertices = tuple(
        index
        for index, generator in enumerate(generators)
        if belongs(value - generator, apery)
    )
    adjacency: dict[int, set[int]] = {index: set() for index in vertices}
    for offset, left in enumerate(vertices):
        for right in vertices[offset + 1 :]:
            if belongs(value - generators[left] - generators[right], apery):
                adjacency[left].add(right)
                adjacency[right].add(left)
    components: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for vertex in vertices:
        if vertex in seen:
            continue
        pending = [vertex]
        seen.add(vertex)
        component: list[int] = []
        while pending:
            current = pending.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current], reverse=True):
                if neighbor not in seen:
                    seen.add(neighbor)
                    pending.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(components)


def betti_data(
    generators: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], dict[int, tuple[tuple[int, ...], ...]]]:
    """Return Apéry set, complete candidate set, and disconnected candidates."""
    if generators == (1,):
        return (0,), (), {}
    apery = apery_set(generators)
    candidates = tuple(
        sorted(
            {
                apery_value + generator
                for apery_value in apery[1:]
                for generator in generators
            }
        )
    )
    disconnected = {
        candidate: components
        for candidate in candidates
        if len(components := generator_graph_components(generators, apery, candidate))
        > 1
    }
    return apery, candidates, disconnected


def delta_periodicity_bound(generators: tuple[int, ...]) -> int:
    """Return the García-García--Moreno-Frías--Vigneron-Tenorio bound."""
    if generators == (1,):
        return 0
    if len(generators) <= 2:
        return generators[0] * generators[-1]
    differences = tuple(right - left for left, right in pairwise(generators))
    delta_gcd = 0
    for difference in differences:
        delta_gcd = gcd(delta_gcd, difference)
    first = generators[0]
    second = generators[1]
    penultimate = generators[-2]
    last = generators[-1]
    embedding_adjustment = len(generators) - 2
    bounds: list[int] = []
    for middle in generators[1:-1]:
        local_gcd = gcd(gcd(abs(middle - first), abs(first - last)), abs(last - middle))
        left = Fraction(
            -second
            * (
                first * delta_gcd * local_gcd
                + embedding_adjustment * (first - middle) * (first - last)
            ),
            (first - second) * local_gcd,
        )
        right = Fraction(
            penultimate
            * (
                embedding_adjustment * (first - last) * (last - middle)
                - delta_gcd * last * local_gcd
            ),
            (penultimate - last) * local_gcd,
        )
        bounds.extend((_ceil_fraction(left), _ceil_fraction(right)))
    return max(bounds)


def _ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


__all__ = [
    "apery_set",
    "belongs",
    "betti_data",
    "catenary_degree_from_factorizations",
    "delta_periodicity_bound",
    "factorization_count",
    "factorization_length_extrema",
    "factorization_lengths",
    "factorization_predecessors",
    "factorizations",
    "generator_graph_components",
    "minimal_generating_system",
    "reconstruct_factorization",
]
