"""Factorization-family operations for numerical semigroups."""

from __future__ import annotations

import networkx as _nx

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.numerical_semigroups._algorithms import (
    factorization_lengths,
    factorizations,
    minimal_generating_system,
)
from jacobian.math.numerical_semigroups._models import (
    FactorizationComputeRequest,
    FactorizationComputeResult,
    FactorizationDistanceRequest,
    FactorizationDistanceResult,
    FactorizationGraphComputeRequest,
    FactorizationGraphComputeResult,
    FactorizationLengthsComputeRequest,
    FactorizationLengthsComputeResult,
)


def _minimal_generators(generators: tuple[str, ...]) -> list[int]:
    """Return the increasing minimal atom axis as native integers."""

    raw = tuple(
        sorted({parse_canonical_integer(generator) for generator in generators})
    )
    return list(minimal_generating_system(raw))


def _enumerate_factorizations(
    generators: list[int], value: int
) -> list[tuple[int, ...]]:
    """Materialize the admitted complete factorization family."""

    return list(factorizations(tuple(generators), value))


def _gcd_factor(first: tuple[int, ...], second: tuple[int, ...]) -> tuple[int, ...]:
    """Return the coordinate-wise greatest common divisor."""

    return tuple(min(left, right) for left, right in zip(first, second, strict=True))


def _factorization_distance(first: tuple[int, ...], second: tuple[int, ...]) -> int:
    """Compute the standard distance between two factorizations."""

    if not first or not second:
        return 0
    common = _gcd_factor(first, second)
    return max(sum(first) - sum(common), sum(second) - sum(common))


def _build_factorization_graph(
    family: list[tuple[int, ...]],
) -> tuple[list[tuple[int, int]], list[list[int]], bool]:
    """Build the standard shared-support factorization graph."""

    if not family:
        return [], [], True
    graph: _nx.Graph[int] = _nx.Graph()
    graph.add_nodes_from(range(len(family)))
    edges = [
        (left, right)
        for left in range(len(family))
        for right in range(left + 1, len(family))
        if sum(_gcd_factor(family[left], family[right])) > 0
    ]
    graph.add_edges_from(edges)
    components = [list(component) for component in _nx.connected_components(graph)]
    return edges, components, len(components) <= 1


def compute_factorizations(
    request: FactorizationComputeRequest,
) -> FactorizationComputeResult:
    """Compute the complete admitted factorization family Z(s)."""

    generators = _minimal_generators(request.generators)
    value = parse_canonical_integer(request.value)
    minimal_generators = tuple(
        format_canonical_integer(generator) for generator in generators
    )
    if value < 0:
        return FactorizationComputeResult(
            value=request.value,
            minimal_generators=minimal_generators,
            in_semigroup=False,
            factorizations=(),
        )
    family = _enumerate_factorizations(generators, value)
    return FactorizationComputeResult(
        value=request.value,
        minimal_generators=minimal_generators,
        in_semigroup=bool(family),
        factorizations=tuple(family),
    )


def compute_factorization_lengths(
    request: FactorizationLengthsComputeRequest,
) -> FactorizationLengthsComputeResult:
    """Compute the complete admitted factorization-length set."""

    generators = _minimal_generators(request.generators)
    value = parse_canonical_integer(request.value)
    minimal_generators = tuple(
        format_canonical_integer(generator) for generator in generators
    )
    if value < 0:
        return FactorizationLengthsComputeResult(
            value=request.value,
            minimal_generators=minimal_generators,
            in_semigroup=False,
            lengths=(),
        )
    lengths = factorization_lengths(tuple(generators), value)
    return FactorizationLengthsComputeResult(
        value=request.value,
        minimal_generators=minimal_generators,
        in_semigroup=bool(lengths),
        lengths=lengths,
    )


def compute_factorization_distance(
    request: FactorizationDistanceRequest,
) -> FactorizationDistanceResult:
    """Compute the standard distance between two admitted factorizations."""

    first = tuple(request.first)
    second = tuple(request.second)
    return FactorizationDistanceResult(
        value=request.value,
        distance=_factorization_distance(first, second),
        first_length=sum(first),
        second_length=sum(second),
    )


def compute_factorization_graph(
    request: FactorizationGraphComputeRequest,
) -> FactorizationGraphComputeResult:
    """Compute the exact shared-support graph of an admitted family."""

    generators = _minimal_generators(request.generators)
    value = parse_canonical_integer(request.value)
    minimal_generators = tuple(
        format_canonical_integer(generator) for generator in generators
    )
    if value < 0:
        return FactorizationGraphComputeResult(
            value=request.value,
            minimal_generators=minimal_generators,
            in_semigroup=False,
            factorizations=(),
            edges=(),
            connected_components=(),
            is_connected=True,
        )
    family = _enumerate_factorizations(generators, value)
    edges, components, connected = _build_factorization_graph(family)
    return FactorizationGraphComputeResult(
        value=request.value,
        minimal_generators=minimal_generators,
        in_semigroup=bool(family),
        factorizations=tuple(family),
        edges=tuple(edges),
        connected_components=tuple(
            tuple(sorted(component)) for component in components
        ),
        is_connected=connected,
    )


__all__ = [
    "compute_factorization_distance",
    "compute_factorization_graph",
    "compute_factorization_lengths",
    "compute_factorizations",
]
