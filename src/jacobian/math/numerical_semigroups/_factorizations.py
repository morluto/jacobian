"""Factorization-family contracts and operations for numerical semigroups."""

from __future__ import annotations

from typing import Self

import networkx as _nx
from pydantic import Field, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.numerical_semigroups._algorithms import (
    factorization_count,
    factorization_lengths,
    factorizations,
    minimal_generating_system,
)
from jacobian.math.numerical_semigroups._models import (
    MAX_GENERATOR,
    MAX_GENERATORS,
    MAX_GRAPH_FACTORIZATIONS,
    MAX_MATERIALIZED_FACTORIZATIONS,
    _require_bounded_value,
    _require_canonical_minimal_axis,
    _require_materializable_factorizations,
    _require_minimal_generators,
    _validation_error,
)


def _require_exact_factorization_family(
    generators: tuple[int, ...],
    value: int,
    family: tuple[tuple[int, ...], ...],
) -> None:
    """Replay one complete factorization family on its canonical axis."""

    if len(set(family)) != len(family):
        raise _validation_error("factorizations must be unique")
    for factorization in family:
        if len(factorization) != len(generators) or any(
            coordinate < 0 for coordinate in factorization
        ):
            raise _validation_error("factorization has invalid coordinates")
        degree = sum(
            coordinate * generator
            for coordinate, generator in zip(factorization, generators, strict=True)
        )
        if degree != value:
            raise _validation_error(
                "factorization does not evaluate to the result value"
            )
    if len(family) != factorization_count(generators, value):
        raise _validation_error("factorizations do not form the complete family")


def _factorization_graph_data(
    family: tuple[tuple[int, ...], ...],
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, ...], ...]]:
    """Construct the canonical shared-support graph representation."""

    edges = tuple(
        (left, right)
        for left in range(len(family))
        for right in range(left + 1, len(family))
        if any(
            a > 0 and b > 0 for a, b in zip(family[left], family[right], strict=True)
        )
    )
    unseen = set(range(len(family)))
    components: list[tuple[int, ...]] = []
    while unseen:
        reached = {min(unseen)}
        while True:
            expanded = reached | {
                right if left in reached else left
                for left, right in edges
                if left in reached or right in reached
            }
            if expanded == reached:
                break
            reached = expanded
        unseen.difference_update(reached)
        components.append(tuple(sorted(reached)))
    return edges, tuple(components)


class FactorizationComputeRequest(StrictModel):
    """Compute the complete factorization family Z(s) for one element."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            f"Positive generators with gcd 1, each at most {MAX_GENERATOR}. "
            "The presentation may be reordered or redundant; returned "
            "coordinates use its increasing minimal generator axis."
        ),
    )
    value: CanonicalInteger

    @model_validator(mode="after")
    def require_complete_materialization(self) -> Self:
        generators = _require_minimal_generators(self.generators)
        value = _require_bounded_value(self.value)
        _require_materializable_factorizations(
            generators, value, MAX_MATERIALIZED_FACTORIZATIONS
        )
        return self


class FactorizationComputeResult(StrictModel):
    """Complete factorization family Z(s) for one element."""

    value: CanonicalInteger
    minimal_generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    in_semigroup: bool
    factorizations: tuple[tuple[int, ...], ...]

    @model_validator(mode="after")
    def require_exact_family(self) -> Self:
        generators = _require_canonical_minimal_axis(self.minimal_generators)
        value = parse_canonical_integer(self.value)
        if self.in_semigroup != bool(self.factorizations):
            raise _validation_error(
                "membership must agree with the factorization family"
            )
        _require_exact_factorization_family(generators, value, self.factorizations)
        return self


class FactorizationLengthsComputeRequest(StrictModel):
    """Compute the complete sorted length set of one element."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            f"Positive generators with gcd 1, each at most {MAX_GENERATOR}. "
            "The presentation may be reordered or redundant; factorization "
            "lengths use its increasing minimal generator axis."
        ),
    )
    value: CanonicalInteger

    @model_validator(mode="after")
    def require_positive_generators_and_bounded_value(self) -> Self:
        _require_minimal_generators(self.generators)
        _require_bounded_value(self.value)
        return self


class FactorizationLengthsComputeResult(StrictModel):
    """Sorted length set of one element."""

    value: CanonicalInteger
    minimal_generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    in_semigroup: bool
    lengths: tuple[int, ...]

    @model_validator(mode="after")
    def require_length_set(self) -> Self:
        generators = _require_canonical_minimal_axis(self.minimal_generators)
        value = parse_canonical_integer(self.value)
        if self.in_semigroup != bool(self.lengths):
            raise _validation_error(
                "membership must agree with the factorization lengths"
            )
        if self.lengths != tuple(sorted(set(self.lengths))):
            raise _validation_error(
                "lengths must be strictly increasing and duplicate-free"
            )
        if any(length < 0 for length in self.lengths):
            raise _validation_error("factorization lengths must be non-negative")
        if self.lengths != factorization_lengths(generators, value):
            raise _validation_error("lengths do not form the complete length set")
        return self


class FactorizationDistanceRequest(StrictModel):
    """Distance between two factorizations of the same element."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            f"Positive generators with gcd 1, each at most {MAX_GENERATOR}. "
            "The presentation may be reordered or redundant, but both "
            "factorization coordinate tuples must use its increasing minimal "
            "generator axis."
        ),
    )
    value: CanonicalInteger
    first: tuple[int, ...] = Field(min_length=1)
    second: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid_request(self) -> Self:
        generators = _require_minimal_generators(self.generators)
        value = _require_bounded_value(self.value)
        if any(c < 0 for c in self.first) or any(c < 0 for c in self.second):
            raise _validation_error("factorization coordinates must be non-negative")
        if len(self.first) != len(generators) or len(self.second) != len(generators):
            raise _validation_error(
                "factorization coordinates must match the minimal generating system"
            )
        first_value = sum(
            coefficient * generator
            for coefficient, generator in zip(self.first, generators, strict=True)
        )
        second_value = sum(
            coefficient * generator
            for coefficient, generator in zip(self.second, generators, strict=True)
        )
        if first_value != value or second_value != value:
            raise _validation_error(
                "both factorizations must evaluate to the declared value"
            )
        return self


class FactorizationDistanceResult(StrictModel):
    """Distance between two factorizations."""

    value: CanonicalInteger
    distance: int = Field(ge=0)
    first_length: int = Field(ge=0)
    second_length: int = Field(ge=0)


class FactorizationGraphComputeRequest(StrictModel):
    """Compute the standard factorization graph of one element."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            f"Positive generators with gcd 1, each at most {MAX_GENERATOR}. "
            "The presentation may be reordered or redundant; graph vertices use "
            "its increasing minimal generator axis."
        ),
    )
    value: CanonicalInteger

    @model_validator(mode="after")
    def require_complete_materialization(self) -> Self:
        generators = _require_minimal_generators(self.generators)
        value = _require_bounded_value(self.value)
        _require_materializable_factorizations(
            generators, value, MAX_GRAPH_FACTORIZATIONS
        )
        return self


class FactorizationGraphComputeResult(StrictModel):
    """Standard factorization graph with connected components."""

    value: CanonicalInteger
    minimal_generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    in_semigroup: bool
    factorizations: tuple[tuple[int, ...], ...]
    edges: tuple[tuple[int, int], ...]
    connected_components: tuple[tuple[int, ...], ...]
    is_connected: bool

    @model_validator(mode="after")
    def require_graph_partition(self) -> Self:
        generators = _require_canonical_minimal_axis(self.minimal_generators)
        value = parse_canonical_integer(self.value)
        vertex_count = len(self.factorizations)
        if self.in_semigroup != bool(self.factorizations):
            raise _validation_error("membership must agree with graph vertices")
        vertices = tuple(
            index for component in self.connected_components for index in component
        )
        if tuple(sorted(vertices)) != tuple(range(vertex_count)):
            raise _validation_error(
                "connected components must partition all graph vertices"
            )
        if self.is_connected != (len(self.connected_components) <= 1):
            raise _validation_error("is_connected must agree with connected components")
        for left, right in self.edges:
            if not 0 <= left < right < vertex_count:
                raise _validation_error("graph edge has invalid vertex indices")
        _require_exact_factorization_family(generators, value, self.factorizations)
        expected_edges, expected_components = _factorization_graph_data(
            self.factorizations
        )
        if self.edges != expected_edges:
            raise _validation_error("edges do not match shared-support adjacency")
        if self.connected_components != expected_components:
            raise _validation_error("connected components do not match the graph")
        return self


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
    "FactorizationComputeRequest",
    "FactorizationComputeResult",
    "FactorizationDistanceRequest",
    "FactorizationDistanceResult",
    "FactorizationGraphComputeRequest",
    "FactorizationGraphComputeResult",
    "FactorizationLengthsComputeRequest",
    "FactorizationLengthsComputeResult",
    "compute_factorization_distance",
    "compute_factorization_graph",
    "compute_factorization_lengths",
    "compute_factorizations",
]
