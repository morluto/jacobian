"""Non-coprimality graph constructor."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.finite_structures.sets._models import FiniteIntegerSet
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.number_theory.non_coprimality_graph._models import (
    MAX_INTEGER_DIGITS,
    MAX_INTEGERS,
    NonCoprimalityGraphResult,
)

__all__ = ["construct_non_coprimality_graph", "verify_non_coprimality_graph"]


@dataclass(frozen=True, slots=True)
class NonCoprimalityGraphAdmission:
    """Canonical source and exact conflict edges computed during admission."""

    source: tuple[int, ...]
    vertices: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]


def _admit_non_coprimality_graph(
    integers: FiniteIntegerSet | tuple[int, ...],
) -> NonCoprimalityGraphAdmission:
    if isinstance(integers, FiniteIntegerSet):
        integer_values = tuple(value for value in integers.elements)
    else:
        integer_values = integers
    if not isinstance(integer_values, tuple):
        raise OperationDomainValidationError(
            location=("integers",),
            code="non_coprimality.integer_type",
            message="integers must be a tuple of integers",
        )
    if not 1 <= len(integer_values) <= MAX_INTEGERS:
        raise OperationDomainValidationError(
            location=("integers",),
            code="non_coprimality.size",
            message=f"integers must contain between 1 and {MAX_INTEGERS} values",
        )

    source: list[int] = []
    values: list[int] = []
    for index, value in enumerate(integer_values):
        if type(value) is not int:
            raise OperationDomainValidationError(
                location=("integers", index),
                code="non_coprimality.integer_type",
                message="integers must contain only integers",
            )
        if value <= 0:
            raise OperationDomainValidationError(
                location=("integers", index),
                code="non_coprimality.must_be_positive",
                message="all integers must be positive",
            )
        # Avoid decimal conversion of huge native integers.  The generous
        # bit-length threshold cannot exclude any value within the exact
        # decimal label bound; the precise check below remains authoritative.
        if value.bit_length() > (MAX_INTEGER_DIGITS + 1) * 4:
            raise OperationDomainValidationError(
                location=("integers", index),
                code="non_coprimality.digits",
                message=(
                    f"integer {index} exceeds the {MAX_INTEGER_DIGITS}-digit bound"
                ),
            )
        label = format_canonical_integer(value)
        if len(label) > MAX_INTEGER_DIGITS:
            raise OperationDomainValidationError(
                location=("integers", index),
                code="non_coprimality.digits",
                message=(
                    f"integer {index} exceeds the {MAX_INTEGER_DIGITS}-digit bound"
                ),
            )
        source.append(value)
        values.append(value)
    if len(set(values)) != len(values):
        raise OperationDomainValidationError(
            location=("integers",),
            code="non_coprimality.must_be_distinct",
            message="integers must be distinct",
        )

    # The retained source owns the graph's vertex axis; preserve its order.
    vertices = tuple(format_canonical_integer(value) for value in source)
    edges: list[tuple[str, str]] = []
    source_pairs = tuple(zip(vertices, values, strict=True))
    for left_index, (left_label, left_value) in enumerate(source_pairs):
        for right_label, right_value in source_pairs[left_index + 1 :]:
            if gcd(left_value, right_value) > 1:
                edges.append(
                    (min(left_label, right_label), max(left_label, right_label))
                )
    canonical_edges = tuple(edges)
    return NonCoprimalityGraphAdmission(tuple(source), vertices, canonical_edges)


def construct_non_coprimality_graph(
    integers: FiniteIntegerSet | tuple[int, ...],
) -> NonCoprimalityGraphResult:
    """Construct the non-coprimality graph of a set of positive integers.

    The graph has one vertex per integer (labelled by the integer's
    canonical string) and an edge between two vertices iff their gcd > 1.
    """
    admission = _admit_non_coprimality_graph(integers)
    graph = SimpleUndirectedGraph(
        vertices=admission.vertices,
        edges=admission.edges,
    )
    return NonCoprimalityGraphResult.model_construct(
        integers=FiniteIntegerSet(elements=admission.source),
        graph=graph,
    )


def verify_non_coprimality_graph(claim: NonCoprimalityGraphResult) -> bool:
    """Check the retained integer source and exact gcd edge relation."""
    source = claim.integers
    if not 1 <= len(source.elements) <= MAX_INTEGERS:
        return False
    try:
        values = tuple(source.elements)
    except (TypeError, ValueError):
        return False
    if any(value <= 0 for value in values) or len(set(values)) != len(values):
        return False
    if claim.graph.vertices != tuple(
        format_canonical_integer(value) for value in source.elements
    ):
        return False
    expected = {
        (min(str(left), str(right)), max(str(left), str(right)))
        for index, left in enumerate(values)
        for right in values[index + 1 :]
        if gcd(left, right) > 1
    }
    return set(claim.graph.edges) == expected
