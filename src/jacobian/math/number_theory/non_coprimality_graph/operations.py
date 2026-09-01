"""Non-coprimality graph constructor."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.number_theory.non_coprimality_graph._models import (
    MAX_INTEGER_DIGITS,
    MAX_INTEGERS,
    NonCoprimalityGraphResult,
)

__all__ = ["construct_non_coprimality_graph"]


@dataclass(frozen=True, slots=True)
class NonCoprimalityGraphAdmission:
    """Canonical source and exact conflict edges computed during admission."""

    source: tuple[str, ...]
    vertices: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]


def _admit_non_coprimality_graph(
    integers: tuple[int, ...],
) -> NonCoprimalityGraphAdmission:
    if not isinstance(integers, tuple):
        raise OperationDomainValidationError(
            location=("integers",),
            code="non_coprimality.integer_type",
            message="integers must be a tuple of integers",
        )
    if not 1 <= len(integers) <= MAX_INTEGERS:
        raise OperationDomainValidationError(
            location=("integers",),
            code="non_coprimality.size",
            message=f"integers must contain between 1 and {MAX_INTEGERS} values",
        )

    source: list[str] = []
    values: list[int] = []
    for index, value in enumerate(integers):
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
        source.append(label)
        values.append(value)
    if len(set(values)) != len(values):
        raise OperationDomainValidationError(
            location=("integers",),
            code="non_coprimality.must_be_distinct",
            message="integers must be distinct",
        )

    sorted_pairs = sorted(zip(source, values, strict=True), key=lambda pair: pair[1])
    vertices = tuple(label for label, _ in sorted_pairs)
    edges: list[tuple[str, str]] = []
    for left_index, (left_label, left_value) in enumerate(sorted_pairs):
        for right_label, right_value in sorted_pairs[left_index + 1 :]:
            if gcd(left_value, right_value) > 1:
                edges.append(
                    (min(left_label, right_label), max(left_label, right_label))
                )
    canonical_edges = tuple(edges)
    return NonCoprimalityGraphAdmission(tuple(source), vertices, canonical_edges)


def construct_non_coprimality_graph(
    integers: tuple[int, ...],
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
        integers=admission.source,
        graph=graph,
    )
