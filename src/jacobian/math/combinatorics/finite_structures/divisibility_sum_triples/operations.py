"""Divisibility-sum triple hypergraph constructor."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

from pydantic_core import PydanticCustomError

from jacobian.canonical import (
    CanonicalLimits,
    encode_strict_json,
    format_canonical_integer,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.finite_structures.divisibility_sum_triples._models import (
    MAX_INTERVAL_SIZE,
    DivisibilitySumTriplesResult,
    _validate_interval_shape,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_LABEL_LENGTH,
    FiniteHypergraph,
)

__all__ = ["construct_divisibility_sum_triples_hypergraph"]

MAX_TRIPLE_ENUMERATION = 1_000_000


def _decimal_digits(value: int) -> int:
    """Count decimal digits without converting an unbounded integer to text."""
    magnitude = abs(value)
    if magnitude == 0:
        return 1
    if magnitude.bit_length() > (MAX_LABEL_LENGTH + 1) * 4:
        return MAX_LABEL_LENGTH + 1
    estimate = magnitude.bit_length() * 30_103 // 100_000 + 1
    while estimate > 1 and magnitude < 10 ** (estimate - 1):
        estimate -= 1
    while magnitude >= 10**estimate:
        estimate += 1
    return estimate


@dataclass(frozen=True, slots=True)
class DivisibilitySumTriplesAdmission:
    """The exact bounded construction shared by admission and execution."""

    vertices: tuple[str, ...]
    edges: tuple[tuple[str, tuple[str, ...]], ...]


def _admit_divisibility_sum_triples(
    lower_bound: int, upper_bound: int
) -> DivisibilitySumTriplesAdmission:
    try:
        _validate_interval_shape(lower_bound, upper_bound)
    except PydanticCustomError as error:
        raise OperationDomainValidationError(
            location=(), code=error.type, message=str(error)
        ) from error

    interval_size = upper_bound - lower_bound + 1
    if interval_size > MAX_INTERVAL_SIZE:
        raise OperationDomainValidationError(
            location=(),
            code="divisibility_sum.interval_too_large",
            message=f"interval size must not exceed {MAX_INTERVAL_SIZE}",
        )
    candidate_count = comb(interval_size, 3) if interval_size >= 3 else 0
    if candidate_count > MAX_TRIPLE_ENUMERATION:
        raise OperationDomainValidationError(
            location=(),
            code="divisibility_sum.enumeration_work_exceeded",
            message=(
                "the triple construction exceeds the "
                f"{MAX_TRIPLE_ENUMERATION:,}-candidate work bound"
            ),
        )

    label_width = max(
        _decimal_digits(lower_bound) + (1 if lower_bound < 0 else 0),
        _decimal_digits(upper_bound) + (1 if upper_bound < 0 else 0),
    )
    if label_width > MAX_LABEL_LENGTH:
        raise OperationDomainValidationError(
            location=(),
            code="divisibility_sum.vertex_label_too_long",
            message=f"interval vertices must be at most {MAX_LABEL_LENGTH} characters",
        )

    vertices = tuple(str(value) for value in range(lower_bound, upper_bound + 1))

    edges: list[tuple[str, tuple[str, ...]]] = []
    for edge_index, triple in enumerate(
        combinations(range(lower_bound, upper_bound + 1), 3)
    ):
        a, b, c = triple
        if a != 0 and (b + c) % a == 0:
            edges.append((f"edge_{edge_index}", tuple(str(value) for value in triple)))
    if len(edges) > 12_000 or 3 * len(edges) > 36_000:
        raise OperationDomainValidationError(
            location=(),
            code="divisibility_sum.output_too_large",
            message="the exact triple family exceeds the hypergraph envelope",
        )

    try:
        payload = {
            "lower_bound": format_canonical_integer(lower_bound),
            "upper_bound": format_canonical_integer(upper_bound),
            "hypergraph": {
                "vertices": list(vertices),
                "edges": [[edge_id, list(members)] for edge_id, members in edges],
            },
        }
        if len(encode_strict_json(payload)) > CanonicalLimits().max_output_bytes:
            raise OperationDomainValidationError(
                location=(),
                code="divisibility_sum.result_bytes_exceeded",
                message="the exact result exceeds the canonical output-byte limit",
            )
    except ValueError as error:
        raise OperationDomainValidationError(
            location=(),
            code="divisibility_sum.result_not_canonical",
            message="the exact result cannot be represented in canonical JSON",
        ) from error
    return DivisibilitySumTriplesAdmission(vertices=vertices, edges=tuple(edges))


def construct_divisibility_sum_triples_hypergraph(
    lower_bound: int,
    upper_bound: int,
) -> DivisibilitySumTriplesResult:
    """Construct the 3-uniform hypergraph of divisibility-sum triples.

    Vertices are the integers in [L, U]. Edges are the increasing triples
    (a, b, c) with L <= a < b < c <= U and a | (b + c).
    """
    admission = _admit_divisibility_sum_triples(lower_bound, upper_bound)
    hypergraph = FiniteHypergraph(
        vertices=admission.vertices,
        edges=admission.edges,
    )
    return DivisibilitySumTriplesResult.model_construct(
        lower_bound=format_canonical_integer(lower_bound),
        upper_bound=format_canonical_integer(upper_bound),
        hypergraph=hypergraph,
    )
