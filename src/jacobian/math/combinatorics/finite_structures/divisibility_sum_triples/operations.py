"""Divisibility-sum triple hypergraph constructor."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

from pydantic_core import PydanticCustomError

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
    vertices = tuple(str(value) for value in range(lower_bound, upper_bound + 1))
    if any(len(vertex) > MAX_LABEL_LENGTH for vertex in vertices):
        raise OperationDomainValidationError(
            location=(),
            code="divisibility_sum.vertex_label_too_long",
            message=f"interval vertices must be at most {MAX_LABEL_LENGTH} characters",
        )

    # If 2U < 3L for a positive interval, every triple satisfies
    # 2a < b+c < 3a, so no divisor test can succeed.  Admit this closed form
    # before the generic interval/work ceilings.
    sparse_no_edges = lower_bound > 0 and 2 * upper_bound < 3 * lower_bound
    candidate_count = (
        0 if sparse_no_edges else comb(interval_size, 3) if interval_size >= 3 else 0
    )
    if candidate_count > MAX_TRIPLE_ENUMERATION:
        raise OperationDomainValidationError(
            location=(),
            code="divisibility_sum.enumeration_work_exceeded",
            message=(
                "the triple construction exceeds the "
                f"{MAX_TRIPLE_ENUMERATION:,}-candidate work bound"
            ),
        )

    edges: list[tuple[str, tuple[str, ...]]] = []
    if not sparse_no_edges:
        for edge_index, triple in enumerate(
            combinations(range(lower_bound, upper_bound + 1), 3)
        ):
            a, b, c = triple
            if a != 0 and (b + c) % a == 0:
                edges.append(
                    (f"edge_{edge_index}", tuple(str(value) for value in triple))
                )
                if len(edges) > 12_000 or 3 * len(edges) > 36_000:
                    raise OperationDomainValidationError(
                        location=(),
                        code="divisibility_sum.output_too_large",
                        message="the exact triple family exceeds the hypergraph envelope",
                    )

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
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        hypergraph=hypergraph,
    )
