"""Construct the k-uniform arithmetic-progression hypergraph of a finite integer interval."""

from __future__ import annotations

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.arithmetic_progression_hypergraph._models import (
    ArithmeticProgressionHypergraphResult,
    _admission_error,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)


def construct_arithmetic_progression_hypergraph(
    lower: int,
    upper: int,
    k: int,
) -> ArithmeticProgressionHypergraphResult:
    """Construct the canonical k-uniform AP hypergraph of the interval [lower, upper].

    Vertices are the integers ``lower, lower+1, ..., upper`` as decimal strings.
    Edges are precisely the increasing k-term arithmetic progressions
    ``a, a+d, ..., a+(k-1)*d`` with ``d >= 1`` and ``a + (k-1)*d <= upper``.

    Each edge is labelled ``(a, d)`` as a string, where ``a`` is the first
    term and ``d`` is the positive common difference.  Edge members are stored
    in canonical sorted order by ``FiniteHypergraph``.
    """
    failure = _admission_error(lower, upper, k)
    if failure is not None:
        field, code, message = failure
        raise OperationDomainValidationError(
            location=(field,),
            code=f"hypergraph.arithmetic_progression.{code}",
            message=message,
        )

    vertices = tuple(str(x) for x in range(lower, upper + 1))

    edges: list[tuple[str, tuple[str, ...]]] = []
    for d in range(1, upper - lower + 1):
        a = lower
        while a + (k - 1) * d <= upper:
            members = tuple(str(a + i * d) for i in range(k))
            edge_id = f"({a},{d})"
            edges.append((edge_id, members))
            a += 1

    hypergraph = FiniteHypergraph(vertices=vertices, edges=tuple(edges))

    return ArithmeticProgressionHypergraphResult(
        lower=lower,
        upper=upper,
        k=k,
        hypergraph=hypergraph,
    )


__all__ = ["construct_arithmetic_progression_hypergraph"]
