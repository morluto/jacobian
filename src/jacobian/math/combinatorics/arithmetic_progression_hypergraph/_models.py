"""Typed wire contracts for k-term arithmetic-progression hypergraph construction."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_EDGES,
    MAX_TOTAL_INCIDENCES,
    MAX_VERTICES,
    FiniteHypergraph,
)


def _edge_count(n: int, k: int) -> int:
    """Return the exact number of k-term AP edges for ``n`` vertices.

    With ``q = floor((n - 1) / (k - 1))`` the closed form is::

        E(n, k) = q*n - (k-1)*q*(q+1) / 2
    """
    if n <= 0 or k < 2:
        return 0
    q = (n - 1) // (k - 1)
    return q * n - (k - 1) * q * (q + 1) // 2


def _max_admitted_interval() -> int:
    """Return the largest ``n`` admitted by the hypergraph representation bound."""

    for n in range(MAX_VERTICES, 0, -1):
        for k in range(3, n + 1):
            edges = _edge_count(n, k)
            if edges <= MAX_EDGES and k * edges <= MAX_TOTAL_INCIDENCES:
                return n
    return 0


MAX_INTERVAL_SIZE: int = _max_admitted_interval()
assert MAX_INTERVAL_SIZE > 0, "no admitted interval size found"


class ArithmeticProgressionHypergraphRequest(StrictModel):
    """Inclusive integer interval ``[lower, upper]`` and arity ``k >= 3``.

    The interval is materialised in the result, so admission is based on
    ``n = upper - lower + 1``, not merely on endpoint digit length.
    """

    lower: int = Field(
        description="Inclusive lower endpoint L of the integer interval."
    )
    upper: int = Field(
        description="Inclusive upper endpoint U of the integer interval."
    )
    k: int = Field(ge=3, description="Arity k >= 3 of each arithmetic progression.")

    @model_validator(mode="after")
    def require_valid_interval(self) -> Self:
        if self.upper < self.lower:
            raise PydanticCustomError(
                "hypergraph.arithmetic_progression.empty_interval",
                "upper must be >= lower",
            )
        n = self.upper - self.lower + 1
        if n > MAX_INTERVAL_SIZE:
            raise PydanticCustomError(
                "hypergraph.arithmetic_progression.interval_too_large",
                f"interval size {n} exceeds the maximum admitted size "
                f"{MAX_INTERVAL_SIZE}",
            )
        edges = _edge_count(n, self.k)
        if edges > MAX_EDGES:
            raise PydanticCustomError(
                "hypergraph.arithmetic_progression.edge_count_exceeds_bound",
                f"edge count {edges} exceeds the {MAX_EDGES}-edge bound",
            )
        incidences = self.k * edges
        if incidences > MAX_TOTAL_INCIDENCES:
            raise PydanticCustomError(
                "hypergraph.arithmetic_progression.incidence_count_exceeds_bound",
                f"incidence count {incidences} exceeds the "
                f"{MAX_TOTAL_INCIDENCES}-incidence bound",
            )
        return self


class ArithmeticProgressionHypergraphResult(StrictModel):
    """The canonical k-uniform arithmetic-progression hypergraph of ``[L, U]``.

    Vertices are the integers ``L, L+1, ..., U`` as decimal strings in numeric
    order.  Each edge is labelled ``(a, d)`` and its members are
    ``(a, a+d, ..., a+(k-1)*d)`` for ``d >= 1`` with ``a + (k-1)*d <= U``.
    """

    lower: int
    upper: int
    k: int
    hypergraph: FiniteHypergraph

    @model_validator(mode="after")
    def require_consistent_arity(self) -> Self:
        if self.k < 3:
            raise PydanticCustomError(
                "hypergraph.arithmetic_progression.invalid_arity",
                "k must be at least 3",
            )
        return self


__all__ = [
    "MAX_INTERVAL_SIZE",
    "ArithmeticProgressionHypergraphRequest",
    "ArithmeticProgressionHypergraphResult",
    "_edge_count",
]
