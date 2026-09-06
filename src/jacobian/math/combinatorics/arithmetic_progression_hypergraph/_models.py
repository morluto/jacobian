"""Typed wire contracts for k-term arithmetic-progression hypergraph construction."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, model_validator
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


def _admission_error(lower: int, upper: int, k: int) -> tuple[str, str, str] | None:
    """Return the first AP construction admission failure, if any."""
    for field, value in (("lower", lower), ("upper", upper), ("k", k)):
        if type(value) is not int:
            return (
                field,
                "invalid_integer",
                f"{field} must be a strict integer",
            )
    if upper < lower:
        return ("upper", "empty_interval", "upper must be >= lower")
    if k < 3:
        return ("k", "invalid_arity", "k must be at least 3")
    n = upper - lower + 1
    if n > MAX_INTERVAL_SIZE:
        return (
            "upper",
            "interval_too_large",
            f"interval size {n} exceeds the maximum admitted size {MAX_INTERVAL_SIZE}",
        )
    edges = _edge_count(n, k)
    if edges > MAX_EDGES:
        return (
            "k",
            "edge_count_exceeds_bound",
            f"edge count {edges} exceeds the {MAX_EDGES}-edge bound",
        )
    incidences = k * edges
    if incidences > MAX_TOTAL_INCIDENCES:
        return (
            "k",
            "incidence_count_exceeds_bound",
            f"incidence count {incidences} exceeds the "
            f"{MAX_TOTAL_INCIDENCES}-incidence bound",
        )

    def decimal_width(value: int) -> int:
        if value == 0:
            return 1
        return (abs(value).bit_length() * 30_103) // 100_000 + 1 + (value < 0)

    vertex_label_bytes = max(decimal_width(lower), decimal_width(upper))
    max_difference = (n - 1) // (k - 1)
    max_start = upper - (k - 1)
    edge_label_bytes = (
        3
        + max(decimal_width(lower), decimal_width(max_start))
        + decimal_width(max_difference)
        if max_difference > 0
        else 0
    )
    if max(vertex_label_bytes, edge_label_bytes) > 64:
        return (
            "upper",
            "label_size_exceeds_bound",
            "generated vertex or edge labels exceed the 64-byte carrier",
        )
    return None


class ArithmeticProgressionHypergraphRequest(StrictModel):
    """Inclusive integer interval ``[lower, upper]`` and arity ``k >= 3``.

    The interval is materialised in the result, so admission is based on
    ``n = upper - lower + 1``, not merely on endpoint digit length.
    """

    lower: StrictInt = Field(
        description="Inclusive lower endpoint L of the integer interval."
    )
    upper: StrictInt = Field(
        description="Inclusive upper endpoint U of the integer interval."
    )
    k: StrictInt = Field(
        ge=3, description="Arity k >= 3 of each arithmetic progression."
    )

    @model_validator(mode="after")
    def require_valid_interval(self) -> Self:
        failure = _admission_error(self.lower, self.upper, self.k)
        if failure is not None:
            _, code, message = failure
            raise PydanticCustomError(
                f"hypergraph.arithmetic_progression.{code}",
                message,
            )
        return self


class ArithmeticProgressionHypergraphResult(StrictModel):
    """The canonical k-uniform arithmetic-progression hypergraph of ``[L, U]``.

    Vertices are the integers ``L, L+1, ..., U`` as decimal strings in numeric
    order.  Each edge is labelled ``(a, d)`` and its members are
    ``(a, a+d, ..., a+(k-1)*d)`` for ``d >= 1`` with ``a + (k-1)*d <= U``.
    """

    lower: StrictInt
    upper: StrictInt
    k: StrictInt
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
    "_admission_error",
    "_edge_count",
]
