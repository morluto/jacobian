"""Typed contracts for divisibility edge profiles with quotient and LPF data."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

MAX_DIVISIBILITY_EDGE_SET_SIZE = 500


class DivisibilityEdgeProfileRequest(StrictModel):
    """Profile quotient and least-prime-factor data on finite divisibility edges."""

    values: tuple[str, ...] = Field(
        min_length=1,
        max_length=MAX_DIVISIBILITY_EDGE_SET_SIZE,
        description=(
            "Ordered source set of positive canonical decimal integers. "
            "The result profiles every proper-divisibility edge a -> b "
            "(a divides b, a != b) with the quotient b/a and its least "
            "prime factor."
        ),
        examples=["2", "4", "6", "12"],
    )


class DivisibilityEdge(StrictModel):
    """One proper-divisibility edge with quotient and least-prime-factor data."""

    source: str
    target: str
    quotient: int = Field(gt=0)
    least_prime_factor: int = Field(gt=1)


class DivisibilityEdgeProfileResult(StrictModel):
    """The complete directed divisibility edge table."""

    values: tuple[str, ...] = Field(min_length=1)
    edges: tuple[DivisibilityEdge, ...] = Field(default=())

    @model_validator(mode="after")
    def require_canonical_edges(self) -> Self:
        for edge in self.edges:
            if edge.source == edge.target:
                raise PydanticCustomError(
                    "divisibility_edge.no_reflexive",
                    "divisibility edges must not be reflexive",
                )
        return self


__all__ = [
    "MAX_DIVISIBILITY_EDGE_SET_SIZE",
    "DivisibilityEdge",
    "DivisibilityEdgeProfileRequest",
    "DivisibilityEdgeProfileResult",
]
