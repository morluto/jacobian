"""Canonical rational V- and H-representation values."""

from __future__ import annotations

from pydantic import Field
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

MAX_RATIONAL_POLYTOPE_DIMENSION = 7
"""Largest ambient axis supported by the canonical rational values."""


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"polytope.{reason}", message)


class Vertex(StrictModel):
    """One rational vertex of a V-representation."""

    coordinates: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_RATIONAL_POLYTOPE_DIMENSION
    )


class Halfspace(StrictModel):
    """One rational half-space ``<a, x> <= b`` of an H-representation."""

    coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_RATIONAL_POLYTOPE_DIMENSION,
        description=(
            "Normal vector a of the half-space <a, x> <= b; at least one "
            "entry must be nonzero (all-zero rows are rejected)."
        ),
    )
    offset: CanonicalRational


__all__ = [
    "MAX_RATIONAL_POLYTOPE_DIMENSION",
    "Halfspace",
    "Vertex",
]
