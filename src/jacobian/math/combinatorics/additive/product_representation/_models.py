"""Typed contracts for the product representation profile operation."""

from pydantic import Field

from jacobian._exact import ExactInteger
from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.sets._models import FiniteIntegerSet


class ProductRepresentationRequest(StrictModel):
    """Request the product representation profile of two finite integer sets."""

    left: FiniteIntegerSet
    right: FiniteIntegerSet


class RepresentationEntry(StrictModel):
    """One product value and its representation multiplicity."""

    product: ExactInteger
    multiplicity: int = Field(ge=1, le=100_000)


class ProductRepresentationResult(StrictModel):
    """The complete product representation profile."""

    left: FiniteIntegerSet
    right: FiniteIntegerSet
    entries: tuple[RepresentationEntry, ...]
    support_cardinality: int = Field(ge=0, le=100_000)


__all__ = [
    "ProductRepresentationRequest",
    "ProductRepresentationResult",
    "RepresentationEntry",
]
