"""Typed contracts for the product representation profile operation."""

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.sets._models import FiniteIntegerSet


class ProductRepresentationRequest(StrictModel):
    """Request the product representation profile of two finite integer sets."""

    left: FiniteIntegerSet
    right: FiniteIntegerSet


class RepresentationEntry(StrictModel):
    """One product value and its representation multiplicity."""

    product: int
    multiplicity: int


class ProductRepresentationResult(StrictModel):
    """The complete product representation profile."""

    left: FiniteIntegerSet
    right: FiniteIntegerSet
    entries: tuple[RepresentationEntry, ...]
    support_cardinality: int


__all__ = [
    "ProductRepresentationRequest",
    "ProductRepresentationResult",
    "RepresentationEntry",
]
