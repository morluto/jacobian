"""Typed wire contracts for formal concept analysis operations."""

from __future__ import annotations

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.math.formal_concept_analysis.values import FormalContext


class DerivationRequest(StrictModel):
    """Derive A' (objects) or B' (attributes)."""

    context: FormalContext
    subset: tuple[int, ...] = Field(default=())


class DerivationResult(StrictModel):
    """The derived set."""

    derived: tuple[int, ...]


class ClosureResult(StrictModel):
    """The closure A'' or B'' with added elements and closed status."""

    closure: tuple[int, ...]
    derived: tuple[int, ...]
    added: tuple[int, ...]
    is_closed: bool


class ConceptRequest(StrictModel):
    """Construct a concept from objects or attributes."""

    context: FormalContext
    subset: tuple[int, ...] = Field(default=())


class ConceptResult(StrictModel):
    """A formal concept (extent, intent)."""

    extent: tuple[int, ...]
    intent: tuple[int, ...]


class EnumerateConceptsRequest(StrictModel):
    """Enumerate all formal concepts."""

    context: FormalContext


class EnumerateConceptsResult(StrictModel):
    """The complete concept family."""

    concepts: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    count: int = Field(ge=0)


class ConceptLatticeResult(StrictModel):
    """The concept lattice with order, covers, top, and bottom."""

    concepts: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    order: tuple[tuple[int, int], ...]
    covers: tuple[tuple[int, int], ...]
    top: int | None = None
    bottom: int | None = None


__all__ = [
    "ClosureResult",
    "ConceptLatticeResult",
    "ConceptRequest",
    "ConceptResult",
    "DerivationRequest",
    "DerivationResult",
    "EnumerateConceptsRequest",
    "EnumerateConceptsResult",
]
