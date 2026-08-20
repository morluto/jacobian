"""Typed wire contracts for formal concept analysis operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.formal_concept_analysis.values import FormalContext


class DerivationRequest(StrictModel):
    """Derive A' (objects) or B' (attributes)."""

    context: FormalContext
    subset: tuple[int, ...] = Field(default=())

    @model_validator(mode="after")
    def require_valid_indices(self) -> Self:
        max_idx = max(len(self.context.objects), len(self.context.attributes))
        for i in self.subset:
            if not 0 <= i < max_idx:
                raise ValueError("subset index out of range")
        return self


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

    @model_validator(mode="after")
    def require_valid_indices(self) -> Self:
        max_idx = max(len(self.context.objects), len(self.context.attributes))
        for i in self.subset:
            if not 0 <= i < max_idx:
                raise ValueError("subset index out of range")
        return self


class ConceptResult(StrictModel):
    """A formal concept (extent, intent)."""

    extent: tuple[int, ...]
    intent: tuple[int, ...]


# Bound the concept enumeration. NextClosure has cost proportional to the
# number of concepts (not 2^|M|), but the number of concepts itself can be
# exponential in the number of attributes.  We bound both the attribute count
# and the number of concepts returned.
MAX_CONCEPT_ATTRIBUTES = 20
MAX_CONCEPTS = 10000


class EnumerateConceptsRequest(StrictModel):
    """Enumerate all formal concepts."""

    context: FormalContext

    @model_validator(mode="after")
    def require_bounded_attribute_count(self) -> Self:
        if len(self.context.attributes) > MAX_CONCEPT_ATTRIBUTES:
            raise ValueError(
                f"concept enumeration supports at most {MAX_CONCEPT_ATTRIBUTES} attributes"
            )
        return self


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
    "MAX_CONCEPTS",
    "MAX_CONCEPT_ATTRIBUTES",
    "ClosureResult",
    "ConceptLatticeResult",
    "ConceptRequest",
    "ConceptResult",
    "DerivationRequest",
    "DerivationResult",
    "EnumerateConceptsRequest",
    "EnumerateConceptsResult",
]
