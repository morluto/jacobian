"""Typed wire contracts for formal concept analysis operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.math.formal_concept_analysis.values import (
    FiniteAttributeImplicationSystem,
    FormalContext,
)


class _SubsetRequest(StrictModel):
    context: FormalContext
    subset: tuple[int, ...] = Field(default=())

    def _require_indices(self, size: int, side: str) -> None:
        for i in self.subset:
            if not 0 <= i < size:
                raise ValueError(f"{side} subset index out of range")


class ObjectSubsetRequest(_SubsetRequest):
    """A subset of the context's object axis."""

    @model_validator(mode="after")
    def require_valid_indices(self) -> Self:
        self._require_indices(len(self.context.objects), "object")
        return self


class AttributeSubsetRequest(_SubsetRequest):
    """A subset of the context's attribute axis."""

    @model_validator(mode="after")
    def require_valid_indices(self) -> Self:
        self._require_indices(len(self.context.attributes), "attribute")
        return self


class ImplicationClosureRequest(StrictModel):
    """Close one canonical attribute subset under a finite implication system."""

    system: FiniteAttributeImplicationSystem
    seed: tuple[StrictInt, ...] = Field(
        default=(),
        description=(
            "Attribute indices initially present. Order is immaterial, duplicate "
            "indices are invalid, and every index refers to system.attributes."
        ),
    )

    @model_validator(mode="after")
    def require_canonical_bounded_seed(self) -> Self:
        if len(set(self.seed)) != len(self.seed):
            raise ValueError("implication seed indices must be unique")
        if any(
            not 0 <= attribute < len(self.system.attributes) for attribute in self.seed
        ):
            raise ValueError(
                "implication seed attribute is outside the declared carrier"
            )
        object.__setattr__(self, "seed", tuple(sorted(self.seed)))
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


class ConceptResult(StrictModel):
    """A formal concept (extent, intent)."""

    extent: tuple[int, ...]
    intent: tuple[int, ...]


# Bound the concept enumeration. NextClosure has cost proportional to the
# number of concepts (not 2^|M|), but the number of concepts itself can be
# exponential in the number of attributes.  We bound both the attribute count
# and the number of concepts returned.
MAX_CONCEPT_ATTRIBUTES = 64
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
    "AttributeSubsetRequest",
    "ClosureResult",
    "ConceptLatticeResult",
    "ConceptResult",
    "DerivationResult",
    "EnumerateConceptsRequest",
    "EnumerateConceptsResult",
    "ImplicationClosureRequest",
    "ObjectSubsetRequest",
]
