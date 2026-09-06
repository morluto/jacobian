"""Typed wire contracts for formal concept analysis operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import ConfigDict, Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.posets.formal_concepts._concepts import MAX_CONCEPTS
from jacobian.math.combinatorics.posets.formal_concepts.basis import (
    MAX_DG_ATTRIBUTES,
    MAX_DG_CANDIDATE_STATES,
    MAX_DG_LOGICAL_WORK,
)
from jacobian.math.combinatorics.posets.formal_concepts.values import (
    MAX_IMPLICATION_MEMBERSHIPS,
    MAX_IMPLICATIONS,
    FiniteAttributeImplicationSystem,
    FormalAttributeSubset,
    FormalConcept,
    FormalContext,
    FormalObjectSubset,
)


class _SubsetRequest(StrictModel):
    context: FormalContext
    subset: tuple[int, ...] = Field(default=())

    def _require_indices(self, size: int, side: str) -> None:
        for i in self.subset:
            if not 0 <= i < size:
                raise PydanticCustomError(
                    "formal_concept_analysis.subset_index_out_of_range",
                    f"{side} subset index out of range",
                    {"axis": side},
                )


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
            raise PydanticCustomError(
                "formal_concept_analysis.seed_indices_not_unique",
                "implication seed indices must be unique",
            )
        if any(
            not 0 <= attribute < len(self.system.attributes) for attribute in self.seed
        ):
            raise PydanticCustomError(
                "formal_concept_analysis.seed_attribute_out_of_range",
                "implication seed attribute is outside the declared carrier",
            )
        object.__setattr__(self, "seed", tuple(sorted(self.seed)))
        return self


class DuquenneGuiguesBasisRequest(StrictModel):
    """Compute the complete canonical implication basis of one context."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Compute all pseudo-intents and the complete Duquenne-Guigues "
                "basis for a FormalContext whose exhaustive candidate carrier "
                f"is at most {MAX_DG_CANDIDATE_STATES} states "
                f"({MAX_DG_ATTRIBUTES} attributes; documented conservative "
                "fallback). Admission probes the exact closure matrix and "
                "basis size, rejects bases beyond the "
                f"{MAX_IMPLICATIONS}-implication canonical implication-system "
                f"carrier ({MAX_IMPLICATION_MEMBERSHIPS:,} memberships), and "
                "reserves one producer plan and closure-equivalence pass bounded by "
                f"{MAX_DG_LOGICAL_WORK:,} logical steps."
            )
        }
    )

    context: FormalContext


class DerivationResult(StrictModel):
    """The derived set."""

    context: FormalContext
    subset: FormalObjectSubset | FormalAttributeSubset
    side: Literal["OBJECT", "ATTRIBUTE"]
    derived: FormalObjectSubset | FormalAttributeSubset

    @model_validator(mode="after")
    def bind_axes_to_context(self) -> Self:
        expected = (
            (FormalObjectSubset, FormalAttributeSubset)
            if self.side == "OBJECT"
            else (FormalAttributeSubset, FormalObjectSubset)
        )
        if not isinstance(self.subset, expected[0]) or not isinstance(
            self.derived, expected[1]
        ):
            raise PydanticCustomError(
                "formal_concept_analysis.derivation_axis_mismatch",
                "derivation subset and result axes must agree with side",
            )
        if self.subset.context != self.context or self.derived.context != self.context:
            raise PydanticCustomError(
                "formal_concept_analysis.result_context_mismatch",
                "derived subsets must use the retained context",
            )
        return self


class ClosureResult(StrictModel):
    """The closure A'' or B'' with added elements and closed status."""

    context: FormalContext
    subset: FormalObjectSubset | FormalAttributeSubset
    side: Literal["OBJECT", "ATTRIBUTE"]
    closure: FormalObjectSubset | FormalAttributeSubset
    derived: FormalObjectSubset | FormalAttributeSubset
    added: FormalObjectSubset | FormalAttributeSubset
    is_closed: bool

    @model_validator(mode="after")
    def bind_axes_to_context(self) -> Self:
        same_axis = (
            FormalObjectSubset if self.side == "OBJECT" else FormalAttributeSubset
        )
        derived_axis = (
            FormalAttributeSubset if self.side == "OBJECT" else FormalObjectSubset
        )
        if not all(
            isinstance(value, same_axis)
            for value in (self.subset, self.closure, self.added)
        ) or not isinstance(self.derived, derived_axis):
            raise PydanticCustomError(
                "formal_concept_analysis.closure_axis_mismatch",
                "closure subsets must agree with the declared side",
            )
        if any(
            value.context != self.context
            for value in (self.subset, self.closure, self.derived, self.added)
        ):
            raise PydanticCustomError(
                "formal_concept_analysis.result_context_mismatch",
                "closure subsets must use the retained context",
            )
        return self


class ConceptResult(StrictModel):
    """A formal concept (extent, intent)."""

    context: FormalContext
    subset: FormalObjectSubset | FormalAttributeSubset
    side: Literal["OBJECT", "ATTRIBUTE"]
    extent: FormalObjectSubset
    intent: FormalAttributeSubset

    @model_validator(mode="after")
    def bind_axes_to_context(self) -> Self:
        subset_axis = (
            FormalObjectSubset if self.side == "OBJECT" else FormalAttributeSubset
        )
        if not isinstance(self.subset, subset_axis):
            raise PydanticCustomError(
                "formal_concept_analysis.concept_axis_mismatch",
                "concept input subset must agree with the declared side",
            )
        if any(
            value.context != self.context
            for value in (self.subset, self.extent, self.intent)
        ):
            raise PydanticCustomError(
                "formal_concept_analysis.result_context_mismatch",
                "concept subsets must use the retained context",
            )
        return self


class EnumerateConceptsRequest(StrictModel):
    """Enumerate all formal concepts."""

    context: FormalContext


class EnumerateConceptsResult(StrictModel):
    """The complete concept family."""

    context: FormalContext
    concepts: tuple[FormalConcept, ...]
    count: int = Field(ge=0)

    @model_validator(mode="after")
    def bind_concepts_to_context(self) -> Self:
        if any(concept.context != self.context for concept in self.concepts):
            raise PydanticCustomError(
                "formal_concept_analysis.result_context_mismatch",
                "enumerated concepts must use the retained context",
            )
        return self


class ConceptLatticeResult(StrictModel):
    """The concept lattice with order, covers, top, and bottom."""

    context: FormalContext
    concepts: tuple[FormalConcept, ...]
    order: tuple[tuple[int, int], ...]
    covers: tuple[tuple[int, int], ...]
    top: int | None = None
    bottom: int | None = None

    @model_validator(mode="after")
    def bind_concepts_to_context(self) -> Self:
        if any(concept.context != self.context for concept in self.concepts):
            raise PydanticCustomError(
                "formal_concept_analysis.result_context_mismatch",
                "lattice concepts must use the retained context",
            )
        return self


__all__ = [
    "MAX_CONCEPTS",
    "AttributeSubsetRequest",
    "ClosureResult",
    "ConceptLatticeResult",
    "ConceptResult",
    "DerivationResult",
    "DuquenneGuiguesBasisRequest",
    "EnumerateConceptsRequest",
    "EnumerateConceptsResult",
    "ImplicationClosureRequest",
    "ObjectSubsetRequest",
]
