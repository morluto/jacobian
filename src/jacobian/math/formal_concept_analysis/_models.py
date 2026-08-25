"""Typed wire contracts for formal concept analysis operations."""

from __future__ import annotations

from typing import Self

from pydantic import ConfigDict, Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.formal_concept_analysis.basis import (
    MAX_DG_ATTRIBUTES,
    MAX_DG_CANDIDATE_STATES,
    MAX_DG_LOGICAL_WORK,
    MAX_DG_RESULT_BYTES,
    _duquenne_guigues_preflight,
)
from jacobian.math.formal_concept_analysis.values import (
    MAX_IMPLICATION_MEMBERSHIPS,
    MAX_IMPLICATIONS,
    FiniteAttributeImplicationSystem,
    FormalContext,
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
                "reserves producer and independent replay work bounded by "
                f"{MAX_DG_LOGICAL_WORK:,} logical steps plus a worst-case "
                f"{MAX_DG_RESULT_BYTES:,}-byte serialized result."
            )
        }
    )

    context: FormalContext

    @model_validator(mode="after")
    def require_complete_preflight(self) -> Self:
        _duquenne_guigues_preflight(self.context)
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


# Bound the concept enumeration by its declared output budget. NextClosure's
# cost is proportional to the number of concepts, and that number is bounded
# by 2^min(|objects|, |attributes|): intents biject with a closure system on
# the attribute axis and extents with one on the object axis.  Admission is
# therefore two-tier and result-sensitive.  When that tight worst case fits
# MAX_CONCEPTS the request is admitted statically.  Otherwise one capped
# NextClosure preflight — bounded by the same budget it guards — counts the
# true family, so sparse wide or square contexts stay admissible and only
# contexts whose actual family overflows are rejected before execution.
MAX_CONCEPTS = 10000


class EnumerateConceptsRequest(StrictModel):
    """Enumerate all formal concepts."""

    context: FormalContext

    @model_validator(mode="after")
    def require_bounded_concept_family(self) -> Self:
        worst_case_concepts = 2 ** min(
            len(self.context.objects), len(self.context.attributes)
        )
        if worst_case_concepts > MAX_CONCEPTS:
            from jacobian.math.formal_concept_analysis.operations import (
                concept_family_size_capped,
            )

            family_size = concept_family_size_capped(self.context, MAX_CONCEPTS)
            if family_size > MAX_CONCEPTS:
                raise PydanticCustomError(
                    "formal_concept_analysis.concept_family_exceeds_bound",
                    f"the context carries more than {MAX_CONCEPTS} concepts "
                    "and concept enumeration returns at most "
                    f"{MAX_CONCEPTS}; narrow the context or split the "
                    "enumeration",
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
