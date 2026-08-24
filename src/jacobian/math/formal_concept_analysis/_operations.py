"""Domain adapter for formal concept analysis operations."""

from __future__ import annotations

from typing import Any, cast

from jacobian.math.formal_concept_analysis._models import (
    AttributeSubsetRequest,
    ClosureResult,
    ConceptLatticeResult,
    ConceptResult,
    DerivationResult,
    DuquenneGuiguesBasisRequest,
    EnumerateConceptsRequest,
    EnumerateConceptsResult,
    ImplicationClosureRequest,
    ObjectSubsetRequest,
)
from jacobian.math.formal_concept_analysis.basis import (
    CanonicalImplicationBasisResult,
)
from jacobian.math.formal_concept_analysis.operations import (
    attribute_derivation,
    concept_from_attributes,
    concept_from_objects,
    concept_lattice,
    duquenne_guigues_basis,
    enumerate_concepts,
    implication_closure,
    object_derivation,
)
from jacobian.math.formal_concept_analysis.values import ImplicationClosureResult

__all__ = [
    "compute_attribute_derivation",
    "compute_concept_from_attributes",
    "compute_concept_from_objects",
    "compute_concept_lattice",
    "compute_duquenne_guigues_basis",
    "compute_enumerate_concepts",
    "compute_implication_closure",
    "compute_object_closure",
    "compute_object_derivation",
]


def compute_object_derivation(request: ObjectSubsetRequest) -> DerivationResult:
    result = object_derivation(request.context, frozenset(request.subset))
    return DerivationResult(derived=tuple(sorted(result)))


def compute_attribute_derivation(request: AttributeSubsetRequest) -> DerivationResult:
    result = attribute_derivation(request.context, frozenset(request.subset))
    return DerivationResult(derived=tuple(sorted(result)))


def compute_object_closure(request: ObjectSubsetRequest) -> ClosureResult:
    fs = frozenset(request.subset)
    derived = object_derivation(request.context, fs)
    closure = attribute_derivation(request.context, derived)
    added = tuple(sorted(set(closure) - set(fs)))
    return ClosureResult(
        closure=tuple(sorted(closure)),
        derived=tuple(sorted(derived)),
        added=added,
        is_closed=set(fs) == set(closure),
    )


def compute_attribute_closure(request: AttributeSubsetRequest) -> ClosureResult:
    fs = frozenset(request.subset)
    derived = attribute_derivation(request.context, fs)
    closure = object_derivation(request.context, derived)
    added = tuple(sorted(set(closure) - set(fs)))
    return ClosureResult(
        closure=tuple(sorted(closure)),
        derived=tuple(sorted(derived)),
        added=added,
        is_closed=set(fs) == set(closure),
    )


def compute_implication_closure(
    request: ImplicationClosureRequest,
) -> ImplicationClosureResult:
    return implication_closure(request.system, frozenset(request.seed))


def compute_duquenne_guigues_basis(
    request: DuquenneGuiguesBasisRequest,
) -> CanonicalImplicationBasisResult:
    return duquenne_guigues_basis(request.context)


def compute_concept_from_objects(request: ObjectSubsetRequest) -> ConceptResult:
    result = concept_from_objects(request.context, frozenset(request.subset))
    return ConceptResult(
        extent=tuple(sorted(result["extent"])),
        intent=tuple(sorted(result["intent"])),
    )


def compute_concept_from_attributes(request: AttributeSubsetRequest) -> ConceptResult:
    result = concept_from_attributes(request.context, frozenset(request.subset))
    return ConceptResult(
        extent=tuple(sorted(result["extent"])),
        intent=tuple(sorted(result["intent"])),
    )


def compute_enumerate_concepts(
    request: EnumerateConceptsRequest,
) -> EnumerateConceptsResult:
    concepts = enumerate_concepts(request.context)
    return EnumerateConceptsResult(
        concepts=tuple(
            (tuple(sorted(c["extent"])), tuple(sorted(c["intent"]))) for c in concepts
        ),
        count=len(concepts),
    )


def compute_concept_lattice(
    request: EnumerateConceptsRequest,
) -> ConceptLatticeResult:
    result = concept_lattice(request.context)
    concepts = cast(list[Any], result["concepts"])
    return ConceptLatticeResult(
        concepts=tuple(
            (tuple(sorted(c["extent"])), tuple(sorted(c["intent"]))) for c in concepts
        ),
        order=result["order"],  # type: ignore[arg-type]
        covers=result["covers"],  # type: ignore[arg-type]
        top=result["top"],  # type: ignore[arg-type]
        bottom=result["bottom"],  # type: ignore[arg-type]
    )
