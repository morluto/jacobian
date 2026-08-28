"""Domain adapter for formal concept analysis operations."""

from __future__ import annotations

from jacobian.math.combinatorics.posets.formal_concepts._concepts import (
    enumerate_concept_pairs,
)
from jacobian.math.combinatorics.posets.formal_concepts._models import (
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
from jacobian.math.combinatorics.posets.formal_concepts.basis import (
    CanonicalImplicationBasisResult,
)
from jacobian.math.combinatorics.posets.formal_concepts.operations import (
    _concept_lattice_from_canonical_concepts,
    attribute_derivation,
    concept_from_attributes,
    concept_from_objects,
    duquenne_guigues_basis,
    implication_closure,
    object_derivation,
)
from jacobian.math.combinatorics.posets.formal_concepts.values import (
    ImplicationClosureResult,
)

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
    return concept_from_objects(request.context, frozenset(request.subset))


def compute_concept_from_attributes(request: AttributeSubsetRequest) -> ConceptResult:
    return concept_from_attributes(request.context, frozenset(request.subset))


def compute_enumerate_concepts(
    request: EnumerateConceptsRequest,
) -> EnumerateConceptsResult:
    concepts = enumerate_concept_pairs(request.context)
    return EnumerateConceptsResult(
        concepts=concepts,
        count=len(concepts),
    )


def compute_concept_lattice(
    request: EnumerateConceptsRequest,
) -> ConceptLatticeResult:
    return _concept_lattice_from_canonical_concepts(
        enumerate_concept_pairs(request.context)
    )
