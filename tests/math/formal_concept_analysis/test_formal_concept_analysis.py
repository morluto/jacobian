"""Tests for formal concept analysis operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.formal_concept_analysis import FormalContext
from jacobian.math.formal_concept_analysis._models import (
    AttributeSubsetRequest,
    EnumerateConceptsRequest,
    ObjectSubsetRequest,
)
from jacobian.math.formal_concept_analysis._operations import (
    compute_attribute_derivation,
    compute_concept_from_attributes,
    compute_concept_from_objects,
    compute_concept_lattice,
    compute_enumerate_concepts,
    compute_object_closure,
    compute_object_derivation,
)
from jacobian.math.formal_concept_analysis._tools import TOOLS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cross_context() -> FormalContext:
    """A cross-shaped context: o0 has a0, o1 has a1. Three concepts."""
    return FormalContext(
        objects=("o0", "o1"),
        attributes=("a0", "a1"),
        incidence=((0, 0), (1, 1)),
    )


def _diagonal_context() -> FormalContext:
    """A diagonal context: o0 has a0 and a1; o1 has a1. Three concepts."""
    return FormalContext(
        objects=("o0", "o1"),
        attributes=("a0", "a1"),
        incidence=((0, 0), (0, 1), (1, 1)),
    )


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def test_catalog_contains_only_audited_agent_outcomes() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "formal_context.objects.derivation.compute",
        "formal_context.attributes.derivation.compute",
        "formal_context.objects.closure.compute",
        "formal_context.concept.from_objects.compute",
        "formal_context.concept.from_attributes.compute",
        "formal_context.concepts.enumerate.compute",
        "formal_context.concept_lattice.compute",
        "formal_context.duquenne_guigues_basis.compute",
        "implication_system.closure.compute",
    }


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------


class TestDerivation:
    def test_empty_object_set_derives_all_attributes(self) -> None:
        result = compute_object_derivation(
            ObjectSubsetRequest(context=_cross_context(), subset=())
        )
        assert result.derived == (0, 1)

    def test_o0_derives_a0_only(self) -> None:
        result = compute_object_derivation(
            ObjectSubsetRequest(context=_cross_context(), subset=(0,))
        )
        assert result.derived == (0,)

    def test_empty_attribute_set_derives_all_objects(self) -> None:
        result = compute_attribute_derivation(
            AttributeSubsetRequest(context=_cross_context(), subset=())
        )
        assert result.derived == (0, 1)


# ---------------------------------------------------------------------------
# Closure
# ---------------------------------------------------------------------------


class TestClosure:
    def test_empty_object_set_closure(self) -> None:
        result = compute_object_closure(
            ObjectSubsetRequest(context=_cross_context(), subset=())
        )
        # A = {}, A' = {a0, a1}, A'' = {} (no object has both attributes).
        # Empty set is closed: A == A'' = {}.
        assert result.is_closed is True
        assert result.derived == (0, 1)
        assert result.closure == ()

    def test_o0_closure(self) -> None:
        result = compute_object_closure(
            ObjectSubsetRequest(context=_cross_context(), subset=(0,))
        )
        # A = {o0}, A' = {a0}, A'' = {g : has a0} = {o0}. So A'' = {o0}, closed.
        assert result.is_closed is True
        assert result.closure == (0,)


# ---------------------------------------------------------------------------
# Concept construction
# ---------------------------------------------------------------------------


class TestConcept:
    def test_concept_from_o0(self) -> None:
        result = compute_concept_from_objects(
            ObjectSubsetRequest(context=_cross_context(), subset=(0,))
        )
        assert result.extent == (0,)
        assert result.intent == (0,)

    def test_concept_from_a0(self) -> None:
        result = compute_concept_from_attributes(
            AttributeSubsetRequest(context=_cross_context(), subset=(0,))
        )
        assert result.extent == (0,)
        assert result.intent == (0,)

    def test_concepts_agree(self) -> None:
        from_objects = compute_concept_from_objects(
            ObjectSubsetRequest(context=_cross_context(), subset=(0,))
        )
        from_attrs = compute_concept_from_attributes(
            AttributeSubsetRequest(context=_cross_context(), subset=(0,))
        )
        assert from_objects == from_attrs


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


class TestEnumeration:
    def test_cross_context_has_three_concepts(self) -> None:
        result = compute_enumerate_concepts(
            EnumerateConceptsRequest(context=_cross_context())
        )
        # Cross context: ({}, {a0,a1}), ({o0}, {a0}), ({o1}, {a1})
        assert result.count == 4

    def test_diagonal_context_has_two_concepts(self) -> None:
        """cl(empty) = {a1} here, so the empty intent is not a concept."""
        result = compute_enumerate_concepts(
            EnumerateConceptsRequest(context=_diagonal_context())
        )
        # Diagonal: o0 has both a0 and a1; o1 has a1.
        # Concepts: ({o0,o1}, {a1}), ({o0}, {a0,a1})
        assert result.count == 2


# ---------------------------------------------------------------------------
# Defining-equation replay (#2266)
# ---------------------------------------------------------------------------


def _assert_defining_equations(context: FormalContext) -> None:
    from jacobian.math.formal_concept_analysis import (
        attribute_derivation,
        object_derivation,
    )

    for extent_tuple, intent_tuple in compute_enumerate_concepts(
        EnumerateConceptsRequest(context=context)
    ).concepts:
        extent = frozenset(extent_tuple)
        intent = frozenset(intent_tuple)
        assert object_derivation(context, extent) == intent
        assert attribute_derivation(context, intent) == extent


@pytest.mark.parametrize(
    "incidence",
    (
        ((0, 0), (0, 1), (1, 1)),
        ((0, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1)),
        ((0, 0),),
        ((0, 0), (0, 1), (0, 2), (1, 1), (2, 2)),
    ),
)
def test_every_emitted_concept_satisfies_defining_equations(
    incidence: tuple[tuple[int, int], ...],
) -> None:
    object_count = max((o for o, _a in incidence), default=-1) + 1
    attribute_count = max((a for _o, a in incidence), default=-1) + 1
    context = FormalContext(
        objects=tuple(f"o{i}" for i in range(object_count)),
        attributes=tuple(f"a{j}" for j in range(attribute_count)),
        incidence=incidence,
    )
    _assert_defining_equations(context)


def test_lattice_embedded_concepts_replay_defining_equations() -> None:
    from jacobian.math.formal_concept_analysis import (
        attribute_derivation,
        object_derivation,
    )

    context = _diagonal_context()
    result = compute_concept_lattice(
        EnumerateConceptsRequest(context=_diagonal_context())
    )
    assert len(result.concepts) == 2
    for extent_tuple, intent_tuple in result.concepts:
        extent = frozenset(extent_tuple)
        intent = frozenset(intent_tuple)
        assert object_derivation(context, extent) == intent
        assert attribute_derivation(context, intent) == extent
    # Extents are pairwise distinct, so no two concepts compare equal.
    extents = [frozenset(extent_tuple) for extent_tuple, _intent in result.concepts]
    assert len(set(extents)) == len(extents)

    def test_sparse_context_beyond_twenty_attributes_is_enumerated(self) -> None:
        # One object with no incidences over 21 attributes carries exactly the
        # two trivial concepts, so admission must not reject it by attribute
        # count before considering actual enumeration work.
        context = FormalContext(
            objects=("o0",),
            attributes=tuple(f"a{index}" for index in range(21)),
            incidence=(),
        )
        result = compute_enumerate_concepts(EnumerateConceptsRequest(context=context))
        assert result.count == 2

    def test_attribute_fallback_boundary_is_admitted_and_rejected(self) -> None:
        wide = FormalContext(
            objects=("o0",),
            attributes=tuple(f"a{index}" for index in range(64)),
            incidence=(),
        )
        assert (
            compute_enumerate_concepts(EnumerateConceptsRequest(context=wide)).count
            == 2
        )
        with pytest.raises(ValidationError):
            EnumerateConceptsRequest(
                context=FormalContext(
                    objects=("o0",),
                    attributes=tuple(f"a{index}" for index in range(65)),
                    incidence=(),
                )
            )


# ---------------------------------------------------------------------------
# Concept lattice
# ---------------------------------------------------------------------------


class TestConceptLattice:
    def test_cross_lattice(self) -> None:
        result = compute_concept_lattice(
            EnumerateConceptsRequest(context=_cross_context())
        )
        assert result.top is not None
        assert result.bottom is not None
        assert len(result.concepts) == 4

    def test_sparse_lattice_beyond_twenty_attributes_is_computed(self) -> None:
        context = FormalContext(
            objects=("o0",),
            attributes=tuple(f"a{index}" for index in range(21)),
            incidence=(),
        )
        result = compute_concept_lattice(EnumerateConceptsRequest(context=context))
        assert len(result.concepts) == 2
        assert result.top is not None
        assert result.bottom is not None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_duplicate_objects_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unique"):
            FormalContext(
                objects=("o0", "o0"),
                attributes=("a0",),
                incidence=(),
            )

    def test_out_of_range_incidence_rejected(self) -> None:
        with pytest.raises(ValidationError, match="out of range"):
            FormalContext(
                objects=("o0",),
                attributes=("a0",),
                incidence=((0, 5),),
            )

    def test_duplicate_incidence_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate-free"):
            FormalContext(
                objects=("o0",),
                attributes=("a0",),
                incidence=((0, 0), (0, 0)),
            )
