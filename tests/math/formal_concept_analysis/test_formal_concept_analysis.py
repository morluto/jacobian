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

    @staticmethod
    def _contranominal(axis_size: int) -> FormalContext:
        """The contranominal scale on n objects and n attributes: object i
        has every attribute except i. It carries exactly 2^n concepts."""
        n = axis_size
        return FormalContext(
            objects=tuple(f"o{index}" for index in range(n)),
            attributes=tuple(f"a{index}" for index in range(n)),
            incidence=tuple(
                (object_index, attribute_index)
                for object_index in range(n)
                for attribute_index in range(n)
                if attribute_index != object_index
            ),
        )

    def test_contranominal_context_beyond_result_budget_is_rejected(self) -> None:
        """A 21x21 contranominal context has exactly 2^21 concepts, beyond
        the declared enumeration budget: admission must reject it with one
        capped preflight instead of raising mid-enumeration."""
        with pytest.raises(ValidationError):
            EnumerateConceptsRequest(context=self._contranominal(21))

    def test_empty_square_context_beyond_worst_case_stays_admissible(self) -> None:
        """A 14x14 empty-incidence context has worst case 2^14 but only two
        actual concepts, so the exact preflight must admit it and the
        operation must return the complete family."""
        context = FormalContext(
            objects=tuple(f"o{index}" for index in range(14)),
            attributes=tuple(f"a{index}" for index in range(14)),
            incidence=(),
        )
        request = EnumerateConceptsRequest(context=context)
        result = compute_enumerate_concepts(request)
        assert result.count == 2

    def test_contranominal_boundary_context_enumerates_complete_family(self) -> None:
        """13 is the static boundary of the smaller axis (2^13 = 8192 fits
        the budget; 2^14 does not), and the boundary context returns the
        complete family as a typed result."""
        result = compute_enumerate_concepts(
            EnumerateConceptsRequest(context=self._contranominal(13))
        )
        assert result.count == 2**13 == 8192

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

    def test_attribute_fallback_boundary_is_admitted(self) -> None:
        wide = FormalContext(
            objects=("o0",),
            attributes=tuple(f"a{index}" for index in range(64)),
            incidence=(),
        )
        assert (
            compute_enumerate_concepts(EnumerateConceptsRequest(context=wide)).count
            == 2
        )

    def test_formal_context_rejects_one_attribute_above_its_contract_cap(self) -> None:
        with pytest.raises(ValidationError) as error:
            FormalContext(
                objects=("o0",),
                attributes=tuple(f"a{index}" for index in range(65)),
                incidence=(),
            )
        detail = error.value.errors()[0]
        assert detail["type"] == "too_long"
        assert detail["loc"] == ("attributes",)
        assert detail["ctx"] == {
            "field_type": "Tuple",
            "max_length": 64,
            "actual_length": 65,
        }

    def test_wide_admission_reuses_its_one_exact_concept_enumeration(self) -> None:
        """The exact wide-context admission probe is the served family."""
        context = FormalContext(
            objects=tuple(f"o{index}" for index in range(14)),
            attributes=tuple(f"a{index}" for index in range(14)),
            incidence=(),
        )
        request = EnumerateConceptsRequest(context=context)
        assert compute_enumerate_concepts(request).count == 2

        assert len(compute_concept_lattice(request).concepts) == 2


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
        with pytest.raises(ValidationError):
            FormalContext(
                objects=("o0", "o0"),
                attributes=("a0",),
                incidence=(),
            )

    def test_out_of_range_incidence_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FormalContext(
                objects=("o0",),
                attributes=("a0",),
                incidence=((0, 5),),
            )

    def test_duplicate_incidence_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FormalContext(
                objects=("o0",),
                attributes=("a0",),
                incidence=((0, 0), (0, 0)),
            )
