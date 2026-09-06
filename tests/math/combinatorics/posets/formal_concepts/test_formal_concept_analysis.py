"""Tests for formal concept analysis operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.combinatorics.posets.formal_concepts import (
    FormalContext,
    verify_closure,
    verify_concept,
    verify_concept_lattice,
    verify_derivation,
    verify_enumerate_concepts,
)
from jacobian.math.combinatorics.posets.formal_concepts._models import (
    AttributeSubsetRequest,
    EnumerateConceptsRequest,
    ObjectSubsetRequest,
)
from jacobian.math.combinatorics.posets.formal_concepts._tools import (
    TOOLS,
    compute_attribute_closure,
    compute_attribute_derivation,
    compute_concept_from_attributes,
    compute_concept_from_objects,
    compute_concept_lattice,
    compute_enumerate_concepts,
    compute_object_closure,
    compute_object_derivation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cross_context() -> FormalContext:
    """A cross-shaped context: o0 has a0, o1 has a1. Four concepts."""
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
        "formal_context.attributes.closure.compute",
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
        assert result.derived.indices == (0, 1)

    def test_o0_derives_a0_only(self) -> None:
        result = compute_object_derivation(
            ObjectSubsetRequest(context=_cross_context(), subset=(0,))
        )
        assert result.derived.indices == (0,)

    def test_result_rejects_subset_from_another_context(self) -> None:
        result = compute_object_derivation(
            ObjectSubsetRequest(context=_cross_context(), subset=(0,))
        )
        forged = result.model_dump(mode="json")
        forged["subset"]["context"] = _diagonal_context().model_dump(mode="json")
        with pytest.raises(ValidationError, match="retained context"):
            type(result).model_validate(forged)

    def test_empty_attribute_set_derives_all_objects(self) -> None:
        result = compute_attribute_derivation(
            AttributeSubsetRequest(context=_cross_context(), subset=())
        )
        assert result.derived.indices == (0, 1)


# ---------------------------------------------------------------------------
# Closure
# ---------------------------------------------------------------------------


class TestClosure:
    def test_serialized_closure_claim_is_source_bound_and_verifiable(self) -> None:
        result = compute_object_closure(
            ObjectSubsetRequest(context=_cross_context(), subset=(0,))
        )
        decoded = type(result).model_validate_json(result.model_dump_json())
        assert verify_closure(decoded)

        forged = result.model_dump(mode="json")
        forged["is_closed"] = False
        assert not verify_closure(type(result).model_validate(forged))

    def test_empty_object_set_closure(self) -> None:
        result = compute_object_closure(
            ObjectSubsetRequest(context=_cross_context(), subset=())
        )
        # A = {}, A' = {a0, a1}, A'' = {} (no object has both attributes).
        # Empty set is closed: A == A'' = {}.
        assert result.is_closed is True
        assert result.derived.indices == (0, 1)
        assert result.closure.indices == ()

    def test_o0_closure(self) -> None:
        result = compute_object_closure(
            ObjectSubsetRequest(context=_cross_context(), subset=(0,))
        )
        # A = {o0}, A' = {a0}, A'' = {g : has a0} = {o0}. So A'' = {o0}, closed.
        assert result.is_closed is True
        assert result.closure.indices == (0,)

    def test_empty_attribute_set_closure(self) -> None:
        result = compute_attribute_closure(
            AttributeSubsetRequest(context=_cross_context(), subset=())
        )
        # B = {}, B' = {o0, o1}, B'' = {} because the objects share no attribute.
        assert result.is_closed is True
        assert result.derived.indices == (0, 1)
        assert result.closure.indices == ()

    def test_attribute_closure_round_trips(self) -> None:
        tool = next(
            tool
            for tool in TOOLS
            if tool.operation_id == "formal_context.attributes.closure.compute"
        )
        request = tool.request_type.model_validate(tool.examples[0].input)
        result = tool.run(request)

        assert tool.result_type.model_validate_json(result.model_dump_json()) == result


# ---------------------------------------------------------------------------
# Concept construction
# ---------------------------------------------------------------------------


class TestConcept:
    def test_concept_from_o0(self) -> None:
        result = compute_concept_from_objects(
            ObjectSubsetRequest(context=_cross_context(), subset=(0,))
        )
        assert result.extent.indices == (0,)
        assert result.intent.indices == (0,)

    def test_concept_from_a0(self) -> None:
        result = compute_concept_from_attributes(
            AttributeSubsetRequest(context=_cross_context(), subset=(0,))
        )
        assert result.extent.indices == (0,)
        assert result.intent.indices == (0,)

    def test_concepts_agree(self) -> None:
        context = _cross_context()
        from_objects = compute_concept_from_objects(
            ObjectSubsetRequest(context=context, subset=(0,))
        )
        from_attrs = compute_concept_from_attributes(
            AttributeSubsetRequest(context=context, subset=(0,))
        )
        assert from_objects.extent == from_attrs.extent
        assert from_objects.intent == from_attrs.intent

    def test_serialized_concept_claim_is_source_bound_and_verifiable(self) -> None:
        result = compute_concept_from_objects(
            ObjectSubsetRequest(context=_cross_context(), subset=(0,))
        )
        decoded = type(result).model_validate_json(result.model_dump_json())
        assert verify_concept(decoded)

        forged = result.model_dump(mode="json")
        forged["extent"]["indices"] = [1]
        assert not verify_concept(type(result).model_validate(forged))


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


class TestEnumeration:
    def test_serialized_derivation_claim_is_source_bound_and_verifiable(self) -> None:
        result = compute_object_derivation(
            ObjectSubsetRequest(context=_cross_context(), subset=(0,))
        )
        decoded = type(result).model_validate_json(result.model_dump_json())
        assert verify_derivation(decoded)

        forged = result.model_dump(mode="json")
        forged["derived"]["indices"] = [1]
        assert not verify_derivation(type(result).model_validate(forged))

    def test_cross_context_has_four_concepts(self) -> None:
        result = compute_enumerate_concepts(
            EnumerateConceptsRequest(context=_cross_context())
        )
        # Cross context: ({}, {a0,a1}), ({o0}, {a0}), ({o1}, {a1}),
        # and ({o0,o1}, {}).
        assert result.count == 4

    def test_serialized_enumeration_claim_is_source_bound_and_verifiable(self) -> None:
        result = compute_enumerate_concepts(
            EnumerateConceptsRequest(context=_cross_context())
        )
        decoded = type(result).model_validate_json(result.model_dump_json())
        assert verify_enumerate_concepts(decoded)

        forged = result.model_dump(mode="json")
        forged["count"] = 3
        assert not verify_enumerate_concepts(type(result).model_validate(forged))

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

    @pytest.mark.scale
    def test_contranominal_context_beyond_result_budget_is_rejected(self) -> None:
        """A 21x21 contranominal context has exactly 2^21 concepts, beyond
        the declared enumeration budget: native admission rejects it during
        execution rather than hiding enumeration inside wire parsing."""
        request = EnumerateConceptsRequest(context=self._contranominal(21))
        with pytest.raises(ValueError, match="concept count exceeds"):
            compute_enumerate_concepts(request)

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

    @pytest.mark.scale
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

    def test_wide_context_is_admitted_by_the_native_operation(self) -> None:
        """Wire parsing stays structural while each native operation admits work."""
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
    from jacobian.math.combinatorics.posets.formal_concepts import (
        attribute_derivation,
        object_derivation,
    )

    for concept in compute_enumerate_concepts(
        EnumerateConceptsRequest(context=context)
    ).concepts:
        extent = frozenset(concept.extent.indices)
        intent = frozenset(concept.intent.indices)
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
    from jacobian.math.combinatorics.posets.formal_concepts import (
        attribute_derivation,
        object_derivation,
    )

    context = _diagonal_context()
    result = compute_concept_lattice(EnumerateConceptsRequest(context=context))
    assert len(result.concepts) == 2
    for concept in result.concepts:
        extent = frozenset(concept.extent.indices)
        intent = frozenset(concept.intent.indices)
        assert object_derivation(context, extent) == intent
        assert attribute_derivation(context, intent) == extent
    # Extents are pairwise distinct, so no two concepts compare equal.
    extents = [frozenset(concept.extent.indices) for concept in result.concepts]
    assert len(set(extents)) == len(extents)


# ---------------------------------------------------------------------------
# Concept lattice
# ---------------------------------------------------------------------------


class TestConceptLattice:
    def test_serialized_lattice_claim_is_source_bound_and_verifiable(self) -> None:
        result = compute_concept_lattice(
            EnumerateConceptsRequest(context=_cross_context())
        )
        decoded = type(result).model_validate_json(result.model_dump_json())
        assert verify_concept_lattice(decoded)

        forged = result.model_dump(mode="json")
        forged["covers"] = []
        assert not verify_concept_lattice(type(result).model_validate(forged))

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
