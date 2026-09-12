"""Complete maximal-chain enumeration from canonical cover relations."""

from itertools import combinations, pairwise
from typing import Any, cast

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.posets.core._maximal_chains import (
    MAX_MAXIMAL_CHAIN_ELEMENT_SLOTS,
    MaximalChainEnumerationResult,
    enumerate_maximal_chains,
)
from jacobian.math.combinatorics.posets.core._models import (
    OrderedPair,
    PresentationPair,
    ReflexivePairPolicy,
    RelationInterpretation,
)
from jacobian.math.combinatorics.posets.core.operations import (
    dual_poset,
    materialize_finite_poset,
    maximal_chains,
)


def test_diamond_chains_endpoints_and_histogram() -> None:
    poset = materialize_finite_poset(
        ("0", "a", "b", "1"),
        tuple(
            PresentationPair(lower=lower, upper=upper)
            for lower, upper in (("0", "a"), ("0", "b"), ("a", "1"), ("b", "1"))
        ),
        RelationInterpretation.COVER_EDGES,
        ReflexivePairPolicy.FORBIDDEN,
    )
    result = enumerate_maximal_chains(poset)
    assert tuple(row.elements for row in result.chains) == (
        ("0", "a", "1"),
        ("0", "b", "1"),
    )
    assert all(
        row.lower_endpoint == "0" and row.upper_endpoint == "1" for row in result.chains
    )
    assert tuple(row.cover_relations for row in result.chains) == (
        (OrderedPair(lower="0", upper="a"), OrderedPair(lower="a", upper="1")),
        (OrderedPair(lower="0", upper="b"), OrderedPair(lower="b", upper="1")),
    )
    assert [(cell.length, cell.count) for cell in result.length_histogram] == [(3, 2)]
    assert maximal_chains(poset) == result


def test_empty_poset_has_single_empty_maximal_chain() -> None:
    poset = materialize_finite_poset(
        (), (), RelationInterpretation.COVER_EDGES, ReflexivePairPolicy.FORBIDDEN
    )
    result = enumerate_maximal_chains(poset)
    assert result.chains[0].elements == ()
    assert result.chains[0].lower_endpoint is None
    assert [(cell.length, cell.count) for cell in result.length_histogram] == [(0, 1)]


def test_singleton_and_disconnected_posets_are_complete() -> None:
    singleton = materialize_finite_poset(
        ("x",), (), RelationInterpretation.COVER_EDGES, ReflexivePairPolicy.FORBIDDEN
    )
    assert enumerate_maximal_chains(singleton).chains[0].elements == ("x",)

    disconnected = materialize_finite_poset(
        ("a", "b"),
        (),
        RelationInterpretation.COVER_EDGES,
        ReflexivePairPolicy.FORBIDDEN,
    )
    result = enumerate_maximal_chains(disconnected)
    assert tuple(row.elements for row in result.chains) == (("a",), ("b",))
    assert [(cell.length, cell.count) for cell in result.length_histogram] == [(1, 2)]


def test_inclusion_maximal_chains_include_different_lengths() -> None:
    poset = materialize_finite_poset(
        ("s", "a", "b", "t"),
        tuple(
            PresentationPair(lower=lower, upper=upper)
            for lower, upper in (("s", "a"), ("s", "b"), ("b", "t"))
        ),
        RelationInterpretation.COVER_EDGES,
        ReflexivePairPolicy.FORBIDDEN,
    )
    result = enumerate_maximal_chains(poset)
    assert tuple(row.elements for row in result.chains) == (
        ("s", "a"),
        ("s", "b", "t"),
    )
    assert [(cell.length, cell.count) for cell in result.length_histogram] == [
        (2, 1),
        (3, 1),
    ]


def test_boolean_lattices_b2_and_b3_are_complete_by_independent_oracle() -> None:
    for dimension, expected_count in ((2, 2), (3, 6)):
        elements = tuple(str(mask) for mask in range(1 << dimension))
        relations = tuple(
            PresentationPair(lower=str(lower), upper=str(upper))
            for lower in range(1 << dimension)
            for upper in range(1 << dimension)
            if lower != upper
            and lower & ~upper == 0
            and (upper ^ lower).bit_count() == 1
        )
        poset = materialize_finite_poset(
            elements,
            relations,
            RelationInterpretation.COVER_EDGES,
            ReflexivePairPolicy.FORBIDDEN,
        )
        result = enumerate_maximal_chains(poset)

        strict = {(pair.lower, pair.upper) for pair in poset.strict_order_pairs}

        def is_chain(
            candidate: tuple[str, ...], *, strict: set[tuple[str, str]] = strict
        ) -> bool:
            return all(
                (lower, upper) in strict for lower, upper in combinations(candidate, 2)
            )

        def is_inclusion_maximal(
            candidate: tuple[str, ...], *, elements: tuple[str, ...] = elements
        ) -> bool:
            return all(
                not is_chain(tuple(sorted((*candidate, element))))
                for element in set(elements) - set(candidate)
            )

        oracle = {
            candidate
            for size in range(1, len(elements) + 1)
            for subset in combinations(elements, size)
            for candidate in (subset,)
            if is_chain(candidate) and is_inclusion_maximal(candidate)
        }
        assert len(result.chains) == expected_count
        assert {row.elements for row in result.chains} == oracle
        assert all(
            tuple(row.cover_relations)
            == tuple(
                OrderedPair(lower=lower, upper=upper)
                for lower, upper in pairwise(row.elements)
            )
            for row in result.chains
        )


def test_dual_reverses_each_chain_and_preserves_histogram() -> None:
    poset = materialize_finite_poset(
        ("0", "a", "b", "1"),
        tuple(
            PresentationPair(lower=lower, upper=upper)
            for lower, upper in (("0", "a"), ("0", "b"), ("a", "1"), ("b", "1"))
        ),
        RelationInterpretation.COVER_EDGES,
        ReflexivePairPolicy.FORBIDDEN,
    )
    dual = dual_poset(poset)
    source = enumerate_maximal_chains(poset)
    reversed_result = enumerate_maximal_chains(dual.poset)
    assert tuple(tuple(reversed(row.elements)) for row in source.chains) == tuple(
        row.elements for row in reversed_result.chains
    )
    assert reversed_result.length_histogram == source.length_histogram


def test_serialized_result_round_trips_and_rejects_tampered_cover_claim() -> None:
    poset = materialize_finite_poset(
        ("a", "b"),
        (PresentationPair(lower="a", upper="b"),),
        RelationInterpretation.COVER_EDGES,
        ReflexivePairPolicy.FORBIDDEN,
    )
    result = enumerate_maximal_chains(poset)
    assert type(result).model_validate_json(result.model_dump_json()) == result

    tampered = poset.model_copy(update={"cover_relations": ()})
    with pytest.raises(OperationDomainValidationError, match="canonical"):
        enumerate_maximal_chains(tampered)


def test_serialized_result_rejects_contradictory_structural_row() -> None:
    poset = materialize_finite_poset(
        ("a", "b"),
        (PresentationPair(lower="a", upper="b"),),
        RelationInterpretation.COVER_EDGES,
        ReflexivePairPolicy.FORBIDDEN,
    )
    result = enumerate_maximal_chains(poset)
    payload = result.model_dump(mode="json")
    payload["chains"][0]["length"] = 1
    with pytest.raises(ValidationError, match="chain length"):
        MaximalChainEnumerationResult.model_validate(payload)


def test_serialized_result_rejects_noncover_chain() -> None:
    poset = materialize_finite_poset(
        ("a", "b", "c"),
        (
            PresentationPair(lower="a", upper="b"),
            PresentationPair(lower="b", upper="c"),
        ),
        RelationInterpretation.COVER_EDGES,
        ReflexivePairPolicy.FORBIDDEN,
    )
    result = enumerate_maximal_chains(poset)
    payload = result.model_dump(mode="json")
    payload["chains"][0]["elements"] = ["a", "c"]
    payload["chains"][0]["cover_relations"] = [{"lower": "a", "upper": "c"}]
    payload["chains"][0]["length"] = 2
    with pytest.raises(ValidationError, match="source Hasse"):
        MaximalChainEnumerationResult.model_validate(payload)


def test_direct_native_guard_rejects_untyped_request() -> None:
    with pytest.raises(OperationDomainValidationError, match="typed finite poset"):
        enumerate_maximal_chains(cast(Any, None))


def test_complete_profile_rejects_predicted_result_explosion() -> None:
    # Sixteen binary layers have 65,536 maximal chains.  Their repeated
    # element slots exceed the semantic complete-result envelope even though
    # the row count itself remains below its limit.
    elements = tuple(f"x{layer}_{branch}" for layer in range(16) for branch in (0, 1))
    relations = tuple(
        PresentationPair(
            lower=f"x{layer}_{branch}", upper=f"x{layer + 1}_{next_branch}"
        )
        for layer in range(15)
        for branch in (0, 1)
        for next_branch in (0, 1)
    )
    poset = materialize_finite_poset(
        elements,
        relations,
        RelationInterpretation.COVER_EDGES,
        ReflexivePairPolicy.FORBIDDEN,
    )
    with pytest.raises(
        OperationResourceAdmissionError,
        match=str(MAX_MAXIMAL_CHAIN_ELEMENT_SLOTS),
    ):
        enumerate_maximal_chains(poset)


def test_complete_profile_accepts_fifteen_layer_binary_boundary() -> None:
    # Fifteen binary layers retain 32,768 chains and 491,520 element slots.
    # This accepted boundary exercises linear histogram validation at the
    # scale that previously caused quadratic repeated-count work.
    elements = tuple(f"x{layer}_{branch}" for layer in range(15) for branch in (0, 1))
    relations = tuple(
        PresentationPair(
            lower=f"x{layer}_{branch}", upper=f"x{layer + 1}_{next_branch}"
        )
        for layer in range(14)
        for branch in (0, 1)
        for next_branch in (0, 1)
    )
    poset = materialize_finite_poset(
        elements,
        relations,
        RelationInterpretation.COVER_EDGES,
        ReflexivePairPolicy.FORBIDDEN,
    )

    result = enumerate_maximal_chains(poset)

    assert len(result.chains) == 2**15
    assert [(cell.length, cell.count) for cell in result.length_histogram] == [
        (15, 2**15)
    ]
