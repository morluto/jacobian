"""Complete maximal-chain enumeration from canonical cover relations."""

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.posets.core._maximal_chains import (
    MAX_MAXIMAL_CHAIN_RESULT_BYTES,
    enumerate_maximal_chains,
)
from jacobian.math.combinatorics.posets.core._models import (
    PosetRequest,
    PresentationPair,
    ReflexivePairPolicy,
    RelationInterpretation,
)
from jacobian.math.combinatorics.posets.core.operations import (
    dual_poset,
    materialize_finite_poset,
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
    result = enumerate_maximal_chains(PosetRequest(poset=poset))
    assert tuple(row.elements for row in result.chains) == (
        ("0", "a", "1"),
        ("0", "b", "1"),
    )
    assert all(
        row.lower_endpoint == "0" and row.upper_endpoint == "1" for row in result.chains
    )
    assert [(cell.length, cell.count) for cell in result.length_histogram] == [(3, 2)]


def test_empty_poset_has_single_empty_maximal_chain() -> None:
    poset = materialize_finite_poset(
        (), (), RelationInterpretation.COVER_EDGES, ReflexivePairPolicy.FORBIDDEN
    )
    result = enumerate_maximal_chains(PosetRequest(poset=poset))
    assert result.chains[0].elements == ()
    assert result.chains[0].lower_endpoint is None
    assert [(cell.length, cell.count) for cell in result.length_histogram] == [(0, 1)]


def test_singleton_and_disconnected_posets_are_complete() -> None:
    singleton = materialize_finite_poset(
        ("x",), (), RelationInterpretation.COVER_EDGES, ReflexivePairPolicy.FORBIDDEN
    )
    assert enumerate_maximal_chains(PosetRequest(poset=singleton)).chains[
        0
    ].elements == ("x",)

    disconnected = materialize_finite_poset(
        ("a", "b"),
        (),
        RelationInterpretation.COVER_EDGES,
        ReflexivePairPolicy.FORBIDDEN,
    )
    result = enumerate_maximal_chains(PosetRequest(poset=disconnected))
    assert tuple(row.elements for row in result.chains) == (("a",), ("b",))
    assert [(cell.length, cell.count) for cell in result.length_histogram] == [(1, 2)]


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
    source = enumerate_maximal_chains(PosetRequest(poset=poset))
    reversed_result = enumerate_maximal_chains(PosetRequest(poset=dual.poset))
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
    result = enumerate_maximal_chains(PosetRequest(poset=poset))
    assert type(result).model_validate_json(result.model_dump_json()) == result

    tampered = poset.model_copy(update={"cover_relations": ()})
    with pytest.raises(OperationDomainValidationError, match="canonical"):
        enumerate_maximal_chains(PosetRequest.model_construct(poset=tampered))


def test_complete_profile_rejects_predicted_result_explosion() -> None:
    # Sixteen binary layers have 65,536 maximal chains.  Their repeated
    # labels alone exceed the complete canonical result envelope.
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
        OperationResourceAdmissionError, match=str(MAX_MAXIMAL_CHAIN_RESULT_BYTES)
    ):
        enumerate_maximal_chains(PosetRequest(poset=poset))
