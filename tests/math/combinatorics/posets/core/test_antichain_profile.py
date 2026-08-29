"""Regression coverage for the antichain-profile execution envelope."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.posets.core._models import (
    MAX_ANTICHAIN_PROFILE_CANDIDATES,
    FinitePoset,
    FinitePosetRequest,
    PresentationPair,
    RelationInterpretation,
)
from jacobian.math.combinatorics.posets.core.operations import (
    antichain_profile,
    materialize_finite_poset,
)


def _chain(size: int) -> FinitePoset:
    elements = tuple(f"v{index}" for index in range(size))
    request = FinitePosetRequest(
        elements=elements,
        relation=tuple(
            PresentationPair(lower=elements[index], upper=elements[index + 1])
            for index in range(size - 1)
        ),
        interpretation=RelationInterpretation.COVER_EDGES,
    )
    return materialize_finite_poset(
        request.elements,
        request.relation,
        request.interpretation,
        request.reflexive_pairs,
    )


def test_antichain_profile_executes_the_admitted_fourteen_element_chain() -> None:
    result = antichain_profile(_chain(14))

    assert result.antichain_count == 15
    assert result.maximum_antichains == (
        ("v0",),
        ("v1",),
        ("v10",),
        ("v11",),
        ("v12",),
        ("v13",),
        ("v2",),
        ("v3",),
        ("v4",),
        ("v5",),
        ("v6",),
        ("v7",),
        ("v8",),
        ("v9",),
    )
    assert MAX_ANTICHAIN_PROFILE_CANDIDATES == 16_384


def test_empty_poset_has_the_empty_antichain_as_its_unique_maximum() -> None:
    result = antichain_profile(_chain(0))

    assert result.maximum_antichain_size == 0
    assert result.antichain_count == 1
    assert result.maximum_antichains == ((),)


def test_antichain_profile_rejects_the_next_exponential_envelope() -> None:
    with pytest.raises(OperationDomainValidationError, match="candidate subsets"):
        antichain_profile(_chain(15))
