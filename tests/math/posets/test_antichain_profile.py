"""Regression coverage for the antichain-profile execution envelope."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.posets._models import (
    MAX_ANTICHAIN_PROFILE_CANDIDATES,
    AntichainProfileRequest,
    FinitePoset,
    FinitePosetRequest,
    PresentationPair,
    RelationInterpretation,
)
from jacobian.math.posets._operations import _antichain_profile, _materialized_poset


def _chain(size: int) -> FinitePoset:
    elements = tuple(f"v{index}" for index in range(size))
    return _materialized_poset(
        FinitePosetRequest(
            elements=elements,
            relation=tuple(
                PresentationPair(lower=elements[index], upper=elements[index + 1])
                for index in range(size - 1)
            ),
            interpretation=RelationInterpretation.COVER_EDGES,
        )
    )


def test_antichain_profile_executes_the_admitted_fourteen_element_chain() -> None:
    result = _antichain_profile(AntichainProfileRequest(poset=_chain(14)))

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


def test_antichain_profile_rejects_the_next_exponential_envelope() -> None:
    with pytest.raises(ValidationError, match="candidate subsets"):
        AntichainProfileRequest(poset=_chain(15))
