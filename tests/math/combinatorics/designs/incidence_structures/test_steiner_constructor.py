"""Bounded exact Steiner triple-system construction tests (#1666)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.combinatorics.designs.incidence_structures._models import (
    SteinerTripleSystemRequest,
)
from jacobian.math.combinatorics.designs.incidence_structures._tools import (
    _steiner_triple_system,
)


def test_construct_fano_plane_and_replay_pairs() -> None:
    result = _steiner_triple_system(
        SteinerTripleSystemRequest(order=7, search_budget=100_000)
    )
    assert result.status == "COMPUTED"
    assert result.design is not None
    assert len(result.design.blocks) == 7
    pairs: list[tuple[str, str]] = []
    for block in result.design.blocks:
        assert len(block) == 3
        pairs.extend((block[i], block[j]) for i in range(3) for j in range(i + 1, 3))
    assert len(pairs) == 21
    assert len(set(pairs)) == 21


def test_construct_trivial_sts3() -> None:
    result = _steiner_triple_system(
        SteinerTripleSystemRequest(order=3, search_budget=100)
    )
    assert result.status == "COMPUTED"
    assert result.design is not None
    assert result.design.blocks == (("p0", "p1", "p2"),)


def test_budget_exhaustion_is_unknown() -> None:
    result = _steiner_triple_system(
        SteinerTripleSystemRequest(order=7, search_budget=1)
    )
    assert result.status == "UNKNOWN"
    assert result.design is None


def test_necessary_parameter_condition_rejects_order() -> None:
    with pytest.raises(ValidationError, match="congruent to 1 or 3"):
        SteinerTripleSystemRequest(order=5, search_budget=100)
