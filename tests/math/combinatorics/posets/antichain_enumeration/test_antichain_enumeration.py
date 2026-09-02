from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.posets.antichain_enumeration._models import (
    AntichainEnumerationRequest,
)
from jacobian.math.combinatorics.posets.antichain_enumeration.operations import (
    enumerate_antichains,
)
from jacobian.math.combinatorics.posets.core._models import (
    FinitePoset,
    PresentationPair,
    ReflexivePairPolicy,
    RelationInterpretation,
)
from jacobian.math.combinatorics.posets.core.operations import (
    materialize_finite_poset,
)


def _make_chain(n: int) -> FinitePoset:
    elements = tuple(str(i) for i in range(n))
    relation = tuple(
        PresentationPair(lower=str(i), upper=str(j))
        for i in range(n)
        for j in range(i + 1, n)
    )
    return materialize_finite_poset(
        elements,
        relation,
        RelationInterpretation.COMPARABLE_PAIRS,
        ReflexivePairPolicy.FORBIDDEN,
    )


def _make_antichain_poset(n: int) -> FinitePoset:
    """A poset with n incomparable elements."""
    elements = tuple(str(i) for i in range(n))
    return materialize_finite_poset(
        elements,
        (),
        RelationInterpretation.COMPARABLE_PAIRS,
        ReflexivePairPolicy.FORBIDDEN,
    )


def test_chain_antichains_size_1() -> None:
    poset = _make_chain(3)
    result = enumerate_antichains(poset, 1, 1)
    assert result.count == 3
    assert ("0",) in result.antichains
    assert ("1",) in result.antichains
    assert ("2",) in result.antichains


def test_chain_antichains_size_2() -> None:
    poset = _make_chain(3)
    result = enumerate_antichains(poset, 2, 2)
    assert result.count == 0


def test_chain_all_sizes() -> None:
    poset = _make_chain(3)
    result = enumerate_antichains(poset, 1, 3)
    assert result.count == 3


def test_empty_set() -> None:
    poset = _make_chain(3)
    result = enumerate_antichains(poset, 0, 0)
    assert result.count == 1
    assert () in result.antichains


def test_antichain_poset_size_1() -> None:
    poset = _make_antichain_poset(3)
    result = enumerate_antichains(poset, 1, 1)
    assert result.count == 3


def test_antichain_poset_size_2() -> None:
    poset = _make_antichain_poset(3)
    result = enumerate_antichains(poset, 2, 2)
    assert result.count == 3  # C(3,2) = 3


def test_antichain_poset_all_sizes() -> None:
    poset = _make_antichain_poset(4)
    result = enumerate_antichains(poset, 1, 4)
    # C(4,1) + C(4,2) + C(4,3) + C(4,4) = 4 + 6 + 4 + 1 = 15
    assert result.count == 15


def test_range_cardinalities() -> None:
    poset = _make_antichain_poset(5)
    result = enumerate_antichains(poset, 2, 3)
    # C(5,2) + C(5,3) = 10 + 10 = 20
    assert result.count == 20


def test_v_poset() -> None:
    """V-shaped poset: 0 < 1, 0 < 2, 1 and 2 incomparable."""
    elements = ("0", "1", "2")
    relation = (
        PresentationPair(lower="0", upper="1"),
        PresentationPair(lower="0", upper="2"),
    )
    poset = materialize_finite_poset(
        elements,
        relation,
        RelationInterpretation.COMPARABLE_PAIRS,
        ReflexivePairPolicy.FORBIDDEN,
    )
    result = enumerate_antichains(poset, 2, 2)
    assert result.count == 1
    assert ("1", "2") in result.antichains


def test_exponential_candidate_family_is_rejected_before_enumeration() -> None:
    poset = _make_antichain_poset(24)

    assert AntichainEnumerationRequest(
        poset=poset, min_cardinality=0, max_cardinality=24
    )

    with pytest.raises(OperationDomainValidationError, match="candidate bound"):
        enumerate_antichains(poset, 0, 24)


def test_request_retains_intrinsic_cardinality_range_shape() -> None:
    with pytest.raises(ValidationError, match="max_cardinality"):
        AntichainEnumerationRequest(
            poset=_make_chain(3), min_cardinality=2, max_cardinality=1
        )
