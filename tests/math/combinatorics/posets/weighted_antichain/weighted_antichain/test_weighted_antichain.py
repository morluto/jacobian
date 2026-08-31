from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.posets.core._models import (
    FinitePoset,
    PresentationPair,
    ReflexivePairPolicy,
    RelationInterpretation,
)
from jacobian.math.combinatorics.posets.core.operations import (
    materialize_finite_poset,
)
from jacobian.math.combinatorics.posets.weighted_antichain.operations import (
    compute_maximum_weight_antichain,
)


def _cr(num: int, den: int = 1) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(num, den))


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


def _make_antichain(n: int) -> FinitePoset:
    elements = tuple(str(i) for i in range(n))
    return materialize_finite_poset(
        elements,
        (),
        RelationInterpretation.COMPARABLE_PAIRS,
        ReflexivePairPolicy.FORBIDDEN,
    )


def test_chain_picks_max_weight() -> None:
    poset = _make_chain(3)
    weights = (_cr(1), _cr(3), _cr(2))
    result = compute_maximum_weight_antichain(poset, weights)
    assert result.maximum_weight.as_fraction() == Fraction(3)
    assert result.antichain == ("1",)


def test_antichain_all() -> None:
    poset = _make_antichain(3)
    weights = (_cr(1), _cr(2), _cr(3))
    result = compute_maximum_weight_antichain(poset, weights)
    assert result.maximum_weight.as_fraction() == Fraction(6)
    assert set(result.antichain) == {"0", "1", "2"}


def test_zero_weights() -> None:
    poset = _make_chain(3)
    weights = (_cr(0), _cr(0), _cr(0))
    result = compute_maximum_weight_antichain(poset, weights)
    assert result.maximum_weight.as_fraction() == Fraction(0)


def test_v_poset() -> None:
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
    # 0 < 1, 0 < 2, 1 and 2 incomparable
    weights = (_cr(5), _cr(3), _cr(4))
    result = compute_maximum_weight_antichain(poset, weights)
    # Best antichain: {1, 2} with weight 7
    assert result.maximum_weight.as_fraction() == Fraction(7)
    assert set(result.antichain) == {"1", "2"}


def test_method() -> None:
    poset = _make_chain(2)
    result = compute_maximum_weight_antichain(poset, (_cr(1), _cr(1)))
    assert result.antichain is not None
    assert result.maximum_weight is not None


def test_equal_maxima_choose_lexicographically_least_antichain() -> None:
    poset = _make_chain(2)
    result = compute_maximum_weight_antichain(poset, (_cr(1), _cr(1)))
    assert result.antichain == ("0",)


def test_weight_axis_must_match_the_poset() -> None:
    with pytest.raises(OperationDomainValidationError, match="one entry per poset element"):
        compute_maximum_weight_antichain(_make_chain(2), (_cr(1),))


def test_exponential_search_envelope_is_enforced() -> None:
    with pytest.raises(Exception, match="at most 64 items"):
        _make_antichain(65)

    poset = _make_antichain(64)
    with pytest.raises(OperationDomainValidationError, match="one entry per poset element"):
        compute_maximum_weight_antichain(poset, tuple(_cr(1) for _ in range(65)))


def test_derived_rational_growth_is_rejected_before_subset_search() -> None:
    poset = _make_antichain(2)
    weights = (_cr(1, 10**20_000 - 1), _cr(1, 10**20_000))

    with pytest.raises(OperationDomainValidationError, match="exceeds the canonical digit envelope"):
        compute_maximum_weight_antichain(poset, weights)
