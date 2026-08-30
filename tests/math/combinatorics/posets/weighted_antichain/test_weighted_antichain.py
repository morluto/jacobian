from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.combinatorics.posets.core._models import (
    PresentationPair,
    ReflexivePairPolicy,
    RelationInterpretation,
)
from jacobian.math.combinatorics.posets.core.operations import materialize_finite_poset
from jacobian.math.combinatorics.posets.weighted_antichain.operations import (
    compute_maximum_weight_antichain,
)


def _cr(num, den=1):
    return CanonicalRational.from_fraction(Fraction(num, den))


def _chain_poset(elements):
    pairs = tuple(
        PresentationPair(lower=elements[i], upper=elements[i + 1])
        for i in range(len(elements) - 1)
    )
    return materialize_finite_poset(
        elements=tuple(elements),
        relation=pairs,
        interpretation=RelationInterpretation.COVER_EDGES,
        reflexive_pairs=ReflexivePairPolicy.FORBIDDEN,
    )


def _antichain_poset(elements):
    return materialize_finite_poset(
        elements=tuple(elements),
        relation=(),
        interpretation=RelationInterpretation.COMPARABLE_PAIRS,
        reflexive_pairs=ReflexivePairPolicy.FORBIDDEN,
    )


def test_chain_single_element() -> None:
    poset = _chain_poset(["a", "b", "c"])
    result = compute_maximum_weight_antichain(poset, (_cr(1), _cr(2), _cr(3)))
    assert result.maximum_weight.as_fraction() == Fraction(3)
    assert result.antichain == ("c",)


def test_antichain_all() -> None:
    poset = _antichain_poset(["a", "b", "c"])
    result = compute_maximum_weight_antichain(poset, (_cr(1), _cr(2), _cr(3)))
    assert result.maximum_weight.as_fraction() == Fraction(6)
    assert set(result.antichain) == {"a", "b", "c"}


def test_empty_poset() -> None:
    poset = _antichain_poset([])
    result = compute_maximum_weight_antichain(poset, ())
    assert result.maximum_weight.as_fraction() == Fraction(0)
    assert result.antichain == ()


def test_zero_weights() -> None:
    poset = _antichain_poset(["a", "b"])
    result = compute_maximum_weight_antichain(poset, (_cr(0), _cr(0)))
    assert result.maximum_weight.as_fraction() == Fraction(0)


def test_witness_is_antichain() -> None:
    poset = _chain_poset(["a", "b", "c", "d"])
    result = compute_maximum_weight_antichain(poset, (_cr(3), _cr(1), _cr(4), _cr(2)))
    assert result.maximum_weight.as_fraction() == Fraction(4)
    assert len(result.antichain) == 1


def test_rational_weights() -> None:
    poset = _chain_poset(["a", "b"])
    result = compute_maximum_weight_antichain(poset, (_cr(1, 2), _cr(3, 4)))
    assert result.maximum_weight.as_fraction() == Fraction(3, 4)


def test_result_preserves_source() -> None:
    poset = _antichain_poset(["a"])
    weights = (_cr(5),)
    result = compute_maximum_weight_antichain(poset, weights)
    assert result.poset == poset
    assert result.weights == weights


def test_rational_arithmetic_work_is_admitted_separately() -> None:
    import pytest

    poset = _antichain_poset([str(index) for index in range(20)])
    digits = 1_600
    weights = tuple(
        CanonicalRational(num="1", den="1" + "0" * digits) for _ in range(20)
    )
    with pytest.raises(ValueError, match="summation exceeds"):
        compute_maximum_weight_antichain(poset, weights)


def test_chain_growth_uses_width_not_carrier_size() -> None:
    digits = 16_384
    weights = tuple(
        CanonicalRational(num="1", den="1" + "0" * digits) for _ in range(2)
    )
    result = compute_maximum_weight_antichain(_chain_poset(["a", "b"]), weights)
    assert result.antichain in (("a",), ("b",))
