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
    """Test that the rational growth check rejects large multi-denominator sums.

    With an antichain of width 20 and weights using different denominators
    each near the 32,768-digit limit, the sum's digit growth exceeds the
    canonical envelope.
    """
    import pytest

    poset = _antichain_poset([str(index) for index in range(20)])
    # Use two different denominators near the limit to trigger the
    # multi-denominator growth bound: width * max_digits + len(str(width)).
    weights = tuple(
        CanonicalRational(
            num="1", den="1" + "0" * 16_400 + ("3" if i % 2 == 0 else "7")
        )
        for i in range(20)
    )
    with pytest.raises(ValueError, match="rational growth exceeds"):
        compute_maximum_weight_antichain(poset, weights)


def test_chain_growth_uses_width_not_carrier_size() -> None:
    digits = 16_384
    weights = tuple(
        CanonicalRational(num="1", den="1" + "0" * digits) for _ in range(2)
    )
    result = compute_maximum_weight_antichain(_chain_poset(["a", "b"]), weights)
    assert result.antichain in (("a",), ("b",))


def test_polynomial_chain_21_elements() -> None:
    """A 21-element chain should be handled by the polynomial algorithm."""
    elements = [str(i) for i in range(21)]
    poset = _chain_poset(elements)
    weights = tuple(_cr(1) for _ in range(21))
    result = compute_maximum_weight_antichain(poset, weights)
    # In a chain, the max antichain is a single element with weight 1
    assert result.maximum_weight.as_fraction() == Fraction(1)
    assert len(result.antichain) == 1


def test_polynomial_antichain_30_elements() -> None:
    """A 30-element antichain should sum all weights."""
    elements = [str(i) for i in range(30)]
    poset = _antichain_poset(elements)
    weights = tuple(_cr(i + 1) for i in range(30))
    result = compute_maximum_weight_antichain(poset, weights)
    assert result.maximum_weight.as_fraction() == Fraction(30 * 31, 2)
    assert len(result.antichain) == 30


def test_polynomial_mixed_poset_25_elements() -> None:
    """A 25-element poset with two independent chains should handle polynomially."""
    from jacobian.math.combinatorics.posets.core._models import (
        PresentationPair,
        ReflexivePairPolicy,
        RelationInterpretation,
    )
    from jacobian.math.combinatorics.posets.core.operations import (
        materialize_finite_poset,
    )

    # Use zero-padded names so lexical and numeric order agree.
    elements = [f"e{i:02d}" for i in range(25)]
    # Two chains: e00<e01<...<e12 and e13<e14<...<e24
    pairs = tuple(
        PresentationPair(lower=elements[i], upper=elements[i + 1]) for i in range(12)
    ) + tuple(
        PresentationPair(lower=elements[i], upper=elements[i + 1])
        for i in range(13, 24)
    )
    poset = materialize_finite_poset(
        elements=tuple(elements),
        relation=pairs,
        interpretation=RelationInterpretation.COVER_EDGES,
        reflexive_pairs=ReflexivePairPolicy.FORBIDDEN,
    )
    # Unit weights: max antichain picks one from each chain = 1 + 1 = 2
    weights = tuple(_cr(1) for _ in range(25))
    result = compute_maximum_weight_antichain(poset, weights)
    assert result.maximum_weight.as_fraction() == Fraction(2)
    assert len(result.antichain) == 2
