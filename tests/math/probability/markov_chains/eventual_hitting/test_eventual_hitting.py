from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.probability.markov_chains.eventual_hitting.operations import (
    compute_eventual_hitting_profile,
)


def _cr(num, den=1):
    return CanonicalRational.from_fraction(Fraction(num, den))


def _matrix(rows):
    return tuple(tuple(_cr(v) for v in row) for row in rows)


def test_two_state_absorbing() -> None:
    """State 0 has 1/2 chance to go to state 1 (absorbing) or stay."""
    matrix = _matrix([[1, 2, 2], [0, 1, 1]])
    # Wait, that's wrong. Let me fix the matrix.
    # matrix[0][0] = 1/2, matrix[0][1] = 1/2
    # matrix[1][0] = 0, matrix[1][1] = 1
    matrix = _matrix([[1, 2, 2], [0, 1, 1]])
    # Actually the _cr takes (num, den), so I need:
    matrix = (
        (_cr(1, 2), _cr(1, 2)),
        (_cr(0), _cr(1)),
    )
    result = compute_eventual_hitting_profile(matrix, (1,))
    assert result.hitting_probabilities[0].as_fraction() == Fraction(1)
    assert result.hitting_probabilities[1].as_fraction() == Fraction(1)


def test_impossible_target() -> None:
    """State 0 cannot reach state 1: h(0) = 0."""
    matrix = (
        (_cr(1), _cr(0)),
        (_cr(0), _cr(1)),
    )
    result = compute_eventual_hitting_profile(matrix, (1,))
    assert result.hitting_probabilities[0].as_fraction() == Fraction(0)
    assert result.hitting_probabilities[1].as_fraction() == Fraction(1)
    assert 0 in result.zero_states
    assert 1 in result.almost_sure_states


def test_all_target() -> None:
    """All states are targets: h(i) = 1 for all."""
    matrix = (
        (_cr(1, 2), _cr(1, 2)),
        (_cr(1, 2), _cr(1, 2)),
    )
    result = compute_eventual_hitting_profile(matrix, (0, 1))
    for i in range(2):
        assert result.hitting_probabilities[i].as_fraction() == Fraction(1)


def test_three_state_chain() -> None:
    """3-state chain: 0 -> 1 -> 2 (absorbing)."""
    matrix = (
        (_cr(0), _cr(1), _cr(0)),
        (_cr(0), _cr(0), _cr(1)),
        (_cr(0), _cr(0), _cr(1)),
    )
    result = compute_eventual_hitting_profile(matrix, (2,))
    assert result.hitting_probabilities[0].as_fraction() == Fraction(1)
    assert result.hitting_probabilities[1].as_fraction() == Fraction(1)
    assert result.hitting_probabilities[2].as_fraction() == Fraction(1)
    assert 0 in result.almost_sure_states


def test_mixed_probabilities() -> None:
    """Chain where some states have proper (0 < p < 1) hitting probability."""
    matrix = (
        (_cr(1, 2), _cr(1, 2), _cr(0)),
        (_cr(0), _cr(1, 2), _cr(1, 2)),
        (_cr(0), _cr(0), _cr(1)),
    )
    result = compute_eventual_hitting_profile(matrix, (2,))
    h = [r.as_fraction() for r in result.hitting_probabilities]
    assert h[2] == Fraction(1)
    assert h[1] == Fraction(1)
    assert h[0] == Fraction(1)


def test_result_preserves_source() -> None:
    matrix = (
        (_cr(1), _cr(0)),
        (_cr(0), _cr(1)),
    )
    result = compute_eventual_hitting_profile(matrix, (0,))
    assert result.matrix == matrix
    assert result.target_states == (0,)
