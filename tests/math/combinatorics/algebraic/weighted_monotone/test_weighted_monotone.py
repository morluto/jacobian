from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.algebraic.weighted_monotone._models import (
    WeightedOrderedWord,
)
from jacobian.math.combinatorics.algebraic.weighted_monotone.operations import (
    compute_endpoint_profile,
)
from jacobian.math.logic.languages.words.values import FiniteWord


def _word(
    alphabet: Sequence[str],
    letters: Sequence[str],
    weights: Sequence[int | Fraction],
) -> WeightedOrderedWord:
    fw = FiniteWord(alphabet=tuple(alphabet), letters=tuple(letters))
    return WeightedOrderedWord(
        word=fw,
        weights=tuple(CanonicalRational.from_fraction(Fraction(w)) for w in weights),
    )


def test_single_letter() -> None:
    """One letter: S_0 = T_0 = w_0."""
    source = _word(["a"], ["a"], [3])
    result = compute_endpoint_profile(source)
    assert len(result.entries) == 1
    assert result.entries[0].increasing_value.as_fraction() == Fraction(3)
    assert result.entries[0].decreasing_value.as_fraction() == Fraction(3)


def test_admission_reserves_a_digit_for_rational_addition_carry() -> None:
    """An admitted profile value must fit the canonical rational carrier."""
    q = 10**16_384 - 1
    r = 10**16_384 - 3
    source = _word(
        ["a"],
        ["a", "a"],
        [Fraction(q - 1, q), Fraction(r - 1, r)],
    )

    with pytest.raises(
        OperationDomainValidationError,
        match="rational growth exceeds",
    ):
        compute_endpoint_profile(source)


def test_increasing_letters() -> None:
    """Word 'ab' with weights 1, 2: S = [1, 3], T = [1, 2]."""
    source = _word(["a", "b"], ["a", "b"], [1, 2])
    result = compute_endpoint_profile(source)
    assert result.entries[0].increasing_value.as_fraction() == Fraction(1)
    assert result.entries[1].increasing_value.as_fraction() == Fraction(3)
    assert result.entries[0].decreasing_value.as_fraction() == Fraction(1)
    assert result.entries[1].decreasing_value.as_fraction() == Fraction(2)


def test_decreasing_letters() -> None:
    """Word 'ba' with weights 1, 2: S = [1, 2], T = [1, 3]."""
    source = _word(["a", "b"], ["b", "a"], [1, 2])
    result = compute_endpoint_profile(source)
    assert result.entries[0].increasing_value.as_fraction() == Fraction(1)
    assert result.entries[1].increasing_value.as_fraction() == Fraction(2)
    assert result.entries[0].decreasing_value.as_fraction() == Fraction(1)
    assert result.entries[1].decreasing_value.as_fraction() == Fraction(3)


def test_replay_dp() -> None:
    """Independently compute S and T and compare."""
    source = _word(["a", "b", "c"], ["b", "a", "c"], [1, 2, 3])
    result = compute_endpoint_profile(source)
    word = source.word
    letters = list(word.letters)
    alphabet = list(word.alphabet)
    n = len(letters)
    weights = [w.as_fraction() for w in source.weights]
    letter_rank = {s: i for i, s in enumerate(alphabet)}
    s = [Fraction(0)] * n
    t = [Fraction(0)] * n
    for i in range(n):
        wi = weights[i]
        ri = letter_rank[letters[i]]
        s_best = Fraction(0)
        t_best = Fraction(0)
        for j in range(i):
            rj = letter_rank[letters[j]]
            if rj <= ri:
                s_best = max(s_best, s[j])
            if rj >= ri:
                t_best = max(t_best, t[j])
        s[i] = wi + s_best
        t[i] = wi + t_best
    for i, entry in enumerate(result.entries):
        assert entry.increasing_value.as_fraction() == s[i]
        assert entry.decreasing_value.as_fraction() == t[i]


def test_empty_word() -> None:
    """Empty word: no entries."""
    source = _word(["a"], [], [])
    result = compute_endpoint_profile(source)
    assert len(result.entries) == 0


def test_equal_letters() -> None:
    """Equal letters: both S and T can extend (weak inequality)."""
    source = _word(["a"], ["a", "a"], [1, 2])
    result = compute_endpoint_profile(source)
    assert result.entries[0].increasing_value.as_fraction() == Fraction(1)
    assert result.entries[1].increasing_value.as_fraction() == Fraction(3)
    assert result.entries[0].decreasing_value.as_fraction() == Fraction(1)
    assert result.entries[1].decreasing_value.as_fraction() == Fraction(3)


def test_result_preserves_source() -> None:
    source = _word(["a"], ["a"], [1])
    result = compute_endpoint_profile(source)
    assert result.source == source
