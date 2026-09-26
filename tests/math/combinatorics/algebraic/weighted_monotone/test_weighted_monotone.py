from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from fractions import Fraction
from itertools import combinations, pairwise, product
from typing import cast

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.algebraic.weighted_monotone._models import (
    MAX_WEIGHTED_MONOTONE_ARITHMETIC_WORK,
    WeightedMaximumRequest,
    WeightedMaximumResult,
    WeightedOrderedWord,
)
from jacobian.math.combinatorics.algebraic.weighted_monotone.operations import (
    compute_endpoint_profile,
    maximum_weight_nondecreasing_subsequence,
    maximum_weight_nonincreasing_subsequence,
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


def _enumerated_maximum(
    source: WeightedOrderedWord, *, nondecreasing: bool
) -> Fraction:
    """Independent subset oracle for short finite words."""
    ranks = {letter: rank for rank, letter in enumerate(source.word.alphabet)}
    best: Fraction | None = None
    n = len(source.word.letters)
    for size in range(1, n + 1):
        for indices in combinations(range(n), size):
            selected = tuple(ranks[source.word.letters[i]] for i in indices)
            if nondecreasing:
                monotone = all(a <= b for a, b in pairwise(selected))
            else:
                monotone = all(a >= b for a, b in pairwise(selected))
            if not monotone:
                continue
            value = sum((source.weights[i].as_fraction() for i in indices), Fraction(0))
            if best is None or value > best:
                best = value
    return best if best is not None else Fraction(0)


@pytest.mark.parametrize(
    ("operation", "nondecreasing"),
    [
        (maximum_weight_nondecreasing_subsequence, True),
        (maximum_weight_nonincreasing_subsequence, False),
    ],
)
def test_weighted_monotone_maximum_matches_exhaustive_oracle(
    operation: Callable[[WeightedOrderedWord], WeightedMaximumResult],
    nondecreasing: bool,
) -> None:
    for n in range(5):
        for letters in product(("a", "b", "c"), repeat=n):
            for raw_weights in product((0, Fraction(1, 2), Fraction(3, 2)), repeat=n):
                source = _word(
                    ("a", "b", "c"),
                    letters,
                    cast(tuple[int | Fraction, ...], raw_weights),
                )
                result = operation(source)
                assert result.weight.as_fraction() == _enumerated_maximum(
                    source, nondecreasing=nondecreasing
                )
                assert result.source == source
                assert (
                    sum(
                        (source.weights[i].as_fraction() for i in result.indices),
                        Fraction(0),
                    )
                    == result.weight.as_fraction()
                )
                ranks = {
                    letter: rank for rank, letter in enumerate(source.word.alphabet)
                }
                selected = tuple(ranks[source.word.letters[i]] for i in result.indices)
                assert all(left < right for left, right in pairwise(result.indices))
                if nondecreasing:
                    assert all(left <= right for left, right in pairwise(selected))
                    assert result.monotonicity == "NONDECREASING"
                else:
                    assert all(left >= right for left, right in pairwise(selected))
                    assert result.monotonicity == "NONINCREASING"


def test_empty_source_zero_weights_and_weak_repetitions() -> None:
    empty = _word(("a",), (), ())
    assert maximum_weight_nondecreasing_subsequence(empty).indices == ()
    assert maximum_weight_nondecreasing_subsequence(empty).weight.as_fraction() == 0

    repeated = _word(("a",), ("a", "a", "a"), (0, 0, 0))
    result = maximum_weight_nondecreasing_subsequence(repeated)
    assert result.indices == (0,)
    assert result.weight.as_fraction() == 0
    decreasing = maximum_weight_nonincreasing_subsequence(
        _word(("a", "b"), ("b", "a"), (Fraction(1, 3), Fraction(2, 5)))
    )
    assert decreasing.indices == (0, 1)
    assert decreasing.weight.as_fraction() == Fraction(11, 15)


def test_exact_arithmetic_work_boundary_is_admitted_before_dp() -> None:
    n = 500
    source = _word(("a",), ("a",) * n, (99_999,) * n)
    pair_count = n * (n - 1) // 2
    boundary_work = pair_count * 2 * 8**2 + n * 12 * 8**2
    assert boundary_work <= MAX_WEIGHTED_MONOTONE_ARITHMETIC_WORK
    result = maximum_weight_nondecreasing_subsequence(source)
    assert result.weight.as_fraction() == Fraction(99_999 * n)
    assert len(result.indices) == n

    too_wide = _word(("a",), ("a",) * n, (999_999,) * n)
    above_boundary_work = pair_count * 2 * 9**2 + n * 12 * 9**2
    assert above_boundary_work > MAX_WEIGHTED_MONOTONE_ARITHMETIC_WORK
    with pytest.raises(OperationResourceAdmissionError, match="rational work"):
        maximum_weight_nondecreasing_subsequence(too_wide)


def test_raw_rational_digits_are_rejected_before_exact_integer_decoding() -> None:
    too_wide = "9" * 257
    raw = {
        "word": {"alphabet": ["a"], "letters": ["a"]},
        "weights": [{"num": too_wide, "den": "1"}],
    }
    with pytest.raises(ValidationError, match="256 decimal digits"):
        WeightedMaximumRequest.model_validate({"source": raw})


def test_result_deserialization_is_structural_for_large_source_weights() -> None:
    denominator = "9" * 1000
    payload = {
        "source": {
            "word": {"alphabet": ["a", "b"], "letters": ["a", "b"]},
            "weights": [
                {"num": "1", "den": denominator},
                {"num": "1", "den": str(int(denominator) - 2)},
            ],
        },
        "monotonicity": "NONDECREASING",
        "weight": {"num": "0", "den": "1"},
        "indices": [0, 1],
        "values": ["a", "b"],
    }
    decoded = WeightedMaximumResult.model_validate_json(json.dumps(payload))
    assert len(decoded.source.weights) == 2
    assert decoded.weight.as_fraction() == 0


def test_serialized_result_composes_through_its_typed_contract() -> None:
    source = _word(("a", "b"), ("b", "a"), (Fraction(1, 2), Fraction(3, 4)))
    result = maximum_weight_nonincreasing_subsequence(source)
    encoded = result.model_dump_json()
    decoded = WeightedMaximumResult.model_validate_json(encoded)
    request = WeightedMaximumRequest.model_validate_json(
        json.dumps({"source": decoded.source.model_dump(mode="json")})
    )
    assert request.source == source
    assert decoded.weight.as_fraction() == Fraction(5, 4)


def test_shared_factor_denominators_use_lcm_growth_bound() -> None:
    q = 10**255 + 7
    source = _word(("a",), ("a",) * 4, tuple(Fraction(1, k * q) for k in (1, 2, 3, 5)))
    result = maximum_weight_nondecreasing_subsequence(source)
    assert result.indices == (0, 1, 2, 3)
    assert result.weight.as_fraction() == sum(
        (Fraction(1, k * q) for k in (1, 2, 3, 5)), Fraction()
    )


def test_result_deserialization_rejects_impossible_witness_shape() -> None:
    source = _word(("a",), ("a",), (1,))
    result = maximum_weight_nondecreasing_subsequence(source).model_dump(mode="json")
    result["values"] = []
    with pytest.raises(ValidationError):
        WeightedMaximumResult.model_validate(result)
