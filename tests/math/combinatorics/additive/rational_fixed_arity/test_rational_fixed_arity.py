from __future__ import annotations

import json
from fractions import Fraction
from itertools import combinations
from math import comb

from jacobian._exact import CanonicalRational
from jacobian.canonical import parse_canonical_integer
from jacobian.math.combinatorics.additive.rational_fixed_arity.operations import (
    compute_rational_fixed_arity_sum_profile,
    verify_rational_fixed_arity_sum_profile,
)


def _cr(num, den=None):
    if den is None:
        return CanonicalRational.from_fraction(Fraction(num))
    return CanonicalRational.from_fraction(Fraction(num, den))


def test_fixture() -> None:
    """Sums of pairs from (1/2, 1/3, 1/6): 5/6, 2/3, 1/2, each with multiplicity 1."""
    values = (_cr(1, 2), _cr(1, 3), _cr(1, 6))
    result = compute_rational_fixed_arity_sum_profile(values, 2)
    sum_map = {r.sum_value.as_fraction(): r.multiplicity for r in result.rows}
    assert sum_map[Fraction(5, 6)] == 1
    assert sum_map[Fraction(2, 3)] == 1
    assert sum_map[Fraction(1, 2)] == 1


def test_multiplicity_sum() -> None:
    """Total multiplicity equals C(n, h)."""
    values = [_cr(i + 1, i + 2) for i in range(6)]
    result = compute_rational_fixed_arity_sum_profile(tuple(values), 3)
    total = sum(r.multiplicity for r in result.rows)
    assert total == comb(6, 3)


def test_equal_values_distinct_indices() -> None:
    """Equal values at different indices are distinct choices."""
    values = (_cr(1, 2), _cr(1, 2), _cr(1, 2))
    result = compute_rational_fixed_arity_sum_profile(values, 2)
    assert len(result.rows) == 1
    assert result.rows[0].sum_value.as_fraction() == Fraction(1)
    assert result.rows[0].multiplicity == 3  # C(3,2) = 3


def test_arity_1() -> None:
    """Arity 1 returns each value individually."""
    values = (_cr(1, 3), _cr(2, 3), _cr(1, 3))
    result = compute_rational_fixed_arity_sum_profile(values, 1)
    sum_map = {r.sum_value.as_fraction(): r.multiplicity for r in result.rows}
    assert sum_map[Fraction(1, 3)] == 2
    assert sum_map[Fraction(2, 3)] == 1


def test_singleton_at_maximum_rational_digit_width_is_admitted() -> None:
    value = CanonicalRational(num=1 * 32_768, den=1)
    result = compute_rational_fixed_arity_sum_profile((value,), 1)
    assert result.rows[0].sum_value == value


def test_replay() -> None:
    """Replay: independently compute all sums."""
    values = (_cr(1, 2), _cr(1, 3), _cr(1, 4), _cr(1, 5))
    arity = 2
    result = compute_rational_fixed_arity_sum_profile(values, arity)
    expected: dict[Fraction, int] = {}
    fracs = [v.as_fraction() for v in values]
    for indices in combinations(range(len(values)), arity):
        s = sum(fracs[i] for i in indices)
        expected[s] = expected.get(s, 0) + 1
    actual = {r.sum_value.as_fraction(): r.multiplicity for r in result.rows}
    assert actual == expected


def test_sorted_output() -> None:
    """Rows are sorted by rational value."""
    values = (_cr(3, 2), _cr(1, 4), _cr(1, 2))
    result = compute_rational_fixed_arity_sum_profile(values, 1)
    sums = [r.sum_value.as_fraction() for r in result.rows]
    assert sums == sorted(sums)


def test_result_preserves_source() -> None:
    """Result retains the source values and arity."""
    values = (_cr(1, 2), _cr(1, 3))
    result = compute_rational_fixed_arity_sum_profile(values, 2)
    assert result.values == values
    assert result.arity == 2


def test_native_admission_rejects_combination_explosion() -> None:
    """The native path rejects a huge complete profile before enumeration."""
    import pytest

    values = tuple(_cr(i + 1) for i in range(50))
    with pytest.raises(ValueError, match="enumeration exceeds"):
        compute_rational_fixed_arity_sum_profile(values, 10)


def test_native_admission_rejects_rational_growth() -> None:
    """The native path rejects sums whose exact denominator will overflow."""
    import pytest

    digits = 17_000
    values = (
        CanonicalRational(num=1, den=parse_canonical_integer("1" + "0" * digits)),
        CanonicalRational(num=1, den=parse_canonical_integer("1" + "0" * digits + "1")),
    )
    with pytest.raises(ValueError, match="rational digit bound"):
        compute_rational_fixed_arity_sum_profile(values, 2)


def test_serialized_forged_profile_is_rejected_by_verifier() -> None:
    result = compute_rational_fixed_arity_sum_profile((_cr(1), _cr(2)), 1)
    payload = result.model_dump(mode="json")
    payload["rows"][0]["multiplicity"] += 1
    decoded = result.model_validate_json(json.dumps(payload))
    assert not verify_rational_fixed_arity_sum_profile(decoded)


def test_native_admission_rejects_negative_arity() -> None:
    """Negative arity is a typed domain rejection rather than a host error."""
    import pytest

    with pytest.raises(ValueError, match="arity must be nonnegative"):
        compute_rational_fixed_arity_sum_profile((_cr(1),), -1)


def test_large_source_and_empty_out_of_range_arity_use_result_sensitive_admission() -> (
    None
):
    repeated = tuple(_cr(0) for _ in range(1_001))
    result = compute_rational_fixed_arity_sum_profile(repeated, 1)
    assert result.rows[0].multiplicity == 1_001

    empty = compute_rational_fixed_arity_sum_profile((_cr(1),), 1_001)
    assert empty.rows == ()


def test_arithmetic_progression_uses_lattice_support_bound() -> None:
    """Distinct values can still have a compact, collision-sensitive support."""
    values = tuple(_cr(index) for index in range(1_000))
    result = compute_rational_fixed_arity_sum_profile(values, 2)
    assert len(result.rows) == 1_997


def test_shared_denominator_does_not_grow_with_arity() -> None:
    denominator = 10**100 + 1
    values = tuple(_cr(1, denominator) for _ in range(1_000))
    result = compute_rational_fixed_arity_sum_profile(values, 1_000)
    assert len(result.rows) == 1
    assert result.rows[0].sum_value.as_fraction() == Fraction(1_000, denominator)


def test_reduced_denominators_use_lcm_growth_bound() -> None:
    values = tuple(_cr(index, 10_000) for index in range(8_192))
    result = compute_rational_fixed_arity_sum_profile(values, len(values))
    assert result.rows[0].sum_value.as_fraction() == sum(
        (value.as_fraction() for value in values), Fraction(0)
    )


def test_exact_cancellation_is_presolved_without_lcm_expansion() -> None:
    digits = 17_001
    first = "1" + "0" * digits
    second = first + "1"
    values = (
        CanonicalRational(num=1, den=parse_canonical_integer(first)),
        CanonicalRational(num=-1, den=parse_canonical_integer(first)),
        CanonicalRational(num=1, den=parse_canonical_integer(second)),
        CanonicalRational(num=-1, den=parse_canonical_integer(second)),
    )
    result = compute_rational_fixed_arity_sum_profile(values, len(values))
    assert result.rows[0].sum_value == _cr(0)


def test_arity_zero_does_not_sum_wide_source_values() -> None:
    values = (
        CanonicalRational(num=1, den=parse_canonical_integer("1" + "0" * 17_001)),
        CanonicalRational(num=1, den=parse_canonical_integer("1" + "0" * 17_001 + "1")),
    )
    result = compute_rational_fixed_arity_sum_profile(values, 0)
    assert result.rows[0].sum_value == _cr(0)


def test_cross_denominator_cancellation_is_preserved() -> None:
    q = 10**11
    r = q + 1
    p = q + r
    values = (
        _cr(1, p * q),
        _cr(1, p * r),
        _cr(-1, q * r),
    )

    result = compute_rational_fixed_arity_sum_profile(values, len(values))

    assert result.rows[0].sum_value == _cr(0)
