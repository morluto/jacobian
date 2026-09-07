from __future__ import annotations

import json
from fractions import Fraction
from itertools import combinations

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.additive.rational_subset_sum.operations import (
    compute_rational_subset_sum_profile,
    verify_rational_subset_sum_profile,
)


def _cr(num: int, den: int | None = None) -> CanonicalRational:
    if den is None:
        return CanonicalRational.from_fraction(Fraction(num))
    return CanonicalRational.from_fraction(Fraction(num, den))


def test_fixture() -> None:
    """Complete profile of (1/2, 1/2, -1)."""
    values = (_cr(1, 2), _cr(1, 2), _cr(-1))
    result = compute_rational_subset_sum_profile(values)
    sum_map = {r.sum_value.as_fraction(): r.multiplicity for r in result.rows}
    assert sum_map[Fraction(-1)] == 1
    assert sum_map[Fraction(-1, 2)] == 2
    assert sum_map[Fraction(0)] == 2
    assert sum_map[Fraction(1, 2)] == 2
    assert sum_map[Fraction(1)] == 1


def test_empty_subset() -> None:
    """The empty subset contributes sum 0 with multiplicity 1."""
    values = (_cr(1, 2),)
    result = compute_rational_subset_sum_profile(values)
    sum_map = {r.sum_value.as_fraction(): r.multiplicity for r in result.rows}
    assert sum_map[Fraction(0)] == 1
    assert sum_map[Fraction(1, 2)] == 1


def test_multiplicity_sum() -> None:
    """Total multiplicity equals 2^n."""
    values = (_cr(1, 3), _cr(2, 3), _cr(1))
    result = compute_rational_subset_sum_profile(values)
    total = sum(r.multiplicity for r in result.rows)
    assert total == 2**3


def test_replay() -> None:
    """Replay: independently compute all subset sums."""
    values = (_cr(1, 2), _cr(1, 3), _cr(-1, 4))
    result = compute_rational_subset_sum_profile(values)
    fracs = [v.as_fraction() for v in values]
    n = len(fracs)
    expected: dict[Fraction, int] = {}
    for r in range(n + 1):
        for indices in combinations(range(n), r):
            s = sum((fracs[i] for i in indices), Fraction(0))
            expected[s] = expected.get(s, 0) + 1
    actual = {r.sum_value.as_fraction(): r.multiplicity for r in result.rows}
    assert actual == expected


def test_sorted_output() -> None:
    """Rows are sorted by rational value."""
    values = (_cr(3, 2), _cr(-1, 4), _cr(1, 2))
    result = compute_rational_subset_sum_profile(values)
    sums = [r.sum_value.as_fraction() for r in result.rows]
    assert sums == sorted(sums)


def test_result_preserves_source() -> None:
    """Result retains the source values."""
    values = (_cr(1, 2), _cr(1, 3))
    result = compute_rational_subset_sum_profile(values)
    assert result.values == values


def test_repeated_zero_values_use_their_small_support_bound() -> None:
    """Twenty equal zeroes have one result row, not 2**20 rows."""
    values = tuple(_cr(0) for _ in range(20))
    result = compute_rational_subset_sum_profile(values)
    assert len(result.rows) == 1
    assert result.rows[0].multiplicity == 2**20


def test_single_max_height_value_is_admitted() -> None:
    value = _cr(1, 10**32767 + 1)

    result = compute_rational_subset_sum_profile((value,))

    assert len(result.rows) == 2


def test_single_max_height_negative_numerator_is_admitted() -> None:
    value = _cr(-(10**32768 - 1))

    result = compute_rational_subset_sum_profile((value,))

    assert len(result.rows) == 2


def test_uncancellable_rational_growth_is_rejected_before_enumeration() -> None:
    values = (
        _cr(1, 10**20000 + 1),
        _cr(1, 10**20000 + 3),
    )

    with pytest.raises(OperationDomainValidationError, match="rational"):
        compute_rational_subset_sum_profile(values)


def test_serialized_forged_profile_is_rejected_by_verifier() -> None:
    result = compute_rational_subset_sum_profile((_cr(1),))
    payload = result.model_dump(mode="json")
    payload["rows"][1]["multiplicity"] += 1
    decoded = result.model_validate_json(json.dumps(payload))
    assert not verify_rational_subset_sum_profile(decoded)
