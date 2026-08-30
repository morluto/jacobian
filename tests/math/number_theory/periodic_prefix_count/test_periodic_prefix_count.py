"""Tests for the periodic congruence union prefix count operation."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory._periodic import normalize_periodic_source
from jacobian.math.number_theory._periodic_models import (
    PeriodicCongruenceSubset,
    PeriodicCongruenceSubsetInput,
    PeriodicCongruenceUnionRequest,
    PeriodicCongruenceUnionSource,
)
from jacobian.math.number_theory.periodic_prefix_count.operations import (
    compute_periodic_union_prefix_count,
)


def _source(
    subsets: Sequence[tuple[str, Sequence[str]]], complement: bool = False
) -> PeriodicCongruenceUnionSource:
    return PeriodicCongruenceUnionSource(
        subsets=tuple(
            PeriodicCongruenceSubset(modulus=m, residues=tuple(r)) for m, r in subsets
        ),
        complement=complement,
    )


def _normalize_source(
    subsets: Sequence[tuple[str, Sequence[str]]], complement: bool = False
) -> PeriodicCongruenceUnionSource:
    """Build a canonical source by normalizing raw (possibly negative) residues."""
    request = PeriodicCongruenceUnionRequest(
        subsets=tuple(
            PeriodicCongruenceSubsetInput(modulus=m, residues=tuple(r))
            for m, r in subsets
        ),
        complement=complement,
    )
    return normalize_periodic_source(request)


def test_fixture_mod2_or_mod3() -> None:
    """On [1,6], the union of 0 mod 2 and 1 mod 3 is {1,2,4,6}, count 4."""
    source = _source([("2", ["0"]), ("3", ["1"])])
    result = compute_periodic_union_prefix_count(source, 6)
    assert result.count == "4"
    assert result.occupied_count == 4


def test_empty_union() -> None:
    """Empty union has count 0."""
    source = PeriodicCongruenceUnionSource(subsets=(), complement=False)
    result = compute_periodic_union_prefix_count(source, 10)
    assert result.count == "0"


def test_mod1_all_integers() -> None:
    """0 mod 1: all positive integers."""
    source = _source([("1", ["0"])])
    result = compute_periodic_union_prefix_count(source, 10)
    assert result.count == "10"


def test_cutoff_zero() -> None:
    """Cutoff 0: count 0."""
    source = _source([("2", ["0"])])
    result = compute_periodic_union_prefix_count(source, 0)
    assert result.count == "0"


def test_periodicity() -> None:
    """Extending the cutoff by one period increases the count by occupied_count."""
    source = _source([("2", ["0"])])
    result_small = compute_periodic_union_prefix_count(source, 10)
    result_large = compute_periodic_union_prefix_count(source, 12)
    assert int(result_large.count) - int(result_small.count) == 1


def test_result_preserves_source() -> None:
    """Result retains the source and cutoff."""
    source = _source([("2", ["0"])])
    result = compute_periodic_union_prefix_count(source, 10)
    assert result.source == source
    assert result.cutoff == "10"


# --- Additional evidence from the issue's evidence plan ---


def test_complement_of_full_set() -> None:
    """Complement of {0 mod 2} (i.e., odd numbers) on [1,10] has count 5."""
    source = _source([("2", ["0"])], complement=True)
    result = compute_periodic_union_prefix_count(source, 10)
    assert result.count == "5"


def test_one_nonzero_residue() -> None:
    """1 mod 3 on [1,10]: {1,4,7,10}, count 4."""
    source = _source([("3", ["1"])])
    result = compute_periodic_union_prefix_count(source, 10)
    assert result.count == "4"


def test_exact_period_endpoint() -> None:
    """Cutoff equal to one full period gives exactly occupied_count."""
    source = _source([("2", ["0"]), ("3", ["1"])])
    result = compute_periodic_union_prefix_count(source, 6)
    assert int(result.count) == result.occupied_count


def test_multiple_periods_and_remainder() -> None:
    """Test X = qL + r across several periods; compare against direct enumeration."""
    source = _source([("2", ["0"]), ("3", ["1"])])
    period = 6
    occupied = {0, 1, 2, 4}  # residues occupied in [0, 6)
    for cutoff in range(50):
        result = compute_periodic_union_prefix_count(source, cutoff)
        expected = sum(1 for t in range(1, cutoff + 1) if (t % period) in occupied)
        assert result.count == str(expected)


def test_overlapping_and_nested_moduli() -> None:
    """Overlapping moduli: {0 mod 2} union {0 mod 3} on [1, 20]."""
    source = _source([("2", ["0"]), ("3", ["0"])])
    result = compute_periodic_union_prefix_count(source, 20)
    expected = sum(1 for t in range(1, 21) if t % 2 == 0 or t % 3 == 0)
    assert result.count == str(expected)


def test_complement_overlapping() -> None:
    """Complement of {0 mod 2} union {0 mod 3} on [1, 20]."""
    source = _source([("2", ["0"]), ("3", ["0"])], complement=True)
    result = compute_periodic_union_prefix_count(source, 20)
    expected = sum(1 for t in range(1, 21) if not (t % 2 == 0 or t % 3 == 0))
    assert result.count == str(expected)


def test_negative_residues_normalized() -> None:
    """Negative input residues normalized by the shared source."""
    source = _normalize_source([("2", ["0"]), ("3", ["-2"])])
    result = compute_periodic_union_prefix_count(source, 10)
    expected = sum(1 for t in range(1, 11) if t % 2 == 0 or t % 3 == 1)
    assert result.count == str(expected)


def test_exhaustive_small_periods() -> None:
    """Exhaustive direct enumeration on small periods/cutoffs."""
    test_cases = [
        [("2", ["0"], False)],
        [("3", ["1"], False)],
        [("2", ["0"], False), ("3", ["1"], False)],
        [("4", ["0", "2"], False)],
        [("5", ["2"], False)],
    ]
    for case in test_cases:
        subsets = [(m, r) for m, r, _ in case]
        complement = case[0][2]
        source = _source(subsets, complement=complement)
        period = 1
        for m, _, _ in case:
            period = math.lcm(period, int(m))
        for cutoff in range(3 * period + 5):
            result = compute_periodic_union_prefix_count(source, cutoff)
            expected = 0
            for t in range(1, cutoff + 1):
                belongs = False
                for m, r, _ in case:
                    if t % int(m) in {int(x) % int(m) for x in r}:
                        belongs = True
                        break
                if complement:
                    belongs = not belongs
                expected += 1 if belongs else 0
            assert result.count == str(expected), (
                f"Failed for {case}, complement={complement}, cutoff={cutoff}"
            )


def test_large_cutoff_small_period() -> None:
    """A huge X remains admissible when L and the result are small; no scan through [1,X]."""
    source = _source([("2", ["0"])])
    cutoff = 10**18
    result = compute_periodic_union_prefix_count(source, cutoff)
    assert result.count == str(cutoff // 2)


def test_divisor_family_488_fixture() -> None:
    """Test the multiples encoding of a divisor family as a #488-shaped fixture."""
    source = _source([("2", ["0"]), ("3", ["0"]), ("5", ["0"])])
    for cutoff in [0, 1, 10, 30, 100]:
        result = compute_periodic_union_prefix_count(source, cutoff)
        expected = sum(
            1 for t in range(1, cutoff + 1) if t % 2 == 0 or t % 3 == 0 or t % 5 == 0
        )
        assert result.count == str(expected)


def test_result_common_period() -> None:
    """Result retains the common period."""
    source = _source([("4", ["0"]), ("6", ["1"])])
    result = compute_periodic_union_prefix_count(source, 100)
    assert result.common_period == "12"
    expected = sum(1 for t in range(1, 101) if t % 4 == 0 or t % 6 == 1)
    assert result.count == str(expected)


def test_negative_cutoff_is_rejected() -> None:
    with pytest.raises(OperationDomainValidationError, match="nonnegative"):
        compute_periodic_union_prefix_count(_source([("2", ["0"])]), -1)


def test_large_period_complement_uses_scalar_rank() -> None:
    source = _source([("1000000", ["0"])], complement=True)

    result = compute_periodic_union_prefix_count(source, 10)

    assert result.common_period == "1000000"
    assert result.occupied_count == 999_999
    assert result.count == "10"
