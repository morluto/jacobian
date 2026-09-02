"""Tests for bounded r-full integer enumeration."""

from __future__ import annotations

import pytest
from sympy.ntheory.factor_ import factorint

from jacobian.canonical import format_canonical_integer
from jacobian.math.number_theory._r_full_enumerate import (
    enumerate_r_full,
    enumerate_r_full_numbers,
)
from jacobian.math.number_theory._r_full_enumerate_kernels import (
    enumerate_r_full as enumerate_r_full_kernel,
)
from jacobian.math.number_theory._r_full_enumerate_models import (
    MAX_R_FULL_FAMILY_SIZE,
    RFullEnumerateRequest,
    RFullEnumerateResult,
    plan_r_full_family,
)


def test_r2_equals_powerful() -> None:
    """r=2 gives the powerful family."""
    result = enumerate_r_full_kernel(10, 2)
    assert result == [1, 4, 8, 9]


def test_r3_cubefull_to_20() -> None:
    """3-full integers in [1,20] are 1, 8, 16."""
    result = enumerate_r_full_kernel(20, 3)
    assert result == [1, 8, 16]


def test_all_results_are_r_full() -> None:
    """Every returned integer has all prime exponents >= r."""
    for r in [2, 3, 4]:
        result = enumerate_r_full_kernel(200, r)
        for n in result:
            for _, exp in factorint(n).items():
                assert exp >= r, f"{n} is not {r}-full"


def test_cross_check_brute_force() -> None:
    """Cross-check against brute-force factorization."""
    for r in [2, 3, 4, 5]:
        result = enumerate_r_full_kernel(300, r)
        brute = [
            n for n in range(1, 301) if all(e >= r for _, e in factorint(n).items())
        ]
        assert result == brute


def test_sorted_and_unique() -> None:
    """The family is sorted with no duplicates."""
    result = enumerate_r_full_kernel(500, 3)
    assert result == sorted(result)
    assert len(result) == len(set(result))


def test_operation_round_trip() -> None:
    """The operation model round-trips through the kernel."""
    request = RFullEnumerateRequest(minimum_exponent=3, cutoff="100")
    result = enumerate_r_full_numbers(request)
    assert result.minimum_exponent == 3
    assert result.cutoff == "100"
    assert result.count == len(result.family)


def test_r2_compatibility() -> None:
    """r=2 result matches the powerful family from #2767."""
    request = RFullEnumerateRequest(minimum_exponent=2, cutoff="100")
    result = enumerate_r_full_numbers(request)
    assert "1" in result.family
    assert "4" in result.family
    assert "8" in result.family
    assert "9" in result.family
    assert "12" not in result.family


def test_result_requires_one_for_positive_cutoff() -> None:
    """A complete positive-cutoff family must contain the universal member 1."""
    with pytest.raises(ValueError, match="must begin with 1"):
        RFullEnumerateResult(
            minimum_exponent=2,
            cutoff="10",
            count=0,
            family=(),
        )


def test_native_api_uses_integer_arguments() -> None:
    """Native callers can enumerate without constructing a wire request."""
    result = enumerate_r_full(3, 20)
    assert result == (1, 8, 16)


def test_high_exponent_admits_cutoff_above_old_ceiling() -> None:
    """A sparse high-exponent family is admitted from its actual prime bound."""
    result = enumerate_r_full(64, 2**64)
    assert result == (1, 18446744073709551616)


def test_exponent_bound_follows_cutoff_width() -> None:
    """Sparse exponents beyond the old fixed cap remain valid requests."""
    result = enumerate_r_full(257, 2**257)
    assert result == (1, 2**257)


def test_native_path_does_not_apply_transport_byte_ceiling() -> None:
    """Native callers retain a family admitted by the mathematical budgets."""
    result = enumerate_r_full(64, 10**128)
    assert len(result) == 148_860


def test_large_admitted_cutoff_uses_canonical_integer_formatting() -> None:
    """Large admitted integers do not hit CPython's decimal conversion cap."""
    cutoff = "1" + "0" * 5_000
    result = enumerate_r_full_numbers(
        RFullEnumerateRequest(minimum_exponent=16_000, cutoff=cutoff)
    )
    assert result.cutoff == cutoff
    assert result.family[0] == "1"
    assert format_canonical_integer(2**16_000) in result.family
    assert result.count == 611


def test_native_path_admits_trivial_high_exponents() -> None:
    """An exponent above the cutoff-derived bit length still has a valid family."""
    assert enumerate_r_full(3, 1) == (1,)


def test_large_complete_family_is_admitted_by_family_size() -> None:
    """A large complete family remains inside the mathematical cardinality bound."""
    result = enumerate_r_full(64, 10**109)
    assert len(result) == 31_377


def test_planner_admits_only_bounded_cumulative_merge_work() -> None:
    plan = plan_r_full_family(2, 10**9)
    assert plan.exceeded
    assert plan.reason == "planning"


def test_high_exponent_family_uses_mathematical_family_bound() -> None:
    result = enumerate_r_full_numbers(
        RFullEnumerateRequest(
            minimum_exponent=64,
            cutoff="1" + "0" * 128,
        )
    )

    assert result.count == len(result.family)
    assert result.count <= MAX_R_FULL_FAMILY_SIZE


def test_result_rejects_oversized_family_member_before_parsing() -> None:
    """Result validation bounds member representations before bigint parsing."""
    with pytest.raises(ValueError, match="canonical width"):
        RFullEnumerateResult(
            minimum_exponent=2,
            cutoff="10",
            count=1,
            family=("9" * 1_000_000,),
        )


def test_result_rejects_oversized_cutoff_before_parsing() -> None:
    """Result validation bounds cutoff representations before bigint parsing."""
    with pytest.raises(ValueError, match="at most 32769 characters"):
        RFullEnumerateResult(
            minimum_exponent=2,
            cutoff="1" * 1_000_000,
            count=1,
            family=("1",),
        )
