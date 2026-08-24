"""Exact bounded friable-count operation tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pydantic import ValidationError

from jacobian.math.number_theory import FriableCountResult, count_friable
from jacobian.math.number_theory._friable_operations import compute_friable_count
from jacobian.math.number_theory._models import (
    MAX_FRIABLE_GENERATED_CUTOFF,
    MAX_FRIABLE_MATERIALIZED_X,
    FriableCountRequest,
)


def _prime_factors(value: int) -> Iterator[int]:
    """Yield the distinct prime factors of one positive integer."""

    divisor = 2
    while divisor * divisor <= value:
        if value % divisor:
            divisor += 1
            continue
        yield divisor
        while value % divisor == 0:
            value //= divisor
        divisor += 1
    if value > 1:
        yield value


def _direct_count(x: int, y: int) -> int:
    return sum(
        1
        for value in range(1, x + 1)
        if all(prime <= y for prime in _prime_factors(value))
    )


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        (0, 0, 0),
        (0, 17, 0),
        (1, 0, 1),
        (25, 0, 1),
        (25, 1, 1),
        (10, 2, 4),
        (10, 3, 7),
        (100, 5, 34),
        (100, 100, 100),
    ],
)
def test_known_counts_and_boundary_conventions(x: int, y: int, expected: int) -> None:
    assert count_friable(x, y) == expected


def test_matches_independent_factorization_oracle_on_small_domain() -> None:
    for x in range(41):
        for y in range(13):
            assert count_friable(x, y) == _direct_count(x, y)


def test_prime_step_recurrence_and_composite_cutoff_invariance() -> None:
    primes = (2, 3, 5, 7, 11)
    for x in range(1, 81):
        assert count_friable(x, 4) == count_friable(x, 3)
        previous = 1
        for prime in primes:
            assert count_friable(x, prime) == count_friable(
                x, previous
            ) + count_friable(x // prime, prime)
            previous = prime


def test_largest_prime_factor_recurrence_from_the_source() -> None:
    for x in range(1, 41):
        for y in range(13):
            primes = (
                candidate
                for candidate in range(2, y + 1)
                if tuple(_prime_factors(candidate)) == (candidate,)
            )
            assert count_friable(x, y) == 1 + sum(
                count_friable(x // prime, prime) for prime in primes
            )


def test_is_monotone_in_source_bound_and_cutoff() -> None:
    assert [count_friable(x, 5) for x in range(30)] == sorted(
        count_friable(x, 5) for x in range(30)
    )
    assert [count_friable(100, y) for y in range(20)] == sorted(
        count_friable(100, y) for y in range(20)
    )


def test_generated_regime_admits_large_sources_when_work_is_small() -> None:
    assert count_friable(10**12, 2) == 40
    assert count_friable(10**30, 5) == 48_207


def test_materialized_regime_reaches_its_cell_boundary() -> None:
    assert (
        count_friable(MAX_FRIABLE_MATERIALIZED_X, 999_999) == MAX_FRIABLE_MATERIALIZED_X
    )


def test_large_direct_cases_do_not_inherit_the_materialized_cap() -> None:
    huge = 10**255
    assert count_friable(huge, 1) == 1
    assert count_friable(huge, huge) == huge


def test_native_api_enforces_the_source_digit_bound_before_work() -> None:
    with pytest.raises(ValueError, match="256 decimal digits"):
        count_friable(10**256, 1)


def test_request_rejects_negative_and_noncanonical_sources() -> None:
    with pytest.raises(ValidationError, match="nonnegative"):
        FriableCountRequest(x="-1", y="2")
    with pytest.raises(ValidationError):
        FriableCountRequest(x="01", y="2")


def test_request_rejects_unbounded_generated_prime_cutoff() -> None:
    with pytest.raises(ValidationError, match="prime cutoff"):
        FriableCountRequest(
            x=str(MAX_FRIABLE_MATERIALIZED_X + 1),
            y=str(MAX_FRIABLE_GENERATED_CUTOFF + 1),
        )


def test_request_rejects_generated_search_above_node_budget() -> None:
    with pytest.raises(ValidationError, match="search-node budget"):
        FriableCountRequest(x="1" + "0" * 255, y="5")


def test_result_rejects_a_nearby_estimate_presented_as_exact() -> None:
    with pytest.raises(ValidationError, match="does not match"):
        FriableCountResult(x="100", y="5", count="35")


def test_result_replays_count_and_binds_both_source_fields() -> None:
    result = compute_friable_count(FriableCountRequest(x="100", y="5"))
    assert result == FriableCountResult(x="100", y="5", count="34")

    with pytest.raises(ValidationError, match="does not match"):
        FriableCountResult(x="125", y="5", count=result.count)
    with pytest.raises(ValidationError, match="does not match"):
        FriableCountResult(x="100", y="3", count=result.count)


def test_operation_is_discoverable_with_one_executable_example() -> None:
    from jacobian.catalog.builtins import BUILTIN_TOOLS

    operation = next(
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id == "number_theory.friable.count.compute"
    )
    assert len(operation.examples) == 1
    example = operation.examples[0]
    request = operation.request_type.model_validate(example.input)
    assert operation.run(request).count == "34"


def test_number_theory_native_api_is_explicit() -> None:
    from jacobian.math import number_theory

    assert tuple(number_theory.__all__) == ("FriableCountResult", "count_friable")
    assert all(hasattr(number_theory, name) for name in number_theory.__all__)
