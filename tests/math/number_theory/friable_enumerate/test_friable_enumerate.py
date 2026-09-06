"""Exact bounded friable-enumerate operation tests."""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

from jacobian.math.number_theory._friable_enumerate import compute_friable_enumerate
from jacobian.math.number_theory._friable_enumerate_kernels import enumerate_friable
from jacobian.math.number_theory._friable_enumerate_models import (
    MAX_FRIABLE_ENUMERATE_FAMILY_SIZE,
    MAX_FRIABLE_ENUMERATE_GENERATED_CUTOFF,
    MAX_FRIABLE_ENUMERATE_MATERIALIZED_X,
    FriableEnumerateRequest,
    FriableEnumerateResult,
)
from jacobian.math.number_theory._friable_kernel import count_friable


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


def _direct_enumerate(x: int, y: int) -> list[int]:
    """Brute-force oracle: factor every integer in 1..x."""

    if x == 0:
        return []
    if y <= 1:
        return [1] if x >= 1 else []
    result = []
    for n in range(1, x + 1):
        if all(p <= y for p in _prime_factors(n)):
            result.append(n)
    return result


# --------------------------------------------------------------------------- #
# Fixture: the positive 5-smooth integers at most 20
# --------------------------------------------------------------------------- #

FIVE_SMOOTH_THROUGH_20 = (1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20)


def test_fixture_five_smooth_through_20() -> None:
    result = enumerate_friable(20, 5)
    assert result == FIVE_SMOOTH_THROUGH_20


def test_known_families_and_boundary_conventions() -> None:
    assert enumerate_friable(0, 5) == ()
    assert enumerate_friable(1, 0) == (1,)
    assert enumerate_friable(25, 1) == (1,)
    assert enumerate_friable(10, 2) == (1, 2, 4, 8)
    assert enumerate_friable(10, 3) == (1, 2, 3, 4, 6, 8, 9)
    assert enumerate_friable(100, 100) == tuple(range(1, 101))
    assert enumerate_friable(100, 2) == tuple(
        2**k
        for k in range(7)  # 1,2,4,8,16,32,64
    )


# --------------------------------------------------------------------------- #
# Tests from the implementation packet
# --------------------------------------------------------------------------- #


def test_factor_each_emitted_value_and_require_smooth() -> None:
    """Factor each emitted value and require every prime factor at most y."""
    family = enumerate_friable(200, 5)
    for value in family:
        assert all(prime <= 5 for prime in _prime_factors(value)), (
            f"{value} has a prime factor exceeding 5"
        )


def test_independently_enumerate_exponent_vectors() -> None:
    """Independently enumerate exponent vectors and compare families."""

    def _exponent_vector_enumerate(x: int, y: int) -> list[int]:
        primes = [p for p in range(2, y + 1) if all(p % d != 0 for d in range(2, p))]
        values = []

        def visit(idx: int, product: int) -> None:
            if idx == len(primes):
                values.append(product)
                return
            prime = primes[idx]
            while product <= x:
                visit(idx + 1, product)
                product *= prime

        visit(0, 1)
        values.sort()
        return values

    for x, y in [(20, 5), (50, 7), (100, 5), (30, 3)]:
        assert tuple(_exponent_vector_enumerate(x, y)) == enumerate_friable(x, y)


def test_absent_values_and_increasing_ordering() -> None:
    """Assert that 7,11,13,14,17,19 are absent and ordering is increasing."""
    family = enumerate_friable(20, 5)
    absent = {7, 11, 13, 14, 17, 19}
    for value in absent:
        assert value not in family, (
            f"{value} should be absent from the 5-smooth family at most 20"
        )

    assert list(family) == sorted(family)
    assert len(family) == len(set(family))


# --------------------------------------------------------------------------- #
# Cross-checks against the existing count operation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("x", [0, 1, 5, 10, 20, 50, 100, 200, 500])
@pytest.mark.parametrize("y", [0, 1, 2, 3, 5, 7, 11, 50])
def test_family_length_matches_count(x: int, y: int) -> None:
    family = enumerate_friable(x, y)
    assert len(family) == count_friable(x, y)


def test_matches_independent_factorization_oracle() -> None:
    for x in range(51):
        for y in range(13):
            assert enumerate_friable(x, y) == tuple(_direct_enumerate(x, y))


def test_family_is_increasing_and_deduplicated() -> None:
    for x in [10, 50, 100, 500, 1000]:
        for y in [3, 5, 7, 13]:
            family = enumerate_friable(x, y)
            assert list(family) == sorted(family)
            assert len(family) == len(set(family))


def test_generated_regime_admits_large_sources_when_work_is_small() -> None:
    family = enumerate_friable(10**12, 2)
    assert len(family) == 40
    assert family[0] == 1
    assert family[-1] == 2**39

    family5 = enumerate_friable(10**30, 5)
    assert len(family5) == 48_207


def test_materialized_regime_reaches_its_boundary() -> None:
    """The direct family remains inside the exact result-size envelope."""
    cutoff = MAX_FRIABLE_ENUMERATE_FAMILY_SIZE
    family = enumerate_friable(cutoff, cutoff)
    assert len(family) == cutoff


def test_operation_request_rejects_negative_sources() -> None:
    from jacobian.catalog.models import OperationDomainValidationError

    with pytest.raises(OperationDomainValidationError, match="must be nonnegative"):
        compute_friable_enumerate(FriableEnumerateRequest(x=-1, y=2))


def test_operation_rejects_unbounded_generated_prime_cutoff() -> None:
    with pytest.raises(ValueError, match="exceeds the admitted prime cutoff"):
        compute_friable_enumerate(
            FriableEnumerateRequest(
                x=MAX_FRIABLE_ENUMERATE_MATERIALIZED_X + 1,
                y=MAX_FRIABLE_ENUMERATE_GENERATED_CUTOFF + 1,
            )
        )


def test_operation_rejects_family_above_result_size_budget() -> None:
    with pytest.raises(ValueError, match="result-size budget"):
        compute_friable_enumerate(
            FriableEnumerateRequest(
                x=10000000,
                y=100,
            )
        )


def test_operation_example_executes() -> None:
    result = compute_friable_enumerate(FriableEnumerateRequest(x=20, y=5))
    assert result.family.elements == FIVE_SMOOTH_THROUGH_20
    assert result.x == 20
    assert result.y == 5


def test_result_rejects_impossible_degenerate_families() -> None:
    with pytest.raises(ValueError, match="must be empty when x is zero"):
        FriableEnumerateResult.model_validate(
            {"x": 0, "y": 5, "family": {"elements": [1]}}
        )

    with pytest.raises(ValueError, match=r"must have family \{1\}"):
        FriableEnumerateResult.model_validate(
            {"x": 5, "y": 0, "family": {"elements": [2]}}
        )


def test_result_binds_zero_sources_by_value_and_requires_increasing_family() -> None:
    result = FriableEnumerateResult.model_validate(
        {"x": 0, "y": 5, "family": {"elements": []}}
    )
    assert result.x == 0

    with pytest.raises(ValueError, match="strictly increasing order"):
        FriableEnumerateResult.model_validate(
            {"x": 5, "y": 5, "family": {"elements": [2, 1]}}
        )


def test_operation_is_discoverable_with_one_executable_example() -> None:
    from jacobian.catalog.builtins import BUILTIN_TOOLS

    operation = next(
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id == "integer.friable.enumerate"
    )
    assert len(operation.examples) == 1
    example = operation.examples[0]
    request = operation.request_type.model_validate_json(
        json.dumps(example.input), strict=True
    )
    result = operation.run(request)
    assert result.family.elements == FIVE_SMOOTH_THROUGH_20
