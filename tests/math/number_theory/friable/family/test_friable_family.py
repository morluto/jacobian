"""Exact bounded friable-family enumeration tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory._friable_kernel import count_friable
from jacobian.math.number_theory.friable.family._models import (
    MAX_FRIABLE_FAMILY_GENERATED_CUTOFF,
    MAX_FRIABLE_FAMILY_MATERIALIZED_X,
    MAX_FRIABLE_FAMILY_ROWS,
    FriableFamilyRequest,
    plan_friable_family,
)
from jacobian.math.number_theory.friable.family.operations import (
    enumerate_friable_family,
)


def _direct_family(x: int, y: int) -> list[int]:
    """Independent oracle: trial-divide every candidate 1..x.

    Uses the same convention as the count operation: 1 is always friable for
    positive x (since 1 has no prime factors), and y <= 1 means only 1.
    """

    def largest_prime_factor(n: int) -> int:
        largest = 1
        d = 2
        while d * d <= n:
            if n % d == 0:
                largest = max(largest, d)
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            largest = max(largest, n)
        return largest

    result = []
    for n in range(1, x + 1):
        if n == 1:
            result.append(n)
            continue
        if y <= 1:
            continue
        if largest_prime_factor(n) <= y:
            result.append(n)
    return result


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        (0, 0, []),
        (0, 17, []),
        (10, 0, [1]),
        (10, 1, [1]),
        (10, 2, [1, 2, 4, 8]),
        (10, 3, [1, 2, 3, 4, 6, 8, 9]),
        (
            100,
            5,
            [
                1,
                2,
                3,
                4,
                5,
                6,
                8,
                9,
                10,
                12,
                15,
                16,
                18,
                20,
                24,
                25,
                27,
                30,
                32,
                36,
                40,
                45,
                48,
                50,
                54,
                60,
                64,
                72,
                75,
                80,
                81,
                90,
                96,
                100,
            ],
        ),
        (100, 100, list(range(1, 101))),
    ],
)
def test_known_families_and_boundary_conventions(
    x: int, y: int, expected: list[int]
) -> None:
    assert enumerate_friable_family(x, y) == expected


def test_matches_independent_factorization_oracle_on_small_domain() -> None:
    for x in range(41):
        for y in range(13):
            assert enumerate_friable_family(x, y) == _direct_family(x, y)


def test_family_length_equals_count() -> None:
    """Defining identity: len(family) = Psi(x, y) from the count operation."""

    for x in range(60):
        for y in range(20):
            assert len(enumerate_friable_family(x, y)) == count_friable(x, y)


def test_family_is_strictly_increasing_and_contains_one() -> None:
    for x in range(1, 51):
        for y in range(15):
            fam = enumerate_friable_family(x, y)
            assert fam[0] == 1
            assert fam == sorted(fam)


def test_generated_regime_admits_large_sources_when_work_is_small() -> None:
    fam = enumerate_friable_family(10**12, 2)
    assert fam == sorted(fam)
    assert len(fam) == 40
    # All 2-smooth numbers are powers of 2.
    assert all(v & (v - 1) == 0 for v in fam)


def test_generated_regime_5_smooth_large_source() -> None:
    fam = enumerate_friable_family(10**30, 5)
    assert len(fam) == 48_207
    assert fam == sorted(fam)


def test_materialized_regime_bound() -> None:
    """The materialized regime admits x up to its cap."""
    fam = enumerate_friable_family(MAX_FRIABLE_FAMILY_MATERIALIZED_X, 999_999)
    assert fam == list(range(1, MAX_FRIABLE_FAMILY_MATERIALIZED_X + 1))


def test_direct_full_interval_still_enforces_the_row_budget() -> None:
    beyond = MAX_FRIABLE_FAMILY_ROWS + 1

    with pytest.raises(ValueError, match="row budget"):
        enumerate_friable_family(beyond, beyond)


def test_rejects_unbounded_generated_prime_cutoff() -> None:
    with pytest.raises(ValueError, match="exceeds the admitted prime cutoff"):
        enumerate_friable_family(
            MAX_FRIABLE_FAMILY_MATERIALIZED_X + 1,
            MAX_FRIABLE_FAMILY_GENERATED_CUTOFF + 1,
        )


def test_rejects_large_source_exceeding_budget() -> None:
    """A request that would exceed an admission budget is rejected."""
    with pytest.raises(ValueError, match="exceeds the"):
        enumerate_friable_family(10**30, 7)


def test_generated_regime_rejects_oversized_serialized_family() -> None:
    with pytest.raises(ValueError, match="serialized result budget"):
        plan_friable_family(10**230, 3)


def test_request_rejects_negative_source() -> None:
    from jacobian.math.number_theory.friable.family._tools import (
        compute_friable_family,
    )

    with pytest.raises(OperationDomainValidationError, match="must be nonnegative"):
        compute_friable_family(FriableFamilyRequest(x="-1", y="2"))


def test_request_rejects_noncanonical_source() -> None:
    with pytest.raises(ValidationError):
        FriableFamilyRequest(x="01", y="2")


def test_operation_is_discoverable_with_one_executable_example() -> None:
    from jacobian.catalog.builtins import BUILTIN_TOOLS

    operation = next(
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id == "number_theory.friable.family.enumerate"
    )
    assert len(operation.examples) == 1
    example = operation.examples[0]
    request = operation.request_type.model_validate(example.input)
    result = operation.run(request)
    assert result.family == (
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "8",
        "9",
        "10",
        "12",
        "15",
        "16",
        "18",
        "20",
        "24",
        "25",
        "27",
        "30",
        "32",
        "36",
        "40",
        "45",
        "48",
        "50",
        "54",
        "60",
        "64",
        "72",
        "75",
        "80",
        "81",
        "90",
        "96",
        "100",
    )
    assert len(result.family) == 34


def test_cross_check_identity_holds_on_generated_regime() -> None:
    """len(family) == count_friable(x, y) in the generated regime too."""
    for x, y in [(10**12, 2), (10**30, 5), (10**15, 3)]:
        assert len(enumerate_friable_family(x, y)) == count_friable(x, y)
