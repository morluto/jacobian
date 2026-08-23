from __future__ import annotations

from math import gcd, lcm

import pytest
from pydantic import ValidationError
from sympy import divisors, mobius

from jacobian.math.number_theory import ramanujan_sum
from jacobian.math.number_theory._ramanujan_sum import (
    RAMANUJAN_SUM_OPERATION,
    RamanujanSumRequest,
    RamanujanSumResult,
)


@pytest.mark.parametrize(
    ("modulus", "frequency", "expected"),
    (
        (0, -17, 0),
        (1, 9, 1),
        (2, 3, -1),
        (2, 4, 1),
        (3, 6, 2),
        (3, 7, -1),
        (4, 0, 2),
        (4, 2, -2),
        (4, 3, 0),
        (5, 0, 4),
        (5, 3, -1),
    ),
)
def test_ramanujan_sum_known_values(
    modulus: int, frequency: int, expected: int
) -> None:
    assert ramanujan_sum(modulus, frequency) == expected


def test_ramanujan_sum_agrees_with_divisor_mobius_formula() -> None:
    for modulus in range(1, 65):
        for frequency in range(-40, 41):
            common_divisor = gcd(modulus, abs(frequency))
            expected = sum(
                divisor * int(mobius(modulus // divisor))
                for divisor in divisors(common_divisor)
            )
            assert ramanujan_sum(modulus, frequency) == expected


def test_ramanujan_sum_periodicity_evenness_and_multiplicativity() -> None:
    for modulus in range(1, 25):
        for frequency in range(-20, 21):
            value = ramanujan_sum(modulus, frequency)
            assert ramanujan_sum(modulus, -frequency) == value
            assert ramanujan_sum(modulus, frequency + modulus) == value

    assert ramanujan_sum(20, 7) == ramanujan_sum(4, 7) * ramanujan_sum(5, 7)


@pytest.mark.parametrize(
    ("left_modulus", "right_modulus", "expected"),
    ((2, 2, 2), (2, 3, 0), (3, 3, 6), (3, 4, 0)),
)
def test_ramanujan_sum_complete_period_orthogonality(
    left_modulus: int, right_modulus: int, expected: int
) -> None:
    period = lcm(left_modulus, right_modulus)
    inner_product = sum(
        ramanujan_sum(left_modulus, frequency) * ramanujan_sum(right_modulus, frequency)
        for frequency in range(period)
    )
    assert inner_product == expected


def test_operation_returns_a_source_bound_exact_result() -> None:
    result = RAMANUJAN_SUM_OPERATION.run(
        RamanujanSumRequest(modulus="4", frequency="2")
    )
    assert result == RamanujanSumResult(modulus="4", frequency="2", value="-2")

    for mutation in (
        {"modulus": "4", "frequency": "1", "value": "-2"},
        {"modulus": "3", "frequency": "2", "value": "-2"},
        {"modulus": "4", "frequency": "2", "value": "2"},
    ):
        with pytest.raises(ValidationError, match="does not match"):
            RamanujanSumResult.model_validate(mutation)


def test_ramanujan_sum_request_bounds_factorization_and_frequency_work() -> None:
    boundary = RamanujanSumRequest(
        modulus="999999999999",
        frequency="9" * 256,
    )
    result = RAMANUJAN_SUM_OPERATION.run(boundary)
    assert int(result.value) == ramanujan_sum(
        int(boundary.modulus), int(boundary.frequency)
    )

    with pytest.raises(ValidationError, match=r"at most 12|12 characters"):
        RamanujanSumRequest(modulus="1000000000000", frequency="0")
    with pytest.raises(ValidationError, match=r"at most 256|256 characters"):
        RamanujanSumRequest(modulus="1", frequency="9" * 257)
    with pytest.raises(ValidationError):
        RamanujanSumRequest(modulus="-1", frequency="0")


def test_ramanujan_sum_rejects_negative_native_modulus() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        ramanujan_sum(-1, 0)
