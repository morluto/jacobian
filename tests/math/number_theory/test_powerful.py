"""Exactness and boundary evidence for the powerful-number decision."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from math import gcd
from typing import Any

import pytest
from pydantic import ValidationError
from sympy import factorint, isprime
from tests.math.number_theory._validation import expect_validation

from jacobian.math.number_theory._models import (
    PowerfulNumberRequest,
    PowerfulNumberResult,
)
from jacobian.math.number_theory._powerful import decide_powerful
from jacobian.math.number_theory._powerful_kernels import (
    _perfect_power_witness,
    decide_powerful_data,
)

PayloadMutation = Callable[[dict[str, Any]], None]


def _full_factorization_oracle(value: int) -> bool:
    return all(exponent >= 2 for exponent in factorint(value).values())


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, True),
        (36, True),
        (2**6 * 3**4, True),
        (12167, True),
        (12168, True),
        (180, False),
        (2**6 * 3**4 * 7, False),
        (7, False),
        (10, False),
        (12166, False),
    ],
)
def test_powerful_known_answers(value: int, expected: bool) -> None:
    result = decide_powerful(PowerfulNumberRequest(value=str(value)))

    assert result.is_powerful is expected
    assert result.conclusion == (
        "POWERFUL"
        if expected
        else (
            "EXPONENT_ONE"
            if any(factor.power == 1 for factor in result.stripped_factors)
            else "ROUGH_NOT_PERFECT_POWER"
        )
    )


def test_one_is_vacuously_powerful_with_empty_reconstruction() -> None:
    result = decide_powerful(PowerfulNumberRequest(value="1"))

    assert result.cutoff == 1
    assert result.checked_through == 1
    assert result.stripped_factors == ()
    assert result.residual == "1"
    assert result.residual_perfect_power is None


def test_partial_factor_certificate_reconstructs_12168() -> None:
    result = decide_powerful(PowerfulNumberRequest(value="12168"))

    stripped_product = 1
    for factor in result.stripped_factors:
        stripped_product *= int(factor.prime) ** factor.power
    assert stripped_product * int(result.residual) == 12168
    assert result.residual_perfect_power is not None
    assert int(
        result.residual_perfect_power.base
    ) ** result.residual_perfect_power.exponent == int(result.residual)


def test_exponent_one_obstruction_records_only_the_processed_range() -> None:
    result = decide_powerful(PowerfulNumberRequest(value=str(2**6 * 3**4 * 7)))

    assert result.conclusion == "EXPONENT_ONE"
    assert result.checked_through == 7
    assert result.checked_through < result.cutoff
    assert result.stripped_factors[-1].prime == "7"
    assert result.stripped_factors[-1].power == 1


def test_rough_non_power_residual_is_an_exact_negative_branch() -> None:
    result = decide_powerful(PowerfulNumberRequest(value="10403"))

    assert result.conclusion == "ROUGH_NOT_PERFECT_POWER"
    assert result.checked_through == result.cutoff
    assert result.residual == "10403"
    assert result.residual_perfect_power is None


def test_p_squared_q_cubed_uses_residual_power_witness() -> None:
    value = 67**2 * 71**3
    result = decide_powerful(PowerfulNumberRequest(value=str(value)))

    assert result.is_powerful is True
    assert [factor.model_dump() for factor in result.stripped_factors] == [
        {"prime": "67", "power": 2}
    ]
    assert result.residual == str(71**3)
    assert result.residual_perfect_power is not None
    assert result.residual_perfect_power.base == "71"
    assert result.residual_perfect_power.exponent == 3


def test_pairwise_gcd_exponents_do_not_fake_a_joint_perfect_power() -> None:
    exponents = (6, 10, 15)
    assert all(
        gcd(left, right) > 1
        for left in exponents
        for right in exponents
        if left != right
    )
    assert gcd(*exponents) == 1
    assert _perfect_power_witness(2**6 * 3**10 * 5**15) is None


def test_source_scale_positive_and_planted_exponent_one_failure() -> None:
    smaller_prime = 60_013
    larger_prime = 60_017
    assert isprime(smaller_prime) and isprime(larger_prime)
    powerful = smaller_prime**2 * larger_prime**3
    spoiled = 3 * powerful

    positive = decide_powerful(PowerfulNumberRequest(value=str(powerful)))
    negative = decide_powerful(PowerfulNumberRequest(value=str(spoiled)))

    assert len(str(powerful)) == 24
    assert len(str(spoiled)) == 25
    assert positive.is_powerful is True
    assert positive.residual_perfect_power is not None
    assert negative.conclusion == "EXPONENT_ONE"
    assert negative.is_powerful is False


def test_near_limit_25_digit_positive_uses_rough_residual_witness() -> None:
    smaller_prime = 99_971
    larger_prime = 99_989
    assert isprime(smaller_prime) and isprime(larger_prime)
    value = smaller_prime**2 * larger_prime**3

    result = decide_powerful(PowerfulNumberRequest(value=str(value)))

    assert len(str(value)) == 25
    assert result.is_powerful is True
    assert result.cutoff == 99_982
    assert result.stripped_factors[0].prime == str(smaller_prime)
    assert result.residual_perfect_power is not None
    assert result.residual_perfect_power.base == str(larger_prime)
    assert result.residual_perfect_power.exponent == 3


@pytest.mark.parametrize(
    ("value", "expected_cutoff"),
    [(31**5 - 1, 31), (31**5, 31), (31**5 + 1, 32)],
)
def test_cutoff_boundary(value: int, expected_cutoff: int) -> None:
    result = decide_powerful(PowerfulNumberRequest(value=str(value)))

    assert result.cutoff == expected_cutoff
    assert result.cutoff**5 >= value
    if result.cutoff > 1:
        assert (result.cutoff - 1) ** 5 < value


def test_25_digit_boundary_is_admitted_and_26_digits_are_rejected() -> None:
    request = PowerfulNumberRequest(value="9999999999999999999999999")
    result = decide_powerful(request)

    assert result.cutoff == 100_000
    with expect_validation("number_theory."):
        PowerfulNumberRequest(value="10000000000000000000000000")


@pytest.mark.parametrize("value", ["0", "-1", "01", 1])
def test_request_rejects_nonpositive_noncanonical_or_nonstring_values(
    value: object,
) -> None:
    with pytest.raises(ValidationError):
        PowerfulNumberRequest.model_validate({"value": value})


@pytest.mark.parametrize(
    ("value", "mutation"),
    [
        (12168, lambda payload: payload.update(value="12167")),
        (12168, lambda payload: payload.update(cutoff=payload["cutoff"] + 1)),
        (
            12168,
            lambda payload: payload["stripped_factors"][0].update(power=2),
        ),
        (12168, lambda payload: payload.update(residual="168")),
        (
            12168,
            lambda payload: payload["residual_perfect_power"].update(exponent=3),
        ),
        (
            2**6 * 3**4 * 7,
            lambda payload: payload.update(checked_through=payload["cutoff"]),
        ),
    ],
)
def test_result_rejects_mutated_source_or_certificate(
    value: int,
    mutation: PayloadMutation,
) -> None:
    genuine = decide_powerful(PowerfulNumberRequest(value=str(value)))
    payload = deepcopy(genuine.model_dump(mode="json"))
    mutation(payload)

    with expect_validation("number_theory."):
        PowerfulNumberResult.model_validate(payload)


@pytest.mark.exhaustive
def test_exhaustive_differential_against_complete_factorization() -> None:
    for value in range(1, 200_001):
        result = decide_powerful_data(value)
        assert (result.conclusion == "POWERFUL") is _full_factorization_oracle(value)
