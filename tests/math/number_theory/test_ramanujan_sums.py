from __future__ import annotations

from math import gcd, lcm

import pytest
from pydantic import TypeAdapter, ValidationError
from sympy import divisors, mobius
from tests.math.number_theory._validation import expect_validation

from jacobian._exact import CanonicalInteger
from jacobian.math.number_theory import ramanujan_sum
from jacobian.math.number_theory._models import _MAX_INTEGER_LENGTH
from jacobian.math.number_theory._ramanujan_sum import (
    RAMANUJAN_SUM_OPERATION,
    RamanujanSumRequest,
    RamanujanSumResult,
)
from jacobian.math.number_theory.ramanujan_sums import _MAX_MODULUS_DIGITS


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
        with expect_validation("number_theory."):
            RamanujanSumResult.model_validate(mutation)


def test_zero_sum_binds_the_canonical_zero_string() -> None:
    result = RAMANUJAN_SUM_OPERATION.run(
        RamanujanSumRequest(modulus="4", frequency="3")
    )
    assert result.value == "0"
    assert result == RamanujanSumResult(modulus="4", frequency="3", value="0")


@pytest.mark.parametrize(
    "noncanonical",
    ("-0", "+0", "00", "007", "-007", " 1", "1 ", "1_0", "", "-", "+1"),
)
def test_result_rejects_noncanonical_value_encodings(noncanonical: str) -> None:
    with pytest.raises(ValidationError):
        RamanujanSumResult.model_validate(
            {"modulus": "4", "frequency": "3", "value": noncanonical}
        )


@pytest.mark.parametrize(
    ("model", "payload"),
    (
        (
            RamanujanSumRequest,
            {"modulus": "4", "frequency": "-0"},
        ),
        (
            RamanujanSumResult,
            {"modulus": "4", "frequency": "-0", "value": "2"},
        ),
    ),
)
def test_negative_zero_frequency_is_rejected_before_source_binding(
    model: type, payload: dict[str, str]
) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(payload)


@pytest.mark.parametrize(
    "noncanonical",
    ("-0", "+0", "00", "007", "-007", " 1", "1 ", "1_0", "", "-", "+1"),
)
def test_request_rejects_noncanonical_frequency_encodings(
    noncanonical: str,
) -> None:
    with pytest.raises(ValidationError):
        RamanujanSumRequest(modulus="4", frequency=noncanonical)


def test_equal_frequencies_share_one_serialized_identity() -> None:
    zero_request = RamanujanSumRequest(modulus="4", frequency="0")
    result = RAMANUJAN_SUM_OPERATION.run(zero_request)
    assert result == RamanujanSumResult(modulus="4", frequency="0", value="2")


@pytest.mark.parametrize(
    "encoding",
    (
        "0",
        "7",
        "-7",
        "9" * _MAX_INTEGER_LENGTH,
        "-0",
        "+0",
        "00",
        "-007",
        "",
        "-",
        "+1",
    ),
)
def test_frequency_grammar_is_owned_by_canonical_integer(encoding: str) -> None:
    owner = TypeAdapter(CanonicalInteger)
    try:
        expected = owner.validate_python(encoding)
        owner_accepts = True
    except ValidationError:
        owner_accepts = False
        expected = None

    try:
        request = RamanujanSumRequest(modulus="4", frequency=encoding)
    except ValidationError:
        assert not owner_accepts
    else:
        assert owner_accepts
        assert request.frequency == expected


@pytest.mark.parametrize(
    "encoding",
    ("0", "7", "9" * 12, "-0", "+0", "00", "-007", "", "-", "+1"),
)
def test_modulus_grammar_is_owned_by_canonical_integer(encoding: str) -> None:
    owner = TypeAdapter(CanonicalInteger)
    try:
        expected = owner.validate_python(encoding)
        owner_accepts = True
    except ValidationError:
        owner_accepts = False
        expected = None

    try:
        request = RamanujanSumRequest(modulus=encoding, frequency="0")
    except ValidationError:
        assert not owner_accepts
    else:
        assert owner_accepts
        assert request.modulus == expected


@pytest.mark.parametrize("negative", ("-1", "-4", "-" + "9" * 11))
def test_owner_grammar_admits_negative_moduli_the_operation_rejects(
    negative: str,
) -> None:
    assert TypeAdapter(CanonicalInteger).validate_python(negative) == negative
    with expect_validation("number_theory."):
        RamanujanSumRequest(modulus=negative, frequency="0")
    with expect_validation("number_theory."):
        RamanujanSumResult.model_validate(
            {"modulus": negative, "frequency": "2", "value": "-2"}
        )


@pytest.mark.parametrize(
    ("modulus", "frequency", "value"),
    (
        ("4", "-2", "-2"),
        ("5", "-3", "-1"),
        ("1", "-9", "1"),
    ),
)
def test_canonical_negative_frequencies_round_trip(
    modulus: str, frequency: str, value: str
) -> None:
    request = RamanujanSumRequest(modulus=modulus, frequency=frequency)
    assert RAMANUJAN_SUM_OPERATION.run(request) == RamanujanSumResult(
        modulus=modulus, frequency=frequency, value=value
    )
    assert RamanujanSumResult.model_validate(
        {"modulus": modulus, "frequency": frequency, "value": value}
    ) == RAMANUJAN_SUM_OPERATION.run(request)


@pytest.mark.parametrize(
    ("modulus", "frequency", "value"),
    (
        ("5", "0", "4"),
        ("4", "2", "-2"),
        ("1", "9", "1"),
        ("549755813888", "274877906944", "-274877906944"),
    ),
)
def test_result_accepts_canonical_nonzero_values(
    modulus: str, frequency: str, value: str
) -> None:
    result = RamanujanSumResult.model_validate(
        {"modulus": modulus, "frequency": frequency, "value": value}
    )
    assert int(result.value) == ramanujan_sum(int(modulus), int(frequency))


def test_ramanujan_sum_request_bounds_factorization_and_frequency_work() -> None:
    boundary = RamanujanSumRequest(
        modulus="9" * _MAX_MODULUS_DIGITS,
        frequency="9" * _MAX_INTEGER_LENGTH,
    )
    result = RAMANUJAN_SUM_OPERATION.run(boundary)
    assert int(result.value) == ramanujan_sum(
        int(boundary.modulus), int(boundary.frequency)
    )

    with expect_validation("number_theory."):
        RamanujanSumRequest(modulus=str(10**_MAX_MODULUS_DIGITS), frequency="0")
    with expect_validation("number_theory."):
        RamanujanSumRequest(modulus="1", frequency="9" * (_MAX_INTEGER_LENGTH + 1))
    with pytest.raises(ValidationError):
        RamanujanSumRequest(modulus="-1", frequency="0")


def test_ramanujan_sum_rejects_negative_native_modulus() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        ramanujan_sum(-1, 0)


def test_native_ramanujan_sum_bounds_factorization_work() -> None:
    with pytest.raises(ValueError, match=rf"at most {_MAX_MODULUS_DIGITS}"):
        ramanujan_sum(10**_MAX_MODULUS_DIGITS, 1)
    assert ramanujan_sum(549755813888, 274877906944) == -274877906944


def test_native_ramanujan_sum_bounds_frequency_magnitude() -> None:
    # Reported failure mode: the native entry point must enforce the same
    # 256-character frequency envelope as the wire request before any
    # factorization or modular reduction.
    with pytest.raises(ValueError, match=rf"at most {_MAX_INTEGER_LENGTH}"):
        ramanujan_sum(4, 10**_MAX_INTEGER_LENGTH)
    with pytest.raises(ValueError, match=rf"at most {_MAX_INTEGER_LENGTH}"):
        ramanujan_sum(4, -(10 ** (_MAX_INTEGER_LENGTH - 1)))
    with pytest.raises(ValueError, match=rf"at most {_MAX_INTEGER_LENGTH}"):
        ramanujan_sum(0, 10**_MAX_INTEGER_LENGTH)

    assert ramanujan_sum(4, 10**_MAX_INTEGER_LENGTH - 2) == -2
    assert ramanujan_sum(4, -(10 ** (_MAX_INTEGER_LENGTH - 1) - 2)) == -2
    assert ramanujan_sum(4, 10**_MAX_INTEGER_LENGTH - 1) == 0


def test_native_frequency_bound_matches_wire_admission() -> None:
    for modulus, frequency in (
        ("4", str(10**_MAX_INTEGER_LENGTH - 2)),
        ("1", "9" * _MAX_INTEGER_LENGTH),
    ):
        request = RamanujanSumRequest(modulus=modulus, frequency=frequency)
        assert RAMANUJAN_SUM_OPERATION.run(request).value == str(
            ramanujan_sum(int(modulus), int(frequency))
        )
