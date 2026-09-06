"""Canonical formal-series contracts for rational generating functions."""

import json
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics._recurrence_admission import (
    _lower_decimal_digits,
    _require_bounded_fraction,
)
from jacobian.math.combinatorics._recurrence_models import (
    RationalGeneratingFunctionCoefficientsRequest,
    RationalGeneratingFunctionCoefficientsResult,
)
from jacobian.math.combinatorics.operations import (
    rational_generating_function_coefficients,
    verify_rational_generating_function_coefficients,
)
from jacobian.math.polynomials.series._models import (
    SeriesTruncateRequest,
    TruncatedSeries,
)
from jacobian.math.polynomials.series.operations import truncate

ONE = CanonicalRational(num=1, den=1)
MINUS_ONE = CanonicalRational(num=-1, den=1)


def _expand(
    denominator: tuple[CanonicalRational, ...], order: int
) -> RationalGeneratingFunctionCoefficientsResult:
    return rational_generating_function_coefficients(
        (ONE,), denominator, "ASCENDING_POWERS_OF_X", 0, order
    )


def test_degree_32_recurrence_returns_a_bound_canonical_series() -> None:
    order = 4_096
    denominator = (ONE, *((MINUS_ONE,) * 32))
    request = RationalGeneratingFunctionCoefficientsRequest(
        numerator=(ONE,),
        denominator=denominator,
        coefficient_convention="ASCENDING_POWERS_OF_X",
        expansion_point=0,
        truncation_order=order,
    )

    result = _expand(request.denominator, request.truncation_order)
    values = tuple(value.as_fraction() for value in result.series.coefficients)
    denominator_values = tuple(value.as_fraction() for value in denominator)

    assert len(values) == order
    assert values[-1] == sum(values[-33:-1])
    assert all(
        sum(
            denominator_values[offset] * values[degree - offset]
            for offset in range(min(degree, len(denominator_values) - 1) + 1)
        )
        == (1 if degree == 0 else 0)
        for degree in range(order)
    )
    assert result.series.variable == "x"
    assert result.series.truncation_order == order
    assert verify_rational_generating_function_coefficients(result)
    assert (
        RationalGeneratingFunctionCoefficientsResult.model_validate_json(
            encode_strict_json(result.model_dump(mode="json"))
        )
        == result
    )
    wire = result.model_dump_json()
    assert '"expansion_point":"0"' in wire
    assert (
        RationalGeneratingFunctionCoefficientsResult.model_validate_json(wire) == result
    )
    with pytest.raises(ValidationError):
        RationalGeneratingFunctionCoefficientsRequest.model_validate_json(
            request.model_dump_json().replace('"0"', "0")
        )

    forged = result.model_dump(mode="json")
    forged["series"]["coefficients"][0] = {"num": "2", "den": "1"}
    decoded = RationalGeneratingFunctionCoefficientsResult.model_validate(forged)
    assert not verify_rational_generating_function_coefficients(decoded)


def test_canonical_series_composes_into_formal_series_truncation() -> None:
    result = rational_generating_function_coefficients(
        (ONE,),
        (ONE, MINUS_ONE),
        "ASCENDING_POWERS_OF_X",
        "0",
        6,
    )
    request = SeriesTruncateRequest.model_validate(
        {"series": result.series.model_dump(mode="json"), "target_order": 3}
    )

    truncated = truncate(request.series, request.target_order)

    assert tuple(value.as_fraction() for value in truncated.result.coefficients) == (
        Fraction(1),
        Fraction(1),
        Fraction(1),
    )


def test_source_relation_survives_canonical_json_roundtrip() -> None:
    result = _expand((ONE, MINUS_ONE), 5)
    wire = json.loads(encode_strict_json(result.model_dump(mode="json")))
    decoded = RationalGeneratingFunctionCoefficientsResult.model_validate(wire)

    assert decoded.numerator == (ONE,)
    assert decoded.denominator == (ONE, MINUS_ONE)
    assert verify_rational_generating_function_coefficients(decoded)


def test_verifier_rejects_model_copy_scalar_corruption() -> None:
    result = _expand((ONE, MINUS_ONE), 5)
    malformed_denominator = result.denominator[0].model_copy(update={"den": "0"})
    malformed = result.model_copy(
        update={"denominator": (malformed_denominator, MINUS_ONE)}
    )
    assert not verify_rational_generating_function_coefficients(malformed)

    malformed_coefficient = result.series.coefficients[0].model_copy(
        update={"num": None}
    )
    malformed_series = result.series.model_copy(
        update={
            "coefficients": (
                malformed_coefficient,
                *result.series.coefficients[1:],
            )
        }
    )
    malformed = result.model_copy(update={"series": malformed_series})
    assert not verify_rational_generating_function_coefficients(malformed)


def test_verifier_rejects_overwork_before_scanning_a_forged_series() -> None:
    order = 250_000
    forged = RationalGeneratingFunctionCoefficientsResult.model_construct(
        numerator=(ONE,),
        denominator=(ONE, *((MINUS_ONE,) * 32)),
        coefficient_convention="ASCENDING_POWERS_OF_X",
        expansion_point="0",
        truncation_order=order,
        series=TruncatedSeries.model_construct(
            variable="x",
            truncation_order=order,
            coefficients=(ONE,) * order,
        ),
    )

    assert not verify_rational_generating_function_coefficients(forged)


def test_decimal_digit_bound_counts_power_of_ten_exactly() -> None:
    below = 10**32_768 - 1
    at = 10**32_768
    assert _lower_decimal_digits(below) == 32_768
    assert _lower_decimal_digits(at) == 32_769
    _require_bounded_fraction(
        Fraction(below),
        label="series coefficient",
        location=("coefficients", 0),
    )
    with pytest.raises(OperationDomainValidationError, match="32768-digit bound"):
        _require_bounded_fraction(
            Fraction(at),
            label="series coefficient",
            location=("coefficients", 0),
        )


def test_denominator_degree_controls_the_recurrence_work_envelope() -> None:
    denominator = (ONE, *((MINUS_ONE,) * 32))

    with pytest.raises(OperationDomainValidationError, match="exact work bound"):
        _expand(denominator, 8_000)


def test_large_constant_prefix_is_admitted_by_recurrence_work() -> None:
    assert len(_expand((ONE,), 250_000).series.coefficients) == 250_000
