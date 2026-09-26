import json
from collections.abc import Sequence
from fractions import Fraction
from math import comb

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials.series._models import (
    MAX_RESULT_RATIONAL_DIGITS,
    MAX_TRUNCATION_ORDER,
    SeriesComposeRequest,
    SeriesDivideRequest,
    SeriesMultiplyResult,
    SeriesPowerRequest,
    SeriesReversionRequest,
    SeriesReversionResult,
    TruncatedSeries,
    _SeriesAddSubtractRequest,
    _SeriesMultiplyRequest,
)
from jacobian.math.polynomials.series.operations import (
    add,
    compose,
    divide,
    multiply,
    power,
    reversion,
)


def _coefficient(num: str = "1", den: str = "1") -> dict[str, str]:
    return {"num": num, "den": den}


def _series(order: int, coefficients: list[dict[str, str]]) -> dict[str, object]:
    return {"variable": "x", "truncation_order": order, "coefficients": coefficients}


def _assert_coefficients_fit_result_envelope(
    coefficients: Sequence[CanonicalRational],
) -> None:
    assert all(
        max(len(str(value.num)), len(str(value.den))) <= MAX_RESULT_RATIONAL_DIGITS
        for value in coefficients
    )


def test_multiplication_bound_does_not_reject_coefficientwise_addition() -> None:
    order = 20
    coefficients = [_coefficient(den=str(2**800)) for _ in range(order)]
    payload = {
        "left": _series(order, coefficients),
        "right": _series(order, coefficients),
    }

    add_request = _SeriesAddSubtractRequest.model_validate_json(json.dumps(payload))
    assert [
        c.as_fraction()
        for c in add(add_request.left, add_request.right).result.coefficients
    ] == [Fraction(1, 2**799)] * order
    request = _SeriesMultiplyRequest.model_validate_json(json.dumps(payload))
    result = multiply(request.left, request.right)
    assert [c.as_fraction() for c in result.result.coefficients] == [
        Fraction(k + 1, 2**1600) for k in range(order)
    ]
    _assert_coefficients_fit_result_envelope(result.result.coefficients)


def test_power_propagates_binary_convolution_growth() -> None:
    order = 8
    coefficients = [_coefficient(den=str(3**500)) for _ in range(order)]
    request = SeriesPowerRequest.model_validate_json(
        json.dumps({"series": _series(order, coefficients), "exponent": 16})
    )
    result = power(request.series, request.exponent)
    assert [c.as_fraction() for c in result.result.coefficients] == [
        Fraction(comb(k + 15, 15), 3**8000) for k in range(order)
    ]
    _assert_coefficients_fit_result_envelope(result.result.coefficients)


def test_division_propagates_inverse_and_residual_growth() -> None:
    # The leading reciprocal contributes 2**700 at each recurrence degree;
    # order 24 genuinely exceeds the result envelope (order 8 does not).
    order = 24
    numerator = [_coefficient() for _ in range(order)]
    denominator = [_coefficient(den=str(2**700)), *[_coefficient()] * (order - 1)]
    request = SeriesDivideRequest.model_validate_json(
        json.dumps(
            {"left": _series(order, numerator), "right": _series(order, denominator)}
        )
    )
    with pytest.raises(OperationDomainValidationError) as error:
        divide(request.left, request.right)
    assert (
        error.value.errors()[0]["type"]
        == "formal_power_series.inverse_coefficient_growth"
    )


def test_composition_with_reduced_coefficients_fits_result_envelope() -> None:
    order = 8
    outer = [_coefficient() for _ in range(order)]
    inner = [_coefficient("0"), *[_coefficient(den=str(5**300))] * (order - 1)]
    request = SeriesComposeRequest.model_validate_json(
        json.dumps({"outer": _series(order, outer), "inner": _series(order, inner)})
    )
    result = compose(request.outer, request.inner)
    expected = [Fraction(1)]
    expected.extend(
        sum(
            (
                Fraction(comb(degree - 1, power_degree - 1), 5 ** (300 * power_degree))
                for power_degree in range(1, degree + 1)
            ),
            start=Fraction(),
        )
        for degree in range(1, order)
    )
    assert [
        coefficient.as_fraction() for coefficient in result.result.coefficients
    ] == expected
    _assert_coefficients_fit_result_envelope(result.result.coefficients)


def test_reversion_propagates_nonlinear_coefficient_growth() -> None:
    order = 24
    coefficients = [
        _coefficient("0"),
        _coefficient("1"),
        _coefficient(num=str(10**255)),
        *[_coefficient()] * (order - 3),
    ]
    request = SeriesReversionRequest.model_validate_json(
        json.dumps(_series(order, coefficients))
    )
    with pytest.raises(OperationDomainValidationError) as error:
        reversion(request.as_series())
    assert (
        error.value.errors()[0]["type"]
        == "formal_power_series.reversion_coefficient_growth"
    )


def test_sparse_linear_reversion_remains_admitted() -> None:
    request = SeriesReversionRequest.model_validate_json(
        json.dumps(
            _series(
                4,
                [
                    _coefficient("0"),
                    _coefficient("2"),
                    _coefficient("0"),
                    _coefficient("0"),
                ],
            )
        )
    )

    result = reversion(request.as_series())
    assert [c.as_fraction() for c in result.result.coefficients] == [
        Fraction(0),
        Fraction(1, 2),
        Fraction(0),
        Fraction(0),
    ]
    assert all(c.num == 0 for c in (*result.left_residual, *result.right_residual))


def test_largest_multiplication_result_fits_shared_output_envelope() -> None:
    numerator = str(10**256 - 1)
    denominator = str(10**255)
    coefficient = _coefficient(numerator, denominator)
    series = TruncatedSeries.model_validate_json(
        json.dumps(_series(MAX_TRUNCATION_ORDER, [coefficient] * MAX_TRUNCATION_ORDER))
    )
    result = multiply(series, series)
    assert [c.as_fraction() for c in result.result.coefficients] == [
        Fraction((degree + 1) * (10**256 - 1) ** 2, 10**510)
        for degree in range(MAX_TRUNCATION_ORDER)
    ]
    _assert_coefficients_fit_result_envelope(result.result.coefficients)


def test_result_round_trips_remain_structural() -> None:
    zero = _coefficient("0")
    one = _coefficient("1")
    source = _series(2, [zero, one])
    fabricated = _series(2, [zero, zero])

    parsed = SeriesReversionResult.model_validate_json(
        json.dumps(
            {
                "source": source,
                "result": fabricated,
                "left_residual": [zero, zero],
                "right_residual": [zero, zero],
            }
        )
    )
    assert parsed.result.coefficients[1].num == 0


def test_multiplication_result_rejects_structural_context_mismatch() -> None:
    series = TruncatedSeries.model_validate_json(
        json.dumps(_series(2, [_coefficient("1"), _coefficient("1")]))
    )
    payload = multiply(series, series).model_dump(mode="json")
    payload["result"]["variable"] = "y"

    with pytest.raises(ValidationError) as error:
        SeriesMultiplyResult.model_validate_json(json.dumps(payload))
    assert (
        error.value.errors()[0]["type"] == "formal_power_series.source_context_mismatch"
    )


def _zero_series_payload(order: int) -> dict[str, object]:
    return _series(order, [_coefficient("0") for _ in range(order)])


def test_all_zero_multiply_results_remain_representable_at_the_envelope_order() -> None:
    order = MAX_TRUNCATION_ORDER
    zeros = [_coefficient("0") for _ in range(order)]
    payload = {
        "left": _zero_series_payload(order),
        "right": _zero_series_payload(order),
        "result": _zero_series_payload(order),
        "convolution_ledger": zeros,
    }
    verdict = SeriesMultiplyResult.model_validate_json(json.dumps(payload))
    assert verdict.result.truncation_order == MAX_TRUNCATION_ORDER


@pytest.mark.parametrize("order, exponent, factorial_input", [(128, 9, True)])
def test_power_shared_denominator_envelope(
    order: int, exponent: int, factorial_input: bool
) -> None:
    from math import factorial

    from jacobian._exact import CanonicalRational

    coefficients = (
        [factorial(j) for j in range(order)]
        if factorial_input
        else [1] + [0] * (order - 1)
    )
    series = TruncatedSeries(
        variable="z",
        truncation_order=order,
        coefficients=tuple(CanonicalRational(num=c, den=1) for c in coefficients),
    )
    expected = [1] + [0] * (order - 1)
    for _ in range(exponent):
        expected = [
            sum(expected[i] * coefficients[k - i] for i in range(k + 1))
            for k in range(order)
        ]
    result = power(series, exponent)
    assert [c.as_fraction() for c in result.result.coefficients] == expected
