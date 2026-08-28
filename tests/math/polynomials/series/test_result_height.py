import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials.series._models import (
    MAX_RATIONAL_DIGITS,
    MAX_TRUNCATION_ORDER,
    InputTruncatedSeries,
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
from jacobian.math.polynomials.series._operations import (
    compute_compose,
    compute_divide,
    compute_multiply,
    compute_power,
    compute_reversion,
)


def _coefficient(num: str = "1", den: str = "1") -> dict[str, str]:
    return {"num": num, "den": den}


def _series(order: int, coefficients: list[dict[str, str]]) -> dict[str, object]:
    return {"variable": "x", "truncation_order": order, "coefficients": coefficients}


def test_multiplication_bound_does_not_reject_coefficientwise_addition() -> None:
    order = 20
    coefficients = [_coefficient(den=str(2**800)) for _ in range(order)]
    payload = {
        "left": _series(order, coefficients),
        "right": _series(order, coefficients),
    }

    assert _SeriesAddSubtractRequest.model_validate(payload)
    request = _SeriesMultiplyRequest.model_validate(payload)
    with pytest.raises(OperationDomainValidationError) as error:
        compute_multiply(request.left, request.right)
    assert (
        error.value.errors()[0]["type"]
        == "formal_power_series.multiplication_coefficient_growth"
    )


def test_power_propagates_binary_convolution_growth() -> None:
    order = 8
    coefficients = [_coefficient(den=str(3**500)) for _ in range(order)]
    request = SeriesPowerRequest.model_validate(
        {"series": _series(order, coefficients), "exponent": 16}
    )
    with pytest.raises(OperationDomainValidationError) as error:
        compute_power(request.series, request.exponent)
    assert (
        error.value.errors()[0]["type"]
        == "formal_power_series.power_coefficient_growth"
    )


def test_division_propagates_inverse_and_residual_growth() -> None:
    order = 8
    numerator = [_coefficient() for _ in range(order)]
    denominator = [_coefficient(den=str(2**700)), *[_coefficient()] * (order - 1)]
    request = SeriesDivideRequest.model_validate(
        {"left": _series(order, numerator), "right": _series(order, denominator)}
    )
    with pytest.raises(OperationDomainValidationError) as error:
        compute_divide(request.left, request.right)
    assert (
        error.value.errors()[0]["type"]
        == "formal_power_series.inverse_coefficient_growth"
    )


def test_composition_propagates_inner_power_growth() -> None:
    order = 8
    outer = [_coefficient() for _ in range(order)]
    inner = [_coefficient("0"), *[_coefficient(den=str(5**300))] * (order - 1)]
    request = SeriesComposeRequest.model_validate(
        {"outer": _series(order, outer), "inner": _series(order, inner)}
    )
    with pytest.raises(OperationDomainValidationError) as error:
        compute_compose(request.outer, request.inner)
    assert (
        error.value.errors()[0]["type"]
        == "formal_power_series.composition_coefficient_growth"
    )


def test_reversion_propagates_linear_coefficient_division() -> None:
    order = 8
    coefficients = [
        _coefficient("0"),
        _coefficient(den=str(7**250)),
        *[_coefficient()] * (order - 2),
    ]
    request = SeriesReversionRequest.model_validate(_series(order, coefficients))
    with pytest.raises(OperationDomainValidationError) as error:
        compute_reversion(request.as_series())
    assert (
        error.value.errors()[0]["type"]
        == "formal_power_series.reversion_coefficient_growth"
    )


def test_sparse_linear_reversion_remains_admitted() -> None:
    request = SeriesReversionRequest.model_validate(
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

    assert request.coefficients[1].num == "2"


def test_small_requests_remain_admitted() -> None:
    series = InputTruncatedSeries.model_validate(
        _series(2, [_coefficient("1"), _coefficient("1")])
    )
    assert SeriesPowerRequest(series=series, exponent=3)


def test_value_carrier_admits_compact_series_beyond_the_input_order_ceiling() -> None:
    order = MAX_TRUNCATION_ORDER + 88
    value = TruncatedSeries.model_validate(_series(order, [_coefficient("1")] * order))
    assert value.truncation_order == order


def test_operation_inputs_keep_the_shared_order_ceiling() -> None:
    series = InputTruncatedSeries.model_validate(
        _series(
            MAX_TRUNCATION_ORDER + 1,
            [_coefficient("1")] * (MAX_TRUNCATION_ORDER + 1),
        )
    )
    with pytest.raises(OperationDomainValidationError):
        compute_power(series, 1)


def test_largest_multiplication_result_fits_shared_output_envelope() -> None:
    numerator = "9" * MAX_RATIONAL_DIGITS
    denominator = "1" + "0" * (MAX_RATIONAL_DIGITS - 1)
    coefficient = _coefficient(numerator, denominator)
    series = InputTruncatedSeries.model_validate(
        _series(MAX_TRUNCATION_ORDER, [coefficient] * MAX_TRUNCATION_ORDER)
    )
    with pytest.raises(OperationDomainValidationError):
        compute_multiply(series, series)


def test_result_round_trips_remain_structural() -> None:
    zero = _coefficient("0")
    one = _coefficient("1")
    source = _series(2, [zero, one])
    fabricated = _series(2, [zero, zero])

    parsed = SeriesReversionResult.model_validate(
        {
            "source": source,
            "result": fabricated,
            "left_residual": [zero, zero],
            "right_residual": [zero, zero],
        }
    )
    assert parsed.result.coefficients[1].num == "0"


def test_multiplication_result_rejects_structural_context_mismatch() -> None:
    series = InputTruncatedSeries.model_validate(
        _series(2, [_coefficient("1"), _coefficient("1")])
    )
    payload = compute_multiply(series, series).model_dump(mode="json")
    payload["result"]["variable"] = "y"

    with pytest.raises(ValidationError) as error:
        SeriesMultiplyResult.model_validate(payload)
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
    verdict = SeriesMultiplyResult.model_validate(payload)
    assert verdict.result.truncation_order == MAX_TRUNCATION_ORDER
