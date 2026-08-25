import pytest
from pydantic import ValidationError

from jacobian.canonical import encode_strict_json
from jacobian.math.formal_power_series._models import (
    MAX_RATIONAL_DIGITS,
    MAX_TRUNCATION_ORDER,
    InputTruncatedSeries,
    SeriesComposeRequest,
    SeriesDivideRequest,
    SeriesDivideResult,
    SeriesInverseResult,
    SeriesMultiplyResult,
    SeriesPowerRequest,
    SeriesReversionRequest,
    SeriesReversionResult,
    TruncatedSeries,
    _SeriesAddSubtractRequest,
    _SeriesMultiplyRequest,
)
from jacobian.math.formal_power_series._operations import (
    compute_divide,
    compute_inverse,
    compute_multiply,
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
    with pytest.raises(ValidationError) as error:
        _SeriesMultiplyRequest.model_validate(payload)
    assert (
        error.value.errors()[0]["type"]
        == "formal_power_series.multiplication_coefficient_growth"
    )


def test_power_propagates_binary_convolution_growth() -> None:
    order = 8
    coefficients = [_coefficient(den=str(3**500)) for _ in range(order)]
    with pytest.raises(ValidationError) as error:
        SeriesPowerRequest.model_validate(
            {"series": _series(order, coefficients), "exponent": 16}
        )
    assert (
        error.value.errors()[0]["type"]
        == "formal_power_series.power_coefficient_growth"
    )


def test_division_propagates_inverse_and_residual_growth() -> None:
    order = 8
    numerator = [_coefficient() for _ in range(order)]
    denominator = [_coefficient(den=str(2**700)), *[_coefficient()] * (order - 1)]
    with pytest.raises(ValidationError) as error:
        SeriesDivideRequest.model_validate(
            {"left": _series(order, numerator), "right": _series(order, denominator)}
        )
    assert (
        error.value.errors()[0]["type"]
        == "formal_power_series.inverse_coefficient_growth"
    )


def test_composition_propagates_inner_power_growth() -> None:
    order = 8
    outer = [_coefficient() for _ in range(order)]
    inner = [_coefficient("0"), *[_coefficient(den=str(5**300))] * (order - 1)]
    with pytest.raises(ValidationError) as error:
        SeriesComposeRequest.model_validate(
            {"outer": _series(order, outer), "inner": _series(order, inner)}
        )
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
    with pytest.raises(ValidationError) as error:
        SeriesReversionRequest.model_validate(_series(order, coefficients))
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
    with pytest.raises(ValidationError):
        InputTruncatedSeries.model_validate(
            _series(MAX_TRUNCATION_ORDER + 1, [_coefficient("1")])
        )


def test_largest_multiplication_result_fits_shared_output_envelope() -> None:
    numerator = "9" * MAX_RATIONAL_DIGITS
    denominator = "1" + "0" * (MAX_RATIONAL_DIGITS - 1)
    coefficient = _coefficient(numerator, denominator)
    series = InputTruncatedSeries.model_validate(
        _series(MAX_TRUNCATION_ORDER, [coefficient] * MAX_TRUNCATION_ORDER)
    )
    result = compute_multiply(series, series)

    assert encode_strict_json(result.model_dump(mode="json"))


def test_reversion_result_rejects_fabricated_conclusion_with_zero_residuals() -> None:
    zero = _coefficient("0")
    one = _coefficient("1")
    source = _series(2, [zero, one])
    fabricated = _series(2, [zero, zero])

    with pytest.raises(ValidationError) as error:
        SeriesReversionResult.model_validate(
            {
                "source": source,
                "result": fabricated,
                "left_residual": [zero, zero],
                "right_residual": [zero, zero],
            }
        )
    assert (
        error.value.errors()[0]["type"]
        == "formal_power_series.reversion_left_residual_mismatch"
    )


def test_multiplication_result_rejects_fabricated_conclusion_and_ledger() -> None:
    series = InputTruncatedSeries.model_validate(
        _series(2, [_coefficient("1"), _coefficient("1")])
    )
    payload = compute_multiply(series, series).model_dump(mode="json")
    fabricated = [_coefficient("0"), _coefficient("0")]
    payload["result"]["coefficients"] = fabricated
    payload["convolution_ledger"] = fabricated

    with pytest.raises(ValidationError) as error:
        SeriesMultiplyResult.model_validate(payload)
    assert (
        error.value.errors()[0]["type"]
        == "formal_power_series.convolution_result_mismatch"
    )


def test_inverse_result_rejects_fabricated_conclusion_with_zero_residual() -> None:
    series = InputTruncatedSeries.model_validate(
        _series(2, [_coefficient("1"), _coefficient("1")])
    )
    payload = compute_inverse(series).model_dump(mode="json")
    payload["result"]["coefficients"] = [_coefficient("1"), _coefficient("0")]

    with pytest.raises(ValidationError) as error:
        SeriesInverseResult.model_validate(payload)
    assert (
        error.value.errors()[0]["type"]
        == "formal_power_series.inverse_residual_mismatch"
    )


def test_division_result_rejects_fabricated_conclusion_with_zero_residual() -> None:
    numerator = InputTruncatedSeries.model_validate(
        _series(2, [_coefficient("1"), _coefficient("0")])
    )
    denominator = InputTruncatedSeries.model_validate(
        _series(2, [_coefficient("1"), _coefficient("1")])
    )
    payload = compute_divide(numerator, denominator).model_dump(mode="json")
    payload["quotient"]["coefficients"] = [_coefficient("1"), _coefficient("0")]

    with pytest.raises(ValidationError) as error:
        SeriesDivideResult.model_validate(payload)
    assert (
        error.value.errors()[0]["type"]
        == "formal_power_series.division_residual_mismatch"
    )


def _zero_series_payload(order: int) -> dict[str, object]:
    return _series(order, [_coefficient("0") for _ in range(order)])


def test_replaying_results_reject_orders_above_the_producer_envelope() -> None:
    order = MAX_TRUNCATION_ORDER + 1
    zeros = [_coefficient("0") for _ in range(order)]
    multiply_payload = {
        "left": _zero_series_payload(order),
        "right": _zero_series_payload(order),
        "result": _zero_series_payload(order),
        "convolution_ledger": zeros,
    }
    with pytest.raises(ValidationError) as error:
        SeriesMultiplyResult.model_validate(multiply_payload)
    assert (
        error.value.errors()[0]["type"]
        == "formal_power_series.multiplication_replay_order"
    )

    reversion_residuals = tuple(_coefficient("0") for _ in range(order))
    with pytest.raises(ValidationError) as error:
        SeriesReversionResult.model_validate(
            {
                "source": _zero_series_payload(order),
                "result": _zero_series_payload(order),
                "left_residual": reversion_residuals,
                "right_residual": reversion_residuals,
            }
        )
    assert (
        error.value.errors()[0]["type"] == "formal_power_series.reversion_replay_order"
    )


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
