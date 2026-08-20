import pytest
from pydantic import ValidationError

from jacobian.math.formal_power_series._models import (
    InputTruncatedSeries,
    SeriesComposeRequest,
    SeriesDivideRequest,
    SeriesPowerRequest,
    SeriesReversionRequest,
    _SeriesAddSubtractRequest,
    _SeriesMultiplyRequest,
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
    with pytest.raises(ValidationError, match="multiplication coefficient growth"):
        _SeriesMultiplyRequest.model_validate(payload)


def test_power_propagates_binary_convolution_growth() -> None:
    order = 8
    coefficients = [_coefficient(den=str(3**500)) for _ in range(order)]
    with pytest.raises(ValidationError, match="power coefficient growth"):
        SeriesPowerRequest.model_validate(
            {"series": _series(order, coefficients), "exponent": 16}
        )


def test_division_propagates_inverse_and_residual_growth() -> None:
    order = 8
    numerator = [_coefficient() for _ in range(order)]
    denominator = [_coefficient(den=str(2**700)), *[_coefficient()] * (order - 1)]
    with pytest.raises(ValidationError, match="inverse coefficient growth"):
        SeriesDivideRequest.model_validate(
            {"left": _series(order, numerator), "right": _series(order, denominator)}
        )


def test_composition_propagates_inner_power_growth() -> None:
    order = 8
    outer = [_coefficient() for _ in range(order)]
    inner = [_coefficient("0"), *[_coefficient(den=str(5**300))] * (order - 1)]
    with pytest.raises(ValidationError, match="composition coefficient growth"):
        SeriesComposeRequest.model_validate(
            {"outer": _series(order, outer), "inner": _series(order, inner)}
        )


def test_reversion_propagates_linear_coefficient_division() -> None:
    order = 8
    coefficients = [
        _coefficient("0"),
        _coefficient(den=str(7**250)),
        *[_coefficient()] * (order - 2),
    ]
    with pytest.raises(ValidationError, match="reversion coefficient growth"):
        SeriesReversionRequest.model_validate(_series(order, coefficients))


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
