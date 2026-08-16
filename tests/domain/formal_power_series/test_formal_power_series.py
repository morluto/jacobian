"""Tests for truncated formal power series operations."""

from jacobian.contracts.formal_power_series import (
    MAX_RATIONAL_DIGITS,
    InputTruncatedSeries,
    TruncatedSeries,
)
from jacobian.domains.formal_power_series.operations import (
    compute_derivative,
    compute_multiply,
)


def _coeff(num: str, den: str = "1") -> dict[str, str]:
    return {"num": num, "den": den}


def test_derivative_of_order_one_is_zero() -> None:
    series = TruncatedSeries(
        variable="x",
        truncation_order=1,
        coefficients=(_coeff("7"),),
    )
    result = compute_derivative(series)
    assert result.result.truncation_order == 1
    assert result.result.coefficients[0].as_fraction() == 0


def test_input_series_rejects_oversized_coefficients() -> None:
    import pytest
    from pydantic import ValidationError

    huge = "1" + "0" * MAX_RATIONAL_DIGITS
    with pytest.raises(ValidationError, match="input coefficient"):
        InputTruncatedSeries(
            variable="x",
            truncation_order=1,
            coefficients=(_coeff(huge),),
        )


def test_product_can_exceed_input_digit_bound() -> None:
    large = "1" + "0" * (MAX_RATIONAL_DIGITS - 1)
    left = TruncatedSeries(
        variable="x",
        truncation_order=1,
        coefficients=(_coeff(large),),
    )
    right = TruncatedSeries(
        variable="x",
        truncation_order=1,
        coefficients=(_coeff(large),),
    )
    result = compute_multiply(left, right)
    value = result.result.coefficients[0]
    assert len(value.num.lstrip("-")) > MAX_RATIONAL_DIGITS
    assert len(value.num.lstrip("-")) <= 4096
