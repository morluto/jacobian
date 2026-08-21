"""Tests for truncated formal power series operations."""

from jacobian.math.formal_power_series import derivative, multiply
from jacobian.math.formal_power_series._models import (
    MAX_RATIONAL_DIGITS,
    InputTruncatedSeries,
    SeriesInverseRequest,
    TruncatedSeries,
)
from jacobian.math.formal_power_series._operations import (
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


def test_native_projection_aliases_call_the_shared_typed_kernels() -> None:
    series = TruncatedSeries(
        variable="x",
        truncation_order=2,
        coefficients=(_coeff("1"), _coeff("2")),
    )

    assert derivative(series) == compute_derivative(series)
    assert multiply(series, series) == compute_multiply(series, series)


def test_power_rejects_result_digit_overflow() -> None:
    import pytest
    from pydantic import ValidationError

    from jacobian.math.formal_power_series._models import SeriesPowerRequest

    huge = "1" + "0" * (MAX_RATIONAL_DIGITS - 1)
    with pytest.raises(ValidationError, match="4096-digit"):
        SeriesPowerRequest(
            series=InputTruncatedSeries(
                variable="x",
                truncation_order=1,
                coefficients=(_coeff(huge),),
            ),
            exponent=1000,
        )


def test_reversion_rejects_nonzero_constant() -> None:
    import pytest
    from pydantic import ValidationError

    from jacobian.math.formal_power_series._models import SeriesReversionRequest

    with pytest.raises(ValidationError, match="zero constant"):
        SeriesReversionRequest(
            variable="x",
            truncation_order=2,
            coefficients=(_coeff("1"), _coeff("1")),
        )


def test_integral_rejects_oversized_output_order() -> None:
    import pytest
    from pydantic import ValidationError

    from jacobian.math.formal_power_series._models import SeriesIntegralRequest

    with pytest.raises(ValidationError, match="source_order"):
        SeriesIntegralRequest(
            series=InputTruncatedSeries(
                variable="x",
                truncation_order=2,
                coefficients=(_coeff("1"), _coeff("0")),
            ),
            output_order=4,
        )


def test_inverse_rejects_zero_constant() -> None:
    import pytest
    from pydantic import ValidationError

    from jacobian.math.formal_power_series._models import SeriesInverseRequest

    with pytest.raises(ValidationError, match="nonzero constant"):
        SeriesInverseRequest(
            variable="x",
            truncation_order=2,
            coefficients=(_coeff("0"), _coeff("1")),
        )


def test_inverse_rejects_result_coefficient_growth() -> None:
    import pytest
    from pydantic import ValidationError

    huge = "1" + "0" * (MAX_RATIONAL_DIGITS - 1)
    with pytest.raises(ValidationError, match="inverse coefficient growth"):
        SeriesInverseRequest(
            variable="x",
            truncation_order=20,
            coefficients=(
                _coeff("1"),
                _coeff("-" + huge),
                *(_coeff("0") for _ in range(18)),
            ),
        )


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
