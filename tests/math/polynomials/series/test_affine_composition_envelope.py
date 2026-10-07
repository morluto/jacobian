"""Affine composition uses the complete linear source prefix."""

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.polynomials.series import compose, truncate
from jacobian.math.polynomials.series._models import (
    MAX_RATIONAL_DIGITS,
    MAX_TRUNCATE_SOURCE_ORDER,
    MAX_TRUNCATION_ORDER,
    TruncatedSeries,
)


def _series(variable: str, values: list[Fraction]) -> TruncatedSeries:
    return TruncatedSeries(
        variable=variable,
        truncation_order=len(values),
        coefficients=tuple(CanonicalRational.from_fraction(value) for value in values),
    )


@pytest.mark.parametrize("order", (MAX_TRUNCATION_ORDER + 1, MAX_TRUNCATE_SOURCE_ORDER))
@pytest.mark.parametrize("slope", (Fraction(-5, 7), Fraction(0)))
def test_affine_composition_preserves_dense_exact_coefficients_and_handoff(
    order: int, slope: Fraction
) -> None:
    outer = _series("Y_2", [Fraction(2, 3), slope] + [Fraction(0)] * (order - 2))
    values = [Fraction(0)] + [Fraction((i % 5) - 2, 7) for i in range(1, order)]
    inner = _series("Y_2", values)
    result = compose(outer, inner).result
    expected = [Fraction(2, 3)] + [slope * value for value in values[1:]]
    assert [value.as_fraction() for value in result.coefficients] == expected
    assert result.variable == "Y_2"
    assert result.truncation_order == order
    decoded = TruncatedSeries.model_validate_json(result.model_dump_json())
    prefix = truncate(decoded, MAX_TRUNCATION_ORDER).result
    assert [value.as_fraction() for value in prefix.coefficients] == expected[
        :MAX_TRUNCATION_ORDER
    ]


def test_affine_composition_admits_full_source_coefficient_height() -> None:
    order = MAX_TRUNCATE_SOURCE_ORDER
    numerator = 10**MAX_RATIONAL_DIGITS - 1
    slope = Fraction(numerator, numerator - 2)
    inner_value = Fraction(numerator - 4, numerator - 6)
    outer = _series("x", [Fraction(-2, 3), slope] + [Fraction(0)] * (order - 2))
    inner = _series("x", [Fraction(0)] + [inner_value] * (order - 1))
    result = compose(outer, inner).result
    assert result.coefficients[0].as_fraction() == Fraction(-2, 3)
    assert all(
        value.as_fraction() == slope * inner_value for value in result.coefficients[1:]
    )
    assert TruncatedSeries.model_validate_json(result.model_dump_json()) == result


def test_nonlinear_composition_keeps_its_existing_order_boundary() -> None:
    order = MAX_TRUNCATION_ORDER + 1
    outer = _series(
        "x", [Fraction(0), Fraction(0), Fraction(1)] + [Fraction(0)] * (order - 3)
    )
    inner = _series("x", [Fraction(0), Fraction(1)] + [Fraction(0)] * (order - 2))
    with pytest.raises(OperationResourceAdmissionError) as error:
        compose(outer, inner)
    assert error.value.errors()[0]["type"] == "formal_power_series.input_order"


def test_native_affine_kernel_enforces_its_clock_without_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian import _execution
    from jacobian._execution import OperationExecutionTimeoutError
    from jacobian.math.polynomials.series import operations

    now = [0.0]
    monkeypatch.setattr(operations, "monotonic", lambda: now[0])
    monkeypatch.setattr("jacobian._execution.time.monotonic", lambda: now[0])
    kernel = operations._compose_coefficients

    def expired(outer: TruncatedSeries, inner: TruncatedSeries) -> list[Fraction]:
        now[0] = 61.0
        return kernel(outer, inner)

    monkeypatch.setattr(operations, "_compose_coefficients", expired)
    series = _series("x", [Fraction(0), Fraction(1)] + [Fraction(0)] * 511)
    with pytest.raises(OperationExecutionTimeoutError):
        compose(series, series)
    assert _execution.current_request_execution() is None


def test_native_affine_kernel_observes_cancellation_without_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from threading import Event

    from jacobian._execution import (
        OperationExecutionCancelledError,
        request_cancellation,
    )
    from jacobian.math.polynomials.series import operations

    signal = Event()
    kernel = operations._compose_coefficients

    def cancelled(outer: TruncatedSeries, inner: TruncatedSeries) -> list[Fraction]:
        signal.set()
        return kernel(outer, inner)

    monkeypatch.setattr(operations, "_compose_coefficients", cancelled)
    series = _series("x", [Fraction(0), Fraction(1)] + [Fraction(0)] * 511)
    with request_cancellation(signal), pytest.raises(OperationExecutionCancelledError):
        compose(series, series)
