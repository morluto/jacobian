"""Affine composition uses the complete linear source prefix."""

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.polynomials.series import add, compose, power, truncate
from jacobian.math.polynomials.series._models import (
    MAX_RATIONAL_DIGITS,
    MAX_RESULT_RATIONAL_DIGITS,
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


def test_affine_composition_consumes_actual_grown_sum_unchanged() -> None:
    height = 10**MAX_RATIONAL_DIGITS - 1
    left = _series("x", [Fraction(0), Fraction(1, height), Fraction(1, height)])
    right = _series(
        "x", [Fraction(0), Fraction(1, height - 2), Fraction(1, height - 2)]
    )
    produced = add(left, right).result
    decoded = TruncatedSeries.model_validate_json(produced.model_dump_json())
    identity = _series("x", [Fraction(0), Fraction(1), Fraction(0)])
    assert compose(identity, decoded).result == produced


@pytest.mark.parametrize("constant", (False, True))
def test_affine_composition_reuses_actual_sparse_high_power(constant: bool) -> None:
    order = 800
    source = _series("x", [Fraction(0), Fraction(10000)] + [Fraction(0)] * (order - 2))
    produced = power(source, order - 1).result
    assert all(not value.num for value in produced.coefficients[:-1])
    assert produced.coefficients[-1].as_fraction() == 10000 ** (order - 1)
    decoded = TruncatedSeries.model_validate_json(produced.model_dump_json())
    values = [Fraction(1), Fraction(0)] if constant else [Fraction(0), Fraction(1)]
    outer = _series("x", values + [Fraction(0)] * (order - 2))
    result = compose(outer, decoded).result
    assert result == (outer if constant else produced)


def test_affine_full_height_sources_have_derived_output_admission() -> None:
    ceiling = 10**MAX_RESULT_RATIONAL_DIGITS
    value = Fraction(ceiling // 10)
    outer = _series("x", [Fraction(0), Fraction(2), Fraction(0)])
    inner = _series("x", [Fraction(0), value, value])
    result = compose(outer, inner).result
    assert [v.as_fraction() for v in result.coefficients] == [0, 2 * value, 2 * value]
    assert TruncatedSeries.model_validate_json(result.model_dump_json()) == result

    oversized = _series("x", [Fraction(0), Fraction(ceiling), Fraction(0)])
    with pytest.raises(OperationResourceAdmissionError) as source_error:
        compose(outer, oversized)
    assert (
        source_error.value.errors()[0]["type"]
        == "formal_power_series.compose_coefficient"
    )

    full = _series("x", [Fraction(0), Fraction(ceiling - 1), Fraction(0)])
    with pytest.raises(OperationResourceAdmissionError) as growth_error:
        compose(outer, full)
    assert (
        growth_error.value.errors()[0]["type"]
        == "formal_power_series.compose_coefficient_growth"
    )


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
