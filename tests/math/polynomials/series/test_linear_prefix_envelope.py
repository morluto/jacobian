"""Complete linear prefixes use coefficientwise known answers."""

from fractions import Fraction
from threading import Event

import pytest

from jacobian._exact import CanonicalRational as Q
from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    request_cancellation,
)
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.polynomials.series import (
    add,
    derivative,
    inverse,
    operations,
    subtract,
    truncate,
)
from jacobian.math.polynomials.series._models import (
    MAX_RATIONAL_DIGITS,
    MAX_RESULT_RATIONAL_DIGITS,
    MAX_TRUNCATE_SOURCE_ORDER,
    TruncatedSeries,
)


def _series(values: list[Fraction]) -> TruncatedSeries:
    return TruncatedSeries(
        variable="t_2",
        truncation_order=len(values),
        coefficients=tuple(Q.from_fraction(value) for value in values),
    )


@pytest.mark.parametrize("order", (513, MAX_TRUNCATE_SOURCE_ORDER))
def test_full_height_linear_prefixes_have_exact_independent_coefficients(
    order: int,
) -> None:
    p = 10**MAX_RATIONAL_DIGITS - 1
    a, b = Fraction(p, p - 2), Fraction(p - 4, p - 6)
    left, right = _series([a] * order), _series([b] * order)
    for operation, expected in ((add, a + b), (subtract, a - b)):
        result = operation(left, right).result
        assert all(value.as_fraction() == expected for value in result.coefficients)
        decoded = TruncatedSeries.model_validate_json(result.model_dump_json())
        assert truncate(decoded, 512).result.coefficients == result.coefficients[:512]
        assert result.variable == "t_2"
    result = derivative(left).result
    assert result.truncation_order == order - 1
    assert [value.as_fraction() for value in result.coefficients] == [
        k * a for k in range(1, order)
    ]
    assert TruncatedSeries.model_validate_json(result.model_dump_json()) == result


def test_derivative_consumes_actual_admitted_inverse_without_rewriting() -> None:
    source = _series([Fraction(1), Fraction(-1)] + [Fraction(0)] * 2046)
    produced = inverse(source).result
    assert all(value.as_fraction() == 1 for value in produced.coefficients)
    decoded = TruncatedSeries.model_validate_json(produced.model_dump_json())
    result = derivative(decoded).result
    assert [value.as_fraction() for value in result.coefficients] == list(
        range(1, 2048)
    )
    zero = _series([Fraction(0)] * 2047)
    assert add(result, zero).result == result


def test_linear_consumers_reuse_grown_inverse_and_sum_coefficients() -> None:
    source_height = 10**MAX_RATIONAL_DIGITS - 1
    source = _series([Fraction(1), Fraction(-source_height), Fraction(0)])
    produced = inverse(source).result
    decoded = TruncatedSeries.model_validate_json(produced.model_dump_json())
    assert decoded.coefficients[2].as_fraction() == source_height**2
    differentiated = derivative(decoded).result
    assert [value.as_fraction() for value in differentiated.coefficients] == [
        source_height,
        2 * source_height**2,
    ]
    zero = _series([Fraction(0)] * 2)
    assert add(differentiated, zero).result == differentiated
    assert subtract(differentiated, zero).result == differentiated

    left = _series([Fraction(1, source_height)] * 2)
    right = _series([Fraction(1, source_height - 2)] * 2)
    summed = add(left, right).result
    decoded_sum = TruncatedSeries.model_validate_json(summed.model_dump_json())
    assert derivative(decoded_sum).result.coefficients[0] == summed.coefficients[1]


def test_full_result_height_zero_and_shared_denominator_arithmetic() -> None:
    height = 10 ** (MAX_RESULT_RATIONAL_DIGITS - 1)
    source = _series([Fraction(0), Fraction(height), Fraction(height)])
    result = derivative(source).result
    assert [value.as_fraction() for value in result.coefficients] == [
        height,
        2 * height,
    ]
    zero = _series([Fraction(0)] * 2)
    assert add(result, zero).result == result
    assert subtract(result, result).result == zero
    assert add(result, result).result.coefficients[1].as_fraction() == 4 * height


def test_source_height_and_derived_result_growth_have_distinct_refusals() -> None:
    limit = 10**MAX_RESULT_RATIONAL_DIGITS
    with pytest.raises(OperationResourceAdmissionError) as source_error:
        derivative(_series([Fraction(0), Fraction(limit)]))
    assert (
        source_error.value.errors()[0]["type"]
        == "formal_power_series.input_coefficient"
    )

    source = _series([Fraction(0), Fraction(0), Fraction(limit - 1)])
    with pytest.raises(OperationResourceAdmissionError) as derivative_error:
        derivative(source)
    assert derivative_error.value.errors()[0]["type"] == (
        "formal_power_series.derivative_coefficient_growth"
    )
    source = _series([Fraction(limit - 1)])
    with pytest.raises(OperationResourceAdmissionError) as addition_error:
        add(source, source)
    assert (
        addition_error.value.errors()[0]["type"]
        == "formal_power_series.sum_coefficient_growth"
    )


def test_linear_source_ceiling_remains_bounded() -> None:
    source = _series([Fraction(0)] * (MAX_TRUNCATE_SOURCE_ORDER + 1))
    for call in (lambda: add(source, source), lambda: derivative(source)):
        with pytest.raises(OperationResourceAdmissionError):
            call()


@pytest.mark.parametrize("operation", (add, subtract, derivative))
def test_native_linear_kernel_enforces_deadline_without_dispatch(
    monkeypatch: pytest.MonkeyPatch, operation: object
) -> None:
    now = [0.0]
    monkeypatch.setattr(operations, "monotonic", lambda: now[0])
    monkeypatch.setattr("jacobian._execution.time.monotonic", lambda: now[0])
    convert = operations._series_fractions

    def expired(series: TruncatedSeries) -> list[Fraction]:
        values = convert(series)
        now[0] = 61.0
        return values

    monkeypatch.setattr(operations, "_series_fractions", expired)
    source = _series([Fraction(1)] * 513)
    with pytest.raises(OperationExecutionTimeoutError):
        if operation is derivative:
            derivative(source)
        elif operation is add:
            add(source, source)
        else:
            subtract(source, source)


def test_native_linear_arithmetic_observes_real_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signal = Event()
    convert = operations._series_fractions

    def cancelled(series: TruncatedSeries) -> list[Fraction]:
        values = convert(series)
        signal.set()
        return values

    monkeypatch.setattr(operations, "_series_fractions", cancelled)
    source = _series([Fraction(1)] * 513)
    with request_cancellation(signal), pytest.raises(OperationExecutionCancelledError):
        add(source, source)
