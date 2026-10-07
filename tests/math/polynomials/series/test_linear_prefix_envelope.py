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
