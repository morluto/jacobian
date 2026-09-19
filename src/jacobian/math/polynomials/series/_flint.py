"""Private FLINT kernels for exact truncated rational series.

``python-flint`` exposes ``fmpq_series`` values with an explicit precision,
but its arithmetic methods also consult the process-global ``ctx.cap``.  The
series operations must not mutate that ambient setting: instead this adapter
uses the per-object series as the typed boundary and FLINT's exact
``fmpq_poly`` arithmetic for the bounded Newton and composition kernels.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from jacobian._execution import BackendFailureReason, OperationBackendError


def _backend_fraction(value: Any) -> Fraction:
    """Decode one FLINT rational without passing backend values onward."""

    numerator = getattr(value, "p", getattr(value, "numerator", None))
    denominator = getattr(value, "q", getattr(value, "denominator", None))
    if numerator is None or denominator is None:
        raise TypeError("FLINT returned a non-rational coefficient")
    denominator_int = int(denominator)
    if denominator_int == 0:
        raise ValueError("FLINT returned a zero rational denominator")
    return Fraction(int(numerator), denominator_int)


def _series_from_fractions(coefficients: tuple[Fraction, ...], order: int) -> Any:
    """Construct an explicitly precise ``fmpq_series`` input object."""

    from flint import fmpq, fmpq_series

    result = fmpq_series(
        [fmpq(value.numerator, value.denominator) for value in coefficients],
        prec=order,
    )
    if int(result.prec) != order:
        raise ValueError("FLINT did not retain the requested series precision")
    return result


def _poly_from_series(series: Any, order: int) -> Any:
    from flint import fmpq_poly

    return fmpq_poly([series[index] for index in range(order)])


def _fractions_from_series(series: Any, order: int) -> list[Fraction]:
    if int(series.prec) != order:
        raise ValueError("FLINT returned a series at the wrong precision")
    return [_backend_fraction(series[index]) for index in range(order)]


def _poly_inverse(source: Any, order: int) -> Any:
    """Invert ``source`` modulo ``x**order`` by exact FLINT Newton steps."""

    from flint import fmpq_poly

    inverse = fmpq_poly([1 / source[0]])
    precision = 1
    two = fmpq_poly([2])
    while precision < order:
        next_precision = min(2 * precision, order)
        truncated = source.truncate(next_precision)
        inverse = (inverse * (two - truncated * inverse)).truncate(next_precision)
        precision = next_precision
    return inverse.truncate(order)


def _poly_compose(outer: Any, inner: Any, order: int) -> Any:
    """Compose two FLINT polynomials and retain only ``order`` coefficients."""

    from flint import fmpq_poly

    result = fmpq_poly(0)
    start = min(max(int(outer.degree()), -1), order - 1)
    for index in range(start, -1, -1):
        result = (result * inner + fmpq_poly([outer[index]])).truncate(order)
    return result


def _poly_reversion(source: Any, order: int) -> Any:
    """Compute a compositional inverse by exact Newton iteration."""

    from flint import fmpq_poly

    # The linear approximation is x / f_1, not the constant reciprocal.
    inverse = fmpq_poly([0, 1 / source[1]])
    precision = 1
    target = fmpq_poly([0, 1])
    while precision < order:
        next_precision = min(2 * precision, order)
        truncated = source.truncate(next_precision)
        error = (_poly_compose(truncated, inverse, next_precision) - target).truncate(
            next_precision
        )
        derivative = _poly_compose(
            source.derivative().truncate(next_precision), inverse, next_precision
        )
        inverse = (
            inverse - error * _poly_inverse(derivative, next_precision)
        ).truncate(next_precision)
        precision = next_precision
    return inverse.truncate(order)


def _poly_coefficients(poly: Any, order: int) -> list[Fraction]:
    return [_backend_fraction(poly[index]) for index in range(order)]


def _inverse_backend(
    coefficients: tuple[Fraction, ...],
) -> tuple[list[Fraction], list[Fraction]]:
    """Return inverse coefficients and its FLINT-computed residual."""

    order = len(coefficients)
    source_series = _series_from_fractions(coefficients, order)
    source = _poly_from_series(source_series, order)
    result = _poly_inverse(source, order)
    residual = (
        source * result
        - _poly_from_series(_series_from_fractions((Fraction(1),), order), order)
    ).truncate(order)
    result_series = _series_from_fractions(
        tuple(_poly_coefficients(result, order)), order
    )
    result_coefficients = _fractions_from_series(result_series, order)
    residual_coefficients = _poly_coefficients(residual, order)
    if any(residual_coefficients):
        raise ValueError("FLINT inverse residual is nonzero")
    return result_coefficients, residual_coefficients


def _reversion_backend(
    coefficients: tuple[Fraction, ...],
) -> tuple[list[Fraction], list[Fraction], list[Fraction]]:
    """Return reversion coefficients and both FLINT-computed residuals."""

    order = len(coefficients)
    source_series = _series_from_fractions(coefficients, order)
    source = _poly_from_series(source_series, order)
    result = _poly_reversion(source, order)
    target = _poly_from_series(
        _series_from_fractions((Fraction(0), Fraction(1)), order), order
    )
    left = (_poly_compose(source, result, order) - target).truncate(order)
    right = (_poly_compose(result, source, order) - target).truncate(order)
    result_series = _series_from_fractions(
        tuple(_poly_coefficients(result, order)), order
    )
    result_coefficients = _fractions_from_series(result_series, order)
    left_coefficients = _poly_coefficients(left, order)
    right_coefficients = _poly_coefficients(right, order)
    if any(left_coefficients) or any(right_coefficients):
        raise ValueError("FLINT reversion residual is nonzero")
    return result_coefficients, left_coefficients, right_coefficients


def inverse_backend(
    coefficients: tuple[Fraction, ...],
) -> tuple[list[Fraction], list[Fraction]]:
    """Run the exact inverse adapter with typed operational failures."""

    try:
        return _inverse_backend(coefficients)
    except OperationBackendError:
        raise
    except (ImportError, ModuleNotFoundError, AttributeError) as exc:
        raise OperationBackendError(BackendFailureReason.INITIALIZATION) from exc
    except Exception as exc:
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT) from exc


def reversion_backend(
    coefficients: tuple[Fraction, ...],
) -> tuple[list[Fraction], list[Fraction], list[Fraction]]:
    """Run the exact reversion adapter with typed operational failures."""

    try:
        return _reversion_backend(coefficients)
    except OperationBackendError:
        raise
    except (ImportError, ModuleNotFoundError, AttributeError) as exc:
        raise OperationBackendError(BackendFailureReason.INITIALIZATION) from exc
    except Exception as exc:
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT) from exc


__all__ = ["inverse_backend", "reversion_backend"]
