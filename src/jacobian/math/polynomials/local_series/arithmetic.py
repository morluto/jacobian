"""Exact bounded arithmetic on Laurent windows."""

from __future__ import annotations

from fractions import Fraction
from math import gcd

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)

from .arithmetic_models import (
    LaurentDeramifyResult,
    LaurentIntegralResult,
    LaurentPrincipalPartResult,
    LaurentResidueResult,
)
from .values import (
    MAX_LOCAL_SERIES_COEFFICIENT_DIGITS,
    MAX_LOCAL_SERIES_EXPONENT,
    MAX_LOCAL_SERIES_TERMS,
    TruncatedLaurentWindow,
)


def _fraction(value: CanonicalRational) -> Fraction:
    return value.as_fraction()


def _is_canonical_rational(value: CanonicalRational) -> bool:
    return (
        type(value.num) is int
        and type(value.den) is int
        and value.den > 0
        and gcd(abs(value.num), value.den) == 1
        and (value.num != 0 or value.den == 1)
    )


def _cr(value: Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(value)


def _check(s: TruncatedLaurentWindow) -> None:
    if not isinstance(s, TruncatedLaurentWindow):
        raise OperationDomainValidationError(
            location=("series",),
            code="local_series.series_type",
            message="series must be a Laurent window",
        )
    try:
        valid = (
            type(s.valuation_lower) is int
            and type(s.precision) is int
            and s.valuation_lower < s.precision
            and isinstance(s.coefficients, tuple)
            and len(s.coefficients) == s.precision - s.valuation_lower
            and _is_canonical_rational(s.center)
            and all(isinstance(value, CanonicalRational) and _is_canonical_rational(value) for value in s.coefficients)
        )
        if valid:
            require_bounded_rational(s.center, max_digits=MAX_LOCAL_SERIES_COEFFICIENT_DIGITS, label="Laurent center")
            for value in s.coefficients:
                require_bounded_rational(value, max_digits=MAX_LOCAL_SERIES_COEFFICIENT_DIGITS, label="Laurent coefficient")
    except (AttributeError, TypeError, ValueError):
        valid = False
    if not valid:
        raise OperationDomainValidationError(location=("series",), code="local_series.series_structure", message="series must expose a canonical dense exponent window")
    if len(s.coefficients) > MAX_LOCAL_SERIES_TERMS:
        raise OperationResourceAdmissionError(
            location=("series",),
            code="local_series.terms",
            message="Laurent arithmetic exceeds the retained-term envelope",
        )


def _pair(left: TruncatedLaurentWindow, right: TruncatedLaurentWindow) -> None:
    _check(left)
    _check(right)
    if (left.variable, left.center) != (right.variable, right.center):
        raise OperationDomainValidationError(
            location=("series",),
            code="local_series.parent_mismatch",
            message="Laurent windows must share variable and center",
        )


def _window(
    source: TruncatedLaurentWindow, lo: int, hi: int, values: dict[int, Fraction]
) -> TruncatedLaurentWindow:
    if (
        hi <= lo
        or hi - lo > MAX_LOCAL_SERIES_TERMS
        or max(abs(lo), abs(hi)) > MAX_LOCAL_SERIES_EXPONENT
    ):
        raise OperationResourceAdmissionError(
            location=("precision",),
            code="local_series.result_bound",
            message="Laurent result exceeds its admitted exponent or term envelope",
        )
    return TruncatedLaurentWindow(
        variable=source.variable,
        center=source.center,
        valuation_lower=lo,
        precision=hi,
        coefficients=tuple(_cr(values.get(k, Fraction(0))) for k in range(lo, hi)),
    )


def _coeff(s: TruncatedLaurentWindow, exponent: int) -> Fraction:
    if s.valuation_lower <= exponent < s.precision:
        return _fraction(s.coefficients[exponent - s.valuation_lower])
    return Fraction(0)


def _nonzero(s: TruncatedLaurentWindow) -> int | None:
    for i, c in enumerate(s.coefficients):
        if _fraction(c):
            return s.valuation_lower + i
    return None


def add(
    left: TruncatedLaurentWindow, right: TruncatedLaurentWindow
) -> TruncatedLaurentWindow:
    _pair(left, right)
    if (left.valuation_lower, left.precision) != (
        right.valuation_lower,
        right.precision,
    ):
        raise OperationDomainValidationError(
            location=("series",),
            code="local_series.window_mismatch",
            message="addition requires identical known exponent windows",
        )
    return _window(
        left,
        left.valuation_lower,
        left.precision,
        {
            k: _coeff(left, k) + _coeff(right, k)
            for k in range(left.valuation_lower, left.precision)
        },
    )


def subtract(
    left: TruncatedLaurentWindow, right: TruncatedLaurentWindow
) -> TruncatedLaurentWindow:
    _pair(left, right)
    if (left.valuation_lower, left.precision) != (
        right.valuation_lower,
        right.precision,
    ):
        raise OperationDomainValidationError(
            location=("series",),
            code="local_series.window_mismatch",
            message="subtraction requires identical known exponent windows",
        )
    return _window(
        left,
        left.valuation_lower,
        left.precision,
        {
            k: _coeff(left, k) - _coeff(right, k)
            for k in range(left.valuation_lower, left.precision)
        },
    )


def multiply(
    left: TruncatedLaurentWindow,
    right: TruncatedLaurentWindow,
    output_precision: int | None = None,
) -> TruncatedLaurentWindow:
    _pair(left, right)
    lo = left.valuation_lower + right.valuation_lower
    safe = min(
        left.precision + right.valuation_lower, right.precision + left.valuation_lower
    )
    hi = safe if output_precision is None else output_precision
    if type(hi) is not int or hi > safe or hi <= lo:
        raise OperationDomainValidationError(
            location=("output_precision",),
            code="local_series.precision_not_supported",
            message=f"output precision must satisfy {lo} < P <= {safe}",
        )
    values = {
        k: sum(
            (
                _coeff(left, i) * _coeff(right, k - i)
                for i in range(left.valuation_lower, left.precision)
                if right.valuation_lower <= k - i < right.precision
            ),
            Fraction(0),
        )
        for k in range(lo, hi)
    }
    return _window(left, lo, hi, values)


def inverse(
    series: TruncatedLaurentWindow, output_precision: int | None = None
) -> TruncatedLaurentWindow:
    _check(series)
    v = _nonzero(series)
    if v is None:
        raise OperationDomainValidationError(
            location=("series",),
            code="local_series.zero_not_invertible",
            message="the zero prefix is not invertible",
        )
    # The unit prefix has exponents 0 through P-v-1. Its reciprocal
    # determines output exponents below P-2v; unknown tails stay unknown.
    safe = series.precision - 2 * v
    hi = safe if output_precision is None else output_precision
    if hi <= -v or hi > safe:
        raise OperationDomainValidationError(
            location=("output_precision",),
            code="local_series.precision_not_supported",
            message="inverse precision requires known unit coefficients",
        )
    unit_hi = series.precision
    unit = [_coeff(series, v + j) for j in range(unit_hi - v)]
    a0 = unit[0]
    out: dict[int, Fraction] = {-v: 1 / a0}
    for n in range(1, hi + v):
        out[-v + n] = (
            -sum(
                (
                    unit[j] * out[-v + n - j]
                    for j in range(1, min(n, len(unit) - 1) + 1)
                ),
                Fraction(0),
            )
            / a0
        )
    return _window(series, -v, hi, out)


def divide(
    numerator: TruncatedLaurentWindow,
    denominator: TruncatedLaurentWindow,
    output_precision: int | None = None,
) -> TruncatedLaurentWindow:
    _pair(numerator, denominator)
    return multiply(numerator, inverse(denominator, output_precision), output_precision)


def power(
    series: TruncatedLaurentWindow, exponent: int, output_precision: int | None = None
) -> TruncatedLaurentWindow:
    _check(series)
    if type(exponent) is not int:
        raise OperationDomainValidationError(location=("exponent",), code="local_series.exponent_type", message="power exponent must be an integer")
    if exponent == 0:
        hi = series.precision if output_precision is None else output_precision
        if type(hi) is not int or hi <= 0:
            raise OperationDomainValidationError(location=("output_precision",), code="local_series.precision_not_supported", message="identity precision must be positive")
        return _window(series, 0, hi, {0: Fraction(1)})
    if exponent < 0:
        return power(inverse(series, output_precision), -exponent, output_precision)
    safe = series.precision + (exponent - 1) * series.valuation_lower
    lo = exponent * series.valuation_lower
    hi = safe if output_precision is None else output_precision
    if type(hi) is not int or hi <= lo or hi > safe:
        raise OperationDomainValidationError(location=("output_precision",), code="local_series.precision_not_supported", message=f"power precision must satisfy {lo} < P <= {safe}")
    if exponent == 1:
        return truncate(series, series.valuation_lower, hi)
    result = series
    for _count in range(2, exponent + 1):
        result = multiply(result, series, min(hi, result.precision + series.valuation_lower))
    return result


def derivative(series: TruncatedLaurentWindow) -> TruncatedLaurentWindow:
    _check(series)
    lo = series.valuation_lower - 1
    hi = series.precision - 1
    return _window(
        series,
        lo,
        hi,
        {
            k - 1: k * _coeff(series, k)
            for k in range(series.valuation_lower, series.precision)
            if k
        },
    )


def integral(series: TruncatedLaurentWindow) -> LaurentIntegralResult:
    _check(series)
    lo = series.valuation_lower + 1
    hi = series.precision + 1
    vals = {
        k + 1: _coeff(series, k) / (k + 1)
        for k in range(series.valuation_lower, series.precision)
        if k != -1
    }
    # The logarithmic term is carried separately and is never silently dropped.
    return LaurentIntegralResult(
        source=series,
        laurent_part=_window(series, lo, hi, vals),
        log_coefficient=_cr(_coeff(series, -1)),
    )


def truncate(
    series: TruncatedLaurentWindow, valuation_lower: int, precision: int
) -> TruncatedLaurentWindow:
    _check(series)
    if (
        valuation_lower < series.valuation_lower
        or precision > series.precision
        or precision <= valuation_lower
    ):
        raise OperationDomainValidationError(
            location=("precision",),
            code="local_series.truncation_window",
            message="truncation must be a subwindow of the known prefix",
        )
    return _window(
        series,
        valuation_lower,
        precision,
        {k: _coeff(series, k) for k in range(valuation_lower, precision)},
    )


def shift(series: TruncatedLaurentWindow, amount: int) -> TruncatedLaurentWindow:
    _check(series)
    return _window(
        series,
        series.valuation_lower + amount,
        series.precision + amount,
        {
            k + amount: _coeff(series, k)
            for k in range(series.valuation_lower, series.precision)
        },
    )


def change_scale(
    series: TruncatedLaurentWindow, scale: CanonicalRational
) -> TruncatedLaurentWindow:
    _check(series)
    c = _fraction(scale)
    if not c:
        raise OperationDomainValidationError(
            location=("scale",),
            code="local_series.zero_scale",
            message="scale must be nonzero",
        )
    return _window(
        series,
        series.valuation_lower,
        series.precision,
        {
            k: _coeff(series, k) * c**k
            for k in range(series.valuation_lower, series.precision)
        },
    )


def ramify(series: TruncatedLaurentWindow, ramification: int) -> TruncatedLaurentWindow:
    _check(series)
    if type(ramification) is not int or ramification < 1:
        raise OperationDomainValidationError(
            location=("ramification",),
            code="local_series.ramification",
            message="ramification must be positive",
        )
    return _window(
        series,
        series.valuation_lower * ramification,
        series.precision * ramification,
        {
            k * ramification: _coeff(series, k)
            for k in range(series.valuation_lower, series.precision)
        },
    )


def deramify(
    series: TruncatedLaurentWindow, ramification: int
) -> LaurentDeramifyResult:
    _check(series)
    if type(ramification) is not int or ramification < 1:
        raise OperationDomainValidationError(location=("ramification",), code="local_series.ramification", message="ramification must be positive")
    bad = tuple(
        k
        for k in range(series.valuation_lower, series.precision)
        if _coeff(series, k) and k % ramification
    )
    if bad:
        return LaurentDeramifyResult(
            status="NOT_IN_IMAGE_OF_RAMIFICATION", offending_exponents=bad
        )
    lo = -((-series.valuation_lower) // ramification)
    hi = (series.precision + ramification - 1) // ramification
    return LaurentDeramifyResult(
        status="IN_IMAGE_OF_RAMIFICATION",
        result=_window(
            series, lo, hi, {k: _coeff(series, k * ramification) for k in range(lo, hi)}
        ),
    )


def residue(series: TruncatedLaurentWindow) -> LaurentResidueResult:
    _check(series)
    return LaurentResidueResult(
        series=series, residue=CanonicalRational.from_fraction(_coeff(series, -1))
    )


def principal_part(series: TruncatedLaurentWindow) -> LaurentPrincipalPartResult:
    _check(series)
    if series.valuation_lower < 0:
        principal = _window(series, series.valuation_lower, min(0, series.precision), {k: _coeff(series, k) for k in range(series.valuation_lower, min(0, series.precision))})
    else:
        principal = _window(series, 0, 1, {0: Fraction(0)})
    if series.precision > 0:
        regular = _window(series, max(0, series.valuation_lower), series.precision, {k: _coeff(series, k) for k in range(max(0, series.valuation_lower), series.precision)})
    else:
        regular = _window(series, 0, 1, {0: Fraction(0)})
    return LaurentPrincipalPartResult(series=series, principal_part=principal, regular_part=regular)


__all__ = [
    "add",
    "change_scale",
    "deramify",
    "derivative",
    "divide",
    "integral",
    "inverse",
    "multiply",
    "power",
    "principal_part",
    "ramify",
    "residue",
    "shift",
    "subtract",
    "truncate",
]
