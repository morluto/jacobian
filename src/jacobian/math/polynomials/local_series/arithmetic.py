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
    MAX_LOCAL_SERIES_POWER_EXPONENT,
    MAX_LOCAL_SERIES_POWER_WORK,
    MAX_LOCAL_SERIES_TERMS,
    TruncatedLaurentWindow,
)


def _fraction(value: CanonicalRational) -> Fraction:
    return value.as_fraction()


def _cr(value: Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(value)


def _strict_int(value: object, location: tuple[str, ...], code: str) -> int:
    if type(value) is not int:
        raise OperationDomainValidationError(
            location=location, code=code, message="value must be an integer"
        )
    return value


def _admit_fraction(value: Fraction) -> None:
    """Admit a Fraction without first constructing an oversized carrier."""

    maximum = max(abs(value.numerator), value.denominator)
    if maximum.bit_length() <= 3 * MAX_LOCAL_SERIES_COEFFICIENT_DIGITS:
        return
    if maximum >= 10**MAX_LOCAL_SERIES_COEFFICIENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("coefficients",),
            code="local_series.coefficient_bound",
            message=(
                "Laurent arithmetic coefficient exceeds the "
                f"{MAX_LOCAL_SERIES_COEFFICIENT_DIGITS}-digit bound"
            ),
        )


def _admit_coefficient(value: CanonicalRational) -> None:
    """Admit a generated coefficient for the Laurent carrier."""

    try:
        require_bounded_rational(
            value,
            max_digits=MAX_LOCAL_SERIES_COEFFICIENT_DIGITS,
            label="Laurent arithmetic coefficient",
        )
    except ValueError as error:
        raise OperationResourceAdmissionError(
            location=("coefficients",),
            code="local_series.coefficient_bound",
            message=str(error),
        ) from error


def _check_canonical_coefficient(value: object) -> None:
    """Revalidate native scalar carriers before any Fraction conversion."""

    if not isinstance(value, CanonicalRational):
        raise OperationDomainValidationError(
            location=("series", "coefficients"),
            code="local_series.coefficient_type",
            message="Laurent coefficients must be canonical rationals",
        )
    num = getattr(value, "num", None)
    den = getattr(value, "den", None)
    if type(num) is not int or type(den) is not int or den <= 0:
        raise OperationDomainValidationError(
            location=("series", "coefficients"),
            code="local_series.coefficient_value",
            message="Laurent coefficients must have a positive denominator",
        )
    try:
        rational = Fraction(num, den)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise OperationDomainValidationError(
            location=("series", "coefficients"),
            code="local_series.coefficient_value",
            message="Laurent coefficients must be valid rationals",
        ) from error
    if (num, den) != (rational.numerator, rational.denominator):
        raise OperationDomainValidationError(
            location=("series", "coefficients"),
            code="local_series.coefficient_value",
            message="Laurent coefficients must be reduced canonical rationals",
        )
    _admit_coefficient(value)


def _check(s: TruncatedLaurentWindow) -> None:
    if not isinstance(s, TruncatedLaurentWindow):
        raise OperationDomainValidationError(
            location=("series",),
            code="local_series.series_type",
            message="series must be a Laurent window",
        )
    if any(
        getattr(s, field, None) is None
        for field in (
            "variable",
            "valuation_lower",
            "precision",
            "coefficients",
            "center",
        )
    ):
        raise OperationDomainValidationError(
            location=("series",),
            code="local_series.series_structure",
            message="Laurent window is missing required structural fields",
        )
    if type(s.variable) is not str:
        raise OperationDomainValidationError(
            location=("series", "variable"),
            code="local_series.variable_type",
            message="Laurent variable must be a strict identifier string",
        )
    if (
        type(s.valuation_lower) is not int
        or type(s.precision) is not int
        or s.precision < s.valuation_lower
        or not isinstance(s.coefficients, tuple)
        or len(s.coefficients) != s.precision - s.valuation_lower
    ):
        raise OperationDomainValidationError(
            location=("series",),
            code="local_series.series_shape",
            message="series has an invalid Laurent window shape",
        )
    if (
        abs(s.valuation_lower) > MAX_LOCAL_SERIES_EXPONENT
        or abs(s.precision) > MAX_LOCAL_SERIES_EXPONENT
    ):
        raise OperationResourceAdmissionError(
            location=("series",),
            code="local_series.exponent_bound",
            message="Laurent window exponents exceed the admitted envelope",
        )
    if len(s.coefficients) > MAX_LOCAL_SERIES_TERMS:
        raise OperationResourceAdmissionError(
            location=("series",),
            code="local_series.terms",
            message="Laurent arithmetic exceeds the retained-term envelope",
        )
    if not isinstance(s.center, CanonicalRational):
        raise OperationDomainValidationError(
            location=("series", "center"),
            code="local_series.center_type",
            message="Laurent center must be a canonical rational",
        )
    center_num = getattr(s.center, "num", None)
    center_den = getattr(s.center, "den", None)
    if type(center_num) is not int or type(center_den) is not int:
        raise OperationDomainValidationError(
            location=("series", "center"),
            code="local_series.center_value",
            message="Laurent center components must be strict integers",
        )
    try:
        center = s.center.as_fraction()
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise OperationDomainValidationError(
            location=("series", "center"),
            code="local_series.center_value",
            message="Laurent center must be a valid canonical rational",
        ) from error
    if center_den <= 0 or (center_num, center_den) != (
        center.numerator,
        center.denominator,
    ):
        raise OperationDomainValidationError(
            location=("series", "center"),
            code="local_series.center_value",
            message="Laurent center must be reduced with a positive denominator",
        )
    # Native callers can supply model_construct() values, so do not rely on
    # the carrier validator having checked coefficient bounds.
    for coefficient in s.coefficients:
        _check_canonical_coefficient(coefficient)


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
        hi < lo
        or hi - lo > MAX_LOCAL_SERIES_TERMS
        or max(abs(lo), abs(hi)) > MAX_LOCAL_SERIES_EXPONENT
    ):
        raise OperationResourceAdmissionError(
            location=("precision",),
            code="local_series.result_bound",
            message="Laurent result exceeds its admitted exponent or term envelope",
        )
    values = {k: values.get(k, Fraction(0)) for k in range(lo, hi)}
    for value in values.values():
        _admit_fraction(value)
    coefficients = tuple(_cr(values[k]) for k in range(lo, hi))
    for coefficient in coefficients:
        _admit_coefficient(coefficient)
    return TruncatedLaurentWindow.model_construct(
        variable=source.variable,
        center=source.center,
        valuation_lower=lo,
        precision=hi,
        coefficients=coefficients,
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


def _product_work(
    left: TruncatedLaurentWindow,
    right: TruncatedLaurentWindow,
    lo: int,
    hi: int,
) -> int:
    """Count dense convolution incidences without constructing them."""

    total = 0
    left_lo, left_hi = left.valuation_lower, left.precision
    right_lo, right_hi = right.valuation_lower, right.precision
    for exponent in range(lo, hi):
        lower = max(left_lo, exponent - right_hi + 1)
        upper = min(left_hi - 1, exponent - right_lo)
        if upper >= lower:
            total += upper - lower + 1
    return total


def _admit_single_product_growth(
    left: TruncatedLaurentWindow,
    right: TruncatedLaurentWindow,
    lo: int,
    hi: int,
) -> None:
    """Reject an unambiguous oversized product before multiplying it.

    A coefficient with several summands can cancel, so only the one-summand
    case is admitted from this cheap bound.  The general result is still
    checked by ``_window`` after exact convolution.
    """

    for exponent in range(lo, hi):
        pairs = [
            (left_index, exponent - left_index)
            for left_index in range(left.valuation_lower, left.precision)
            if right.valuation_lower <= exponent - left_index < right.precision
            and _coeff(left, left_index)
            and _coeff(right, exponent - left_index)
        ]
        if len(pairs) != 1:
            continue
        left_index, right_index = pairs[0]
        left_value = left.coefficients[left_index - left.valuation_lower]
        right_value = right.coefficients[right_index - right.valuation_lower]
        digits = max(
            len(str(abs(left_value.num))) + len(str(abs(right_value.num))),
            len(str(left_value.den)) + len(str(right_value.den)),
        )
        if digits > MAX_LOCAL_SERIES_COEFFICIENT_DIGITS:
            raise OperationResourceAdmissionError(
                location=("coefficients",),
                code="local_series.coefficient_bound",
                message="Laurent product coefficients exceed the admitted digit bound",
            )


def multiply(
    left: TruncatedLaurentWindow,
    right: TruncatedLaurentWindow,
    output_precision: int | None = None,
) -> TruncatedLaurentWindow:
    _pair(left, right)
    if output_precision is not None:
        _strict_int(
            output_precision, ("output_precision",), "local_series.precision_type"
        )
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
    work = _product_work(left, right, lo, hi)
    if work > MAX_LOCAL_SERIES_POWER_WORK:
        raise OperationResourceAdmissionError(
            location=("series",),
            code="local_series.arithmetic_work",
            message="Laurent multiplication exceeds the bounded convolution work limit",
        )
    _admit_single_product_growth(left, right, lo, hi)
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
    if output_precision is not None:
        _strict_int(
            output_precision, ("output_precision",), "local_series.precision_type"
        )
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
    unit_width = unit_hi - v
    if unit_width * unit_width > MAX_LOCAL_SERIES_POWER_WORK:
        raise OperationResourceAdmissionError(
            location=("series",),
            code="local_series.arithmetic_work",
            message="Laurent inversion exceeds the bounded recurrence work limit",
        )
    unit = [_coeff(series, v + j) for j in range(unit_width)]
    a0 = unit[0]
    out: dict[int, Fraction] = {-v: 1 / a0}
    _admit_fraction(out[-v])
    for n in range(1, hi + v):
        candidate = (
            -sum(
                (
                    unit[j] * out[-v + n - j]
                    for j in range(1, min(n, len(unit) - 1) + 1)
                ),
                Fraction(0),
            )
            / a0
        )
        # Admit each recurrence intermediate before it can feed the next
        # coefficient; otherwise a long inverse could grow outside the
        # canonical carrier before final window construction.
        _admit_fraction(candidate)
        out[-v + n] = candidate
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
        raise OperationDomainValidationError(
            location=("exponent",),
            code="local_series.exponent_type",
            message="exponent must be an integer",
        )
    if abs(exponent) > MAX_LOCAL_SERIES_POWER_EXPONENT:
        raise OperationResourceAdmissionError(
            location=("exponent",),
            code="local_series.exponent_bound",
            message=(
                "Laurent power exponent exceeds the "
                f"{MAX_LOCAL_SERIES_POWER_EXPONENT}-step envelope"
            ),
        )
    if output_precision is not None and type(output_precision) is not int:
        raise OperationDomainValidationError(
            location=("output_precision",),
            code="local_series.precision_type",
            message="output precision must be an integer",
        )
    if (
        output_precision is not None
        and abs(output_precision) > MAX_LOCAL_SERIES_EXPONENT
    ):
        raise OperationResourceAdmissionError(
            location=("output_precision",),
            code="local_series.precision_bound",
            message="Laurent output precision exceeds the exponent envelope",
        )
    if exponent == 0:
        hi = max(1, series.precision) if output_precision is None else output_precision
        if hi <= 0:
            raise OperationDomainValidationError(
                location=("output_precision",),
                code="local_series.precision_not_supported",
                message="identity precision must include exponent zero",
            )
        return _window(series, 0, hi, {0: Fraction(1)})
    if exponent < 0:
        return power(inverse(series, output_precision), -exponent, output_precision)
    if exponent == 1:
        if output_precision is None:
            return series
        return truncate(series, series.valuation_lower, output_precision)
    source_width = max(1, len(series.coefficients))
    output_width = (
        max(1, output_precision - series.valuation_lower)
        if output_precision is not None
        else min(MAX_LOCAL_SERIES_TERMS, exponent * source_width)
    )
    if exponent * source_width * output_width > MAX_LOCAL_SERIES_POWER_WORK:
        raise OperationResourceAdmissionError(
            location=("exponent",),
            code="local_series.arithmetic_work",
            message="Laurent powering exceeds the bounded arithmetic work limit",
        )
    result = series
    for _ in range(exponent - 1):
        result = multiply(result, series, output_precision)
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
    _strict_int(valuation_lower, ("valuation_lower",), "local_series.precision_type")
    _strict_int(precision, ("precision",), "local_series.precision_type")
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
    _strict_int(amount, ("shift",), "local_series.shift_type")
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
    numerator = getattr(scale, "num", None)
    denominator = getattr(scale, "den", None)
    if (
        not isinstance(scale, CanonicalRational)
        or type(numerator) is not int
        or type(denominator) is not int
        or denominator <= 0
        or gcd(abs(numerator), denominator) != 1
        or (numerator == 0 and denominator != 1)
    ):
        raise OperationDomainValidationError(
            location=("scale",),
            code="local_series.scale_type",
            message="scale must be a reduced canonical rational with positive denominator",
        )
    c = Fraction(numerator, denominator)
    if not c:
        raise OperationDomainValidationError(
            location=("scale",),
            code="local_series.zero_scale",
            message="scale must be nonzero",
        )
    scale_digits = max(len(str(abs(c.numerator))), len(str(c.denominator)))
    # Exponentiation can otherwise create an enormous temporary before the
    # result carrier gets a chance to reject it.  Reduction can only decrease
    # component widths, so this is a sound pre-admission bound.
    max_source_digits = max(
        (
            max(len(str(abs(value.num))), len(str(value.den)))
            for value in series.coefficients
        ),
        default=1,
    )
    if any(
        max_source_digits + abs(k) * scale_digits > MAX_LOCAL_SERIES_COEFFICIENT_DIGITS
        for k in range(series.valuation_lower, series.precision)
    ):
        raise OperationResourceAdmissionError(
            location=("scale",),
            code="local_series.coefficient_bound",
            message="scaled Laurent coefficients exceed the admitted digit bound",
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
        raise OperationDomainValidationError(
            location=("ramification",),
            code="local_series.ramification",
            message="ramification must be positive",
        )
    bad = tuple(
        k
        for k in range(series.valuation_lower, series.precision)
        if _coeff(series, k) and k % ramification
    )
    if bad:
        return LaurentDeramifyResult(
            status="NOT_IN_IMAGE_OF_RAMIFICATION", offending_exponents=bad
        )
    # Only aligned exponents whose multiples are inside the known source
    # window are known in the preimage.  Floor division would invent a value
    # from an omitted source tail at the lower boundary.
    lo = -((-series.valuation_lower) // ramification)
    hi = -((-series.precision) // ramification)
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
    split = max(series.valuation_lower, min(series.precision, 0))
    principal_hi = split
    regular_lo = split
    principal = _window(
        series,
        series.valuation_lower,
        principal_hi,
        {k: _coeff(series, k) for k in range(series.valuation_lower, principal_hi)},
    )
    regular = _window(
        series,
        regular_lo,
        series.precision,
        {k: _coeff(series, k) for k in range(regular_lo, series.precision)},
    )
    return LaurentPrincipalPartResult(
        series=series, principal_part=principal, regular_part=regular
    )


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
