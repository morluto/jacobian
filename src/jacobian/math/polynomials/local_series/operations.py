"""Exact native valuation profiles for truncated Laurent windows."""

from __future__ import annotations

import re
from fractions import Fraction
from math import comb, lcm

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.local_series._models import (
    NonzeroValuation,
    ValuationProfileResult,
    ZeroValuation,
)
from jacobian.math.polynomials.local_series.arithmetic import (
    _admit_fraction,
    add,
    change_scale,
    deramify,
    derivative,
    divide,
    integral,
    inverse,
    multiply,
    power,
    principal_part,
    ramify,
    residue,
    shift,
    subtract,
    truncate,
)
from jacobian.math.polynomials.local_series.arithmetic import (
    _check as _check_laurent,
)
from jacobian.math.polynomials.local_series.arithmetic_models import (
    LaurentToPowerSeriesResult,
    PuiseuxResidueResult,
    RationalFunctionExpansionResult,
)
from jacobian.math.polynomials.local_series.puiseux_arithmetic import (
    MAX_PUISEUX_OUTPUT_BYTES,
)
from jacobian.math.polynomials.local_series.puiseux_arithmetic import (
    _check as _check_puiseux,
)
from jacobian.math.polynomials.local_series.puiseux_arithmetic import (
    add as puiseux_add,
)
from jacobian.math.polynomials.local_series.puiseux_arithmetic import (
    derivative as puiseux_derivative,
)
from jacobian.math.polynomials.local_series.puiseux_arithmetic import (
    inverse as puiseux_inverse,
)
from jacobian.math.polynomials.local_series.puiseux_arithmetic import (
    multiply as puiseux_multiply,
)
from jacobian.math.polynomials.local_series.puiseux_arithmetic import (
    subtract as puiseux_subtract,
)
from jacobian.math.polynomials.local_series.puiseux_values import TruncatedPuiseuxWindow
from jacobian.math.polynomials.local_series.values import (
    MAX_LOCAL_SERIES_COEFFICIENT_DIGITS,
    MAX_LOCAL_SERIES_EXPONENT,
    MAX_LOCAL_SERIES_TERMS,
    TruncatedLaurentWindow,
)
from jacobian.math.polynomials.rational_functions._models import (
    require_canonical_rational_function,
)
from jacobian.math.polynomials.series._models import (
    MAX_RATIONAL_DIGITS,
    MAX_TRUNCATION_ORDER,
    TruncatedSeries,
)
from jacobian.math.polynomials.values import RationalFunction


def laurent_valuation_profile(
    series: TruncatedLaurentWindow,
) -> ValuationProfileResult:
    """Return ZERO_AT_PRECISION or the exact valuation profile of a window."""

    if not isinstance(series, TruncatedLaurentWindow):
        raise OperationDomainValidationError(
            location=("series",),
            code="local_series.valuation_profile_series_type",
            message="series must be a truncated Laurent window value",
        )
    if len(series.coefficients) > MAX_LOCAL_SERIES_TERMS:
        raise OperationResourceAdmissionError(
            location=("series", "coefficients"),
            code="local_series.valuation_profile_window_bound",
            message=(
                "Laurent window length exceeds the "
                f"{MAX_LOCAL_SERIES_TERMS}-term valuation envelope"
            ),
        )
    for index, coefficient in enumerate(series.coefficients):
        if coefficient.as_fraction() != 0:
            valuation = series.valuation_lower + index
            return ValuationProfileResult._from_kernel(
                series,
                conclusion=NonzeroValuation(
                    status="NONZERO",
                    valuation=valuation,
                    leading_coefficient=coefficient,
                    pole_order=max(-valuation, 0),
                    zero_order=max(valuation, 0),
                ),
            )
    return ValuationProfileResult._from_kernel(
        series, conclusion=ZeroValuation(status="ZERO_AT_PRECISION")
    )


def _check_expansion_precision(precision: int) -> None:
    if type(precision) is not int or precision < 1:
        raise OperationDomainValidationError(
            location=("precision",),
            code="local_series.expansion_precision",
            message="precision must be a positive exclusive exponent cutoff",
        )
    if precision > MAX_LOCAL_SERIES_EXPONENT:
        raise OperationResourceAdmissionError(
            location=("precision",),
            code="local_series.expansion_precision_bound",
            message="requested precision exceeds the local-series exponent envelope",
        )
    if precision > MAX_LOCAL_SERIES_TERMS:
        raise OperationResourceAdmissionError(
            location=("precision",),
            code="local_series.expansion_work_bound",
            message=(
                "requested rational-function expansion exceeds the "
                f"{MAX_LOCAL_SERIES_TERMS}-coefficient work envelope"
            ),
        )


def _check_expansion_function(function: RationalFunction) -> None:
    if not isinstance(function, RationalFunction) or len(function.variables) != 1:
        raise OperationDomainValidationError(
            location=("function",),
            code="local_series.rational_function_univariate",
            message="rational-function expansion requires a univariate QQ function",
        )
    try:
        require_canonical_rational_function(
            function,
            maximum_terms=256,
            maximum_exponent=128,
            maximum_coefficient_digits=128,
            label="local-series expansion function",
        )
    except ValueError as error:
        raise OperationResourceAdmissionError(
            location=("function",),
            code="local_series.expansion_function_bound",
            message=str(error),
        ) from error


def _rational_digits(value: Fraction) -> int:
    return max(len(str(abs(value.numerator))), len(str(value.denominator)))


def _admit_expansion_recurrence(
    numerator: list[Fraction],
    denominator: list[Fraction],
    count: int,
    *,
    source_bytes_bound: int,
) -> None:
    """Bound recurrence coefficient height and serialized output before division."""
    if count == 0:
        return
    values = (*numerator, *denominator)
    common_denominator_digits = sum(len(str(value.denominator)) for value in values)
    if common_denominator_digits > MAX_LOCAL_SERIES_COEFFICIENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("function",),
            code="local_series.expansion_denominator_bound",
            message=(
                "translated polynomial coefficient denominators exceed the "
                f"{MAX_LOCAL_SERIES_COEFFICIENT_DIGITS}-digit common-denominator envelope"
            ),
        )
    common_denominator = lcm(*(value.denominator for value in values))
    common_denominator_digit_count = len(str(common_denominator))
    numerator_digit_bound = max(
        (len(str(abs(value.numerator))) for value in values),
        default=1,
    )
    if (
        common_denominator_digit_count + numerator_digit_bound
        > MAX_LOCAL_SERIES_COEFFICIENT_DIGITS
    ):
        raise OperationResourceAdmissionError(
            location=("function",),
            code="local_series.expansion_integer_coefficient_bound",
            message=(
                "common-denominator polynomial coefficients exceed the "
                f"{MAX_LOCAL_SERIES_COEFFICIENT_DIGITS}-digit envelope"
            ),
        )
    integer_digit_bound = max(
        (
            len(str(abs(value.numerator * (common_denominator // value.denominator))))
            for value in values
        ),
        default=1,
    )
    numerator_height = [integer_digit_bound]
    maximum_coefficient_digits = integer_digit_bound
    for n in range(1, count):
        summands = min(n, len(denominator) - 1)
        recurrence_height = (
            max(
                integer_digit_bound + n * integer_digit_bound,
                max(
                    (
                        numerator_height[n - j]
                        + (j - 1) * integer_digit_bound
                        + integer_digit_bound
                        for j in range(1, summands + 1)
                    ),
                    default=1,
                ),
            )
            + (summands + 1).bit_length()
            + 2
        )
        numerator_height.append(recurrence_height)
        maximum_coefficient_digits = max(
            maximum_coefficient_digits,
            recurrence_height,
            (n + 1) * integer_digit_bound,
        )
    if maximum_coefficient_digits > MAX_LOCAL_SERIES_COEFFICIENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("precision",),
            code="local_series.expansion_coefficient_bound",
            message=(
                "requested expansion exceeds the admitted exact coefficient "
                f"height of {MAX_LOCAL_SERIES_COEFFICIENT_DIGITS} digits"
            ),
        )
    # The result carries both the shifted series and its normalized unit
    # quotient, plus the admitted source rational function.
    output_bound = (
        source_bytes_bound + 1024 + count * (4 * maximum_coefficient_digits + 192)
    )
    if output_bound > MAX_PUISEUX_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("result",),
            code="local_series.expansion_output_bound",
            message=(
                "requested Laurent prefix exceeds the "
                f"{MAX_PUISEUX_OUTPUT_BYTES}-byte output envelope"
            ),
        )


def _check_expansion_center(center: CanonicalRational) -> Fraction:
    if not isinstance(center, CanonicalRational):
        raise OperationDomainValidationError(
            location=("center",),
            code="local_series.expansion_center",
            message="center must be a canonical rational",
        )
    if type(center.num) is not int or type(center.den) is not int or center.den <= 0:
        raise OperationDomainValidationError(
            location=("center",),
            code="local_series.expansion_center",
            message="center must be a reduced rational with positive denominator",
        )
    value = center.as_fraction()
    if (center.num, center.den) != (value.numerator, value.denominator):
        raise OperationDomainValidationError(
            location=("center",),
            code="local_series.expansion_center",
            message="center must be in canonical reduced form",
        )
    center_digits = max(len(str(abs(center.num))), len(str(center.den)))
    if center_digits > 30:
        raise OperationResourceAdmissionError(
            location=("center",),
            code="local_series.expansion_center_bound",
            message="center exceeds the 30-digit translation envelope",
        )
    return value


def _source_function_output_bytes(function: RationalFunction) -> int:
    """Conservatively price the source function repeated in the result."""
    # At most 256 terms in each polynomial; each term has two 128-digit
    # rational components, an exponent, field names, and JSON punctuation.
    return (
        8192
        + 2 * (len(function.numerator.terms) + len(function.denominator.terms)) * 512
    )


def rational_function_at_point(
    function: RationalFunction, center: CanonicalRational, precision: int
) -> RationalFunctionExpansionResult:
    """Expand a univariate rational function at ``x = center`` through t^N."""
    _check_expansion_function(function)
    _check_expansion_precision(precision)
    x0 = _check_expansion_center(center)

    def translate(polynomial: object) -> list[Fraction]:
        terms = polynomial.terms
        degree = max((term.exponents[0] for term in terms), default=0)
        coefficients = [Fraction(0) for _ in range(degree + 1)]
        for term in terms:
            exponent = term.exponents[0]
            coefficient = term.coefficient.as_fraction()
            for index in range(exponent + 1):
                coefficients[index] += (
                    coefficient * comb(exponent, index) * x0 ** (exponent - index)
                )
        return coefficients

    numerator = translate(function.numerator)
    denominator = translate(function.denominator)
    numerator_order = next((i for i, c in enumerate(numerator) if c), None)
    denominator_order = next((i for i, c in enumerate(denominator) if c), None)
    if denominator_order is None:
        raise OperationDomainValidationError(
            location=("function", "denominator"),
            code="local_series.expansion_zero_denominator",
            message="rational-function denominator cannot vanish identically",
        )
    if numerator_order is None:
        series = TruncatedLaurentWindow(
            variable="t",
            place="FINITE",
            center=center,
            valuation_lower=0,
            precision=precision,
            coefficients=tuple(
                CanonicalRational(num=0, den=1) for _ in range(precision)
            ),
        )
        return RationalFunctionExpansionResult(
            function=function,
            series=series,
            numerator_order=None,
            denominator_order=denominator_order,
            valuation=None,
            pole_order=0,
            zero_order=0,
            normalized_unit_quotient=None,
            product_residual_precision=precision + denominator_order,
        )

    valuation = numerator_order - denominator_order
    coefficient_count = max(0, precision - valuation)
    if coefficient_count > MAX_LOCAL_SERIES_TERMS:
        raise OperationResourceAdmissionError(
            location=("precision",),
            code="local_series.expansion_work_bound",
            message=(
                "requested rational-function expansion exceeds the "
                f"{MAX_LOCAL_SERIES_TERMS}-coefficient work envelope"
            ),
        )
    b = denominator[denominator_order:]
    a = numerator[numerator_order:]
    _admit_expansion_recurrence(
        a,
        b,
        coefficient_count,
        source_bytes_bound=_source_function_output_bytes(function),
    )
    quotient: list[Fraction] = []
    for n in range(coefficient_count):
        value = a[n] if n < len(a) else Fraction(0)
        for j in range(1, min(n, len(b) - 1) + 1):
            value -= b[j] * quotient[n - j]
        value /= b[0]
        _admit_fraction(value)
        quotient.append(value)

    if valuation >= precision:
        lower = 0
        output = [Fraction(0)] * precision
    else:
        lower = valuation
        output = quotient
    if len(output) > MAX_LOCAL_SERIES_TERMS:
        raise OperationResourceAdmissionError(
            location=("result",),
            code="local_series.expansion_result_bound",
            message="rational-function Laurent window exceeds the retained-term envelope",
        )
    series = TruncatedLaurentWindow.model_construct(
        variable="t",
        place="FINITE",
        center=center,
        valuation_lower=lower,
        precision=precision,
        coefficients=tuple(CanonicalRational.from_fraction(c) for c in output),
    )
    unit = None
    if coefficient_count:
        unit = TruncatedLaurentWindow.model_construct(
            variable="t",
            place="FINITE",
            center=center,
            valuation_lower=0,
            precision=coefficient_count,
            coefficients=tuple(CanonicalRational.from_fraction(c) for c in quotient),
        )
    return RationalFunctionExpansionResult(
        function=function,
        series=series,
        numerator_order=numerator_order,
        denominator_order=denominator_order,
        valuation=valuation,
        pole_order=max(-valuation, 0),
        zero_order=max(valuation, 0),
        normalized_unit_quotient=unit,
        product_residual_precision=precision + denominator_order,
    )


def rational_function_at_infinity(
    function: RationalFunction, precision: int
) -> RationalFunctionExpansionResult:
    """Expand a univariate rational function in the reciprocal parameter t=1/x."""
    _check_expansion_function(function)
    _check_expansion_precision(precision)

    def reverse(polynomial: object) -> list[Fraction]:
        degree = max((term.exponents[0] for term in polynomial.terms), default=0)
        coefficients = [Fraction(0) for _ in range(degree + 1)]
        for term in polynomial.terms:
            exponent = term.exponents[0]
            coefficients[degree - exponent] = term.coefficient.as_fraction()
        return coefficients

    numerator = reverse(function.numerator)
    denominator = reverse(function.denominator)
    numerator_degree = max(
        (term.exponents[0] for term in function.numerator.terms), default=-1
    )
    denominator_degree = max(
        (term.exponents[0] for term in function.denominator.terms), default=0
    )
    if numerator_degree < 0:
        lower = 0
        coefficients = [Fraction(0)] * precision
        valuation = None
        numerator_order = None
        denominator_order = -denominator_degree
        quotient = []
        coefficient_count = 0
    else:
        valuation = denominator_degree - numerator_degree
        numerator_order = -numerator_degree
        denominator_order = -denominator_degree
        coefficient_count = max(0, precision - valuation)
        if coefficient_count > MAX_LOCAL_SERIES_TERMS:
            raise OperationResourceAdmissionError(
                location=("precision",),
                code="local_series.expansion_work_bound",
                message=(
                    "requested rational-function expansion exceeds the "
                    f"{MAX_LOCAL_SERIES_TERMS}-coefficient work envelope"
                ),
            )
        quotient: list[Fraction] = []
        _admit_expansion_recurrence(
            numerator,
            denominator,
            coefficient_count,
            source_bytes_bound=_source_function_output_bytes(function),
        )
        for n in range(coefficient_count):
            value = numerator[n] if n < len(numerator) else Fraction(0)
            for j in range(1, min(n, len(denominator) - 1) + 1):
                value -= denominator[j] * quotient[n - j]
            value /= denominator[0]
            _admit_fraction(value)
            quotient.append(value)
        if valuation >= precision:
            lower = 0
            coefficients = [Fraction(0)] * precision
        else:
            lower = valuation
            coefficients = quotient
    series = TruncatedLaurentWindow.model_construct(
        variable="t",
        place="INFINITY",
        center=CanonicalRational(num=0, den=1),
        valuation_lower=lower,
        precision=precision,
        coefficients=tuple(CanonicalRational.from_fraction(c) for c in coefficients),
    )
    unit = None
    if coefficient_count:
        unit = TruncatedLaurentWindow.model_construct(
            variable="t",
            place="INFINITY",
            center=CanonicalRational(num=0, den=1),
            valuation_lower=0,
            precision=coefficient_count,
            coefficients=tuple(CanonicalRational.from_fraction(c) for c in quotient),
        )
    return RationalFunctionExpansionResult(
        function=function,
        series=series,
        numerator_order=numerator_order,
        denominator_order=denominator_order,
        valuation=valuation,
        pole_order=max(-(valuation or 0), 0),
        zero_order=max(valuation or 0, 0),
        normalized_unit_quotient=unit,
        product_residual_precision=precision + denominator_order,
    )


def add_puiseux(
    left: TruncatedPuiseuxWindow, right: TruncatedPuiseuxWindow
) -> TruncatedPuiseuxWindow:
    return puiseux_add(left, right)


def subtract_puiseux(
    left: TruncatedPuiseuxWindow, right: TruncatedPuiseuxWindow
) -> TruncatedPuiseuxWindow:
    return puiseux_subtract(left, right)


def multiply_puiseux(
    left: TruncatedPuiseuxWindow, right: TruncatedPuiseuxWindow
) -> TruncatedPuiseuxWindow:
    return puiseux_multiply(left, right)


def differentiate_puiseux(series: TruncatedPuiseuxWindow) -> TruncatedPuiseuxWindow:
    """Differentiate a bounded exact Puiseux prefix."""
    return puiseux_derivative(series)


def inverse_puiseux(series: TruncatedPuiseuxWindow) -> TruncatedPuiseuxWindow:
    """Invert a bounded exact Puiseux prefix with a retained leading term."""
    return puiseux_inverse(series)


def residue_puiseux(series: TruncatedPuiseuxWindow) -> PuiseuxResidueResult:
    """Extract the t^-1 coefficient when it is known by the retained window."""
    _, lower, terms, precision = _check_puiseux(series)
    exponent = Fraction(-1)
    if not lower <= exponent < precision:
        raise OperationDomainValidationError(
            location=("series", "precision"),
            code="local_series.puiseux_residue_precision",
            message="Puiseux window must contain exponent -1 to determine its residue",
        )
    value = next(
        (coefficient for term_exp, coefficient in terms if term_exp == exponent),
        Fraction(0),
    )

    # The result repeats the source window and adds a rational scalar. Bound
    # its compact JSON size from the already admitted canonical values before
    # constructing that output.
    def rational_bytes(number: Fraction) -> int:
        return len(str(abs(number.numerator))) + len(str(number.denominator)) + 24

    output_bound = (
        512
        + len(series.variable.encode("utf-8"))
        + rational_bytes(series.center.as_fraction())
        + rational_bytes(lower)
        + rational_bytes(precision)
        + len(str(series.ramification_index))
        + rational_bytes(value)
        + sum(
            64 + rational_bytes(term_exponent) + rational_bytes(coefficient)
            for term_exponent, coefficient in terms
        )
    )
    if output_bound > MAX_PUISEUX_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("result",),
            code="local_series.puiseux_residue_output_bound",
            message=(
                "source-bound Puiseux residue exceeds the "
                f"{MAX_PUISEUX_OUTPUT_BYTES}-byte output envelope"
            ),
        )
    return PuiseuxResidueResult(
        series=series, residue=CanonicalRational.from_fraction(value)
    )


def from_power_series(series: TruncatedSeries) -> TruncatedLaurentWindow:
    """Embed QQ[[x]]/(x^N) into the finite-center Laurent carrier at zero."""
    if not isinstance(series, TruncatedSeries):
        raise OperationDomainValidationError(
            location=("series",),
            code="local_series.power_series_type",
            message="series must be an exact truncated power-series value",
        )
    if (
        type(series.truncation_order) is not int
        or series.truncation_order < 1
        or type(series.variable) is not str
        or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", series.variable) is None
        or not isinstance(series.coefficients, tuple)
        or len(series.coefficients) != series.truncation_order
    ):
        raise OperationDomainValidationError(
            location=("series",),
            code="local_series.power_series_shape",
            message="power-series value has an invalid dense coefficient shape",
        )
    if series.truncation_order > MAX_TRUNCATION_ORDER:
        raise OperationResourceAdmissionError(
            location=("series", "truncation_order"),
            code="local_series.power_series_order_bound",
            message=(
                f"power-series conversion is limited to order {MAX_TRUNCATION_ORDER}"
            ),
        )
    for coefficient in series.coefficients:
        if (
            not isinstance(coefficient, CanonicalRational)
            or type(coefficient.num) is not int
            or type(coefficient.den) is not int
            or coefficient.den <= 0
        ):
            raise OperationDomainValidationError(
                location=("series", "coefficients"),
                code="local_series.power_series_coefficient_value",
                message="power-series coefficients must be canonical rationals",
            )
        rational = Fraction(coefficient.num, coefficient.den)
        if (coefficient.num, coefficient.den) != (
            rational.numerator,
            rational.denominator,
        ):
            raise OperationDomainValidationError(
                location=("series", "coefficients"),
                code="local_series.power_series_coefficient_value",
                message="power-series coefficients must use reduced canonical form",
            )
        try:
            require_bounded_rational(
                coefficient,
                max_digits=MAX_RATIONAL_DIGITS,
                label="power-series coefficient",
            )
        except ValueError as error:
            raise OperationResourceAdmissionError(
                location=("series", "coefficients"),
                code="local_series.power_series_coefficient_bound",
                message=str(error),
            ) from error
    first_nonzero = next(
        (
            index
            for index, coefficient in enumerate(series.coefficients)
            if coefficient.num
        ),
        None,
    )
    if first_nonzero is None:
        lower = 0
        coefficients = series.coefficients
    else:
        lower = first_nonzero
        coefficients = series.coefficients[first_nonzero:]
    return TruncatedLaurentWindow.model_construct(
        variable=series.variable,
        place="FINITE",
        center=CanonicalRational(num=0, den=1),
        valuation_lower=lower,
        precision=series.truncation_order,
        coefficients=coefficients,
    )


def to_power_series(series: TruncatedLaurentWindow) -> LaurentToPowerSeriesResult:
    """Convert a finite-origin Laurent prefix when it has no negative terms."""
    _check_laurent(series)
    if series.place != "FINITE" or series.center.as_fraction() != 0:
        raise OperationDomainValidationError(
            location=("series",),
            code="local_series.power_series_parent",
            message="power-series conversion requires the finite center zero parent",
        )
    if not 1 <= series.precision <= MAX_TRUNCATION_ORDER:
        raise OperationResourceAdmissionError(
            location=("series", "precision"),
            code="local_series.power_series_order_bound",
            message=(
                f"power-series conversion requires precision at most {MAX_TRUNCATION_ORDER}"
            ),
        )
    for coefficient in series.coefficients:
        try:
            require_bounded_rational(
                coefficient,
                max_digits=MAX_RATIONAL_DIGITS,
                label="power-series coefficient",
            )
        except ValueError as error:
            raise OperationResourceAdmissionError(
                location=("series", "coefficients"),
                code="local_series.power_series_coefficient_bound",
                message=str(error),
            ) from error
    first_negative = next(
        (
            series.valuation_lower + index
            for index, coefficient in enumerate(series.coefficients)
            if series.valuation_lower + index < 0 and coefficient.num != 0
        ),
        None,
    )
    if first_negative is not None:
        return LaurentToPowerSeriesResult(
            status="HAS_NEGATIVE_EXPONENTS",
            source=series,
            first_negative_exponent=first_negative,
        )
    coefficients = [CanonicalRational(num=0, den=1) for _ in range(series.precision)]
    for index, coefficient in enumerate(series.coefficients):
        exponent = series.valuation_lower + index
        if 0 <= exponent < series.precision:
            coefficients[exponent] = coefficient
    power_series = TruncatedSeries.model_construct(
        variable=series.variable,
        truncation_order=series.precision,
        coefficients=tuple(coefficients),
    )
    return LaurentToPowerSeriesResult(
        status="CONVERTED",
        source=series,
        result=power_series,
    )


__all__ = [
    "add",
    "add_puiseux",
    "change_scale",
    "deramify",
    "derivative",
    "differentiate_puiseux",
    "divide",
    "from_power_series",
    "integral",
    "inverse",
    "inverse_puiseux",
    "laurent_valuation_profile",
    "multiply",
    "multiply_puiseux",
    "power",
    "principal_part",
    "ramify",
    "rational_function_at_infinity",
    "rational_function_at_point",
    "residue",
    "residue_puiseux",
    "shift",
    "subtract",
    "subtract_puiseux",
    "to_power_series",
    "truncate",
]
