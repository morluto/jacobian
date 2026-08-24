"""Supported exact truncated formal-power-series API.

Native callables admit their inputs through the same typed request envelopes
as the wire operations before calling one shared domain kernel, so direct
Python callers cannot start unbounded kernel work either.
"""

from jacobian._exact import CanonicalRational
from jacobian.math.formal_power_series._models import (
    SeriesArithmeticResult,
    SeriesComposeRequest,
    SeriesComposeResult,
    SeriesDerivativeResult,
    SeriesDivideRequest,
    SeriesDivideResult,
    SeriesIdentityCheckResult,
    SeriesIntegralRequest,
    SeriesIntegralResult,
    SeriesInverseRequest,
    SeriesInverseResult,
    SeriesMultiplyResult,
    SeriesPowerRequest,
    SeriesPowerResult,
    SeriesReversionRequest,
    SeriesReversionResult,
    SeriesScalarMultiplyRequest,
    SeriesScalarMultiplyResult,
    SeriesToPolynomialResult,
    SeriesTruncateRequest,
    SeriesTruncateResult,
    TruncatedSeries,
    TruncateSourceSeries,
    _SeriesAddSubtractRequest,
    _SeriesIdentityCheckRequest,
    _SeriesMultiplyRequest,
    as_input_series,
)
from jacobian.math.formal_power_series._operations import (
    compute_add,
    compute_compose,
    compute_derivative,
    compute_divide,
    compute_from_polynomial,
    compute_identity_check,
    compute_integral,
    compute_inverse,
    compute_multiply,
    compute_power,
    compute_reversion,
    compute_scalar_multiply,
    compute_subtract,
    compute_to_polynomial,
    compute_truncate,
)


def add(left: TruncatedSeries, right: TruncatedSeries) -> SeriesArithmeticResult:
    """Add two series coefficientwise modulo x^N."""
    _SeriesAddSubtractRequest(left=as_input_series(left), right=as_input_series(right))
    return compute_add(left, right)


def subtract(left: TruncatedSeries, right: TruncatedSeries) -> SeriesArithmeticResult:
    """Subtract two series coefficientwise modulo x^N."""
    _SeriesAddSubtractRequest(left=as_input_series(left), right=as_input_series(right))
    return compute_subtract(left, right)


def multiply(left: TruncatedSeries, right: TruncatedSeries) -> SeriesMultiplyResult:
    """Multiply two series modulo x^N via Cauchy convolution."""
    _SeriesMultiplyRequest(left=as_input_series(left), right=as_input_series(right))
    return compute_multiply(left, right)


def scalar_multiply(
    series: TruncatedSeries, scalar: CanonicalRational
) -> SeriesScalarMultiplyResult:
    """Multiply a series by an exact rational scalar."""
    request = SeriesScalarMultiplyRequest(
        series=as_input_series(series),
        scalar=scalar,
    )
    return compute_scalar_multiply(series, request.scalar)


def power(series: TruncatedSeries, exponent: int) -> SeriesPowerResult:
    """Compute series^exponent via binary exponentiation modulo x^N."""
    request = SeriesPowerRequest(series=as_input_series(series), exponent=exponent)
    return compute_power(series, request.exponent)


def inverse(series: TruncatedSeries) -> SeriesInverseResult:
    """Invert a truncated unit series modulo x^N."""
    SeriesInverseRequest(
        variable=series.variable,
        truncation_order=series.truncation_order,
        coefficients=series.coefficients,
    )
    return compute_inverse(series)


def divide(
    numerator: TruncatedSeries, denominator: TruncatedSeries
) -> SeriesDivideResult:
    """Divide two series modulo x^N when the denominator is a unit."""
    SeriesDivideRequest(
        left=as_input_series(numerator),
        right=as_input_series(denominator),
    )
    return compute_divide(numerator, denominator)


def compose(outer: TruncatedSeries, inner: TruncatedSeries) -> SeriesComposeResult:
    """Compose two series F(G(x)) modulo x^N when G(0) = 0."""
    SeriesComposeRequest(outer=as_input_series(outer), inner=as_input_series(inner))
    return compute_compose(outer, inner)


def reversion(series: TruncatedSeries) -> SeriesReversionResult:
    """Compute the compositional inverse of a series modulo x^N."""
    SeriesReversionRequest(
        variable=series.variable,
        truncation_order=series.truncation_order,
        coefficients=series.coefficients,
    )
    return compute_reversion(series)


def derivative(series: TruncatedSeries) -> SeriesDerivativeResult:
    """Formal derivative of a series."""
    return compute_derivative(as_input_series(series))


def integral_zero_constant(
    series: TruncatedSeries, output_order: int
) -> SeriesIntegralResult:
    """Zero-constant formal antiderivative with output_order coefficients."""
    request = SeriesIntegralRequest(
        series=as_input_series(series),
        output_order=output_order,
    )
    return compute_integral(series, request.output_order)


def truncate(series: TruncatedSeries, target_order: int) -> SeriesTruncateResult:
    """Truncate an admitted source series to a smaller bounded order."""
    request = SeriesTruncateRequest(
        series=TruncateSourceSeries(
            variable=series.variable,
            truncation_order=series.truncation_order,
            coefficients=series.coefficients,
        ),
        target_order=target_order,
    )
    return compute_truncate(request.series, request.target_order)


def identity_check(
    left: TruncatedSeries, right: TruncatedSeries
) -> SeriesIdentityCheckResult:
    """Check whether two series are equal modulo x^N."""
    _SeriesIdentityCheckRequest(
        left=as_input_series(left),
        right=as_input_series(right),
    )
    return compute_identity_check(left, right)


from_polynomial = compute_from_polynomial


def to_polynomial(series: TruncatedSeries) -> SeriesToPolynomialResult:
    """Return the canonical truncated polynomial representative of the series."""
    return compute_to_polynomial(as_input_series(series))


__all__ = [
    "TruncatedSeries",
    "add",
    "compose",
    "derivative",
    "divide",
    "from_polynomial",
    "identity_check",
    "integral_zero_constant",
    "inverse",
    "multiply",
    "power",
    "reversion",
    "scalar_multiply",
    "subtract",
    "to_polynomial",
    "truncate",
]
