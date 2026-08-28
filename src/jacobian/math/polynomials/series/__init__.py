"""Supported exact truncated formal-power-series API.

Native callables take canonical series values and semantic scalar parameters.
They delegate to owner-local operation entrypoints that admit once before
running the direct kernels; wire request models remain an implementation detail
of the catalog and MCP projection.
"""

from jacobian._exact import CanonicalRational
from jacobian.math.polynomials.series._models import (
    SeriesArithmeticResult,
    SeriesComposeResult,
    SeriesDerivativeResult,
    SeriesDivideResult,
    SeriesFromPolynomialResult,
    SeriesIdentityCheckResult,
    SeriesIntegralResult,
    SeriesInverseResult,
    SeriesMultiplyResult,
    SeriesPowerResult,
    SeriesReversionResult,
    SeriesScalarMultiplyResult,
    SeriesToPolynomialResult,
    SeriesTruncateResult,
    TruncatedSeries,
)
from jacobian.math.polynomials.series._operations import (
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
    return compute_add(left, right)


def subtract(left: TruncatedSeries, right: TruncatedSeries) -> SeriesArithmeticResult:
    """Subtract two series coefficientwise modulo x^N."""
    return compute_subtract(left, right)


def multiply(left: TruncatedSeries, right: TruncatedSeries) -> SeriesMultiplyResult:
    """Multiply two series modulo x^N via Cauchy convolution."""
    return compute_multiply(left, right)


def scalar_multiply(
    series: TruncatedSeries, scalar: CanonicalRational
) -> SeriesScalarMultiplyResult:
    """Multiply a series by an exact rational scalar."""
    return compute_scalar_multiply(series, scalar)


def power(series: TruncatedSeries, exponent: int) -> SeriesPowerResult:
    """Compute series^exponent via binary exponentiation modulo x^N."""
    return compute_power(series, exponent)


def inverse(series: TruncatedSeries) -> SeriesInverseResult:
    """Invert a truncated unit series modulo x^N."""
    return compute_inverse(series)


def divide(
    numerator: TruncatedSeries, denominator: TruncatedSeries
) -> SeriesDivideResult:
    """Divide two series modulo x^N when the denominator is a unit."""
    return compute_divide(numerator, denominator)


def compose(outer: TruncatedSeries, inner: TruncatedSeries) -> SeriesComposeResult:
    """Compose two series F(G(x)) modulo x^N when G(0) = 0."""
    return compute_compose(outer, inner)


def reversion(series: TruncatedSeries) -> SeriesReversionResult:
    """Compute the compositional inverse of a series modulo x^N."""
    return compute_reversion(series)


def derivative(series: TruncatedSeries) -> SeriesDerivativeResult:
    """Formal derivative of a series."""
    return compute_derivative(series)


def integral_zero_constant(
    series: TruncatedSeries, output_order: int
) -> SeriesIntegralResult:
    """Zero-constant formal antiderivative with output_order coefficients."""
    return compute_integral(series, output_order)


def truncate(series: TruncatedSeries, target_order: int) -> SeriesTruncateResult:
    """Truncate an admitted source series to a smaller bounded order."""
    return compute_truncate(series, target_order)


def identity_check(
    left: TruncatedSeries, right: TruncatedSeries
) -> SeriesIdentityCheckResult:
    """Check whether two series are equal modulo x^N."""
    return compute_identity_check(left, right)


def from_polynomial(
    variable: str,
    coefficients: tuple[CanonicalRational, ...],
    truncation_order: int,
) -> SeriesFromPolynomialResult:
    """Construct a truncated series from canonical dense coefficients."""
    series = TruncatedSeries(
        variable=variable,
        truncation_order=truncation_order,
        coefficients=coefficients,
    )
    return compute_from_polynomial(
        series.variable, series.coefficients, series.truncation_order
    )


def to_polynomial(series: TruncatedSeries) -> SeriesToPolynomialResult:
    """Return the canonical truncated polynomial representative of the series."""
    return compute_to_polynomial(series)


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
