"""Domain-owned exact truncated formal power series operations.

All kernel functions operate on the immutable :class:`TruncatedSeries`
contract value defined in :mod:`jacobian.contracts.formal_power_series`.
Arithmetic is exact rational (Python ``Fraction``); no CAS series object
crosses the boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from jacobian.canonical import format_canonical_integer
from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.formal_power_series import (
    MAX_TRUNCATION_ORDER,
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

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _wire(value: Fraction) -> CanonicalRational:
    return CanonicalRational(
        num=format_canonical_integer(value.numerator),
        den=format_canonical_integer(value.denominator),
    )


def _series_fractions(series: TruncatedSeries) -> list[Fraction]:
    return [c.as_fraction() for c in series.coefficients]


def _series_result(
    variable: str, order: int, coeffs: Sequence[Fraction]
) -> TruncatedSeries:
    if len(coeffs) != order:
        raise ValueError("internal coefficient length does not match truncation order")
    return TruncatedSeries(
        variable=variable,
        truncation_order=order,
        coefficients=tuple(_wire(c) for c in coeffs),
    )


def _cauchy_convolve(
    a: Sequence[Fraction], b: Sequence[Fraction], n: int
) -> list[Fraction]:
    """c_k = sum_{i=0}^{k} a_i * b_{k-i} for 0 <= k < n."""
    return [sum(a[i] * b[k - i] for i in range(k + 1)) for k in range(n)]  # type: ignore[misc]


def _require_matching(
    left: TruncatedSeries, right: TruncatedSeries, label: str
) -> None:
    if left.variable != right.variable:
        raise ValueError(f"{label} must share the same variable")
    if left.truncation_order != right.truncation_order:
        raise ValueError(f"{label} must share the same truncation order")


# ---------------------------------------------------------------------------
# Arithmetic: add / subtract / multiply / scalar multiply
# ---------------------------------------------------------------------------


def compute_add(
    left: TruncatedSeries, right: TruncatedSeries
) -> SeriesArithmeticResult:
    """Add two series coefficientwise modulo x^N."""
    _require_matching(left, right, "operands")
    n = left.truncation_order
    a = _series_fractions(left)
    b = _series_fractions(right)
    return SeriesArithmeticResult(
        result=_series_result(left.variable, n, [a[i] + b[i] for i in range(n)])
    )


def compute_subtract(
    left: TruncatedSeries, right: TruncatedSeries
) -> SeriesArithmeticResult:
    """Subtract two series coefficientwise modulo x^N."""
    _require_matching(left, right, "operands")
    n = left.truncation_order
    a = _series_fractions(left)
    b = _series_fractions(right)
    return SeriesArithmeticResult(
        result=_series_result(left.variable, n, [a[i] - b[i] for i in range(n)])
    )


def compute_multiply(
    left: TruncatedSeries, right: TruncatedSeries
) -> SeriesMultiplyResult:
    """Multiply two series modulo x^N via Cauchy convolution."""
    _require_matching(left, right, "operands")
    n = left.truncation_order
    a = _series_fractions(left)
    b = _series_fractions(right)
    result = _cauchy_convolve(a, b, n)
    return SeriesMultiplyResult(
        result=_series_result(left.variable, n, result),
        convolution_ledger=tuple(_wire(c) for c in result),
    )


def compute_scalar_multiply(
    series: TruncatedSeries, scalar: CanonicalRational
) -> SeriesScalarMultiplyResult:
    """Multiply a series by an exact rational scalar."""
    a = _series_fractions(series)
    scalar_val = scalar.as_fraction()
    n = series.truncation_order
    result = [a[i] * scalar_val for i in range(n)]
    return SeriesScalarMultiplyResult(result=_series_result(series.variable, n, result))


# ---------------------------------------------------------------------------
# Power (binary exponentiation)
# ---------------------------------------------------------------------------


def compute_power(series: TruncatedSeries, exponent: int) -> SeriesPowerResult:
    """Compute series^exponent via binary exponentiation modulo x^N."""
    if exponent < 0:
        raise ValueError("negative exponents are not supported in this operation")
    n = series.truncation_order
    a = _series_fractions(series)

    result_coeffs = [Fraction(1)] + [Fraction(0)] * (n - 1)
    base = a[:]
    multiplications = 0
    e = exponent
    while e > 0:
        if e & 1:
            result_coeffs = _cauchy_convolve(result_coeffs, base, n)
            multiplications += 1
        e >>= 1
        if e > 0:
            base = _cauchy_convolve(base, base, n)
            multiplications += 1

    return SeriesPowerResult(
        result=_series_result(series.variable, n, result_coeffs),
        multiplication_count=multiplications,
    )


# ---------------------------------------------------------------------------
# Inverse
# ---------------------------------------------------------------------------


def compute_inverse(series: TruncatedSeries) -> SeriesInverseResult:
    """Compute the multiplicative inverse of a series modulo x^N.

    Requires a_0 != 0.  Computes B such that A*B = 1 (mod x^N) via the
    standard recurrence: b_0 = 1/a_0; b_n = -(1/a_0) * sum_{i=1}^{n} a_i b_{n-i}.
    """
    n = series.truncation_order
    a = _series_fractions(series)
    if a[0] == 0:
        raise ValueError("series with zero constant term is not a unit")
    inv = [Fraction(0)] * n
    inv[0] = Fraction(1) / a[0]
    for k in range(1, n):
        s = Fraction(0)
        for i in range(1, k + 1):
            s += a[i] * inv[k - i]
        inv[k] = -inv[0] * s
    # Compute residual A*B - 1
    product = _cauchy_convolve(a, inv, n)
    product[0] -= Fraction(1)
    return SeriesInverseResult(
        result=_series_result(series.variable, n, inv),
        residual_coefficients=tuple(_wire(c) for c in product),
    )


# ---------------------------------------------------------------------------
# Divide
# ---------------------------------------------------------------------------


def compute_divide(
    numerator: TruncatedSeries, denominator: TruncatedSeries
) -> SeriesDivideResult:
    """Compute Q = A / B mod x^N where b_0 != 0."""
    _require_matching(numerator, denominator, "operands")
    if denominator.coefficients[0].as_fraction() == 0:
        raise ValueError("denominator with zero constant term is not a unit")
    n = numerator.truncation_order
    a = _series_fractions(numerator)
    b = _series_fractions(denominator)

    inv = compute_inverse(denominator)
    b_inv = _series_fractions(inv.result)
    q = _cauchy_convolve(a, b_inv, n)
    bq = _cauchy_convolve(b, q, n)
    residual = [bq[i] - a[i] for i in range(n)]
    return SeriesDivideResult(
        quotient=_series_result(numerator.variable, n, q),
        residual_coefficients=tuple(_wire(c) for c in residual),
    )


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


def compute_compose(
    outer: TruncatedSeries, inner: TruncatedSeries
) -> SeriesComposeResult:
    """Compute F(G(x)) mod x^N where G(0) = 0.

    Composes by iteratively computing G^k (powers) and multiplying by f_k:
    F(G) = sum_{k=0}^{N-1} f_k * G^k mod x^N.
    """
    _require_matching(outer, inner, "outer and inner series")
    if inner.coefficients[0].as_fraction() != 0:
        raise ValueError(
            "inner series must have zero constant term for composition with a finite prefix"
        )
    n = outer.truncation_order
    f = _series_fractions(outer)
    g = _series_fractions(inner)

    g_power = [Fraction(1)] + [Fraction(0)] * (n - 1)  # G^0 = 1
    result = [f[0] * g_power[i] for i in range(n)]
    for k in range(1, n):
        g_power = _cauchy_convolve(g_power, g, n)  # G^k
        for i in range(n):
            result[i] += f[k] * g_power[i]
    return SeriesComposeResult(result=_series_result(outer.variable, n, result))


# ---------------------------------------------------------------------------
# Reversion (compositional inverse)
# ---------------------------------------------------------------------------


def compute_reversion(series: TruncatedSeries) -> SeriesReversionResult:
    """Compute the compositional inverse G(x) of F(x) mod x^N.

    Requires F(0) = 0 and F'(0) = f_1 != 0.  Returns G such that:
      - F(G(x)) = x mod x^N (left identity)
      - G(F(x)) = x mod x^N (right identity)

    Uses a coefficient-by-coefficient recurrence to determine coefficients
    of G one at a time.
    """
    n = series.truncation_order
    f = _series_fractions(series)
    if f[0] != 0:
        raise ValueError("reversion requires zero constant term")
    if n < 2:
        raise ValueError("reversion requires truncation order >= 2")
    if f[1] == 0:
        raise ValueError("reversion requires nonzero linear coefficient")

    # G(x) = g_0 + g_1 x + g_2 x^2 + ... such that F(G(x)) = x mod x^N
    # g_0 = 0, g_1 = 1/f_1
    g = [Fraction(0)] * n
    g[1] = Fraction(1) / f[1]
    # Compute G powers up to N-1 and solve for g_k one at a time
    for k in range(2, n):
        target = Fraction(0)
        # For j = 2 to k:
        #   compute G^j using g_0..g_{k-1} and read off coefficient of x^k
        g_powers = [[Fraction(0)] * (k + 1) for _ in range(k + 1)]
        g_powers[0] = [Fraction(1)] + [Fraction(0)] * k  # G^0 = 1
        g_powers[1] = [*list(g[:k]), Fraction(0)]
        for j in range(2, k + 1):
            g_powers[j] = [
                sum(g_powers[j - 1][m] * g_powers[1][i - m] for m in range(i + 1))  # type: ignore[misc]
                for i in range(k + 1)
            ]
        known = Fraction(0)
        for j in range(2, k + 1):
            fj = f[j] if j < len(f) else Fraction(0)
            known += fj * g_powers[j][k]
        g[k] = (target - known) / f[1]

    # Compute residuals F(G) and G(F)
    fg = compute_compose(
        _series_result(series.variable, n, f),
        _series_result(series.variable, n, g),
    )
    fg_coeffs = _series_fractions(fg.result)
    left_residual = [
        fg_coeffs[i] - (Fraction(1) if i == 1 else Fraction(0)) for i in range(n)
    ]

    gf = compute_compose(
        _series_result(series.variable, n, g),
        _series_result(series.variable, n, f),
    )
    gf_coeffs = _series_fractions(gf.result)
    right_residual = [
        gf_coeffs[i] - (Fraction(1) if i == 1 else Fraction(0)) for i in range(n)
    ]

    return SeriesReversionResult(
        result=_series_result(series.variable, n, g),
        left_residual=tuple(_wire(c) for c in left_residual),
        right_residual=tuple(_wire(c) for c in right_residual),
    )


# ---------------------------------------------------------------------------
# Derivative
# ---------------------------------------------------------------------------


def compute_derivative(series: TruncatedSeries) -> SeriesDerivativeResult:
    """Formal derivative: b_n = (n+1) * a_{n+1}.

    Output order convention: max(N-1, 1).
    """
    n = series.truncation_order
    a = _series_fractions(series)
    output_order = max(n - 1, 1)
    if n == 1:
        result = [Fraction(0)]
    else:
        result = [Fraction((i + 1) * a[i + 1]) for i in range(output_order)]
    return SeriesDerivativeResult(
        result=_series_result(series.variable, output_order, result)
    )


# ---------------------------------------------------------------------------
# Integral (zero constant)
# ---------------------------------------------------------------------------


def compute_integral(
    series: TruncatedSeries, output_order: int
) -> SeriesIntegralResult:
    """Zero-constant formal antiderivative with output_order coefficients.

    B(x) = sum_{n=1}^{output_order-1} (a_{n-1} / n) x^n + 0
    (the constant is zero).
    """
    n = series.truncation_order
    a = _series_fractions(series)
    if output_order > n + 1:
        raise ValueError("output_order must not exceed source_order + 1")
    result = [Fraction(0)] * output_order
    for i in range(1, output_order):
        if i - 1 < n:
            result[i] = a[i - 1] / i
    return SeriesIntegralResult(
        result=_series_result(series.variable, output_order, result)
    )


# ---------------------------------------------------------------------------
# Truncate
# ---------------------------------------------------------------------------


def compute_truncate(
    series: TruncatedSeries, target_order: int
) -> SeriesTruncateResult:
    """Truncate a series to a smaller order."""
    if target_order > series.truncation_order:
        raise ValueError("target_order must not exceed source truncation order")
    if target_order > MAX_TRUNCATION_ORDER:
        raise ValueError("target_order exceeds the public bound")
    a = _series_fractions(series)
    result = a[:target_order]
    return SeriesTruncateResult(
        result=_series_result(series.variable, target_order, result)
    )


# ---------------------------------------------------------------------------
# Identity check
# ---------------------------------------------------------------------------


def compute_identity_check(
    left: TruncatedSeries, right: TruncatedSeries
) -> SeriesIdentityCheckResult:
    """Check if two series are equal mod x^N."""
    _require_matching(left, right, "operands")
    a = _series_fractions(left)
    b = _series_fractions(right)
    for i in range(left.truncation_order):
        if a[i] != b[i]:
            diff = a[i] - b[i]
            return SeriesIdentityCheckResult(
                status="NOT_EQUAL",
                first_differing_index=i,
                exact_difference=_wire(diff),
            )
    return SeriesIdentityCheckResult(status="EQUAL_MOD_X_TO_N")


# ---------------------------------------------------------------------------
# Polynomial conversions
# ---------------------------------------------------------------------------


def compute_from_polynomial(
    variable: str,
    coefficients: Sequence[CanonicalRational],
    truncation_order: int,
) -> SeriesFromPolynomialResult:
    """Convert a dense rational coefficient tuple into a truncated series."""
    from jacobian.contracts.formal_power_series import SeriesFromPolynomialRequest

    request = SeriesFromPolynomialRequest(
        variable=variable,
        coefficients=tuple(coefficients),
        truncation_order=truncation_order,
    )
    coeffs = [c.as_fraction() for c in request.coefficients]
    return SeriesFromPolynomialResult(
        result=_series_result(request.variable, request.truncation_order, coeffs)
    )


def compute_to_polynomial(series: TruncatedSeries) -> SeriesToPolynomialResult:
    """Return the canonical truncated polynomial representative of the series."""
    return SeriesToPolynomialResult(
        result=TruncatedSeries(
            variable=series.variable,
            truncation_order=series.truncation_order,
            coefficients=series.coefficients,
        )
    )
