# Rational functions to Laurent prefixes

`local_series.from_rational_function_at_point.compute` expands a canonical
univariate rational function over `QQ` around a rational point `a`, using
`t = x - a`. Its `precision` is the exclusive exponent cutoff: the returned
window contains the exact coefficients with exponent less than that cutoff.
The omitted tail is unknown.

`local_series.at_infinity.compute` uses the same window carrier with an
explicit `place: INFINITY` and reciprocal parameter `t = 1/x`. Its center is
canonically zero, and Laurent arithmetic preserves this place and rejects
mixing it with a finite-point window.

The operation factors the exact vanishing orders of numerator and denominator
at `a`; their difference is the Laurent valuation. It then divides the two
remaining power series by an exact coefficient recurrence. A pole at `a` is
therefore represented by negative exponents. The canonical window is returned
in the `series` field, so its coefficients and parent compose with Laurent
arithmetic.

Both operations admit at most 256 source terms, degree 128, coefficient
components of at most 128 decimal digits, and 4096 output coefficients. The
finite-point operation additionally caps each center component at 30 decimal
digits. Recurrence coefficient height and serialized output size are admitted
before expansion. These operations do not expand multivariate functions or
algebraic centers.

The source-bound result reports numerator and denominator local orders, their
difference (the valuation), pole and zero orders, and the normalized unit
quotient prefix when it is determined by the requested precision. It also
reports the exact residual cutoff: for the returned prefix `f_N`, the source
relation `Q*f_N - P` vanishes below `product_residual_precision` in the local
parameter. The `series` field is the composable Laurent value.
