# Rational quadratic-form scaling

[Documentation home](../../../index.md) · [Tool surface](../../tools.md) · [Operation references](../index.md) · [This domain](index.md)

`quadratic_form.scale.compute` multiplies every diagonal and cross coefficient of
a rational quadratic form by one exact rational factor, on the same ordered
coordinate axis. For `Q(x) = a_ii x_i^2 + a_ij x_i x_j` and an exact
`f = p/q`, the result is `f * Q` coefficientwise, so the axis and every
coefficient slot are preserved.

A cross term whose scaled numerator cancels to zero is dropped, which is the
only way the result can have fewer cross terms than the source. Cancellation is
performed with `gcd` on the four numerator and denominator factors *before*
multiplying, so a reducible factor such as `2/2` leaves the coefficient exactly
unchanged rather than passing through a division that is not exact at every
intermediate step.

Admission charges the ordered axis length, the retained coefficient support
(diagonal plus cross terms), the factor's own digit width, and the exact
numerator and denominator growth of every product. Growth is checked with
integer division, so a product that would exceed the coefficient digit bound is
refused before the product is formed rather than after.

Because the operation is linear, the result is checkable against an independent
evaluation: for any point `x` over `QQ`, `Q_scaled(x) = f * Q(x)`. The test
suite verifies that identity by cross-multiplication over the integers involved,
so it holds exactly without floating point.

`quadratic_form.direct_sum.compute` combines forms on disjoint axes, and
`quadratic_form.variable_change.compute` composes a basis change with a form.
Scaling composes with both, since it fixes the axis and rescales coefficients
only.
