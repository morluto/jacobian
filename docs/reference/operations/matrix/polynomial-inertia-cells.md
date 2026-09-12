# Exact inertia cells for polynomial matrices

[Documentation home](../../../index.md) · [Matrix operations](index.md) ·
[Tool surface](../../tools.md)

`matrix.polynomial.inertia_cells.compute` returns the complete exact inertia
profile of one symmetric matrix over `QQ[t]` on a closed rational interval.
The result is source-bound: it echoes the polynomial matrix and interval and
alternates singleton point cells with open interval cells. Every domain
endpoint is a point cell, every rank-drop parameter is retained as a point
cell, and each cell carries `(n_positive, n_negative, n_zero)` whose sum is the
matrix order.

The input uses the canonical `RationalPolynomialMatrix` and
`ClosedRationalInterval` values. Matrix entries must use one univariate
variable and the matrix must be square and symmetric. Empty and zero matrices,
constant matrices, tangential roots such as `t²`, persistent nullspaces, and a
singleton interval are supported within the operation's admitted envelope.

Irrational transition parameters are represented by a primitive real-algebraic
root identity (`RealAlgebraicValue`) together with a rational isolating
interval. Rational transitions use canonical rational values and singleton
isolating intervals. The partition is exact; it is not a sampled eigenvalue
plot. In particular, `diag(t² - 2, 0)` on `[-2, 2]` retains both `±√2` even
though its determinant is identically zero.

The kernel first bounds the source, characteristic-polynomial growth,
transition isolation, exact coefficient-sign work, sampling, and output size.
It then processes disconnected support blocks independently, isolates all roots
of the lowest nonzero characteristic coefficient, and uses exact specialization
at point cells. The eigenvalue signs are obtained from the real-rooted
characteristic polynomial's Descartes sign variations; no floating-point
eigenvalue calculation is used. A request that exceeds any admitted bound or
does not complete exactly is rejected rather than returning a partial profile.

The operation is available natively as
`jacobian.math.matrices.inertia_cells.compute_inertia_cells` and through
`math.find`/`math.run`. Its catalog example is the persistent-nullspace
`diag(t² - 2, 0)` fixture.
