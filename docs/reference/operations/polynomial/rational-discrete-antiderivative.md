# Rational discrete antiderivatives

`polynomial.rational.discrete_antiderivative.compute` returns the unique
canonical polynomial `Q` satisfying

```text
Q(0) = 0
Q(x + 1) - Q(x) = P(x)
```

for the selected variable `x`. The input is a sparse `RationalPolynomial` over
`QQ`; the selected variable must be one of its declared ordered axes. Other
axes remain coefficient parameters, so a multivariate source composes directly
with the ordinary polynomial consumers without rendering or reparsing an
expression.

The result retains the request, the normalized antiderivative, and the exact
reconstructed difference. The reconstruction is a mathematical witness for
the defining relation, not a generic verification envelope. Empty and zero
sources preserve their declared axes and return the zero polynomial.

Admission is performed before the triangular exact solve. It bounds the
selected-axis output support, exponent increase, the current deterministic
quadratic coefficient-slice work envelope, common-denominator growth,
intermediate rational height, and the final canonical coefficient height. The
current pure-Python kernel admits at most 1,000,000 charged exact updates; a
larger degree requires a future maintained dense backend or a new bounded
algorithm regime. The implementation uses a deterministic binomial/monomial
triangular solve; no symbolic expression parser or backend object crosses the
public boundary.

For the motivating source

```text
P(k, N) = k^2 (N-k)^2
```

the caller can evaluate the returned `Q(k, N)` at `k=N` and `k=1` to obtain
the finite sum over `k=1,...,N-1`; the operation itself returns only the
reusable polynomial and does not encode that theorem-specific endpoint
workflow.

[Polynomial operations index](index.md) · [Python API](../../python-api.md)
