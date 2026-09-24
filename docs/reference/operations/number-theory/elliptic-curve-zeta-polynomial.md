# Finite-field elliptic zeta numerator

`elliptic_curve.finite_field.zeta_polynomial.compute` returns the numerator
of the zeta function of a nonsingular short-Weierstrass curve over a supported
finite field:

```text
P(T) = 1 - a*T + q*T^2
a = q + 1 - #E(F_q)
```

The result carries the exact curve, its point count, trace, and a canonical
`IntegerPolynomial`. That polynomial serializes coefficients in descending
degree order, so the tuple is `(q, -a, 1)`. The operation derives `a` from an
exact quadratic-character sum over the declared field. It admits fields of
order at most 4096 and preflights arithmetic work and integer growth before
counting. Result construction checks the polynomial/count relation without
recounting the curve.

For `y^2 = x^3 + x + 1`, the result over `F5` is
`P(T) = 1 + 3*T + 5*T^2`; its extension count satisfies
`#E(F_(q^2)) = q^2 + 1 - (a^2 - 2q)`. The operation returns the zeta
numerator, not an extension-field presentation or point set.

[Finite-field elliptic extension counts](elliptic-curve-extension-counts.md) ·
[Number theory operations](index.md) · [Operation reference](../index.md)
