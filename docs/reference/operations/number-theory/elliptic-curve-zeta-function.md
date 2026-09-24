# Finite-field elliptic zeta function

`elliptic_curve.finite_field.zeta.compute` returns the full Hasse-Weil zeta
function of an admitted nonsingular short-Weierstrass curve over a finite
field `F_q`:

```text
Z(E/F_q, T) = (1 - a*T + q*T^2) / ((1 - T)(1 - q*T))
a = q + 1 - #E(F_q)
```

This is the genus-one case of the curve zeta-function formula; see [Hoffman's
notes on elliptic curves over finite fields](https://www.math.lsu.edu/~hoffman/papers/elmod.pdf).
The operation reuses the exact zeta-numerator computation and returns the
fraction as Jacobian's canonical `RationalFunction` over `QQ(T)`, together
with the exact curve, base-field point count, and Frobenius trace.

The denominator is monic in the polynomial carrier's descending-degree order.
Thus the serialized rational function stores numerator `Z` and denominator
`D`, each divided by `q`; this gives the same rational function while meeting
the carrier's monic-denominator convention. The fraction is reduced: its
numerator does not vanish at `T=1` or `T=1/q`, since those values are
`#E(F_q)` and `#E(F_q)/q`, respectively, and a nonsingular elliptic curve has
at least its identity point.

The operation inherits the numerator kernel's preflight: `q <= 4096`, with
the finite-field presentation and character-sum work admitted before the
count. Its output has at most three terms in each polynomial, degree two, and
coefficients with at most four decimal digits. For
`y^2 = x^3 + x + 1` over `F_5`, the function is
`(1 + 3*T + 5*T^2)/((1-T)(1-5*T))`.

[Zeta numerator](elliptic-curve-zeta-polynomial.md) ·
[Extension counts](elliptic-curve-extension-counts.md) ·
[Number theory operations](index.md)
