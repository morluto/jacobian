# Rational function-field Riemann–Roch membership

[Number theory operations](index.md) · [Tool surface](../../tools.md)

`function_field.riemann_roch.membership.compute` decides whether an element
`f` belongs to `L(D)` for a finite divisor `D` over the prime rational function
field `GF(p)(x)`. Its exact criterion is

```text
f ∈ L(D)  iff  v_P(f) + D(P) ≥ 0 for every place P.
```

For a nonzero rational function, the operation returns a row for every place in
the union of the supplied divisor support and the complete support of
`div(f)`. Finite places in `div(f)` are obtained by exact factorization of its
canonical numerator and denominator over `GF(p)`; the unique place at infinity
is included when its valuation is nonzero. Each row retains the place, the
function valuation, divisor multiplicity, and their exact sum. A negative sum
identifies a concrete place proving nonmembership. Places outside the union
have both contributions zero, so the returned profile decides the complete
valuation criterion. The associated principal divisor is the sum of the
orders of vanishing at prime divisors, as in the [Stacks Project's Weil divisor
definition](https://stacks.math.columbia.edu/tag/02AR).

Zero has valuation `+∞` at every place and belongs to every `L(D)`. It is
returned as the structural `IN_SPACE` branch with an empty profile; the zero
function has no principal divisor and is not passed to the principal-divisor
operation.

This contract is limited to `GF(p)(x)`, where every finite place is represented
by a monic irreducible polynomial and every nonzero rational function has
finitely many zeros and poles plus the single infinite place. The operation
admits at most 256 divisor places, degree-12 numerator and denominator
polynomials, a 281-place support profile, 5,000,000 factor-work units, and a
4 MiB conservative result bound. Algebraic extensions require their own
complete place and principal-divisor support before membership can be offered.
