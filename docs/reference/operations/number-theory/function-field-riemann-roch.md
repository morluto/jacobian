# Rational function-field Riemann–Roch spaces

[Documentation home](../../../index.md) · [Tool surface](../../tools.md) · [Operation references](../index.md) · [This domain](index.md)

[Number theory operations](index.md) · [Tool surface](../../tools.md)

`function_field.riemann_roch_space.compute` returns the complete space `L(D)`
for finite divisors over `GF(p)(x)`. It also supports divisors `m P∞` on the
odd-degree hyperelliptic model `y²=f(x)` when `p` is odd, `f` is squarefree,
and `3 ≤ deg(f) ≤ 11`. This extension-model slice accepts only the unique
infinite place; finite places and even-degree models remain outside its
contract. Multiplicities are limited to 4096 bits and the returned basis to
13 elements.

Write

```text
D = Σ_P n_P P + n_∞ ∞
h = ∏_P P(x)^n_P
d = deg(D) = Σ_P n_P deg(P) + n_∞
```

Finite places are represented by their monic irreducible polynomials. Negative
exponents in `h` are exact denominator factors. The operation uses
`div(P(x)) = P - deg(P)∞`, so `div(h) = Σ n_P P - (Σ n_P deg(P))∞`.
For `f=g/h`, the finite-place inequalities `(f)+D ≥ 0` hold exactly when
`g` is a polynomial, and the infinity inequality is exactly `deg(g) ≤ d`.
Consequently `L(D) = h⁻¹ GF(p)[x]_{≤d}` when `d ≥ 0`, and `L(D)={0}` when
`d < 0`; the canonical basis is `x^i/h` for `0≤i≤d`.

For a hyperelliptic model of odd degree `d=2g+1`, the unique point at infinity
has `v∞(x)=-2` and `v∞(y)=-d`. Therefore

```text
L(m P∞) = span {x^i : 2i≤m} ∪ {x^j y : 2j+d≤m}
```

with an empty basis for `m<0`. The two families have distinct pole-order
parities, so their valuations establish linear independence; each listed
element satisfies the required pole bound. This gives dimension
`floor(m/2)+1 + max(0, floor((m-d)/2)+1)` for `m≥0`, and agrees with
Riemann–Roch `m+1-g` for `m≥2g-1`.

The result value carries the admitted divisor, its dimension, and exact
`FiniteFunctionFieldElement` basis values. If the positive finite part of `h`
is `A` and its negative finite part is `B`, the operation forms `h=A/B` and
returns reduced fractions `x^i B/A`. The only possible common factor in these
fractions is a power of `x` shared with `A`; it is cancelled directly. Distinct
prime support makes the other numerator and denominator factors coprime.

The exact input profile is admitted before place factorization or polynomial
products. Positive and negative finite degree contributions are each limited
to 12 when a basis is required, the basis dimension is limited to 13, and the
largest reduced numerator and denominator degrees must fit the carrier's
degree-12 polynomial bound. Products over the prime field have at most 12
factor degrees per sign and at most `24 × 13² = 4056` coefficient multiply-add
steps. Coefficients are reduced modulo `p` after each operation, so their
height is bounded by the characteristic. If `d<0`, the operation returns the
empty basis without constructing `h`.

The hyperelliptic support does not yet include finite places, arbitrary
extension fields, or Riemann–Roch membership checks. Differential spaces and
canonical divisors also remain unsupported.
