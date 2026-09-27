# Polynomials over rational cyclotomic fields

`CyclotomicPolynomial` represents a sparse polynomial over the explicitly
declared field `QQ(zeta_n) = QQ[x]/(Phi_n(x))`. Its coefficients reuse
`RationalCyclotomicElement`, with exactly `phi(n)` ascending power-basis
coordinates. The value retains the field, ordered variables, and canonical
descending lexicographic monomial support.

`polynomial.cyclotomic_coefficient.embed.compute` maps a
`RationalPolynomial` coefficientwise through the canonical inclusion
`QQ -> QQ(zeta_n)`. A rational coefficient `a` maps to `(a, 0, ..., 0)` in the
fixed power basis. Variables and exponents are unchanged. The zero polynomial
therefore stays empty while retaining both its variable axis and selected
field. The order-one field is `QQ` itself and has degree one.

This value is a target representation prerequisite for exact coefficient
transforms such as Dirichlet-character polynomial twists. The embedding
operation claims only the coefficient-ring map; it makes no claim about a
polynomial's origin or other identities.

The current carrier uses the shared rational cyclotomic field envelope: field
order at most 128, and each coordinate numerator and denominator at most 256
decimal digits. A polynomial contains at most 16,384 exact rational
coordinates across all coefficients, at most 4,096 monomials, and at most 10
MiB of admitted embedding output. Admission counts field degree times source
term count, checks coefficient height, and bounds the complete output before
constructing coefficients.

The exact convention is the standard quotient presentation `QQ[x]/(Phi_n)`;
PARI/GP documents the nth cyclotomic polynomial `Phi_n` as `polcyclo(n)` in its
[polynomial reference](https://pari.math.u-bordeaux.fr/dochtml/html/Polynomials_and_power_series.html).
