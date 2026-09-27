# Polynomial-coefficient shift-operator addition and scaling

The two operations use left-coefficient normal form

\[
P=\sum_i p_i(n)S^i,
\qquad S^i a(n)=a(n+i)S^i.
\]

`ore.shift.operator.add.compute` computes the exact sparse sum. Its bounded
coefficient domain is the closed subalgebra `QQ[n]<S>`: every coefficient must
be a polynomial with denominator one. The output merges equal shift exponents,
adds exact rational polynomial coefficients, removes zero polynomial
coefficients, and retains strictly increasing exponents.

`ore.operator.scalar_left_multiply.compute` left-multiplies a polynomial-
coefficient shift operator by a rational constant in `QQ`. The scalar is
represented as a constant rational function on the `QQ(n)` axis. A nonconstant
`QQ(n)` scalar or an operator with rational-function coefficients is outside
these operation contracts.

`ore.shift.operator.normalize.compute` writes an operator as `scale` times a
primitive integer-polynomial operator. It takes the positive gcd of all
coefficient numerators over the least common multiple of their denominators;
the sign of `scale` makes the coefficient at the greatest shift exponent and
then greatest polynomial degree positive. The zero operator maps to itself
with unit scale.

Both operations admit sparse work, coefficient digits, and complete serialized
results before rational arithmetic. Addition bounds each overlapping
polynomial coefficient using the exact cross-product numerator and product
denominator digit heights; scaling bounds every rational product. Because the
accepted coefficient values are canonical `RationalFunction` values and the
results remain polynomials over `QQ`, zero removal and canonical rational
coefficients provide the output normal form. This normalization does not clear
rational-function coefficient denominators or establish a minimal annihilator.

[Finite-prefix action](ore-shift-sequence-prefix.md) applies a shift operator
to exact sequence values with an explicit index origin and finite boundary
exclusions.

[Operation catalog](../../tools.md) · [Polynomial operations](index.md)
