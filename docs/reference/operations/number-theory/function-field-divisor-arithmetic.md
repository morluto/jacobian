# Rational function-field divisor arithmetic

`function_field.divisor.add.compute`, `function_field.divisor.negate.compute`,
and `function_field.divisor.scale.compute` implement the additive group and
integer scaling on finite divisors of the explicitly represented rational
function field `GF(p)(x)`. Finite places are monic irreducible polynomials in
`GF(p)[x]`; the infinite place has degree one. Every place is checked against
the same exact field parent before an operation returns its divisor.

Addition combines equal places and removes zero coefficients. Negation changes
each coefficient's sign. Scaling accepts an exact integer. The support is
limited to 256 places; input multiplicities and the scalar are limited to 4096
bits, and operation results are limited to 4097 bits for addition and 8192
bits for scaling.

These operations do not extend place arithmetic to algebraic function fields.
The current place value for an extension does not identify a prime ideal in its
maximal order, so that requires a stronger place representation.
