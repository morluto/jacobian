# First-order differential GCRD

`ore.operator.gcrd.compute` accepts two operators in the existing
`QQ(x)<D>` carrier, restricted to differential order at most one and rational
constant coefficients. It returns their monic greatest common right divisor,
left cofactors, and left Bézout coefficients. All values retain the `x` axis
and use the existing exact rational-function representation.

For nonzero first-order inputs

\[
A=aD+b,\qquad B=cD+d,
\]

the operation forms the exact rational constant `bc-ad`. If it is nonzero,
\[
\frac{c}{bc-ad}A-\frac{a}{bc-ad}B=1,
\]
so the operators have no nonunit common right divisor. If it is zero, then
`b/a=d/c`; the normalized operator `D+b/a` is a common right divisor and both
inputs are left scalar multiples of it. Since both inputs have order one, no
common right divisor can have larger order. These arguments establish the
greatest-divisor postcondition in the ambient `QQ(x)<D>` ring, not just in the
constant-coefficient subring. Order-zero and zero inputs use the field-unit
convention; the zero-zero pair returns zero.

The operation preflights order, coefficient type, rational scalar height, and
the output scalar carrier before rational arithmetic. With input scalar height
`h`, every computed rational scalar has at most `5h+2` decimal digits. The
accepted input limit is 20 digits, below the shared 128-digit rational
coefficient carrier. No CAS backend is needed for this closed-form slice.

Higher-order inputs and nonconstant `QQ(x)` coefficients are rejected as
outside this operation's current exact domain. They need an admitted Ore
Euclidean algorithm, division in `QQ(x)<D>`, and corresponding coefficient
growth bounds; polynomial-coefficient monic division alone does not supply
those semantics.
