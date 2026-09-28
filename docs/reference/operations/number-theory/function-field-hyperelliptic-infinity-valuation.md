# Hyperelliptic valuation at infinity

[Number theory operations](index.md) · [Tool surface](../../tools.md)

`function_field.hyperelliptic_infinity_place.valuation.compute` evaluates a
function at the unique point above infinity for a squarefree odd-degree model
`y^2=f(x)` over an odd prime field. The typed place retains the complete
`FiniteFunctionField` parent and its degree-one residue field `GF(p)`. It
composes directly with existing `FiniteFunctionFieldElement` values.

For `d = deg(f)` odd, the pole orders at this place are `v(x)=-2` and
`v(y)=-d`. If an element is `a(x)+b(x)y`, then the valuations of its two
nonzero terms have opposite parity: `2 v_infinity(a)` is even, while
`2 v_infinity(b)-d` is odd. They cannot cancel, so the answer is the minimum
of those terms' valuations. The rational-function orders are computed exactly
from normalized numerator and denominator degrees. This also makes the
valuation multiplicative for products of represented elements.

For a basis and infinity-place treatment of odd-degree hyperelliptic function
fields, see Michael Stoll's primary lecture notes, [Arithmetic of Hyperelliptic
Curves, §2](https://mathe2.uni-bayreuth.de/stoll/teaching/ArithHypKurven-SS2014/Skript-ArithHypCurves-pub-print.pdf).

This carrier deliberately covers only the unique rational point at infinity
for odd `deg(f)` from 3 through the existing coefficient bound. Even-degree
models can have two geometric points over infinity or a single point with a
larger residue field, so they need a different place value. The operation does
not yet unify affine and infinite hyperelliptic places into divisors; divisor,
residue, and Riemann–Roch carriers still target the rational-function-field
place representation.
