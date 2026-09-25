# Function-field genus

[Number theory operations](index.md) · [Tool surface](../../tools.md)

`function_field.genus.compute` returns the genus of either `GF(p)(x)` or a
supported quadratic hyperelliptic function field. A hyperelliptic input must
have odd prime characteristic and exact monic presentation
`GF(p)(x)[y]/(y^2-f(x))`, where `f` is a squarefree polynomial of degree 3
through the admitted degree bound 12. Rational-function coefficients,
characteristic two, repeated roots, and other algebraic extensions are not in
this operation's domain. The existing `FiniteFunctionField` and
`FiniteFunctionFieldElement` values preserve the equation and exact parent;
field arithmetic can therefore consume the same parent without an isomorphism
or lossy curve-to-field conversion.

The operation admits the field carrier's characteristic, coefficient degrees,
extension degree, and estimated work against the function-field resource
envelope. It then checks the normalized quadratic equation and branch
polynomial squarefreeness. These checks prove the needed algebraic properties
for this family: a simple root remains a simple root over the algebraic
closure, so `f` is not a square in `overline{GF(p)}(x)` and the quadratic
extension is geometrically irreducible; odd characteristic makes the defining
quadratic separable in `y`. The
smooth projective model maps with degree two to `P^1`; its branch points are
the roots of `f`, together with infinity exactly when `deg(f)` is odd. In odd
characteristic Riemann–Hurwitz therefore gives
`2g - 2 = -4 + r`, hence `g = floor((deg(f)-1)/2)`. The formula is the usual
hyperelliptic genus formula; the exact general framework is the
[Stacks Project genus definition](https://stacks.math.columbia.edu/tag/0BY6)
and [Riemann–Hurwitz theorem](https://stacks.math.columbia.edu/tag/0C1D).

The computation does not invent curve places or claim that the existing
`FunctionFieldPlace` represents points on this model. Current place,
valuation, residue, principal-divisor, and Riemann–Roch operations remain
restricted to `GF(p)(x)`. A useful next representation is a typed curve-point
place retaining the coordinates and residue field, together with local
uniformizer data; merely attaching an `(x,y)` pair to the current irreducible
polynomial-in-`x` place would be unsound.
