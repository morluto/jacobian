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

The operation performs the normal bounded function-field parent admission,
including separability and irreducibility, then checks the normalized
quadratic equation and the branch polynomial's squarefreeness. The branch
polynomial has a simple zero over the algebraic closure, so it is not a square
in `GF(p)(x)` and the quadratic extension remains geometrically integral. The
smooth projective model maps with degree two to `P^1`; its branch points are
the roots of `f`, together with infinity exactly when `deg(f)` is odd. In odd
characteristic Riemann–Hurwitz therefore gives
`2g - 2 = -4 + r`, hence `g = floor((deg(f)-1)/2)`. The formula is the usual
hyperelliptic genus formula; the exact general framework is the
[Stacks Project genus definition](https://stacks.math.columbia.edu/tag/0BY6)
and [Riemann–Hurwitz theorem](https://stacks.math.columbia.edu/tag/0C1D).

The genus computation does not construct place values. Separate operations
represent rational affine points and, for odd-degree models, the unique
rational point at infinity. These typed values retain the hyperelliptic field
parent and residue field; they are not instances of the rational-field
`FunctionFieldPlace`. Divisor, residue, principal-divisor, and Riemann–Roch
operations do not yet consume these curve-place values.
