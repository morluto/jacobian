# Rational affine hyperelliptic valuations

`function_field.hyperelliptic_affine_place.valuation.compute` computes the
normalized discrete valuation of an exact element at a `GF(p)`-rational affine
point on an admitted odd-characteristic model `y^2 = f(x)`, with squarefree
polynomial `f`. The typed point retains the exact function-field parent,
coordinates, residue parent `GF(p)`, and a local parameter.

At a point `(a,b)` with `b != 0`, the operation uses `x-a` as uniformizer and
solves `y(x)^2=f(x)` in the corresponding formal power-series ring. At a branch
point `(a,0)`, squarefreeness gives `f'(a) != 0`; the operation uses `y` as
uniformizer and solves `f(x(y))=y^2`. For an element represented as
`(u(x)+v(x)y)/d(x)`, the norm `u(x)^2-v(x)^2 f(x)` bounds the finite expansion
needed to find its exact order. Resource admission precedes that expansion.
Both function-field valuation operations return finite results in the closed
tagged shape `{ "kind": "FINITE", "value": n }`, including when `n=0`. The
zero function uses the distinct shape
`{ "kind": "POSITIVE_INFINITY" }`; this branch has no numeric field and does
not serialize as `null`.

The rational-function-field `function_field.place.valuation.compute` operation
uses the same tagged result for finite and infinite rational-function-field
places.

This slice covers only rational affine points of the supported hyperelliptic
model. It does not represent points over extension residue fields or points at
infinity, and it does not infer either from a rational-function-field place.

The local-ring definition and normalized discrete valuation at a nonsingular
curve point are standard; see [Handbook of Elliptic and Hyperelliptic Curve
Cryptography, chapter 4](https://www.hyperelliptic.org/HEHCC/chapters/chap04.pdf).
The choices `x-a` off the branch locus and `y` at a simple branch point are
also used in explicit hyperelliptic local expansions; see the
[hyperelliptic-curve arithmetic notes](https://mathe2.uni-bayreuth.de/stoll/teaching/ArithHypKurven-SS2019/Skript-ArithHypCurves-pub-screen.pdf).
