# Rational affine hyperelliptic place enumeration

`function_field.hyperelliptic_affine_places.enumerate` returns every
`GF(p)`-rational affine point on a supported odd-characteristic squarefree
model `y^2=f(x)`. Each point is returned as the existing typed
`HyperellipticAffinePlace`, retaining its exact function field, coordinates,
residue field `GF(p)`, and local parameter. The values can be passed unchanged
to `function_field.hyperelliptic_affine_place.valuation.compute`.

The result is complete for affine points with rational coordinates. It does not
enumerate points at infinity or places with larger residue fields, and so it is
not a complete place enumeration for the global function field. The operation
scans each `x` in `GF(p)`, evaluates `f(x)`, and returns the one or two square
roots in `GF(p)` (or none). Its work is bounded by `p` times the polynomial
length, and its output contains at most `2p` places.

The affine/projective distinction follows the standard hyperelliptic model:
for odd degree, the smooth projective curve also has a rational point at
infinity, so an affine scan must not claim to enumerate all global places.
See [Stoll, *Rational points on hyperelliptic curves*](https://www.mathe2.uni-bayreuth.de/stoll/papers/CAR-2014-01.pdf).
