# Rational coordinate-map composition

[Polynomial operations](index.md) · [Operation references](../index.md)

`rational_function_map.compose.compute` returns the exact composite of two
canonical maps over `QQ` while retaining the source-bound domain of the
construction.

For

\[
G : X=(x_1,\ldots,x_n) \to Y=(y_1,\ldots,y_m), \qquad
F : Y \to Z=(z_1,\ldots,z_r),
\]

the request is valid only when `G.target_coordinates` and
`F.source_variables` are equal as ordered axes. The result contains `outer`,
`inner`, and `composite = F o G`; the composite has the source variables of
`G` and target coordinates of `F`.

Its `construction_locus_guard` is a canonical ordered tuple of nonconstant
monic rational polynomials in the source variables. Every guard is required to
be nonzero. The tuple records the conjunction of:

- all nonconstant denominators supplied by the components of `G`; and
- the numerator of every normalized nonconstant outer denominator after
  substitution into `G`.

This is the pre-cancellation locus. If a common factor cancels from a
composite component, the guard remains, because the returned value only claims
the pointwise identity `F(G(x))` on the locus where both supplied maps and all
substitution denominators were defined. A substituted outer denominator that
is identically zero is a domain error, not a cancelled result.

The operation admits source, substitution, denominator-clearing, exact
normalization, cancellation, guard, intermediate, work, coefficient-height,
term-support, and complete-result bounds before invoking the maintained exact
polynomial backend. It does not compute inverses, images, fibers, dominance,
Jacobians, projective atlas gluing, or analytic domains.

The native equivalent is:

```python
from jacobian.math.polynomials.rational_functions.composition import compose_maps

result = compose_maps(outer, inner)
```

Native callers supply `RationalFunctionMap` values directly. The Pydantic
`RationalMapCompositionRequest` is the transport request model used by the
catalog adapter.
