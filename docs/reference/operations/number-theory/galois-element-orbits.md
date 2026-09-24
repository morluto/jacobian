# Exact element orbits in supported quadratic splitting fields

`number_field.element.embedding_orbit.compute` takes an exact element of a
retained splitting field over `QQ` and returns its complete action under every
`QQ`-automorphism supported by that field. The current field carrier is limited
to `QQ` and quadratic extensions, so an accepted action contains one or two
maps.

The result retains the source element and each exact automorphism/image pair.
It also returns the distinct image orbit in deterministic automorphism order,
the subgroup fixing the source element, the orbit size, and the minimal
polynomial over `QQ`. Thus the map family remains usable by the automorphism,
subgroup, fixed-field, and element-application operations without reconstructing
maps from image labels.

For a quadratic field `QQ(alpha)` with defining polynomial
`alpha^2 + c1*alpha + c0`, an element `a+b*alpha` has trace
`2a-c1*b` and norm `a^2-c1*a*b+c0*b^2`. The returned polynomial is the
linear polynomial when the element is rational and the quadratic characteristic
polynomial otherwise. Exact rational arithmetic is used throughout; the
field, element-coordinate, automorphism-count, and polynomial coefficient
bounds are enforced before expansion.

Examples include `alpha=sqrt(2)`, whose orbit is `{sqrt(2), -sqrt(2)}` and
minimal polynomial is `x^2-2`, and the element `3` in the same field, whose
orbit has size one, stabilizer is the full group, and minimal polynomial is
`x-3`.

This operation does not construct embeddings into a larger normal closure,
return minimal polynomials relative to arbitrary intermediate fields, or
support fields above degree two. Those require broader exact field and map
carriers.
