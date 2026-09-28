# Active terms of a tropical polynomial

`tropical.polynomial.active_terms.compute` evaluates a canonical sparse
tropical polynomial at one exact labelled point and returns every source term
attaining the result. `MIN_PLUS` selects the minimum and `MAX_PLUS` selects the
maximum under the polynomial's own semiring. Ties are retained.

Each witness carries the zero-based position in the canonical source term
tuple, the complete source term, and that monomial's exact value at the point.
The result also retains the unchanged polynomial and point, so indices are
bound to the same variable axis and canonical term order. A coordinate at the
licensed tropical infinity makes a monomial with positive exponent in that
coordinate unreachable. If no monomial is reachable, or the source polynomial
is zero, the result is the semiring's additive identity with no active terms.

Admission checks the source and point, bounds all monomial rational growth and
aggregate digit-weighted arithmetic before evaluation, then bounds the complete
source-bound witness size before expansion. Work is limited to 250,000,000
digit-product units and the serialized result estimate to 12 MiB. The current
polynomial limits also cap terms at 512, variables at 128, and exponents at
1,024. Requests whose conservative intermediate bound exceeds the scalar
carrier are rejected before monomial arithmetic.

This operation supplies indexed source-term witnesses. The existing
`tropical.polynomial.evaluate.compute` remains the smaller projection that
returns the value and active exponent vectors.
