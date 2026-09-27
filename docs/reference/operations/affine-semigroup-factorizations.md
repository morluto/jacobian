# Parent-bound affine-semigroup factorizations

`affine_semigroup.factorization.evaluate` maps one nonnegative coefficient
vector on a positive affine semigroup's ordered generator axis to its exact
ambient vector. Its result is an `AffineFactorization` that retains the
semigroup parent, coefficients, and derived target. This is a reusable typed
value for downstream operations; a bare tuple has no generator-axis or parent
identity.

For a labelled configuration with columns `a_i`, the operation returns
`(S,u,Au)`, where `S` is the retained positive semigroup and `u` is the
coefficient vector. The result does not claim uniqueness, primitivity, or
completeness of the target's fiber. Use `affine_semigroup.factorizations.compute`
to enumerate that complete fiber within its separate admission envelope.
This follows the standard factorization homomorphism
`phi_A: N^n -> S`, `u |-> A u`; the defining map and factorization set are stated
for affine semigroups in García-Sánchez, Ojeda, and Rosales, [*Affine semigroups
having a unique Betti element*](https://arxiv.org/abs/1203.4138), Section 1.

The exact matrix product is computed once. A decoded caller-supplied
`AffineFactorization` has its claimed relation checked when validated, while
the trusted operation constructor does not replay the product. The operation
admits at most 8 ambient rows and 10 generators, 32 decimal digits per
coefficient, an arithmetic work estimate, and the corresponding exact output
digit bound before multiplication.

For generators `(1,-1)`, `(-1,2)`, and `(0,1)`, coefficients `(2,3,4)` map
to `(-1,8)`. The positive grading `(3,2)` certifies that all three generators
belong to a positive affine semigroup even though one ambient coordinate is
negative.
