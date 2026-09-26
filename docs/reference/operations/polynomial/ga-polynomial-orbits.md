# Additive-group polynomial orbits

`algebraic_group.ga.polynomial_orbit.compute` applies a checked polynomial
`G_a` coaction to any source polynomial in the action's ordered `QQ` ring. If
the action parameter is `t`, the result is the exact polynomial
`rho(f) = f(rho(x_1), ..., rho(x_n))` in the explicitly ordered ring
`QQ[x_1, ..., x_n, t]`. It returns the existing `RationalPolynomial` value so
the orbit polynomial can be supplied directly to later exact polynomial
operations.

The action's counit and additive composition law are rechecked before
substitution, since they establish that the supplied generator images define a
`G_a` action. Source and action variable axes must be identical, including
order; the result appends the action parameter as the final axis. This is the
coordinate-ring coaction for the action associated to a locally nilpotent
derivation, whose generator images are `exp(tD)(x_i)`.

Admission bounds source terms, total degree, the complete Cartesian monomial
expansion, repeated-product work, intermediate allocation, coefficient height,
result exponents, and serialized output before expansion. A resource refusal
does not return a truncated orbit.

The algebraic basis is the coordinate-ring description of a `G_a` action as a
coaction `rho: R -> R[t]` obeying the counit and additive coassociativity laws;
see Freudenburg, [*Algebraic Theory of Locally Nilpotent Derivations*](https://link.springer.com/book/10.1007/978-3-662-55350-3), §1.3.
