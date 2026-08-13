# Polynomial-map Keller and inverse-obstruction verification

Jacobian exposes two narrow verification operations for the exact
polynomial-map moves that recur in published Jacobian-conjecture
counterexamples. They are claim binders, not a Jacobian-conjecture solver.

## Keller condition

`polynomial.map.keller_condition.verify` accepts one square sparse polynomial
map over `QQ`. It computes the map's Jacobian through the existing exact
polynomial producer, then creates a claim and certificate bound to the exact
map and Jacobian artifacts. The independent checker replays every partial
derivative and the determinant with sparse rational arithmetic and decides
whether the determinant is a nonzero constant.

The result distinguishes a verified `TRUE` condition from a verified `FALSE`
condition. A malformed or substituted map/Jacobian/certificate is rejected as
`UNKNOWN`; no failed replay is interpreted as a conjecture disproof. The
checked claim is only over the registered `QQ` map semantics, with the
existing dimension, term, exponent, and determinant-expansion bounds.

The checker does not certify injectivity, surjectivity, polynomial invertibility,
or any statement over arbitrary fields. A nonzero constant determinant is the
Keller condition only. It is an input fact for further mathematical reasoning,
not the conclusion of the Jacobian conjecture.

## Collision-to-inverse obstruction

`polynomial.map.inverse.refute_by_collision` accepts a map, two distinct exact
rational points, and a claimed common image. Its independent witness checker
re-evaluates both points and binds the collision to the exact claim
`POLYNOMIAL_MAP_NO_TWO_SIDED_INVERSE` over `QQ`.

The logical bridge is elementary and explicit: a two-sided polynomial inverse
would make the map injective, so two distinct points with the same image rule
out such an inverse for this exact map and domain. The operation does not
claim that a bounded search failed, that no rational inverse exists under some
other semantics, or that the original conjecture is false in every field.

## Evidence and independence

Both operations preserve the source map, claim, candidate/witness,
certificate, and verification record. Checker authorization remains in the
operator-managed registry. The producer uses the pinned SymPy installation;
the checker uses a separate exact sparse-rational implementation in
`jacobian_checkers.polynomial_maps` and runs in the bounded checker worker.

The public reference fixture
`benchmarks/datasets/public-reproductions-v1/jacobian-inverse-obstruction/` binds the
published three-variable map, its determinant `-2`, a rational collision, and
a wrong-image negative control. It is scored false and is intended for
reproduction and contract regression, not for a hidden model-evaluation claim.

The v1 contract deliberately does not generalize to arbitrary characteristic,
complex-algebraic coordinates, Dixmier/Poisson transfer, fiber classification,
or symmetry-class minimality. Those require new domain semantics and separate
bounded completeness obligations.
