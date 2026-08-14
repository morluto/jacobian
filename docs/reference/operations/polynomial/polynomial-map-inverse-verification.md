# Polynomial-map inverse synthesis and verification

`polynomial.map.inverse.candidate_synthesize` searches one finite polynomial
ansatz over `QQ`. The request fixes the forward map, ordered source and target
variables, inverse degree bound, either explicit coordinate supports or the
deterministically generated full total-degree support, the `sympy.solve`
backend, and explicit timeout, unknown, equation, degree, and residual-term
limits.

The synthesis artifact records the complete ordered support and coefficient
symbols, every exact coefficient equation from both compositions, solver
provenance, and—when found—the candidate and both residual families. Supported
statuses are:

- `FOUND`;
- `NO_CANDIDATE_WITHIN_ANSATZ`;
- `UNDERDETERMINED`;
- `TIMEOUT`;
- `BUDGET_EXHAUSTED`;
- `UNSUPPORTED`.

`NO_CANDIDATE_WITHIN_ANSATZ` means only that the declared finite coefficient
system has no solution. It never proves noninvertibility. Synthesis alone
creates no verification record; every found candidate is submitted to
`polynomial.map.inverse.verify`, and the synthesis result records either that
verifier's certificate/output or an explicit verification failure.

`polynomial.map.inverse.verify` verifies a proposed inverse of a square sparse
polynomial map over `QQ`. It is a verification operation, not an inverse
search or synthesis operation.

The request supplies:

- a forward map whose ordered input variables equal `source_variables`;
- an inverse map whose ordered input variables equal `target_variables`;
- explicit source and target variable orders of the same dimension.

The adapter computes and stores both residual families:

1. `inverse_after_forward`, in the source-variable ring;
2. `forward_after_inverse`, in the target-variable ring.

Every residual coordinate is sent through `polynomial.identity.verify` against
zero, and the resulting checker-record URIs are bound into the residual
artifact and the aggregate certificate. The authorized aggregate checker does
not trust those records as a substitute for checking: in a clean process it
parses both source maps, recomputes both compositions using independent sparse
rational arithmetic, compares every declared residual exactly, and accepts an
inverse only if every residual in both directions is zero.

The output binds the two source-map artifacts, coefficient domain, both
variable orders, both residual families, both checker-record families, the
claim, certificate, and aggregate verification record. A nonzero residual
produces a verified `FALSE`; malformed, substituted, incomplete, or
inconsistently ordered evidence fails closed as `UNKNOWN`.

The inverse operation accepts source and target dimensions through four,
canonical reduced rational coefficients, and monomials in the declared variable
order. The shared `RationalPolynomialMap` representation is more general: it
can represent rectangular maps and does not turn these inverse-specific bounds
into universal validity rules. This request is rejected before artifact creation
when conservative composition bounds exceed 1,024 residual terms or total
degree 127; these are operation costs for both the producer and independent
replay.
The v1 synthesis and verification contracts support polynomial maps over `QQ`
only. Rational-map inverses, denominator ansatzes, and pole-domain reasoning
are outside scope.
