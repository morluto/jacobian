# Continuous spike integral/series separation

This Regression benchmark freezes CounterMATH test row 635 at revision
`d4e9f8ca877552f4491a9c2d52e0d230c0fca620`. The agent must construct a
strictly positive continuous function on `[1,infinity)` whose improper integral
diverges while its integer-sample series converges.

The accepted family adds disjoint triangular spikes to the baseline `x^-2`.
The rational width scale is not fixed. The clean-room verifier accepts every
canonical rational `0 < alpha <= 1/4`, reconstructs twelve spike supports and
areas, checks that no integer lies in a support, and validates the symbolic
harmonic-area and p-series separation.

Quality score: **86/100**. Difficulty is **Hard (provisional)**: an agent must
simultaneously preserve strict positivity and continuity, hide every spike from
integer samples, and force divergent total area. The public existential claim
does not reveal a usable certificate, and a finite table alone cannot establish
the two infinite conclusions.

The verifier trusts the standard harmonic- and p-series criteria and therefore
reports only `COMPUTED`. It does not machine-prove locally finite sums on the
real line or claim proof-assistant verification.

## Shortcut audit

Numeric quadrature, a finite set of spikes, and zero-valued integer samples are
insufficient. The baseline, free parameter, general support formula, twelve
exact instances, and both series classifications must agree independently.
