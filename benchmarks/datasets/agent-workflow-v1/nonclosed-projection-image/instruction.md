# Construct a closed subspace with nonclosed projected image

Give an explicit Hilbert space `H`, closed linear subspace `M`, and orthogonal
projection `P` such that `P(M)` is not closed.

Certify every link of the counterexample: the mechanism establishing that `M`
is closed, identification of the projection image `P(M)`, a sequence inside
that image converging to a declared limit, a tail bound proving convergence,
and a proof that the limit has no `ell2` preimage.  For every prefix through
`prefix_length`, submit the exact rational weight, limit coordinate, forced
preimage coordinate, and partial squared norms of the limit and its forced
preimage.

Put the five explanations in the agent-visible `result.proof_obligations`
fields (`boundedness`, `closedness`, `range_identification`, `convergence`, and
`absent_preimage`). The evidence file is a human-readable inspectability
artifact summarizing boundedness, closedness, the projected range, the tail
bound, and the divergent forced preimage; it does not require hidden JSON
marker lines.

A finite-dimensional matrix, a bare theorem citation, or a single approximate
sequence is insufficient.  Fractions must be canonical.  Bind
`/app/evidence/answer.txt` and do not claim `VERIFIED`.
