# Closed-set distance strengthening audit

Public answer-visible reproduction of ProofNetVerif test row 18. The natural theorem says
that disjoint closed subsets of a metric space are separated. The rejected Lean prediction silently
strengthens this to a uniform positive distance, which requires an additional compactness hypothesis.

The task requires a parameterized, locally finite rational-plane counterexample family and several
freely chosen epsilon witnesses. The verifier reconstructs every point and distance exactly, checks
the quantitative zero-infimum witnesses, and distinguishes the valid separation conclusion from the
unsupported metric strengthening.

This case is intentionally not a hidden-answer operation measurement; its canonical family is
published in the instruction and solution for reproducible contract and verifier checks.

Quality score: **89/100**. Difficulty is provisional pending baseline calibration. Assurance remains
`COMPUTED`: the standard theorem that locally finite subsets of Euclidean space are closed is trusted.
