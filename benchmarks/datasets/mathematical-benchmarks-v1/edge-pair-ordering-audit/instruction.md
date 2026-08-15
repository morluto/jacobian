# Audit edge-pair ordering in a graph sum

Read `/app/input.json`, including its complete definition of the edge-pair polynomials, the summation domain, and the source derivation being audited. Determine the coefficient of `x` in their sum over all labeled simple graphs, respecting the declared pair semantics. Diagnose the ordered-versus-unordered counting mismatch from the supplied source derivation; do not introduce a different polynomial family.

Write `/app/submission.json` using `/app/submission_schema.json`. Give the
symbolic incident-pair factor, the number of free graph-edge bits, the corrected
formula, and exact coefficients for every requested `n`. The symbolic formula
applies for `n>=3`, where at least one distinct incident edge pair exists; for
`n<3` the coefficient is zero.

The verifier independently enumerates every labeled graph for `n=3,4,5,6` and
its ordered edge pairs, and separately checks the symbolic factors.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
