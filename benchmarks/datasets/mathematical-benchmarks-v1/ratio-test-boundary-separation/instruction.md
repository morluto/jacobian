# Separate the two outcomes at the ratio-test boundary

Use the harmonic series `1/n` as the divergent witness and the telescoping
series `1/(n*(n+1))` as the convergent witness. Both are positive rational
sequences indexed by `n >= 1` whose consecutive-term ratios tend to one, but
the harmonic series diverges while the telescoping series converges.

Supply the exact ratio and ratio-error identities for each witness, nine
dyadic lower-bound blocks for the divergent witness, and at least four freely
chosen partial-sum checkpoints for the convergent witness.

The verifier independently evaluates every rational checkpoint and replays the
submitted symbolic identities. A conclusion label or numerical sampling alone
is insufficient: the result must demonstrate why a ratio limit equal to one is
inconclusive through both outcomes.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
