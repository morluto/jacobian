# Smallest finite-magma countermodel

The offline input states a universally quantified premise and target identity
for one binary operation. Decide whether the premise implies the target over
nonempty finite magmas. If not, return a smallest countermodel, a valuation
that refutes the target, and the smaller carrier orders exhaustively checked.

Write `submission.json` to the exact agent-visible `submission_schema.json`.
The submitted operation, refuting assignment, and smaller-carrier check are the
certificate.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
