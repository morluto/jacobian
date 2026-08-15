# Audit an expectation claim

Audit the claim in the offline input using exact arithmetic. Account explicitly
for the dependence in `f(f(x))` when `f(x)=x`; return the relevant exact point
probabilities, the ordered squared-difference sum, and the exact expectation.
Represent each rational as an integer `numerator` and positive integer
`denominator` in lowest terms.

Write `submission.json` to the exact agent-visible `submission_schema.json`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
