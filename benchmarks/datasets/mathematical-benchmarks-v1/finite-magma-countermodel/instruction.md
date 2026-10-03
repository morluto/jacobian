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

The verifier replays the task-specific mathematical predicate from the submitted result. The order must be one of the public input search_orders (1 or 2). For order n, table must have exactly n rows and n columns, and each table entry and each refuting_assignment coordinate x and y must be an integer from 0 through n-1. minimality_checked_orders must list exactly the public search_orders smaller than n, in their published order: [] for order 1 and [1] for order 2. These structural requirements do not establish the premise, target refutation, or minimality; the verifier checks those mathematical claims.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
