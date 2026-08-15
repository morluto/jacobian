# Repair a triplewise-empty extremal bound

The frozen source claims the maximum family size is `2n`. Audit that argument for distinct subsets, including the empty set.

Submit the corrected general formula, the incidence-budget facts used for the upper bound, and extremal families on ground sets of sizes 7, 8, and 11. Represent each subset as a list of distinct zero-based elements; element order is not scored. The verifier independently checks uniqueness, range, triplewise-empty intersections, element frequencies, and the claimed maximum size.

The general symbolic incidence argument is replayed through its typed certificate and the finite constructions are checked directly.

Write one JSON submission to `/app/submission.json` using the schema in
`/app/submission_schema.json`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
