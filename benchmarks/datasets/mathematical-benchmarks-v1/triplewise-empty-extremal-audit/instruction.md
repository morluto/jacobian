# Repair a triplewise-empty extremal bound

The frozen source claims the maximum family size is `2n`. Audit that argument for distinct subsets, including the empty set.

Submit the corrected general formula and extremal families on ground sets of sizes 7, 8, and 11. Represent each subset as a list of distinct zero-based elements; element order is not scored. The verifier derives the general upper bound from the submitted constructions and formula: it checks uniqueness, range, triplewise-empty intersections, element frequencies, and that the formula matches the construction sizes at 7, 8, and 11.

The finite constructions are checked directly. Do not submit a separate incidence-budget certificate; the verifier does not score a tautological bound object.

Write one JSON submission to `/app/submission.json` using the schema in
`/app/submission_schema.json`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
