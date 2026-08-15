# Discover and replay an inversion-complementing involution

For permutations of `1..7`, select one of the three declared transformations and use it to certify the total inversion count over all `7!` permutations. Submit the selected transformation, its exact value-map parameters, six distinct freely chosen permutation traces with transformed permutations and both inversion counts, the constant pair sum, fixed-point count, pair count, and total inversion sum.


<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/permutation-involution-certificate.json`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `application/json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
