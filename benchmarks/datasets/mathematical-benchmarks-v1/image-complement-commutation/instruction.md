# Classify direct-image/complement commutation

For each frozen finite mapping, determine whether `f(S \ X) = T \ f(X)` for every subset `X` of the domain. Classify the mapping as `BIJECTIVE`, `INJECTIVE_NOT_SURJECTIVE`, or `SURJECTIVE_NOT_INJECTIVE`; report the complete number of subsets checked; and, when commutation fails, give the first failing subset in increasing bitmask order together with both unequal sides. For a commuting case, all three failure fields must be null.

The three cases may be reported in any order; the verifier matches rows by case `id`. The set-valued arrays `first_failure`, `left_image`, and `right_complement` may list their integer elements in any order; the verifier compares them as sorted sets. The `commutes` field must be a JSON boolean (not `0`/`1`), and every integer field must be a JSON integer (not `true`/`false`).

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
