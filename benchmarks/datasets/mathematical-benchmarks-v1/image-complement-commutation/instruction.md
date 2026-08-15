# Classify direct-image/complement commutation

For each frozen finite mapping, determine whether `f(S \ X) = T \ f(X)` for every subset `X` of the domain. Classify the mapping as `BIJECTIVE`, `INJECTIVE_NOT_SURJECTIVE`, or `SURJECTIVE_NOT_INJECTIVE`; report the complete number of subsets checked; and, when commutation fails, give the first failing subset in increasing bitmask order together with both unequal sides. For a commuting case, all three failure fields must be null.

The three cases may be reported in any order; the verifier matches rows by case `id`. The set-valued arrays `first_failure`, `left_image`, and `right_complement` may list their integer elements in any order; the verifier compares them as sorted sets. The `commutes` field must be a JSON boolean (not `0`/`1`), and every integer field must be a JSON integer (not `true`/`false`).

The single task-specific witness file at `evidence/image-complement-certificate.json` must be a JSON object with exactly three keys: `schema_version` (the string `"1"`), `task_id` (the task identifier), and `result` (an exact copy of the submission `result`). The task-specific witness file must not exceed 16 MiB. The verifier exhaustively replays the powerset semantics; sampled subsets are incomplete.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/image-complement-certificate.json`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `application/json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
