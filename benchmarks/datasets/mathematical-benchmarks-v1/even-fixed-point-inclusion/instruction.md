# Certify permutations with no even fixed point

Count permutations of `{1,...,8}` in which none of `2,4,6,8` occupies its natural position.

Submit the five signed inclusion–exclusion terms, their sum, and the complete histogram giving the number of permutations with exactly `k=0,...,4` even fixed points. The verifier independently enumerates all `8!` permutations and separately recomputes the symbolic inclusion–exclusion terms. A scalar answer alone is incomplete.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
