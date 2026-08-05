# Audit an autoformalized orthogonality statement

Compare the frozen informal theorem, its reference formalization, and a proposed
Lean formalization. Determine whether the proposal preserves the dimension
premise and the meaning of the dot product.

Supply exact integer-vector certificates for both semantic defects:

1. a one-dimensional instance showing why omitting `k ≥ 2` changes the claim;
2. a two-dimensional instance with a nonzero vector orthogonal in the genuine
   dot-product sense, but not coordinatewise annihilated.

Do not claim that either Lean declaration compiles: no Lean runtime is part of
this task. Write `submission.json` to the exact agent-visible
`submission_schema.json`. Put a concise audit in `evidence/answer.txt`, include a
`RESULT_JSON:` line containing the submitted result as JSON, and bind that file
with its SHA-256 digest.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `PROPOSED_FORMALIZATION_FAITHFUL`, `PROPOSED_FORMALIZATION_NOT_FAITHFUL`, `INSUFFICIENT_EVIDENCE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** a string value
- **Completeness:** one of `COMPLETE`, `PARTIAL`, `UNKNOWN`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
